from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from elasticsearch import Elasticsearch, helpers
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import os
import logging
import re
from typing import Generator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize embedding model (all-MiniLM-L6-v2 produces 384-dim embeddings)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMS = 384
embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy load the embedding model."""
    global embedding_model
    if embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully")
    return embedding_model


class TextChunker:
    """Text chunker for splitting documents into smaller chunks."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        if not text or len(text.strip()) < self.min_chunk_size:
            return [text.strip()] if text and text.strip() else []
        
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If text is smaller than chunk size, return as is
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If not at the end, try to break at a sentence or word boundary
            if end < len(text):
                # Try to find a sentence boundary
                last_period = text.rfind('.', start, end)
                last_question = text.rfind('?', start, end)
                last_exclaim = text.rfind('!', start, end)
                
                sentence_end = max(last_period, last_question, last_exclaim)
                
                if sentence_end > start + self.min_chunk_size:
                    end = sentence_end + 1
                else:
                    # Fall back to word boundary
                    last_space = text.rfind(' ', start, end)
                    if last_space > start + self.min_chunk_size:
                        end = last_space
            
            chunk = text[start:end].strip()
            if chunk and len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap if end < len(text) else len(text)
        
        return chunks


def generate_embeddings(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Generate embeddings for a list of texts."""
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return embeddings.tolist()

app = FastAPI(
    title="Elasticsearch Data Loader",
    description="Service to manage Elasticsearch data and load MS MARCO dataset",
    version="1.0.0"
)

# Add CORS middleware for Web UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Elasticsearch configuration
ES_HOST = os.getenv("ELASTICSEARCH_HOST", "elasticsearch")
ES_PORT = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
ES_INDEX = os.getenv("ELASTICSEARCH_INDEX", "msmarco")

# Track loading status
loading_status = {
    "is_loading": False,
    "progress": 0,
    "total": 0,
    "message": "Idle"
}


def get_es_client() -> Elasticsearch:
    """Create and return Elasticsearch client."""
    return Elasticsearch(
        hosts=[{"host": ES_HOST, "port": ES_PORT, "scheme": "http"}],
        request_timeout=30
    )


class LoadConfig(BaseModel):
    index_name: str = "msmarco"
    dataset_split: str = "train"
    max_documents: int | None = None
    batch_size: int = 500
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_batch_size: int = 32


class ESCILoadConfig(BaseModel):
    index_name: str = "products"
    max_documents: int | None = None
    batch_size: int = 500
    embedding_batch_size: int = 32
    locale: str = "us"  # us, es, jp


class ProductSearchConfig(BaseModel):
    index_name: str = "product-corpus"
    max_documents: int | None = None
    batch_size: int = 500
    embedding_batch_size: int = 32


class ChunkConfig(BaseModel):
    text: str
    chunk_size: int = 512
    chunk_overlap: int = 50


class EmbeddingRequest(BaseModel):
    texts: list[str]
    batch_size: int = 32


class IndexConfig(BaseModel):
    index_name: str = "msmarco"


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "elasticsearch-dataloader"}


@app.get("/health")
def health_check():
    """Check service and Elasticsearch health."""
    try:
        es = get_es_client()
        es_health = es.cluster.health()
        return {
            "service": "healthy",
            "elasticsearch": {
                "status": es_health["status"],
                "cluster_name": es_health["cluster_name"],
                "number_of_nodes": es_health["number_of_nodes"]
            }
        }
    except Exception as e:
        return {
            "service": "healthy",
            "elasticsearch": {"status": "unavailable", "error": str(e)}
        }


@app.get("/status")
def get_loading_status():
    """Get current data loading status."""
    return loading_status


@app.post("/chunk")
def chunk_text(config: ChunkConfig):
    """Chunk text into smaller pieces with overlap."""
    try:
        chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        chunks = chunker.chunk_text(config.text)
        return {
            "status": "success",
            "chunk_count": len(chunks),
            "chunks": chunks
        }
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed")
def embed_texts(request: EmbeddingRequest):
    """Generate embeddings for a list of texts."""
    try:
        embeddings = generate_embeddings(request.texts, request.batch_size)
        return {
            "status": "success",
            "model": EMBEDDING_MODEL_NAME,
            "dimensions": EMBEDDING_DIMS,
            "count": len(embeddings),
            "embeddings": embeddings
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
def get_model_info():
    """Get information about the embedding model."""
    return {
        "model_name": EMBEDDING_MODEL_NAME,
        "dimensions": EMBEDDING_DIMS,
        "model_loaded": embedding_model is not None
    }


@app.post("/model/load")
def preload_model():
    """Pre-load the embedding model into memory."""
    try:
        get_embedding_model()
        return {
            "status": "success",
            "message": "Model loaded successfully",
            "model_name": EMBEDDING_MODEL_NAME
        }
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear")
def clear_all_data():
    """Clear all data from Elasticsearch (delete all indices)."""
    try:
        es = get_es_client()
        # Get all indices except system indices
        indices = es.indices.get(index="*")
        deleted_indices = []
        
        for index_name in indices:
            if not index_name.startswith("."):
                es.indices.delete(index=index_name)
                deleted_indices.append(index_name)
                logger.info(f"Deleted index: {index_name}")
        
        return {
            "status": "success",
            "message": "All data cleared",
            "deleted_indices": deleted_indices
        }
    except Exception as e:
        logger.error(f"Error clearing data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear/{index_name}")
def clear_index(index_name: str):
    """Clear a specific index from Elasticsearch."""
    try:
        es = get_es_client()
        
        if not es.indices.exists(index=index_name):
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
        
        es.indices.delete(index=index_name)
        logger.info(f"Deleted index: {index_name}")
        
        return {
            "status": "success",
            "message": f"Index '{index_name}' deleted"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clearing index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/indices")
def list_indices():
    """List all indices in Elasticsearch."""
    try:
        es = get_es_client()
        indices = es.cat.indices(format="json")
        return {
            "indices": [
                {
                    "name": idx["index"],
                    "health": idx["health"],
                    "docs_count": idx["docs.count"],
                    "store_size": idx["store.size"]
                }
                for idx in indices
                if not idx["index"].startswith(".")
            ]
        }
    except Exception as e:
        logger.error(f"Error listing indices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def load_msmarco_background(config: LoadConfig):
    """Background task to load MS MARCO dataset with embeddings."""
    global loading_status
    
    try:
        loading_status["is_loading"] = True
        loading_status["message"] = "Initializing..."
        loading_status["progress"] = 0
        
        es = get_es_client()
        
        # Pre-load embedding model
        loading_status["message"] = "Loading embedding model..."
        logger.info("Loading embedding model...")
        model = get_embedding_model()
        
        # Initialize chunker
        chunker = TextChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        # Create index with vector field mapping
        loading_status["message"] = "Creating index with vector mapping..."
        index_mapping = {
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "passage_id": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": "standard"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIMS,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "query_id": {"type": "keyword"},
                    "query_text": {"type": "text", "analyzer": "standard"},
                    "is_selected": {"type": "boolean"},
                    "chunk_index": {"type": "integer"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "30s"
            }
        }
        
        if es.indices.exists(index=config.index_name):
            es.indices.delete(index=config.index_name)
        
        es.indices.create(index=config.index_name, body=index_mapping)
        logger.info(f"Created index with vector field: {config.index_name}")
        
        # Load MS MARCO dataset
        loading_status["message"] = "Loading MS MARCO dataset from Hugging Face..."
        logger.info("Loading MS MARCO dataset...")
        
        dataset = load_dataset("ms_marco", "v1.1", split=config.dataset_split, streaming=True)
        
        # Process and index documents with embeddings
        loading_status["message"] = "Processing and embedding documents..."
        
        batch_texts = []
        batch_docs = []
        doc_count = 0
        chunk_count = 0
        query_count = 0
        
        for item in dataset:
            if config.max_documents and query_count >= config.max_documents:
                break
            
            query_id = str(item.get("query_id", query_count))
            query_text = item.get("query", "")
            passages = item.get("passages", {})
            
            passage_texts = passages.get("passage_text", [])
            is_selected_list = passages.get("is_selected", [])
            
            for idx, passage_text in enumerate(passage_texts):
                is_selected = is_selected_list[idx] if idx < len(is_selected_list) else 0
                
                # Chunk the passage text
                chunks = chunker.chunk_text(passage_text)
                
                for chunk_idx, chunk_text in enumerate(chunks):
                    batch_texts.append(chunk_text)
                    batch_docs.append({
                        "chunk_id": f"{query_id}_{idx}_{chunk_idx}",
                        "passage_id": f"{query_id}_{idx}",
                        "text": chunk_text,
                        "query_id": query_id,
                        "query_text": query_text,
                        "is_selected": bool(is_selected),
                        "chunk_index": chunk_idx
                    })
                    
                    # Process batch when full
                    if len(batch_texts) >= config.embedding_batch_size:
                        # Generate embeddings for batch
                        embeddings = model.encode(
                            batch_texts,
                            batch_size=config.embedding_batch_size,
                            show_progress_bar=False,
                            convert_to_numpy=True
                        )
                        
                        # Create bulk actions
                        actions = []
                        for doc, embedding in zip(batch_docs, embeddings):
                            doc["embedding"] = embedding.tolist()
                            actions.append({
                                "_index": config.index_name,
                                "_source": doc
                            })
                        
                        # Bulk index
                        helpers.bulk(
                            es,
                            actions,
                            chunk_size=config.batch_size,
                            request_timeout=120,
                            raise_on_error=False
                        )
                        
                        chunk_count += len(batch_texts)
                        batch_texts = []
                        batch_docs = []
            
            query_count += 1
            doc_count += len(passage_texts)
            
            if query_count % 50 == 0:
                loading_status["progress"] = query_count
                loading_status["message"] = f"Processed {query_count} queries, {chunk_count} chunks indexed..."
                logger.info(f"Processed {query_count} queries, {chunk_count} chunks")
        
        # Process remaining batch
        if batch_texts:
            embeddings = model.encode(
                batch_texts,
                batch_size=config.embedding_batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            actions = []
            for doc, embedding in zip(batch_docs, embeddings):
                doc["embedding"] = embedding.tolist()
                actions.append({
                    "_index": config.index_name,
                    "_source": doc
                })
            
            helpers.bulk(
                es,
                actions,
                chunk_size=config.batch_size,
                request_timeout=120,
                raise_on_error=False
            )
            chunk_count += len(batch_texts)
        
        # Refresh index
        es.indices.refresh(index=config.index_name)
        
        loading_status["message"] = f"Completed! Indexed {chunk_count} chunks from {query_count} queries"
        loading_status["progress"] = query_count
        logger.info(f"Indexing complete: {chunk_count} chunks from {query_count} queries")
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        loading_status["message"] = f"Error: {str(e)}"
    finally:
        loading_status["is_loading"] = False


@app.post("/load/msmarco")
def load_msmarco(config: LoadConfig, background_tasks: BackgroundTasks):
    """
    Load MS MARCO dataset into Elasticsearch.
    
    This runs as a background task. Use /status to check progress.
    """
    global loading_status
    
    if loading_status["is_loading"]:
        raise HTTPException(
            status_code=409,
            detail="A loading operation is already in progress"
        )
    
    background_tasks.add_task(load_msmarco_background, config)
    
    return {
        "status": "started",
        "message": "MS MARCO dataset loading started",
        "config": config.model_dump(),
        "check_progress": "/status"
    }


@app.get("/search/{index_name}")
def search(index_name: str, q: str, size: int = 10):
    """Search documents in the specified index."""
    try:
        es = get_es_client()
        
        if not es.indices.exists(index=index_name):
            raise HTTPException(status_code=404, detail=f"Index '{index_name}' not found")
        
        result = es.search(
            index=index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": q,
                        "fields": ["text", "query_text"]
                    }
                },
                "size": size
            }
        )
        
        return {
            "total": result["hits"]["total"]["value"],
            "search_type": "text",
            "hits": [
                {
                    "score": hit["_score"],
                    "source": {k: v for k, v in hit["_source"].items() if k != "embedding"}
                }
                for hit in result["hits"]["hits"]
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class VectorSearchRequest(BaseModel):
    query: str
    index_name: str = "msmarco"
    size: int = 10
    num_candidates: int = 100


@app.post("/search/vector")
def vector_search(request: VectorSearchRequest):
    """
    Perform semantic vector search using the query embedding.
    
    This uses kNN search with cosine similarity.
    """
    try:
        es = get_es_client()
        
        if not es.indices.exists(index=request.index_name):
            raise HTTPException(status_code=404, detail=f"Index '{request.index_name}' not found")
        
        # Generate embedding for the query
        model = get_embedding_model()
        query_embedding = model.encode(request.query, convert_to_numpy=True).tolist()
        
        # Perform kNN search
        result = es.search(
            index=request.index_name,
            body={
                "knn": {
                    "field": "embedding",
                    "query_vector": query_embedding,
                    "k": request.size,
                    "num_candidates": request.num_candidates
                },
                "_source": {
                    "excludes": ["embedding"]
                }
            }
        )
        
        return {
            "total": result["hits"]["total"]["value"],
            "search_type": "vector",
            "model": EMBEDDING_MODEL_NAME,
            "hits": [
                {
                    "score": hit["_score"],
                    "source": hit["_source"]
                }
                for hit in result["hits"]["hits"]
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class HybridSearchRequest(BaseModel):
    query: str
    index_name: str = "msmarco"
    size: int = 10
    num_candidates: int = 100
    vector_boost: float = 0.7
    text_boost: float = 0.3


@app.post("/search/hybrid")
def hybrid_search(request: HybridSearchRequest):
    """
    Perform hybrid search combining vector similarity and text search.
    
    Uses RRF (Reciprocal Rank Fusion) to combine results.
    """
    try:
        es = get_es_client()
        
        if not es.indices.exists(index=request.index_name):
            raise HTTPException(status_code=404, detail=f"Index '{request.index_name}' not found")
        
        # Generate embedding for the query
        model = get_embedding_model()
        query_embedding = model.encode(request.query, convert_to_numpy=True).tolist()
        
        # Perform hybrid search with sub_searches and RRF
        result = es.search(
            index=request.index_name,
            body={
                "size": request.size,
                "query": {
                    "bool": {
                        "should": [
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": f"cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                        "params": {"query_vector": query_embedding}
                                    },
                                    "boost": request.vector_boost
                                }
                            },
                            {
                                "multi_match": {
                                    "query": request.query,
                                    "fields": ["text", "query_text"],
                                    "boost": request.text_boost
                                }
                            }
                        ]
                    }
                },
                "_source": {
                    "excludes": ["embedding"]
                }
            }
        )
        
        return {
            "total": result["hits"]["total"]["value"],
            "search_type": "hybrid",
            "model": EMBEDDING_MODEL_NAME,
            "vector_boost": request.vector_boost,
            "text_boost": request.text_boost,
            "hits": [
                {
                    "score": hit["_score"],
                    "source": hit["_source"]
                }
                for hit in result["hits"]["hits"]
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in hybrid search: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def load_esci_background(config: ESCILoadConfig):
    """Background task to load ESCI (Amazon Shopping Queries) dataset with embeddings."""
    global loading_status
    
    try:
        loading_status["is_loading"] = True
        loading_status["message"] = "Initializing ESCI dataset load..."
        loading_status["progress"] = 0
        
        es = get_es_client()
        
        # Pre-load embedding model
        loading_status["message"] = "Loading embedding model..."
        logger.info("Loading embedding model...")
        model = get_embedding_model()
        
        # Create index with product mapping
        loading_status["message"] = "Creating product index with vector mapping..."
        index_mapping = {
            "mappings": {
                "properties": {
                    "product_id": {"type": "keyword"},
                    "product_title": {"type": "text", "analyzer": "standard"},
                    "product_description": {"type": "text", "analyzer": "standard"},
                    "product_bullet_point": {"type": "text", "analyzer": "standard"},
                    "product_brand": {"type": "keyword"},
                    "product_color": {"type": "keyword"},
                    "product_locale": {"type": "keyword"},
                    "product_class": {"type": "keyword"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIMS,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "combined_text": {"type": "text", "analyzer": "standard"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "30s"
            }
        }
        
        if es.indices.exists(index=config.index_name):
            es.indices.delete(index=config.index_name)
        
        es.indices.create(index=config.index_name, body=index_mapping)
        logger.info(f"Created product index: {config.index_name}")
        
        # Load ESCI product catalog
        loading_status["message"] = "Loading Amazon ESCI product catalog from Hugging Face..."
        logger.info("Loading ESCI product catalog...")
        
        # Load only the product catalog (no queries/labels)
        dataset = load_dataset("tasksource/esci", "product_catalogue", split="train", streaming=True)
        
        # Process and index products with embeddings
        loading_status["message"] = "Processing and embedding products..."
        
        batch_texts = []
        batch_docs = []
        product_count = 0
        seen_products = set()  # Avoid duplicates
        
        for item in dataset:
            if config.max_documents and product_count >= config.max_documents:
                break
            
            # Filter by locale if specified
            product_locale = item.get("product_locale", "us")
            if config.locale and product_locale != config.locale:
                continue
            
            product_id = item.get("product_id", "")
            
            # Skip duplicates
            if product_id in seen_products:
                continue
            seen_products.add(product_id)
            
            product_title = item.get("product_title", "") or ""
            product_description = item.get("product_description", "") or ""
            product_bullet_point = item.get("product_bullet_point", "") or ""
            product_brand = item.get("product_brand", "") or ""
            product_color = item.get("product_color", "") or ""
            product_class = item.get("product_class", "") or ""
            
            # Create combined text for embedding (product catalog info only)
            combined_text = f"{product_title}. {product_brand}. {product_description} {product_bullet_point}".strip()
            
            if not combined_text or len(combined_text) < 10:
                continue
            
            doc = {
                "product_id": product_id,
                "product_title": product_title,
                "product_description": product_description,
                "product_bullet_point": product_bullet_point,
                "product_brand": product_brand,
                "product_color": product_color,
                "product_class": product_class,
                "product_locale": product_locale,
                "combined_text": combined_text
            }
            
            batch_texts.append(combined_text[:1000])  # Limit text length for embedding
            batch_docs.append(doc)
            product_count += 1
            
            # Update progress
            if product_count % 100 == 0:
                loading_status["message"] = f"Processing products... {product_count} processed"
                loading_status["progress"] = product_count
                loading_status["total"] = config.max_documents or 0
            
            # Process batch when full
            if len(batch_texts) >= config.embedding_batch_size:
                # Generate embeddings
                embeddings = model.encode(batch_texts, show_progress_bar=False)
                
                # Add embeddings to docs
                for i, doc in enumerate(batch_docs):
                    doc["embedding"] = embeddings[i].tolist()
                
                # Bulk index
                actions = []
                for doc in batch_docs:
                    actions.append({
                        "_index": config.index_name,
                        "_id": doc["product_id"],
                        "_source": doc
                    })
                
                helpers.bulk(
                    es,
                    actions,
                    chunk_size=config.batch_size,
                    request_timeout=120,
                    raise_on_error=False
                )
                
                batch_texts = []
                batch_docs = []
        
        # Process remaining batch
        if batch_texts:
            embeddings = model.encode(batch_texts, show_progress_bar=False)
            for i, doc in enumerate(batch_docs):
                doc["embedding"] = embeddings[i].tolist()
            
            actions = []
            for doc in batch_docs:
                actions.append({
                    "_index": config.index_name,
                    "_id": doc["product_id"],
                    "_source": doc
                })
            
            helpers.bulk(
                es,
                actions,
                chunk_size=config.batch_size,
                request_timeout=120,
                raise_on_error=False
            )
        
        # Refresh index
        es.indices.refresh(index=config.index_name)
        
        loading_status["message"] = f"Completed! Indexed {product_count} products"
        loading_status["progress"] = product_count
        logger.info(f"ESCI indexing complete: {product_count} products")
        
    except Exception as e:
        logger.error(f"Error loading ESCI dataset: {e}")
        loading_status["message"] = f"Error: {str(e)}"
    finally:
        loading_status["is_loading"] = False


@app.post("/load/esci")
def load_esci(config: ESCILoadConfig, background_tasks: BackgroundTasks):
    """
    Load ESCI (Amazon Shopping Queries) dataset into Elasticsearch.
    
    This runs as a background task. Use /status to check progress.
    """
    global loading_status
    
    if loading_status["is_loading"]:
        raise HTTPException(
            status_code=409,
            detail="A loading operation is already in progress"
        )
    
    loading_status = {
        "is_loading": True,
        "progress": 0,
        "total": config.max_documents or 0,
        "message": "Starting ESCI dataset load..."
    }
    
    background_tasks.add_task(load_esci_background, config)
    
    return {
        "status": "started",
        "message": "ESCI product data loading started in background",
        "index_name": config.index_name,
        "max_documents": config.max_documents,
        "locale": config.locale
    }


def load_product_search_background(config: ProductSearchConfig):
    """Background task to load Product Search Corpus dataset with embeddings."""
    global loading_status
    
    try:
        loading_status["is_loading"] = True
        loading_status["message"] = "Initializing Product Search Corpus load..."
        loading_status["progress"] = 0
        
        es = get_es_client()
        
        # Pre-load embedding model
        loading_status["message"] = "Loading embedding model..."
        logger.info("Loading embedding model...")
        model = get_embedding_model()
        
        # Create index with product mapping
        loading_status["message"] = "Creating product index with vector mapping..."
        index_mapping = {
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "title": {"type": "text", "analyzer": "standard"},
                    "text": {"type": "text", "analyzer": "standard"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": EMBEDDING_DIMS,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "combined_text": {"type": "text", "analyzer": "standard"}
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "30s"
            }
        }
        
        if es.indices.exists(index=config.index_name):
            es.indices.delete(index=config.index_name)
        
        es.indices.create(index=config.index_name, body=index_mapping)
        logger.info(f"Created product index: {config.index_name}")
        
        # Load Product Search Corpus dataset
        loading_status["message"] = "Loading Product Search Corpus from Hugging Face..."
        logger.info("Loading Product Search Corpus dataset...")
        
        dataset = load_dataset("spacemanidol/product-search-corpus", split="train", streaming=True)
        
        # Process and index products with embeddings
        loading_status["message"] = "Processing and embedding products..."
        
        batch_texts = []
        batch_docs = []
        product_count = 0
        
        for item in dataset:
            if config.max_documents and product_count >= config.max_documents:
                break
            
            doc_id = item.get("docid", "") or item.get("doc_id", "") or str(product_count)
            title = item.get("title", "") or ""
            text = item.get("text", "") or item.get("body", "") or ""
            
            # Create combined text for embedding
            combined_text = f"{title}. {text}".strip()
            
            if not combined_text or len(combined_text) < 10:
                continue
            
            doc = {
                "doc_id": doc_id,
                "title": title,
                "text": text,
                "combined_text": combined_text
            }
            
            batch_texts.append(combined_text[:1000])  # Limit text length for embedding
            batch_docs.append(doc)
            product_count += 1
            
            # Update progress
            if product_count % 100 == 0:
                loading_status["message"] = f"Processing products... {product_count} processed"
                loading_status["progress"] = product_count
                loading_status["total"] = config.max_documents or 0
            
            # Process batch when full
            if len(batch_texts) >= config.embedding_batch_size:
                # Generate embeddings
                embeddings = model.encode(batch_texts, show_progress_bar=False)
                
                # Add embeddings to docs
                for i, doc in enumerate(batch_docs):
                    doc["embedding"] = embeddings[i].tolist()
                
                # Bulk index
                actions = []
                for doc in batch_docs:
                    actions.append({
                        "_index": config.index_name,
                        "_id": doc["doc_id"],
                        "_source": doc
                    })
                
                helpers.bulk(
                    es,
                    actions,
                    chunk_size=config.batch_size,
                    request_timeout=120,
                    raise_on_error=False
                )
                
                batch_texts = []
                batch_docs = []
        
        # Process remaining batch
        if batch_texts:
            embeddings = model.encode(batch_texts, show_progress_bar=False)
            for i, doc in enumerate(batch_docs):
                doc["embedding"] = embeddings[i].tolist()
            
            actions = []
            for doc in batch_docs:
                actions.append({
                    "_index": config.index_name,
                    "_id": doc["doc_id"],
                    "_source": doc
                })
            
            helpers.bulk(
                es,
                actions,
                chunk_size=config.batch_size,
                request_timeout=120,
                raise_on_error=False
            )
        
        # Refresh index
        es.indices.refresh(index=config.index_name)
        
        loading_status["message"] = f"Completed! Indexed {product_count} products"
        loading_status["progress"] = product_count
        logger.info(f"Product Search Corpus indexing complete: {product_count} products")
        
    except Exception as e:
        logger.error(f"Error loading Product Search Corpus: {e}")
        loading_status["message"] = f"Error: {str(e)}"
    finally:
        loading_status["is_loading"] = False


@app.post("/load/product-search")
def load_product_search(config: ProductSearchConfig, background_tasks: BackgroundTasks):
    """
    Load Product Search Corpus dataset into Elasticsearch.
    
    This runs as a background task. Use /status to check progress.
    """
    global loading_status
    
    if loading_status["is_loading"]:
        raise HTTPException(
            status_code=409,
            detail="A loading operation is already in progress"
        )
    
    loading_status = {
        "is_loading": True,
        "progress": 0,
        "total": config.max_documents or 0,
        "message": "Starting Product Search Corpus load..."
    }
    
    background_tasks.add_task(load_product_search_background, config)
    
    return {
        "status": "started",
        "message": "Product Search Corpus loading started in background",
        "index_name": config.index_name,
        "max_documents": config.max_documents
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
