from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from elasticsearch import Elasticsearch, helpers
from datasets import load_dataset
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    batch_size: int = 1000


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
    """Background task to load MS MARCO dataset."""
    global loading_status
    
    try:
        loading_status["is_loading"] = True
        loading_status["message"] = "Initializing..."
        loading_status["progress"] = 0
        
        es = get_es_client()
        
        # Create index with mapping
        loading_status["message"] = "Creating index mapping..."
        index_mapping = {
            "mappings": {
                "properties": {
                    "passage_id": {"type": "keyword"},
                    "passage_text": {"type": "text", "analyzer": "standard"},
                    "query_id": {"type": "keyword"},
                    "query_text": {"type": "text", "analyzer": "standard"},
                    "is_selected": {"type": "boolean"}
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
        logger.info(f"Created index: {config.index_name}")
        
        # Load MS MARCO dataset
        loading_status["message"] = "Loading MS MARCO dataset from Hugging Face..."
        logger.info("Loading MS MARCO dataset...")
        
        dataset = load_dataset("ms_marco", "v1.1", split=config.dataset_split, streaming=True)
        
        # Process and index documents
        loading_status["message"] = "Indexing documents..."
        
        def generate_actions():
            count = 0
            for item in dataset:
                if config.max_documents and count >= config.max_documents:
                    break
                
                query_id = str(item.get("query_id", count))
                query_text = item.get("query", "")
                passages = item.get("passages", {})
                
                passage_texts = passages.get("passage_text", [])
                is_selected_list = passages.get("is_selected", [])
                
                for idx, passage_text in enumerate(passage_texts):
                    is_selected = is_selected_list[idx] if idx < len(is_selected_list) else 0
                    
                    yield {
                        "_index": config.index_name,
                        "_source": {
                            "passage_id": f"{query_id}_{idx}",
                            "passage_text": passage_text,
                            "query_id": query_id,
                            "query_text": query_text,
                            "is_selected": bool(is_selected)
                        }
                    }
                
                count += 1
                if count % 100 == 0:
                    loading_status["progress"] = count
                    loading_status["message"] = f"Processing query {count}..."
                    logger.info(f"Processed {count} queries")
        
        # Bulk index
        success, failed = helpers.bulk(
            es,
            generate_actions(),
            chunk_size=config.batch_size,
            request_timeout=120,
            raise_on_error=False
        )
        
        # Refresh index
        es.indices.refresh(index=config.index_name)
        
        loading_status["message"] = f"Completed! Indexed {success} documents, {failed} failed"
        loading_status["progress"] = success
        logger.info(f"Indexing complete: {success} success, {failed} failed")
        
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
                        "fields": ["passage_text", "query_text"]
                    }
                },
                "size": size
            }
        )
        
        return {
            "total": result["hits"]["total"]["value"],
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
        logger.error(f"Error searching: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
