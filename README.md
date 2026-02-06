ElasticSearch DataLoader Modernization
Replaced ELSER ingest pipeline with a custom Python FastAPI microservice, deployable on Kubernetes (microk8s).
Text Chunking: Documents are split into overlapping, sentence-aware chunks for better semantic search and retrieval.
Embedding Generation: Each chunk is embedded using Sentence Transformers (all-MiniLM-L6-v2, 384-dim vectors), enabling semantic similarity search.
Elasticsearch Vector Indexing: Data is indexed in Elasticsearch with a dense_vector field (cosine similarity, 384 dims), supporting efficient kNN and hybrid search.
API Endpoints: The service exposes endpoints for chunking, embedding, loading data, and searching (text, vector, and hybrid).
Web UI: A modern dashboard allows users to trigger data loading, monitor progress, clear indices, and perform semantic or hybrid search—all from the browser.
Kubernetes Ready: All components (DataLoader, Web UI, Elasticsearch) are containerized and have deployment/service YAMLs for microk8s, including LoadBalancer support for the UI.