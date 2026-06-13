import logging
import threading
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-safe singleton locks and containers
_embeddings_lock = threading.Lock()
_embeddings_instance = None

def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Initializes and returns the HuggingFaceEmbeddings instance with the 
    specified 'sentence-transformers/all-MiniLM-L6-v2' model.
    Utilizes a thread-safe singleton pattern to prevent redundant loads.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        with _embeddings_lock:
            if _embeddings_instance is None:
                model_name = "sentence-transformers/all-MiniLM-L6-v2"
                logger.info(f"Initializing HuggingFaceEmbeddings singleton with model: {model_name}")
                try:
                    _embeddings_instance = HuggingFaceEmbeddings(
                        model_name=model_name,
                        model_kwargs={"device": "cpu"},  # Explicitly run on CPU for local/cloud CPU environments
                        encode_kwargs={"normalize_embeddings": True}  # Normalize embeddings for cosine similarity
                    )
                except Exception as e:
                    logger.error(f"Failed to load HuggingFace Embeddings model: {e}")
                    raise RuntimeError(
                        f"Could not load HuggingFace Embeddings model. Details: {str(e)}"
                    )
    return _embeddings_instance
