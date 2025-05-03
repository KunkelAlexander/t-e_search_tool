### **`config.py`**
import os
# Embedding Model
EMBEDDING_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE           = 1500                       # Number of words per text chunk - This determines the context size vs granularity trade-off (I suggest testing between 100 to 1000)
FAISS_TOP_K          = 5                         # Number of top results to retrieve
SEARCH_RESULT_K      = 5

# Where you stored the artefacts when building the index
INDEX_PATH   = "./embeddings/docs.index"               # faiss.write_index(...)
MAP_PATH     = "./embeddings/faiss_mapping.parquet"    # vector_id → doc_id / offsets
PAGES_PATH   = "./embeddings/metadata_with_fulltext.parquet"            # fulltext + metadata

HF_CACHE_DIR = "./hf_model_cache"

os.environ["HF_HOME"] = HF_CACHE_DIR

# Streamlit App
LOGO_PATH = "assets/logo.png"     # Path to your company logo
APP_TITLE = "Internal Knowledge Database"  # Title for the Streamlit app

# Other Settings
DEFAULT_QUERY = "Type your search query here..."  # Placeholder text for the search bar
