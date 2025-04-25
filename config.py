### **`config.py`**

# Embedding Model
EMBEDDING_MODEL      = "all-mpnet-base-v2"
EMBEDDING_MODEL_PATH = r"embeddings/faiss_data.tar.gz"
FAISS_INDEX_FILE     = r"faiss_index_l2.bin"
METADATA_FILE        = r"metadata.json"          # File to store document metadata
CHUNKS_FILE          = r"chunks"
CHUNK_SIZE           = 500                       # Number of words per text chunk - This determines the context size vs granularity trade-off (I suggest testing between 100 to 1000)
FAISS_TOP_K          = 5                         # Number of top results to retrieve
SEARCH_RESULT_K      = 5

# Streamlit App
LOGO_PATH = "frontend/assets/logo.png"     # Path to your company logo
APP_TITLE = "Internal Knowledge Database"  # Title for the Streamlit app

# Other Settings
DEFAULT_QUERY = "Type your search query here..."  # Placeholder text for the search bar
