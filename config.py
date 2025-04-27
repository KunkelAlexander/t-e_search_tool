### **`config.py`**

# Embedding Model
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
PERSIST_DIR          = "../t-and-e-webscraper/chroma_db"
COLLECTION_NAME      = "pdf_collection"
CHUNK_SIZE           = 500                       # Number of words per text chunk - This determines the context size vs granularity trade-off (I suggest testing between 100 to 1000)
FAISS_TOP_K          = 5                         # Number of top results to retrieve
SEARCH_RESULT_K      = 5

# Streamlit App
LOGO_PATH = "assets/logo.png"     # Path to your company logo
APP_TITLE = "Internal Knowledge Database"  # Title for the Streamlit app

# Other Settings
DEFAULT_QUERY = "Type your search query here..."  # Placeholder text for the search bar
