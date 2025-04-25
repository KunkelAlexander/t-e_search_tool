### **`config.py`**
# Configuration file for Internal Knowledge Database

# Paths
SCRAPING_OUTPUT_DIR           = r"C:\Users\TE\Documents\20250103_tande_publications\html"  # Directory to save fetched HTML files
PUBLICATIONS_DIR              = r"C:\Users\TE\Documents\20250103_tande_publications\pdf"    # Directory to store downloaded PDFs
PUBLICATIONS_METADATA_FILE    = r"C:\Users\TE\Documents\20250103_tande_publications\metadata.csv"    # Directory to store PDF metadata

# Simple Search
SNIPPET_LENGTH  = 500
TEXT_INDEX_FILE = f"data/search_index.txt"  # Text index file

# Embedding Model
METADATA_FILE = r"embeddings/20250601_all-mpnet-base-v2/metadata.json"      # File to store document metadata
CHUNKS_FILE   = r"embeddings/20250601_all-mpnet-base-v2/chunks"
FAISS_INDEX_FILE = r"embeddings/20250601_all-mpnet-base-v2/faiss_index.bin"  # FAISS index file
USE_TEXT_INDEX_FILE = True
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"   # Model for generating embeddings - More accurate than "all-MiniLM-L6-v2"
CHUNK_SIZE = 500                             # Number of words per text chunk - This determines the context size vs granularity trade-off
FAISS_TOP_K = 5                              # Number of top results to retrieve

# Streamlit App
LOGO_PATH = "frontend/assets/logo.png"     # Path to your company logo
APP_TITLE = "Internal Knowledge Database"  # Title for the Streamlit app

# Logging
LOGGING_ENABLED = True
LOG_FILE = "logs/application.log"

# Other Settings
DEFAULT_QUERY = "Type your search query here..."  # Placeholder text for the search bar
