import os
import json
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import faiss

# Configurations
from config import (
    PUBLICATIONS_DIR,
    PUBLICATIONS_METADATA_FILE,
    TEXT_INDEX_FILE,
    USE_TEXT_INDEX_FILE,
    CHUNK_SIZE
)

# File paths
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
FAISS_INDEX_FILE = "faiss_index_with_ids.bin"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"
CHUNKS_FILE = "chunks.json"

# Load metadata from CSV
def load_metadata(metadata_file):
    metadata = pd.read_csv(metadata_file)
    metadata = metadata.drop_duplicates(subset="PDF Filename", keep="first")
    return metadata.set_index("PDF Filename").to_dict(orient="index")

# Extract text from a single PDF
def extract_text_from_pdf(file_path):
    try:
        reader = PdfReader(file_path)
        title = reader.metadata.get('/Title', "Unknown Title")
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        return text, title
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return "", "Unknown Title"

# Chunk text into smaller parts
def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    for i in range(0, len(words), chunk_size):
        yield " ".join(words[i:i + chunk_size])

# Create FAISS index with metadata mapping
def create_faiss_index_with_metadata(chunks, model, metadata_list, num_threads=4):
    print("Creating embeddings...")

    def embed_chunk(chunk):
        return model.encode(chunk).astype("float32")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        embeddings = list(tqdm(executor.map(embed_chunk, chunks), total=len(chunks), desc="Generating Embeddings"))

    embeddings = np.array(embeddings)
    ids = np.arange(len(embeddings))  # Create unique IDs for each embedding

    print("Creating FAISS index with IDMap...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, ids)

    return index, embeddings, metadata_list

# Save FAISS index, embeddings, metadata, and chunks
def save_index_and_metadata(index, embeddings, metadata, chunks):
    print("Saving FAISS index with metadata...")
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(EMBEDDINGS_FILE, embeddings)

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)

# Build and save the FAISS index
def build_faiss_index_with_metadata():
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    all_chunks = []
    metadata_list = []

    metadata_dict = load_metadata(PUBLICATIONS_METADATA_FILE)

    for filename in os.listdir(PUBLICATIONS_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(PUBLICATIONS_DIR, filename)
            print(f"Processing: {filename}")

            metadata_entry = metadata_dict.get(filename)
            if not metadata_entry:
                print(f"No metadata found for {filename}. Skipping...")
                continue

            text, _ = extract_text_from_pdf(pdf_path)
            chunks = list(chunk_text(text))

            all_chunks.extend(chunks)
            metadata_list.extend([metadata_entry] * len(chunks))

    index, embeddings, metadata = create_faiss_index_with_metadata(all_chunks, model, metadata_list)
    save_index_and_metadata(index, embeddings, metadata, all_chunks)

if __name__ == "__main__":
    build_faiss_index_with_metadata()
