import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from datetime import datetime

# Configuration
FAISS_INDEX_FILE = "faiss_index_with_ids.bin"
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "metadata.json"
CHUNKS_FILE = "chunks.json"
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# Load Existing Index, Embeddings, and Metadata
def load_existing_index():
    print("Loading existing FAISS index, embeddings, and metadata...")
    index = faiss.read_index(FAISS_INDEX_FILE)
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, embeddings, metadata, chunks


# Save Updated Index, Metadata, and Chunks
def save_updated_index(index, embeddings, metadata, chunks):
    print("Saving updated FAISS index, embeddings, and metadata...")
    faiss.write_index(index, FAISS_INDEX_FILE)
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4)


# Add New Entries to the Index
def add_new_entries(new_text_chunks, new_metadata_entries):
    print("Adding new entries to the FAISS index...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Embed new chunks
    new_embeddings = np.array([model.encode(chunk).astype("float32") for chunk in new_text_chunks])

    index, embeddings, metadata, chunks = load_existing_index()

    # Generate new unique IDs
    start_id = index.ntotal
    new_ids = np.arange(start_id, start_id + len(new_embeddings))

    # Add embeddings and IDs to the index
    index.add_with_ids(new_embeddings, new_ids)

    # Append to existing metadata and chunks
    metadata.extend(new_metadata_entries)
    chunks.extend(new_text_chunks)
    embeddings = np.vstack((embeddings, new_embeddings))

    save_updated_index(index, embeddings, metadata, chunks)


# Remove Entries from the Index
def remove_entries_by_criteria(criteria_function):
    print("Removing entries from the FAISS index based on criteria...")
    index, embeddings, metadata, chunks = load_existing_index()

    # Find IDs to keep based on criteria
    keep_indices = [
        idx for idx, meta in enumerate(metadata)
        if not criteria_function(meta)
    ]

    # Filter embeddings, metadata, and chunks
    embeddings = embeddings[keep_indices]
    metadata = [metadata[i] for i in keep_indices]
    chunks = [chunks[i] for i in keep_indices]

    # Rebuild the FAISS index with remaining embeddings
    print("Rebuilding FAISS index after removal...")
    new_index = faiss.IndexFlatL2(embeddings.shape[1])
    new_index = faiss.IndexIDMap(new_index)
    new_index.add_with_ids(embeddings, np.arange(len(embeddings)))

    save_updated_index(new_index, embeddings, metadata, chunks)


# Update Metadata for Existing Entries
def update_metadata(update_function):
    print("Updating metadata entries...")
    _, embeddings, metadata, chunks = load_existing_index()

    # Apply update function to each metadata entry
    for meta in metadata:
        update_function(meta)

    save_updated_index(_, embeddings, metadata, chunks)


# Example Usage
if __name__ == "__main__":
    index, embeddings, metadata, chunks = load_existing_index()

    # Example 1: Add new entries
    new_text_chunks = [
        "This is a new chunk of text for testing.",
        "Another new chunk added to the index."
    ]
    new_metadata_entries = [
        {"Title": "New Document 1", "Publication Type": "Report", "Publication Date": "2024-06-15"},
        {"Title": "New Document 2", "Publication Type": "Briefing", "Publication Date": "2024-06-20"}
    ]
    add_new_entries(new_text_chunks, new_metadata_entries)

    # Example 2: Remove entries older than a certain date
    def is_old(meta):
        try:
            pub_date = datetime.strptime(meta.get("Publication Date", "1900-01-01"), "%Y-%m-%d")
            return (datetime.now() - pub_date).days > 365 * 5  # Older than 5 years
        except ValueError:
            return True  # Remove if date is invalid

    remove_entries_by_criteria(is_old)

    # Example 3: Update metadata
    def add_author(meta):
        if "Author" not in meta:
            meta["Author"] = "Unknown Author"

    update_metadata(add_author)
