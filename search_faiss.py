import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
import faiss
import config
import tarfile
import streamlit as st

@st.cache_resource(show_spinner="Loading search index and embedding model...")
def extract_and_load_faiss(archive_path, extract_dir="faiss_temp"):
    print(f"Extracting {archive_path} to {extract_dir}...")

    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    index_file = os.path.join(extract_dir, "faiss.index")
    metadata_file = os.path.join(extract_dir, "metadata.json")
    chunk_file = os.path.join(extract_dir, "chunks.json")

    return load_faiss_index(index_file, metadata_file, chunk_file)

# Load FAISS index, metadata, and chunks
def load_faiss_index(index_file, metadata_file, chunk_file):

    print(f"Loading FAISS index ({index_file}), metadata ({metadata_file}), and chunks ({chunk_file})...")

    if not os.path.exists(index_file) or not os.path.exists(metadata_file) or not os.path.exists(chunk_file):
        raise ValueError("No FAISS index available. Please create it using embeddings/build_faiss.py")

    index = faiss.read_index(index_file)
    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    return index, metadata, chunks


# Initialize the FAISS index (build or load)
def initialize_search_index():
    embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    faiss_index, metadata_list, chunks = extract_and_load_faiss(config.EMBEDDING_MODEL_PATH)

    return faiss_index, embedding_model, chunks, metadata_list, "L2" != "L2"

# Perform semantic search
def search_pdfs(query, index, model, chunks, metadata, normalise = False, alpha=0.00, publication_types=None, chunks_before=0, chunks_after=0, max_snippet_length=500):
    """
    Perform a semantic search with date-based weighting on results.

    Parameters:
        query (str): The search query.
        index (faiss.Index): The FAISS search index.
        model: The embedding model.
        chunks (list): List of text chunks.
        metadata (list): List of metadata for each chunk.
        normalise (bool): Whether to normalise query embeddings.
        alpha (float): Decay factor for date weighting.
        chunks_before (int): Number of chunks to include before the main chunk.
        chunks_after (int): Number of chunks to include after the main chunk.
        max_snippet_length (int): Maximum character length of the final snippet.

    Returns:
        list: A list of search results with date-weighted scoring.
    """
    query_embedding = model.encode(query).astype("float32").reshape(1, -1)

    # Debug dimension mismatch in search_pdfs
    if normalise:
        faiss.normalize_L2(query_embedding)

    # Filter IDs based on metadata
    filtered_ids = [
        idx for idx, meta in enumerate(metadata)
        if meta.get("Publication Type", "Unknown Type") in publication_types
    ]

    # Return no results for empty selection
    if not filtered_ids:
        return []

    # Create IDSelector for filtering
    id_selector = faiss.IDSelectorArray(filtered_ids)

    print(f"Performing search with IDSelector...")

    # Results filtered out by id_selector are returned with negative index and infinite distance
    distances, indices = index.search(query_embedding, config.FAISS_TOP_K, params=faiss.SearchParametersIVF(sel=id_selector))

    # Prepare the results with metadata and date weighting
    results = []
    current_date = datetime.now()

    for i, distance in zip(indices[0], distances[0]):
        if i >= 0 and i < len(chunks):  # Ensure the index is within bounds
            metadata_entry = metadata[i]  # Get metadata for the chunk

            # Extract publication date
            publication_date_str = metadata_entry.get("Publication Date", "Unknown Date")
            publication_date = datetime.strptime(publication_date_str, "%b %d, %Y, %I:%M:%S %p")
            days_since_pub = (current_date - publication_date).days

            # Calculate date weight using exponential decay
            date_weight = np.exp(-alpha * days_since_pub/365)

            # Combine similarity distance (lower is better) and date weight
            similarity_score = 1 / (1 + distance)  # Convert FAISS distance to a similarity score
            weighted_score = similarity_score * date_weight


            # -------------------------
            #   Build extended snippet
            # -------------------------
            # Determine which chunk indices to include
            start_idx = max(0, i - chunks_before)
            end_idx = min(len(chunks), i + chunks_after + 1)

            # Join the relevant chunks
            extended_context = " ".join(chunks[start_idx:end_idx])
            extended_context = extended_context.replace("\n", " ")

            # Optionally truncate the extended snippet to avoid overly long text
            extended_context = extended_context[:max_snippet_length]

            results.append({
                "filename": metadata_entry.get("PDF URL", "No PDF URL").split('/')[-1],
                "title": metadata_entry.get("Title", "Unknown Title"),
                "summary": metadata_entry.get("Summary", "No Summary"),
                "publication_date": publication_date.strftime("%B %d, %Y"),
                "publication_type": metadata_entry.get("Publication Type", "Unknown Type"),
                "url": metadata_entry.get("Article URL", "No URL"),
                "pdf_url": metadata_entry.get("PDF URL", "No PDF URL"),
                "snippet": extended_context,
                "score": similarity_score,
                "date_weight": date_weight,
                "weighted_score": weighted_score,
            })

    # Sort results by weighted score in descending order
    results = sorted(results, key=lambda x: x['weighted_score'], reverse=True)[:config.SEARCH_RESULT_K]

    return results
