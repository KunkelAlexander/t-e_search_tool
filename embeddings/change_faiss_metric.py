import faiss
import numpy as np
import json
import argparse

# Load L2 index and convert to cosine similarity index
def normalize_and_rebuild_index(input_index_file, output_index_file):
    # Step 1: Load Existing Index
    print(f"Loading FAISS index from {input_index_file}...")
    index = faiss.read_index(input_index_file)

    # Step 2: Extract Embeddings
    print("Extracting embeddings from the index...")
    embeddings = np.array([index.reconstruct(i) for i in range(index.ntotal)])

    # Step 3: Normalize Embeddings for Cosine Similarity
    print("Normalizing embeddings for cosine similarity...")
    faiss.normalize_L2(embeddings)

    # Step 4: Create a New Index with Inner Product Metric
    print("Creating a new FAISS index with Inner Product (cosine similarity approximation)...")
    d = embeddings.shape[1]
    new_index = faiss.IndexFlatIP(d)  # Inner Product approximates cosine similarity

    # Step 5: Add Normalized Embeddings to the New Index
    print("Adding normalized embeddings to the new FAISS index...")
    new_index.add(embeddings)

    # Step 6: Save the New Index
    print(f"Saving the new FAISS index to {output_index_file}...")
    faiss.write_index(new_index, output_index_file)


    print("New FAISS index with cosine similarity (via Inner Product) created and saved successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild a FAISS index with cosine similarity (approximated via Inner Product)."
    )
    parser.add_argument(
        "--input_index", required=True, help="Path to the input FAISS index file."
    )
    parser.add_argument(
        "--output_index", required=True, help="Path to save the output FAISS index file."
    )

    args = parser.parse_args()

    normalize_and_rebuild_index(
        input_index_file=args.input_index,
        output_index_file=args.output_index
    )


if __name__ == "__main__":
    main()
