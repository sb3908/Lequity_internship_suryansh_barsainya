import faiss
import numpy as np

def create_faiss_index(embeddings, save_path):
    """
    Creates a FAISS index for fast similarity search and saves it to a .bin file.
    
    Args:
        embeddings (np.ndarray): Numpy array containing the document embeddings.
        save_path (str): Path to save the FAISS index (.bin file).

    Returns:
        faiss.IndexFlatL2: A FAISS index containing the embeddings.
    """
    # Get the dimension of the embeddings
    dimension = embeddings.shape[1]
    
    # Create a FAISS index with L2 distance
    index = faiss.IndexFlatL2(dimension)
    
    # Add the embeddings to the index
    index.add(embeddings)
    
    # Save the index to a .bin file
    faiss.write_index(index, save_path)
    print(f"FAISS index saved to: {save_path}")
    
    return index

def search_faiss_index(query_embedding, index, top_k=5):
    """
    Searches the FAISS index for the most similar documents.
    
    Args:
        query_embedding (np.ndarray): Embedding of the query (1D array or single embedding).
        index (faiss.IndexFlatL2): FAISS index containing the document embeddings.
        top_k (int): Number of top results to retrieve.

    Returns:
        (np.ndarray, np.ndarray): Distances and indices of the top-k results.
    """
    query_embedding = np.expand_dims(query_embedding, axis=0)  # Ensure query is 2D
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices

def load_embeddings(embedding_file):
    """
    Loads embeddings and associated file names from a saved .npz file.
    
    Args:
        embedding_file (str): Path to the .npz file containing embeddings and file names.

    Returns:
        tuple: A tuple (embeddings, file_names).
    """
    data = np.load(embedding_file)
    return data["embeddings"], data["file_names"]

if __name__ == "__main__":
    # File paths
    embedding_file = r"D:\RTI Judgments\embeddings.npz"
    faiss_index_path = r"D:\RTI Judgments\faiss_index.bin"

    # Load the saved embeddings and file names
    embeddings, file_names = load_embeddings(embedding_file)

    # Create the FAISS index and save it
    print("Creating FAISS index...")
    faiss_index = create_faiss_index(embeddings, faiss_index_path)

    # Example query: Assume the query is already embedded
    query_text_embedding = embeddings[0]  # Example: Using one document's embedding as a query
    
    # Search for top-k similar documents
    top_k = 5
    distances, indices = search_faiss_index(query_text_embedding, faiss_index, top_k)

    # Display the results
    print(f"\nTop-{top_k} results for the query:")
    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. File: {file_names[idx]} - Distance: {distances[0][i]}")
