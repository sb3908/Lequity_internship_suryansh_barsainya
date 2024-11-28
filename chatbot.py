import numpy as np
import faiss
from transformers import pipeline

# Load pre-trained LLM for text generation
generator = pipeline("text-generation", model="gpt2", device=0)  # Replace with your preferred model

def load_faiss_index(index_path, metadata_path):
    """
    Loads a FAISS index and metadata for retrieval.

    Args:
        index_path (str): Path to the FAISS index file.
        metadata_path (str): Path to the metadata file (e.g., document names or content).

    Returns:
        tuple: (FAISS index, metadata list)
    """
    # Load the FAISS index
    index = faiss.read_index(index_path)
    
    # Load metadata (e.g., file names or text content)
    data = np.load(metadata_path, allow_pickle=True)
    file_names=data["file_names"].tolist()
    
    return index, file_names

def generate_query_embedding(query, embedding_model):
    """
    Generates an embedding for the user query using the pre-trained embedding model.

    Args:
        query (str): The user's query.
        embedding_model: The pre-trained model for embedding generation.

    Returns:
        np.ndarray: The embedding vector for the query.
    """
    inputs = embedding_model.tokenizer(query, return_tensors="pt", truncation=True, padding=True)
    outputs = embedding_model.model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return query_embedding

def query_bot(query, index, file_names,  top_k=5):
    """
    Answers user queries using retrieval-augmented generation.
    
    Args:
        query (str): User's query string.
        index (faiss.IndexFlatL2): FAISS index for similarity search.
        file_names (list): List of file_names (e.g., document texts or names).
        embedding_model: The embedding model used for generating query embeddings.
        top_k (int): Number of top results to retrieve.

    Returns:
        str: Generated response from the chatbot.
    """
    # Generate query embedding
    query_embedding = generate_query_embedding(query, embedding_model)
    
    # Retrieve relevant documents
    distances, indices = index.search(query_embedding, top_k)
    retrieved_docs = [file_names[idx] for idx in indices[0]]
    
    # Concatenate retrieved text for context
    context = " ".join(retrieved_docs)
    
    # Generate response with context
    prompt = f"Based on the following documents: {context}\n\nAnswer the question: {query}"
    response = generator(prompt, max_length=512, num_return_sequences=1)
    return response[0]["generated_text"]

if __name__ == "__main__":
    # File paths
    index_path = r"D:\RTI Judgments\faiss_index.bin"
    file_names_path = r"D:\RTI Judgments\embeddings.npz"

    # Load FAISS index and file_names
    print("Loading FAISS index and file_names...")
    faiss_index, file_names = load_faiss_index(index_path, file_names_path)

    # Simulated embedding model setup (replace with your actual model)
    from transformers import AutoTokenizer, AutoModel
    class EmbeddingModel:
        def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)

    embedding_model = EmbeddingModel()

    # Example user interaction
    print("Chatbot ready!")
    while True:
        user_query = input("\nAsk your question (type 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        # Query the bot and generate a response
        response = query_bot(user_query, faiss_index, file_names, embedding_model)
        print("\nResponse:\n", response)
