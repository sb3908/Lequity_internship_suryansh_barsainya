import os
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Load pre-trained model and tokenizer for generating embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def read_cleaned_text_files(directory_path):
    """
    Reads all .txt files from a directory and loads the cleaned text.
    
    Args:
        directory_path (str): Path to the directory containing .txt files.

    Returns:
        dict: Dictionary where keys are filenames and values are the file content.
    """
    texts = {}
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                texts[file_name] = file.read()
    return texts

def generate_embeddings(texts):
    """
    Generates embeddings for a list of texts using a pre-trained model.
    
    Args:
        texts (list of str): List of cleaned text strings.

    Returns:
        np.ndarray: Numpy array containing vector embeddings.
    """
    embeddings = []
    for text in texts:
        # Tokenize the input text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        # Pass through the model
        outputs = model(**inputs)
        # Perform mean pooling to get a single embedding vector per input
        embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
        embeddings.append(embedding)
    # Stack all embeddings into a single numpy array
    return np.vstack(embeddings)

def process_text_files_to_embeddings(input_directory, output_path):
    """
    Processes .txt files in a directory to generate embeddings and save to a file.
    
    Args:
        input_directory (str): Path to the directory containing .txt files.
        output_path (str): Path to save the embeddings as a NumPy file.
    """
    print("Reading cleaned text files...")
    texts = read_cleaned_text_files(input_directory)
    file_names = list(texts.keys())
    cleaned_texts = list(texts.values())
    
    print("Generating embeddings...")
    embeddings = generate_embeddings(cleaned_texts)
    
    # Save embeddings and filenames
    np.savez(output_path, embeddings=embeddings, file_names=file_names)
    print(f"Embeddings saved to: {output_path}")

if __name__ == "__main__":
    # Path to the directory containing cleaned .txt files
    input_directory = r"D:\RTI Judgments\Extracted_Text"

    # Path to save the embeddings
    output_file_path = r"D:\RTI Judgments\embeddings.npz"

    # Generate and save embeddings
    process_text_files_to_embeddings(input_directory, output_file_path)

    print("Embedding generation completed.")
