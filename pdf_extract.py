import os
import re
from PyPDF2 import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a machine-readable PDF using PyPDF2.
    
    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text()
    except Exception as e:
        print(f"An error occurred while processing {pdf_path}: {e}")
    return text

def clean_text(text):
    """
    Cleans and refines the extracted text.
    
    Args:
        text (str): Raw extracted text.

    Returns:
        str: Refined text.
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters (optional, adjust as needed)
    text = re.sub(r'[^\w\s.,:;!?\'"-]', '', text)
    # Strip leading/trailing whitespace
    return text.strip()

def process_multiple_pdfs(directory_path):
    """
    Processes multiple PDFs in a given directory, extracts, and cleans text.

    Args:
        directory_path (str): Path to the directory containing PDF files.

    Returns:
        dict: A dictionary where keys are PDF filenames and values are the refined text.
    """
    pdf_texts = {}
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, file_name)
            print(f"Processing: {pdf_path}")
            # Extract and clean text
            raw_text = extract_text_from_pdf(pdf_path)
            refined_text = clean_text(raw_text)
            pdf_texts[file_name] = refined_text
    return pdf_texts

def save_texts_to_files(output_dir, pdf_texts):
    """
    Saves the refined text for each PDF into separate text files.

    Args:
        output_dir (str): Directory where text files will be saved.
        pdf_texts (dict): Dictionary with PDF filenames as keys and refined text as values.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for file_name, text in pdf_texts.items():
        output_file = os.path.join(output_dir, file_name.replace('.pdf', '.txt'))
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
    print(f"Saved extracted text to: {output_dir}")

if __name__ == "__main__":
    # Path to the directory containing PDFs
    pdf_directory = r"D:\RTI Judgments"

    # Path to save the output text files
    output_directory = r"D:\RTI Judgments\Extracted_Text"

    # Process all PDFs in the directory
    processed_texts = process_multiple_pdfs(pdf_directory)

    # Save the processed texts to individual files
    save_texts_to_files(output_directory, processed_texts)

    # Print summary of processed files
    print("\n--- Summary of Processed PDFs ---")
    for file_name in processed_texts:
        print(f"Processed: {file_name}")
