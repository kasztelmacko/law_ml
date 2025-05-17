from sentence_transformers import SentenceTransformer
import os
import numpy as np
import pickle
from tqdm import tqdm
import re

def preprocess_text(text, company_name=None):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', text)     # emails
    text = re.sub(r'http[s]?://\S+|www\.\S+', ' ', text)        # URLs
    text = re.sub(r'\b(?:p\.?\s?o\.?\s?box|suite|floor|building|road|avenue|st\.?|street|zip|zipcode|city|state|country)\b[\w\s,.]*', ' ', text)  # physical addresses
    text = re.sub(r'\+?\d[\d\s\-\(\)]{7,}\d', ' ', text)        # phone numbers
    text = re.sub(r'\b\d+(\.\d+)*[.)]?', ' ', text)             # numbered sections
    text = re.sub(r'\b[a-z]\)|[a-z][.)]', ' ', text)            # lettered subsections
    if company_name:
        base = re.escape(company_name.lower())
        variants = [base, base.replace('-', ' '), base.replace(' ', ''), base.replace('.', ' ')]
        for variant in variants:
            text = re.sub(r'\b' + variant + r'\b', ' ', text)
    return text

def generate_embeddings(data_dir, output_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = {}
    file_list = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    for filename in tqdm(file_list, desc="Generating sentence-transformer embeddings"):
        with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
            company_name = filename.replace('.txt', '').lower()
            text = preprocess_text(f.read(), company_name=company_name)
            if text:
                embedding = model.encode(text, show_progress_bar=False, convert_to_numpy=True)
                embeddings[filename] = embedding

    with open(output_path, 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings

# Example usage
if __name__ == "__main__":
    generate_embeddings("data/companies/", "data/company_embeddings/embeddings_sentence_trans.pkl")
