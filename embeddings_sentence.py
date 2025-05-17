from sentence_transformers import SentenceTransformer
import os
import numpy as np
import pickle
from tqdm import tqdm
from clean_text import preprocess_text

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
