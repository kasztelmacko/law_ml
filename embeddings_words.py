from transformers import BertTokenizer, BertModel
import torch
import os
import numpy as np
from tqdm import tqdm
import re
import pickle
from clean_text import preprocess_text

def generate_embeddings(data_dir, output_dir):
    """Generate and save BERT embeddings for all documents"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 8 if torch.cuda.is_available() else 4

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    if device.type == 'cuda':
        model = model.half()
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    def process_batch(texts):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding='longest',
            max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).cpu()
    
    file_list = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    embeddings = {}
    
    batch_texts = []
    batch_names = []
    
    for filename in tqdm(file_list, desc="Generating embeddings"):
        with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as f:
            company_name = filename.replace('.txt', '').lower()
            text = preprocess_text(f.read(), company_name=company_name)
            if text:
                batch_texts.append(text)
                batch_names.append(filename)
                
                if len(batch_texts) >= batch_size:
                    batch_embeddings = process_batch(batch_texts)
                    for emb, name in zip(batch_embeddings, batch_names):
                        embeddings[name] = emb.numpy()
                    batch_texts = []
                    batch_names = []
    
    if batch_texts:
        batch_embeddings = process_batch(batch_texts)
        for emb, name in zip(batch_embeddings, batch_names):
            embeddings[name] = emb.numpy()

    with open(os.path.join(output_dir, 'embeddings_cleaned.pkl'), 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings

def save_similarity_matrix(filenames, matrix, output_path):
    """Save similarity matrix with labels"""
    np.savez(output_path, filenames=filenames, matrix=matrix)

def main():
    data_dir = "data/companies/"
    output_dir = "data/company_embeddings/"
    
    generate_embeddings(data_dir, output_dir)

if __name__ == "__main__":
    main()