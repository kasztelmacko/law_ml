import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_and_save_similarity_matrix(embeddings_path, metadata_csv, output_csv):
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)

    df = pd.read_csv(metadata_csv)
    companies = df['Company'].tolist()

    valid_companies = []
    embedding_matrix = []

    for company in companies:
        filename = f"{company.lower().replace(' ', '')}.txt"
        if filename in embeddings:
            valid_companies.append(company)
            embedding_matrix.append(embeddings[filename])

    if not embedding_matrix:
        print("No matching embeddings found.")
        return

    embedding_matrix = np.array(embedding_matrix)
    similarity_matrix = cosine_similarity(embedding_matrix)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=valid_companies,
        columns=valid_companies
    )

    similarity_df.to_csv(output_csv)

compute_and_save_similarity_matrix(
    embeddings_path='data/company_embeddings/embeddings_cleaned.pkl',
    metadata_csv='data/terms_of_service.csv',
    output_csv='data/similarity_cleaned.csv'
)

compute_and_save_similarity_matrix(
    embeddings_path='data/company_embeddings/embeddings_sentence_trans.pkl',
    metadata_csv='data/terms_of_service.csv',
    output_csv='data/similarity_sentence_trans.csv'
)