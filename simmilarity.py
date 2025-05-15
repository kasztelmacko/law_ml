import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

with open('data/company_embeddings/embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

df = pd.read_csv('data/terms_of_service.csv')
companies = df['Company'].tolist()

valid_companies = []
embedding_matrix = []

for company in companies:
    filename = f"{company.lower().replace(' ', '')}.txt"
    if filename in embeddings:
        valid_companies.append(company)
        embedding_matrix.append(embeddings[filename])

embedding_matrix = np.array(embedding_matrix)

similarity_matrix = cosine_similarity(embedding_matrix)

similarity_df = pd.DataFrame(
    similarity_matrix,
    index=valid_companies,
    columns=valid_companies
)

print(similarity_df)

similarity_df.to_csv('data/company_similarity_matrix.csv')