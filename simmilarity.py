import pickle
from sklearn.metrics.pairwise import cosine_similarity

with open('data/company_embeddings/embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

activtrak_embedding = embeddings["activtrak.txt"]

similarity = cosine_similarity(
    embeddings["activtrak.txt"].reshape(1, -1),
    embeddings["openai.txt"].reshape(1, -1)
)[0][0]

print(f"Similarity between ActivTrak and Aircall: {similarity:.4f}")