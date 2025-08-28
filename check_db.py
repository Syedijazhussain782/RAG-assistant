import chromadb

client = chromadb.PersistentClient(path="./news_db")
collection = client.get_collection("news_articles")

print("Total chunks:", collection.count())



results = collection.peek(limit=5)
print(results)


from langchain_huggingface import HuggingFaceEmbeddings
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)

query = "How do governments respond to inflation?"
q_vec = emb.embed_query(query)

results = collection.query(
    query_embeddings=[q_vec],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

for i, doc in enumerate(results["documents"][0]):
    print(f"Result {i+1}")
    print("Content:", doc)
    print("Metadata:", results["metadatas"][0][i])
    print("Distance:", results["distances"][0][i])
    print("-"*40)
