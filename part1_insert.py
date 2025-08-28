# part1_insert.py

import os
import chromadb
import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader


# 1. Initialize ChromaDB client
client = chromadb.PersistentClient(path="./climate_change_chroma_db")
collection = client.get_or_create_collection(
    name="climate_change_articles",
    metadata={"hnsw:space": "cosine"}
)


# 2. Define text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)


# 3. Define embedding model (GPU/CPU auto-detect)
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
)


# 4. Load, chunk, embed, and insert
def load_and_insert(folder_path):
    doc_counter = collection.count()

    for file in os.listdir(folder_path):
        if not file.endswith(".txt"):
            continue
        path = os.path.join(folder_path, file)
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()

        for doc in docs:
            chunks = text_splitter.split_text(doc.page_content)

            ids = [f"climate_{doc_counter+i}" for i in range(len(chunks))]
            embeddings = embeddings_model.embed_documents(chunks)

            collection.add(
                ids=ids,
                documents=chunks,
                metadatas=[{"title": file, "chunk_id": i} for i in range(len(chunks))],
                embeddings=embeddings
            )
            doc_counter += len(chunks)
            print(f"Inserted {len(chunks)} chunks from {file}")

    print(f"Total chunks in DB: {collection.count()}")


if __name__ == "__main__":
    load_and_insert("./climate_change_data")  # put your txt files here
