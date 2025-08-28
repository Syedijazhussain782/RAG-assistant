# part2_query.py

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
import torch
import os 


# 1. Reconnect to DB
client = chromadb.PersistentClient(path="./climate_change_chroma_db")
collection = client.get_collection("climate_change_articles")


# 2. Re-init embeddings (must match Part 1)
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
)


# 3. Define search helper
def search_db(query, top_k=3):
    q_vec = embeddings_model.embed_query(query)
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "content": doc,
            "title": results["metadatas"][0][i]["title"],
            "similarity": 1 - results["distances"][0][i]
        })
    return chunks


# 4. Define answer generator
def answer_question(query, llm):
    chunks = search_db(query, top_k=3)
    if not chunks:
        return "No relevant data found.", []

    context = "\n\n".join([f"From {c['title']}:\n{c['content']}" for c in chunks])

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the following  snippets to answer the user’s question.

Context:
{context}

Question: {question}

Answer in a clear, evidence-based way.
"""
    )

    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)

    return response.content, chunks


# 5. Run a test query
if __name__ == "__main__":
    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )

    question = "Compare how inflation and climate change both create inequality across populations?"
    answer, refs = answer_question(question, llm)

    print("Answer:", answer)
    print("\nBased on sources:")
    for r in refs:
        print(f"- {r['title']} (similarity={r['similarity']:.2f})")
