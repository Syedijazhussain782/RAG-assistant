import os
import torch
import chromadb
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate

# Load environment variables (API key from .env)
load_dotenv()

# 1. Reconnect to ChromaDB
client = chromadb.PersistentClient(path="./climate_change_chroma_db")
collection = client.get_collection("climate_change_articles")

# 2. Re-init embeddings (must match insert step)
device = "cuda" if torch.cuda.is_available() else "cpu"
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
)

# 3. Helper: search function
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

# 4. Helper: answer generator
def answer_question(query, llm):
    chunks = search_db(query, top_k=3)
    if not chunks:
        return "No relevant data found.", []

    context = "\n\n".join([f"From {c['title']}:\n{c['content']}" for c in chunks])

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
Use the following research snippets to answer the question.

Context:
{context}

Question: {question}

Answer clearly and base it only on the context above.
"""
    )

    prompt = prompt_template.format(context=context, question=query)
    response = llm.invoke(prompt)

    return response.content, chunks

# 5. Main test runner
if __name__ == "__main__":
    # Initialize LLM
    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=os.getenv("GROQ_API_KEY")
    )

    test_questions = [
        "How did the oil crises of the 1970s influence inflation and central bank policies?",
        "What are the key differences between EU and US AI regulations?",
        "How does climate change drive human migration?",
    ]

    for q in test_questions:
        print("=" * 80)
        print(f"Q: {q}\n")
        answer, refs = answer_question(q, llm)
        print(f"Answer: {answer}\n")
        print("Based on sources:")
        for r in refs:
            print(f"- {r['title']} (similarity={r['similarity']:.2f})")
        print()
