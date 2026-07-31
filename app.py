

# A sample document — later this becomes a real file upload
document = """
Our company offers a 30-day return policy on all items.
Refunds are processed within 5 business days.
Shipping is free for orders over $50.
We ship internationally to over 40 countries.
Customer support is available Monday to Friday, 9am to 5pm.
"""

# Chunk it: split into pieces. Simplest version — split by line.
chunks = [line.strip() for line in document.strip().split("\n") if line.strip()]

# Prove it worked — look at your chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i}: {chunk}")

import chromadb

# Create a local vector database and a "collection" (like a table)
db = chromadb.Client()
collection = db.create_collection("company_docs")

# Store each chunk. Chroma embeds them automatically with a built-in model.
collection.add(
    documents=chunks,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)

print(f"Stored {collection.count()} chunks in the vector database.")

question = "How long do I have to return something?"

results = collection.query(
    query_texts=[question],
    n_results=1          # return the single most relevant chunk
)

print(f"\nQuestion: {question}")
print(f"Most relevant chunk: {results['documents'][0][0]}")

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

retrieved_chunk = results['documents'][0][0]

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "Answer the user's question using ONLY the context provided. If the context doesn't contain the answer, say you don't know."},
        {"role": "user", "content": f"Context: {retrieved_chunk}\n\nQuestion: {question}"}
    ]
)

print(f"\nAnswer: {response.choices[0].message.content}")