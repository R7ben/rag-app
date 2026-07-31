import os
import streamlit as st
from pypdf import PdfReader
import chromadb
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.title("Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # STAGE 1 — extract text from the PDF, then chunk it
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    # Split into ~500-character chunks with 50-char overlap
    text = text.replace("\n", " ")
    chunk_size, overlap = 500, 50
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]
    chunks = [c.strip() for c in chunks if len(c.strip()) > 30]

    # STAGE 2 — embed & store (fresh collection each upload)
    db = chromadb.Client()
    try:
        db.delete_collection("pdf_docs")   # clear old data on re-upload
    except:
        pass
    collection = db.create_collection("pdf_docs")
    collection.add(documents=chunks, ids=[f"c{i}" for i in range(len(chunks))])

    st.success(f"Loaded {len(chunks)} chunks. Ask a question below.")

    # STAGE 3 & 4 — retrieve then generate
    question = st.text_input("Your question:")
    if question:
        results = collection.query(query_texts=[question], n_results=5)
        context = "\n".join(results["documents"][0])

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Answer using ONLY the context provided. If it's not in the context, say you don't know."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )
        st.write(response.choices[0].message.content)