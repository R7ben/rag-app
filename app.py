import os
import st_yled
import streamlit as st
import time
from pypdf import PdfReader
import chromadb
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
st_yled.init()

st.sidebar.title("Tools",)
page = st.sidebar.radio("Choose a tool:", ["Chat with your PDF", "Meeting Notes Assistant"])


def pdf_chat():
    st_yled.title("Chat with your PDF", color="#F57272FF")
    container = st.container(border=True, width="stretch", autoscroll=True)
    container.write("Upload a PDF and ask questions about its content. The app will extract text, chunk it, and use an LLM to answer your questions based on the document.")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        st.pdf(uploaded_file, height=700)
        # STAGE 1 — extract text from the PDF, then chunk it
        reader = PdfReader(uploaded_file)
        text = ""
        with st.spinner("Extracting text from PDF..."):
            time.sleep(1)  # simulate processing time
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
            with st.container(border=True):
                st.write(response.choices[0].message.content)

    sentiment_mapping = ["one", "two", "three", "four", "five"]
    selected = st.feedback("stars")
    if selected is not None:
        st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")
    st.badge("Made by ruben(Github:R7brn)", color="blue", width="stretch")


def meeting_notes_assistant():
    st_yled.title("Meeting Notes Assistant", color="#F57272FF")
    container = st.container(border=True, width="stretch", autoscroll=True)
    container.write("Paste in raw, messy meeting notes and ask questions about its content")
    meeting_notes = st.text_area("Paste in raw, messy meeting notes here:", height=200, placeholder="Paste your meeting notes here...")
    if meeting_notes:

        def ask(query, instruction):
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": f"Answer using ONLY the context provided. If it's not in the context, say you don't know. {instruction}"},
                    {"role": "user", "content": f"Context:\n{meeting_notes}\n\nQuestion: {query}"}
                ]
            )
            with st.container(border=True):
                st.write(response.choices[0].message.content)

        if st.button("Summarize into 3 bullets"):
            ask("Summarize the meeting", "Give a clean 3-bullet summary.")

        if st.button("List action items (who does what)"):
            ask("List the action items", "Give a list of action items (who does what).")

        question = st.text_input("Or ask your own question:")
        if st.button("Answer a question") and question:
            ask(question, "Answer the user's question about the meeting notes.")


if page == "Chat with your PDF":
    pdf_chat()
else:
    meeting_notes_assistant()
