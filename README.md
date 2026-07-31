# rag-app

A minimal Retrieval-Augmented Generation (RAG) example. Upload a PDF, and it chunks the text, stores the chunks in a local vector database, retrieves the chunks most relevant to your question, and asks an LLM to answer using only that context.

## How it works

1. **Extract & chunk** — [pypdf](https://pypi.org/project/pypdf/) pulls the text out of the uploaded PDF, which is then split into ~500-character chunks with a 50-character overlap (overlap keeps context from getting lost across chunk boundaries).
2. **Embed & store** — [ChromaDB](https://www.trychroma.com/) embeds each chunk with its built-in default model and stores it in an in-memory collection, reset on every new upload.
3. **Retrieve & answer** — the question is embedded, the 5 closest chunks are retrieved, and they're passed as context to a Groq-hosted LLM (`llama-3.3-70b-versatile`), which answers using only that context.

See [explaination](explaination) for a more detailed walkthrough of each step.

## Setup

1. Create and activate a virtual environment:

   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:

   ```
   pip install streamlit pypdf chromadb python-dotenv groq
   ```

3. Create a `.env` file in the project root with your [Groq API key](https://console.groq.com/keys):

   ```
   GROQ_API_KEY=your_key_here
   ```

## Usage

```
streamlit run app.py
```

Upload a PDF in the browser UI, then ask a question. The app shows how many chunks were loaded, and answers your question using only the retrieved context.

## Next steps

- Persist the Chroma collection to disk instead of using an in-memory client.
- Let the user tune chunk size/overlap and `n_results` from the UI.
- Show which chunks were retrieved alongside the answer, for transparency.
