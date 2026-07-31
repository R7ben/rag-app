# rag-app

A minimal Retrieval-Augmented Generation (RAG) example. It chunks a sample document, stores the chunks in a local vector database, retrieves the most relevant chunk for a question, and asks an LLM to answer using only that context.

## How it works

1. **Chunk** — the sample document is split into lines, each treated as a chunk.
2. **Embed & store** — [ChromaDB](https://www.trychroma.com/) embeds each chunk with its built-in default model and stores it in an in-memory collection.
3. **Retrieve & answer** — the question is embedded, the closest chunk is retrieved, and it's passed as context to a Groq-hosted LLM (`llama-3.3-70b-versatile`), which answers using only that context.

See [explaination](explaination) for a more detailed walkthrough of each step.

## Setup

1. Create and activate a virtual environment:

   ```
   python -m venv venv
   venv\Scripts\activate
   ```

2. Install dependencies:

   ```
   pip install chromadb python-dotenv groq
   ```

3. Create a `.env` file in the project root with your [Groq API key](https://console.groq.com/keys):

   ```
   GROQ_API_KEY=your_key_here
   ```

## Usage

```
python app.py
```

This prints the generated chunks, how many were stored, the sample question, the retrieved chunk, and the LLM's answer.

## Next steps

- Replace the hardcoded `document` string with real file uploads.
- Swap the line-based chunking for a smarter splitter (e.g. by paragraph or token count).
- Persist the Chroma collection to disk instead of using an in-memory client.
