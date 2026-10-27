# Medical RAG Assistant

A Streamlit chat interface for asking general health questions against the UTF-8 text references in `data/`. It retrieves relevant passages with FAISS, reranks them, and asks a local Ollama model to answer with source citations.

> **Important:** This is an educational information tool, not a medical device or a substitute for a licensed healthcare professional. It does not diagnose conditions or prescribe treatment. Seek urgent care for severe or rapidly worsening symptoms. Do not enter personal health information.

## Requirements

- Python 3.10 or newer
- Ollama installed and running locally
- A chat model downloaded in Ollama (the default is `phi3`)

## Setup

From the project directory, create and activate a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Pull the default local chat model and start Ollama:

```powershell
ollama pull phi3
ollama serve
```

In another terminal, launch Streamlit:

```powershell
streamlit run app.py
```

The embedding model (`sentence-transformers/all-MiniLM-L6-v2`) and reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) are downloaded by their libraries on first use. Internet access is needed for that initial download; subsequent runs use the local cached model files.

## Configuration

The following optional environment variables configure the Ollama connection:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_MODEL` | `phi3` | Name of the model available in Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

Put UTF-8 `.txt` reference files in `data/`. Nested folders are supported. The app loads and indexes these files once per Streamlit process; restart the app after adding or changing reference files.

## How it works

1. Load text references and split them into overlapping passages.
2. Create embeddings and retrieve the closest passages from FAISS.
3. Rerank the retrieved passages with a cross-encoder.
4. Generate a cautious answer with Ollama and display the source filenames.

The conversation is kept in Streamlit's session state and can be cleared from the sidebar. Questions should be answered only from the local reference corpus; when the corpus does not contain an answer, the assistant is instructed to say so.
