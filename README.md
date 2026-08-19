# RAG Assistant for Academic Books

An advanced Retrieval-Augmented Generation (RAG) pipeline designed to extract insights and answer complex queries from academic textbooks, specifically focusing on economics and time series analysis. 

Built with **LangChain**, **ChromaDB**, and **Groq** for ultra-fast, context-aware responses.

## Features
- **Intelligent Retrieval:** Uses HuggingFace embeddings to accurately search through dense academic text.
- **Ultra-Fast LLM Inference:** Powered by Groq API (`ChatGroq`) for near-instantaneous generation.
- **Local Vector Storage:** Utilizes ChromaDB for efficient and scalable document retrieval.
- **Containerized:** Fully Dockerized for seamless deployment and reproducibility.

## Tech Stack
- **Framework:** [LangChain](https://www.langchain.com/)
- **Embeddings:** HuggingFace (`sentence-transformers`)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/)
- **LLM:** [Groq](https://groq.com/)
- **Deployment:** Docker

## Setup & Installation

### 1. Clone the repository
```bash
git clone [https://github.com/GZempilis/RAG-Assistant-for-Academic-Books.git](https://github.com/GZempilis/RAG-Assistant-for-Academic-Books.git)
cd RAG-Assistant-for-Academic-Books
```

### 2. Environment Variables
Create a `.env` file in the root directory and add your API keys. Make sure this file is included in your `.gitignore`.
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Local Run (Without Docker)
Install the dependencies and run the pipeline:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python RAG.py
```

### 4. Run with Docker
Build and run the container:
```bash
docker build -t rag-assistant .
docker run --env-file .env rag-assistant
```


