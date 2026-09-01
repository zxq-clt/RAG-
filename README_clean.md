# Multi-Agent-Medical-Assistant

**AI-powered multi-agentic system for medical knowledge Q&A and research assistance.**

This project builds a **multi-agent RAG conversation system** for medical knowledge Q&A. A **LangGraph** workflow orchestrates three specialized agents - conversation, RAG (retrieval-augmented generation), and web search - and dynamically routes each user query to the most suitable agent based on intent analysis. A complete RAG pipeline (PDF parsing -> semantic chunking -> query expansion -> hybrid retrieval -> reranking) grounds answers in a private medical literature knowledge base, with automatic degradation to real-time web search when the knowledge base cannot provide a confident answer.

## Table of Contents
- [Overview](#overview)
- [Technical Workflow](#technical-workflow)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Contributions](#contributions)
- [License](#license)

---

## Overview

The **Multi-Agent Medical Assistant** is an AI-powered chatbot designed for **medical knowledge Q&A and research assistance**.

**Powered by Multi-Agent Intelligence**, this system integrates:
- **Large Language Models (LLMs)**
- **Retrieval-Augmented Generation (RAG)** leveraging a private vector database
- **Real-time Web Search** for up-to-date medical insights

### What You'll Learn from This Project
- **Multi-Agent Orchestration** with a structured LangGraph workflow
- **Advanced RAG Techniques** - hybrid retrieval, semantic chunking, query expansion, and reranking
- **Confidence-Based Routing** with automatic **RAG -> Web Search** degradation
- **Input & Output Guardrails** for safe and compliant medical responses

For a detailed breakdown of the agents and workflow, check out `agents/README_clean.md`.

---

## Technical Workflow

The system is built as a **6-node LangGraph workflow**:

| # | Node | Responsibility |
|---|------|----------------|
| 1 | `analyze_input` | Analyzes the user input and applies **input guardrails** (blocks PII leakage, prompt injection, and unsafe/harmful content) |
| 2 | `route_to_agent` | LLM-based **intent routing** - decides which agent should handle the query, with confidence scoring |
| 3 | `CONVERSATION_AGENT` | Handles general chat, greetings, and non-medical questions with multi-turn context |
| 4 | `RAG_AGENT` | Answers medical knowledge questions from the private knowledge base; computes a retrieval confidence score |
| 5 | `WEB_SEARCH_PROCESSOR_AGENT` | Rewrites the query and searches the web for time-sensitive or out-of-knowledge-base questions; also the fallback when RAG confidence is low |
| 6 | `apply_guardrails` | Applies **output guardrails** to verify medical compliance before returning the response |

**Conditional routing:**
- `analyze_input` -> `apply_guardrails` (blocked input) / `route_to_agent` (allowed)
- `route_to_agent` -> `CONVERSATION_AGENT` / `RAG_AGENT` / `WEB_SEARCH_PROCESSOR_AGENT` (low confidence falls back to RAG)
- `RAG_AGENT` -> `WEB_SEARCH_PROCESSOR_AGENT` (low retrieval confidence or insufficient info) / `apply_guardrails`
- Conversation & web-search agents -> `apply_guardrails`

---
## Key Features

- **Multi-Agent Architecture**: Specialized agents working in harmony to handle information retrieval, reasoning, and conversational responses

- **Advanced RAG Retrieval System**:
  - Docling-based parsing to extract text and tables from PDFs
  - LLM-based semantic chunking with structural boundary awareness
  - Multi-turn query rewriting so follow-up questions ("how is it treated?") are answered with full context
  - LLM-based query expansion with related medical domain terms
  - Qdrant hybrid search combining BM25 sparse keyword search with dense embedding vector search
  - Cross-Encoder based reranking of retrieved chunks for accurate LLM responses
  - Answers carry inline `[n]` citations plus a structured source panel (title, link, relevance score, evidence snippet)
  - Per-session semantic cache that reuses retrieval/generation results for similar follow-up questions
  - Input-output guardrails to ensure safe and relevant responses
  - Confidence-based handoff between RAG and Web Search to prevent hallucinations

- **Real-time Research Integration**: Web search agent that retrieves the latest medical research papers and findings

- **Confidence-Based Routing**: Retrieval confidence scoring ensures answers are grounded, with automatic fallback to web search when confidence is low

- **Input & Output Guardrails**: Ensures safe, unbiased, and reliable medical responses while filtering out harmful or misleading content

- **Multi-Turn Conversation**: Session state persistence enables context-aware multi-turn dialogue; the RAG agent rewrites follow-ups and caches per-session results
- **Answer-Level Evaluation**: a 30-question benchmark (`eval_questions.json`) plus scripts (`eval_rag.py` retrieval metrics, `eval_answers.py` LLM-judged faithfulness/relevance) and a GitHub Actions workflow for regression testing

- **Intuitive User Interface**: Designed for users with minimal technical expertise

---

## Tech Stack

| Component | Technologies |
|-----------|-------------|
| **Backend Framework** | FastAPI |
| **Agent Orchestration** | LangGraph |
| **Document Parsing** | Docling |
| **Knowledge Storage** | Qdrant Vector Database |
| **Guardrails** | LangChain |
| **Frontend** | HTML, CSS, JavaScript |
| **Deployment** | Docker, GitHub Actions CI/CD |

---

## Model Download

The model weights required by this project are **not pushed to the repository** (they are large and excluded via `.gitignore`). Download them first and place them in the corresponding directories:

| Model | Purpose | Download Link | Local Path |
|-------|---------|---------------|------------|
| `BAAI/bge-small-zh-v1.5` | Embedding model, generates dense query/document vectors (Chinese medical semantic retrieval) | https://huggingface.co/BAAI/bge-small-zh-v1.5 | `./models/bge-small-zh-v1.5` |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking retrieved chunks | https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2 | `./ms-marco-MiniLM-L-6-v2` |
| `ds4sd/docling-models` | Docling PDF parsing models (layout analysis / table recognition) | https://huggingface.co/ds4sd/docling-models | `./docling-models` |

Download with `huggingface_hub` (or set `HF_ENDPOINT=https://hf-mirror.com` inside mainland China):

```bash
pip install -U huggingface_hub

huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir ./models/bge-small-zh-v1.5
huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2 --local-dir ./ms-marco-MiniLM-L-6-v2
huggingface-cli download ds4sd/docling-models --local-dir ./docling-models
```

> Tip: the embedding/reranker model paths can also be overridden via the `EMBEDDING_MODEL` / `RERANKER_MODEL` environment variables (see `config.py`).
## Installation & Setup

### Option 1: Using Docker

**Prerequisites:**
- [Docker](https://docs.docker.com/get-docker/) installed on your system
- API keys for the required services

**1. Clone the Repository**
```bash
git clone <your-repo-url>
cd Multi-Agent-Medical-Assistant
```

**2. Create Environment File**

Create a `.env` file in the root directory and add the following API keys:

> **Note:** The project uses the DeepSeek API (OpenAI-compatible) by default. To use another LLM, update the model definitions in `config.py` and provide the matching env variables.

```bash
# LLM Configuration (DeepSeek API, OpenAI-compatible interface)
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-v4-flash

# Web Search API Key (Free credits available with new Tavily Account)
TAVILY_API_KEY=

# (OPTIONAL) If using Qdrant server version, local does not require API key
QDRANT_URL=
QDRANT_API_KEY=
```

**3. Build the Docker Image**
```bash
docker build -t medical-assistant .
```

**4. Run the Docker Container**
```bash
docker run -d --name medical-assistant-app -p 8000:8000 --env-file .env medical-assistant
```
The application will be available at: http://localhost:8000

**5. Ingest Data into Vector DB from Docker Container**

- To ingest a single document:
```bash
docker exec medical-assistant-app python ingest_rag_data.py --file ./data/raw/brain_tumors_ucni.pdf
```

- To ingest multiple documents from a directory:
```bash
docker exec medical-assistant-app python ingest_rag_data.py --dir ./data/raw
```

### Option 2: Without Using Docker

**1. Clone the Repository**
```bash
git clone <your-repo-url>
cd Multi-Agent-Medical-Assistant
```

**2. Create & Activate Virtual Environment**
- If using conda:
```bash
conda create --name <environment-name> python=3.11
conda activate <environment-name>
```
- If using python venv:
```bash
python -m venv <environment-name>
source <environment-name>/bin/activate  # For Mac/Linux
<environment-name>\Scripts\activate     # For Windows
```

**3. Install Dependencies**
```bash
pip install -r requirements.txt
```

**4. Set Up API Keys**

Create a `.env` file and add the required API keys as shown in Option 1.

**5. Run the Application**
```bash
python app.py
```
The application will be available at: http://localhost:8000

**6. Ingest additional data into the Vector DB**
- To ingest one document at a time:
```bash
python ingest_rag_data.py --file ./data/raw/brain_tumors_ucni.pdf
```
- To ingest multiple documents from a directory:
```bash
python ingest_rag_data.py --dir ./data/raw
```

---

## Usage

> **Note:**
> 1. The first run downloads the reranker model and loads the local embedding model - be patient.
> 2. Once the models are loaded, everything should work seamlessly.

- Ask medical queries to leverage **retrieval-augmented generation (RAG)** from the private knowledge base, or **web-search** to retrieve the latest information.

---

## Contributions

Contributions are welcome! Please check the [issues](<your-repo-url>/issues) tab for feature requests and improvements.

---

## License

This project is licensed under the **Apache-2.0 License**. See the [LICENSE](LICENSE) file for details.
