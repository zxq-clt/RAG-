# Agent Details

Detailed breakdown of the implemented agents and the LangGraph workflow of the Multi-Agent Medical Assistant.

## Table of Contents
- [Workflow Overview](#workflow-overview)
- [Implemented Agents](#implemented-agents)
- [RAG Pipeline](#rag-pipeline)
- [Research Papers and Documents Used for RAG](#research-papers-and-documents-used-for-rag)
---

## Workflow Overview

The system is a **6-node LangGraph workflow** that orchestrates three agents:

| # | Node | Responsibility |
|---|------|----------------|
| 1 | `analyze_input` | Analyzes the user input and applies **input guardrails**. Blocks PII leakage, prompt injection, unsafe/harmful content, and abuse attempts; if blocked, the workflow jumps directly to `apply_guardrails` |
| 2 | `route_to_agent` | LLM-based **intent routing**. Reviews the query together with recent conversation context and returns the target agent plus a confidence score; low-confidence decisions fall back to `RAG_AGENT` |
| 3 | `CONVERSATION_AGENT` | Handles general chat, greetings, and non-medical questions using multi-turn conversation history |
| 4 | `RAG_AGENT` | Answers medical knowledge questions from the private vector knowledge base and computes a `retrieval_confidence` score; flags the response when the retrieved context is insufficient |
| 5 | `WEB_SEARCH_PROCESSOR_AGENT` | Rewrites the user query with conversation context and searches the web via Tavily for time-sensitive questions; also serves as the automatic fallback when RAG confidence is low or information is insufficient |
| 6 | `apply_guardrails` | Applies **output guardrails** to review the generated response for medical compliance and filter unsafe or inappropriate content before returning it |

**Conditional routing (LangGraph conditional edges):**
- `analyze_input` -> `apply_guardrails` (blocked input) / `route_to_agent` (allowed)
- `route_to_agent` -> `CONVERSATION_AGENT` / `RAG_AGENT` / `WEB_SEARCH_PROCESSOR_AGENT` (low confidence -> `RAG_AGENT` fallback)
- `RAG_AGENT` -> `WEB_SEARCH_PROCESSOR_AGENT` (low retrieval confidence or insufficient info) / `apply_guardrails`
- `CONVERSATION_AGENT` / `WEB_SEARCH_PROCESSOR_AGENT` -> `apply_guardrails`
---

## Implemented Agents

### 1. CONVERSATION_AGENT

Handles general conversation, greetings, and non-medical questions. It builds the reply from the complete conversation history so that multi-turn follow-ups keep context. For medical questions it stays conservative: it avoids fabricating facts, provides general knowledge only, and reminds the user to consult a licensed medical professional for serious concerns. It does not attempt to analyze images on its own.

### 2. RAG_AGENT

The core medical knowledge agent. It runs the full RAG pipeline against the private knowledge base:

1. Rewrites the latest user question into a standalone query using the conversation history, so multi-turn follow-ups ("how is it treated?") are searchable on their own.
2. Expands the query with related medical terms to improve recall.
3. Retrieves candidate document chunks via **Qdrant hybrid search** (dense vectors + BM25 sparse vectors).
4. Reranks the candidates with a cross-encoder and keeps the most relevant chunks.
5. Generates the final answer from the selected context with the LLM, numbering each retrieved chunk so the answer can cite its sources with inline `[n]` references.

Answers include a structured source list (title, path, score, and evidence snippet), and a session-level semantic cache reuses previous retrieval/generation results for semantically similar follow-up questions.

After retrieval it computes a `retrieval_confidence` score from the relevance of the top retrieved chunks. If the score is below the configured threshold, or the model explicitly reports that the context is insufficient, the workflow automatically degrades to `WEB_SEARCH_PROCESSOR_AGENT`.

### 3. WEB_SEARCH_PROCESSOR_AGENT

Handles time-sensitive queries (latest medical developments, current outbreaks, etc.) and acts as the fallback when RAG cannot answer confidently. It rewrites the user query using recent conversation context, calls the Tavily search API to fetch up-to-date web results, and lets the LLM summarize and integrate them into a final response.
---
## RAG Pipeline

The RAG pipeline (`agents/rag_agent/`) is composed of the following components:

1. **Document parsing** (`doc_parser.py`) - uses Docling to extract text and tables from medical PDFs.
2. **Content processing** (`content_processor.py`) - LLM-based semantic chunking that respects structural boundaries so tables and sections are not split incoherently.
3. **Query rewriting** (`query_rewriter.py`) - rewrites multi-turn follow-up questions into standalone, searchable queries using the conversation history (conversational RAG).
4. **Query expansion** (`query_expander.py`) - an LLM expands the user query with related medical domain terms to improve retrieval recall.
5. **Vector store** (`vectorstore_qdrant.py`) - Qdrant collection with a dense vector index (cosine distance) and a BM25 sparse vector index; retrieval runs in hybrid mode so semantic and keyword signals are combined.
6. **Reranking** (`reranker.py`) - a cross-encoder reranks the retrieved candidates and selects the top chunks.
7. **Response generation** (`response_generator.py`) - the LLM generates the final answer from the reranked context and the conversation history, with inline `[n]` citations and a structured source list.

`MedicalRAG.process_query` also maintains a per-session **semantic cache**: similar questions within one session reuse the stored response instead of repeating retrieval and generation (configurable via `config.rag.semantic_cache_*`).

`ingest_rag_data.py` ingests PDFs (single file or whole directory) into the vector database.

> **Model download:** the local models used by RAG (embedding model `BAAI/bge-small-zh-v1.5`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, and Docling models `ds4sd/docling-models`) are large and **not pushed to the repository**; see the "Model Download" section in the root `README_clean.md` for links and paths.

---

## Research Papers and Documents Used for RAG

The private knowledge base is built from the following medical literature:

1. Saeedi, S., Rezayi, S., Keshavarz, H. et al. MRI-based brain tumor detection using convolutional deep learning methods and chosen machine learning techniques. BMC Med Inform Decis Mak 23, 16 (2023). https://doi.org/10.1186/s12911-023-02114-6

2. Babu Vimala, B., Srinivasan, S., Mathivanan, S.K. et al. Detection and classification of brain tumor using hybrid deep learning models. Sci Rep 13, 23029 (2023). https://doi.org/10.1038/s41598-023-50505-6

3. Khaliki, M.Z., Basarslan, M.S. Brain tumor detection from images and comparison with transfer learning methods and 3-layer CNN. Sci Rep 14, 2664 (2024). https://doi.org/10.1038/s41598-024-52823-9

4. Brain Tumors: an Introduction basic level, Mayfield Clinic, UCNI

5. Cleverley J, Piper J, Jones M M. The role of chest radiography in confirming covid-19 pneumonia BMJ 2020; 370 :m2426 https://doi.org/10.1136/bmj.m2426

6. Yasin, R., Gouda, W. Chest X-ray findings monitoring COVID-19 disease course and severity. Egypt J Radiol Nucl Med 51, 193 (2020). https://doi.org/10.1186/s43055-020-00296-x

7. Cozzi, D., Albanesi, M., Cavigli, E. et al. Chest X-ray in new Coronavirus Disease 2019 (COVID-19) infection: findings and correlation with clinical outcome. Radiol med 125, 730-737 (2020). https://doi.org/10.1007/s11547-020-01232-9

8. Jain, R., Gupta, M., Taneja, S. et al. Deep learning based detection and analysis of COVID-19 on chest X-ray images. Appl Intell 51, 1690-1700 (2021). https://doi.org/10.1007/s10489-020-01902-1

9. El Houby, E.M.F. COVID-19 detection from chest X-ray images using transfer learning. Sci Rep 14, 11639 (2024). https://doi.org/10.1038/s41598-024-61693-0

10. Diabetes mellitus - https://www.researchgate.net/publication/270283336_Diabetes_mellitus

11. Skin Lesion Analysis Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC). Noel C. F. Codella et al. https://doi.org/10.48550/arXiv.1710.05006

12. Zahra Mirikharaji et al. A survey on deep learning for skin lesion segmentation. Medical Image Analysis, Volume 88, 2023, 102863, ISSN 1361-8415. https://doi.org/10.1016/j.media.2023.102863
