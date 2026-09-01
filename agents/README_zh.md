# 智能体详解

多智能体医疗助手中已实现智能体与 LangGraph 工作流的详细介绍。

## 目录
- [工作流概览](#工作流概览)
- [已实现智能体](#已实现智能体)
- [RAG 流水线](#rag-流水线)
- [RAG 使用的文献与文档](#rag-使用的文献与文档)
---

## 工作流概览

系统采用 **6 节点 LangGraph 工作流**，编排三个智能体：

| # | 节点 | 职责 |
|---|------|------|
| 1 | `analyze_input` | 分析用户输入并应用**输入护栏**：拦截 PII 泄露、提示注入、不安全/有害内容与滥用行为；被拦截时直接跳转到 `apply_guardrails` |
| 2 | `route_to_agent` | 基于 LLM 的**意图路由**：结合最近对话上下文分析查询，返回目标智能体与置信度；低置信度决策回退到 `RAG_AGENT` |
| 3 | `CONVERSATION_AGENT` | 利用多轮对话历史处理日常闲聊、问候与非医学问题 |
| 4 | `RAG_AGENT` | 从私有向量知识库回答医学知识问题，并计算 `retrieval_confidence` 检索置信度；检索上下文不足时标记响应 |
| 5 | `WEB_SEARCH_PROCESSOR_AGENT` | 结合对话上下文改写用户查询，通过 Tavily 搜索时效性问题；同时作为 RAG 置信度低或信息不足时的自动兜底 |
| 6 | `apply_guardrails` | 应用**输出护栏**：对生成回答进行医疗合规性审核，过滤不安全或不恰当内容后返回 |

**条件路由（LangGraph 条件边）：**
- `analyze_input` → `apply_guardrails`（输入被拦截）/ `route_to_agent`（输入通过）
- `route_to_agent` → `CONVERSATION_AGENT` / `RAG_AGENT` / `WEB_SEARCH_PROCESSOR_AGENT`（低置信度 → `RAG_AGENT` 兜底）
- `RAG_AGENT` → `WEB_SEARCH_PROCESSOR_AGENT`（检索置信度低或信息不足）/ `apply_guardrails`
- `CONVERSATION_AGENT` / `WEB_SEARCH_PROCESSOR_AGENT` → `apply_guardrails`
---

## 已实现智能体

### 1. CONVERSATION_AGENT（对话智能体）

处理日常对话、问候与非医学问题。它基于完整对话历史构建回复，保证多轮追问保持上下文。面对医学问题保持谨慎：不编造事实，只提供一般性知识，并提醒用户严重问题应咨询持证医疗专业人员。它不会自行分析图片。

### 2. RAG_AGENT（医学文献检索智能体）

核心医学知识智能体，针对私有知识库执行完整 RAG 流水线：

1. 结合对话历史，将用户最新问题改写为独立可检索的问题，使"它怎么治疗？"这类多轮追问也能独立检索。
2. 用相关医学术语扩展查询，提升召回率。
3. 通过 **Qdrant 混合检索**（稠密向量 + BM25 稀疏向量）召回候选文档块。
4. 用交叉编码器对候选块重排序，保留最相关的块。
5. 由 LLM 基于精选上下文与对话历史生成最终回答，并为每个检索块编号，使回答能以 `[n]` 内联引用标注来源。

回答附带结构化来源列表（标题、路径、得分与证据片段）；同时，会话级语义缓存会为语义相似的追问复用之前的检索与生成结果。

检索后系统根据 top 块的相关性计算 `retrieval_confidence` 置信度。当得分低于配置阈值，或模型明确表示上下文不足时，工作流自动降级到 `WEB_SEARCH_PROCESSOR_AGENT`。

### 3. WEB_SEARCH_PROCESSOR_AGENT（网络搜索智能体）

处理时效性问题（最新医学进展、当前疫情等），并在 RAG 无法置信作答时充当兜底。它利用最近对话上下文改写用户查询，调用 Tavily 搜索 API 获取最新网页结果，再由 LLM 汇总整合为最终回答。
---
## RAG 流水线

RAG 流水线（`agents/rag_agent/`）由以下组件构成：

1. **文档解析**（`doc_parser.py`）- 使用 Docling 从医学 PDF 中提取文本与表格。
2. **内容处理**（`content_processor.py`）- 基于 LLM 的语义分块，尊重结构边界，避免表格与章节被不合理拆散。
3. **查询改写**（`query_rewriter.py`）- 结合对话历史，将多轮追问改写为独立、可检索的问题（对话式 RAG）。
4. **查询扩展**（`query_expander.py`）- LLM 使用相关医学术语扩展用户查询，提升检索召回率。
5. **向量存储**（`vectorstore_qdrant.py`）- Qdrant 集合包含稠密向量索引（余弦距离）与 BM25 稀疏向量索引；检索以混合模式运行，融合语义与关键词信号。
6. **重排序**（`reranker.py`）- 交叉编码器对召回候选重排序并选出 top 块。
7. **回答生成**（`response_generator.py`）- LLM 基于重排序后的上下文与对话历史生成最终回答，带 `[n]` 内联引用与结构化来源列表。

`MedicalRAG.process_query` 还维护了会话级**语义缓存**：同一会话内的相似问题直接复用已存储的回答，而无需重复检索与生成（可通过 `config.rag.semantic_cache_*` 配置）。

`ingest_rag_data.py` 用于将 PDF（单个文件或整个目录）导入向量数据库。

> **模型下载：** RAG 使用的本地模型（嵌入模型 `BAAI/bge-small-zh-v1.5`、重排序模型 `cross-encoder/ms-marco-MiniLM-L-6-v2`、Docling 模型 `ds4sd/docling-models`）体积较大，**不会随仓库推送**；下载链接与放置路径见根目录 `README_zh.md` 的「模型下载」章节。

---

## RAG 使用的文献与文档

私有知识库由以下医学文献构建：

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