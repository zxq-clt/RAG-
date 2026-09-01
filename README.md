# **面向医学知识问答与科研辅助的 AI 多智能体系统**

本项目构建了一个**多智能体 RAG 对话系统**，用于医学知识问答。基于 **LangGraph** 的工作流编排了三个专职智能体——对话（conversation）、RAG（检索增强生成）与网络搜索（web search）——并根据意图分析将每个用户请求动态路由到最合适的智能体。完整的 RAG 流水线（PDF 解析 → 语义分块 → 查询扩展 → 混合检索 → 重排序）将回答锚定在私有医学文献知识库中；当知识库无法给出置信回答时，系统自动降级到实时网络搜索。

## 目录
- [项目概览](#项目概览)
- [技术工作流](#技术工作流)
- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [安装与部署](#安装与部署)
- [使用说明](#使用说明)
- [贡献](#贡献)
- [许可证](#许可证)

---

## 项目概览

**多智能体医疗助手**是一个面向**医学知识问答与科研辅助**的 AI 对话系统。

**由多智能体智能驱动**，本系统集成了：
- **大语言模型（LLM）**
- 基于私有向量数据库的**检索增强生成（RAG）**
- 获取最新医学资讯的**实时网络搜索**

### 本项目的技术亮点
- 基于 LangGraph 结构化工作流的**多智能体编排**
- **进阶 RAG 技术**——混合检索、语义分块、查询扩展与重排序
- 基于置信度的路由，支持 **RAG → 网络搜索** 自动降级
- 面向安全合规医疗回答的**输入/输出双层护栏**

智能体与工作流的详细介绍见 `agents/README_zh.md`。

---

## 技术工作流

系统采用 **6 节点 LangGraph 工作流**：

| # | 节点 | 职责 |
|---|------|------|
| 1 | `analyze_input` | 分析用户输入并应用**输入护栏**（拦截 PII 泄露、提示注入、不安全/有害内容） |
| 2 | `route_to_agent` | 基于 LLM 的**意图路由**——结合最近对话上下文判断目标智能体并给出置信度；低置信度决策自动回退到 `RAG_AGENT` |
| 3 | `CONVERSATION_AGENT` | 利用多轮对话历史处理日常闲聊、问候与非医学问题 |
| 4 | `RAG_AGENT` | 从私有向量知识库回答医学知识问题，并计算 `retrieval_confidence` 检索置信度；上下文不足时标记响应 |
| 5 | `WEB_SEARCH_PROCESSOR_AGENT` | 结合对话上下文改写用户查询，通过 Tavily 搜索时效性问题；同时作为 RAG 置信度低或信息不足时的自动兜底 |
| 6 | `apply_guardrails` | 应用**输出护栏**，对生成回答做医疗合规性审核，过滤不安全或不恰当内容后返回 |

**条件路由（LangGraph 条件边）：**
- `analyze_input` → `apply_guardrails`（输入被拦截）/ `route_to_agent`（输入通过）
- `route_to_agent` → `CONVERSATION_AGENT` / `RAG_AGENT` / `WEB_SEARCH_PROCESSOR_AGENT`（低置信度回退 `RAG_AGENT`）
- `RAG_AGENT` → `WEB_SEARCH_PROCESSOR_AGENT`（检索置信度低或信息不足）/ `apply_guardrails`
- `CONVERSATION_AGENT` / `WEB_SEARCH_PROCESSOR_AGENT` → `apply_guardrails`

---
## 核心特性

- **多智能体架构**：专职智能体协同完成信息检索、推理与对话应答

- **进阶 RAG 检索系统**：
  - 基于 Docling 的 PDF 解析，提取文本与表格
  - 基于 LLM 的语义分块，感知结构边界，避免表格与章节被不合理拆散
  - 多轮查询改写：让"它怎么治疗？"这类追问也能结合上下文独立检索
  - 基于 LLM 的医学术语查询扩展，提升召回率
  - Qdrant 混合检索：BM25 稀疏关键词检索 + 稠密向量语义检索
  - 基于交叉编码器的检索结果重排序，保证进入 LLM 的上下文质量
  - 回答内联 `[n]` 引用，并附结构化来源面板（标题、链接、相关度得分、证据片段）
  - 会话级语义缓存：相似追问直接复用检索与生成结果，降低延迟
  - 输入/输出护栏确保回答安全、合规、相关
  - 基于置信度的 RAG 与网络搜索自动切换，避免幻觉与"不知道"式空答

- **实时科研集成**：网络搜索智能体检索最新医学研究论文与动态

- **基于置信度的路由**：检索置信度打分确保回答有据可依，置信度低时自动降级到网络搜索

- **多轮对话**：会话状态持久化实现上下文感知的多轮对话；RAG 智能体对追问做改写并缓存会话内结果

- **答案级评测**：30 题基准集（`eval_questions.json`）+ 评测脚本（`eval_rag.py` 检索指标、`eval_answers.py` LLM 判分的忠实度/相关性）+ GitHub Actions 回归工作流

- **直观的用户界面**：为无技术背景的用户设计

---

## 技术栈

| 组件 | 技术 |
|-----------|-------------|
| **后端框架** | FastAPI |
| **智能体编排** | LangGraph |
| **文档解析** | Docling |
| **知识存储** | Qdrant 向量数据库 |
| **护栏** | LangChain |
| **前端** | HTML, CSS, JavaScript |
| **部署** | Docker, GitHub Actions CI/CD |

---

## 模型下载

本项目所需的模型权重，请先按下表下载并放置到对应目录：

| 模型 | 用途 | 下载链接 | 放置路径 |
|------|------|----------|----------|
| `BAAI/bge-small-zh-v1.5` | 嵌入模型，生成查询/文档稠密向量（中文医学语义检索） | https://huggingface.co/BAAI/bge-small-zh-v1.5 | `./models/bge-small-zh-v1.5` |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 交叉编码器，对检索结果重排序 | https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2 | `./ms-marco-MiniLM-L-6-v2` |
| `ds4sd/docling-models` | Docling PDF 解析模型（版面分析 / 表格识别） | https://huggingface.co/ds4sd/docling-models | `./docling-models` |

推荐使用 `huggingface_hub` 下载（国内环境可设置 `HF_ENDPOINT=https://hf-mirror.com` 使用镜像）：

```bash
pip install -U huggingface_hub

huggingface-cli download BAAI/bge-small-zh-v1.5 --local-dir ./models/bge-small-zh-v1.5
huggingface-cli download cross-encoder/ms-marco-MiniLM-L-6-v2 --local-dir ./ms-marco-MiniLM-L-6-v2
huggingface-cli download ds4sd/docling-models --local-dir ./docling-models
```

> 提示：也可以通过环境变量 `EMBEDDING_MODEL`、`RERANKER_MODEL` 指定嵌入/重排序模型路径（见 `config.py`）。
## 安装与部署

### 方式一：使用 Docker

**前置条件：**
- 已安装 [Docker](https://docs.docker.com/get-docker/)
- 所需服务的 API Key

**1. 克隆仓库**
```bash
git clone <your-repo-url>
cd Multi-Agent-Medical-Assistant
```

**2. 创建环境变量文件**

在项目根目录创建 `.env` 文件并填入以下 API Key：

> **注意：** 项目默认使用 DeepSeek API（兼容 OpenAI 接口）。如需更换其他 LLM，请修改 `config.py` 中的模型定义并配置相应的环境变量。

```bash
# LLM 配置（DeepSeek API，兼容 OpenAI 接口）
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-v4-flash

# 网络搜索 API Key（注册 Tavily 可获得免费额度）
TAVILY_API_KEY=

#（可选）使用 Qdrant 服务端时填写；本地模式无需 API Key
QDRANT_URL=
QDRANT_API_KEY=
```

**3. 构建 Docker 镜像**
```bash
docker build -t medical-assistant .
```

**4. 运行 Docker 容器**
```bash
docker run -d --name medical-assistant-app -p 8000:8000 --env-file .env medical-assistant
```
应用访问地址：http://localhost:8000

**5. 在容器内导入知识库数据**

- 导入单个文档：
```bash
docker exec medical-assistant-app python ingest_rag_data.py --file ./data/raw/brain_tumors_ucni.pdf
```

- 导入目录下的多个文档：
```bash
docker exec medical-assistant-app python ingest_rag_data.py --dir ./data/raw
```

### 方式二：不使用 Docker

**1. 克隆仓库**
```bash
git clone <your-repo-url>
cd Multi-Agent-Medical-Assistant
```

**2. 创建并激活虚拟环境**
- 使用 conda：
```bash
conda create --name <environment-name> python=3.11
conda activate <environment-name>
```
- 使用 python venv：
```bash
python -m venv <environment-name>
source <environment-name>/bin/activate  # Mac/Linux
<environment-name>\Scripts\activate     # Windows
```

**3. 安装依赖**
```bash
pip install -r requirements.txt
```

**4. 配置 API Key**

创建 `.env` 文件，填入与方式一相同的 API Key。

**5. 运行应用**
```bash
python app.py
```
应用访问地址：http://localhost:8000

**6. 向向量数据库导入更多数据**
- 一次导入一个文档：
```bash
python ingest_rag_data.py --file ./data/raw/brain_tumors_ucni.pdf
```
- 导入目录下的多个文档：
```bash
python ingest_rag_data.py --dir ./data/raw
```

---

## 使用说明

> **提示：**
> 1. 首次运行会下载重排序模型并加载本地嵌入模型，请耐心等待。
> 2. 模型加载完成后即可流畅使用。

- 向系统提问医学问题：利用私有知识库的**检索增强生成（RAG）**回答，或通过**网络搜索**获取最新信息。

---

## 贡献

欢迎贡献！请在 [issues](<your-repo-url>/issues) 页面查看功能请求与改进建议。

---

## 许可证

本项目基于 **Apache-2.0 许可证** 开源。详见 [LICENSE](LICENSE) 文件。
