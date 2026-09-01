"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:
you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.

Each llm definition has unique temperature value relevant to the specific class.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables from .env file
load_dotenv()

# DeepSeek 通用配置（所有 LLM 共用）
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"


class AgentDecisoinConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.1  # Deterministic
        )


class ConversationConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.7  # Creative but factual
        )


class WebSearchConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.3  # Slightly creative but factual
        )
        self.context_limit = 20


class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = 512  # bge-small-zh 输出维度是 512
        self.distance_metric = "Cosine"
        self.use_local = True
        self.vector_local_path = "data/qdrant_db"
        self.doc_local_path = "data/docs_db"
        self.parsed_content_dir = "data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_assistance_rag"
        self.chunk_size = 512
        self.chunk_overlap = 50

        # 本地开源嵌入模型（免费，无需API）
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=os.getenv("EMBEDDING_MODEL", "./models/bge-small-zh-v1.5"),
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

        self.llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.3
        )
        self.summarizer_model = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.5
        )
        self.chunker_model = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.0
        )
        self.response_generator_model = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.3
        )
        # 检索阶段召回候选数（第一阶段）
        self.retrieval_top_k = 20
        self.vector_search_type = 'similarity'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        #self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        #self.reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        # 默认使用本地重排模型，可通过环境变量 RERANKER_MODEL 覆盖（如 CI 环境用在线模型）
        self.reranker_model = os.getenv("RERANKER_MODEL", "./ms-marco-MiniLM-L-6-v2")
        self.reranker_top_k = 3

        self.max_context_length = 8192
        self.include_sources = True
        self.min_retrieval_confidence = 0.40
        self.context_limit = 20

        # 对话式 RAG：多轮查询改写
        self.rewrite_query = True
        # 会话级语义缓存：相似问题直接复用检索/生成结果
        self.semantic_cache_enabled = True
        self.semantic_cache_threshold = 0.92
        self.semantic_cache_capacity = 200


class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        self.llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
            temperature=0.1
        )


class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"


class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"


class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 8000
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5


class UIConfig:
    def __init__(self):
        self.theme = "light"
        self.enable_speech = False
        self.enable_image_upload = True


class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20
