"""
Agent Decision System for Multi-Agent Medical Chatbot

This module handles the orchestration of different agents using LangGraph.
It dynamically routes user queries to the appropriate agent based on content and context.
"""

import json
from typing import Dict, List, Optional, Any, Literal, TypedDict, Union, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import MessagesState, StateGraph, END
import os, getpass
from dotenv import load_dotenv
from agents.rag_agent import MedicalRAG
from agents.web_search_processor_agent import WebSearchProcessorAgent
from agents.image_analysis_agent import ImageAnalysisAgent
from agents.guardrails.local_guardrails import LocalGuardrails

from langgraph.checkpoint.memory import MemorySaver

import cv2
import numpy as np

from config import Config

load_dotenv()

# Load configuration
config = Config()

# Initialize memory
memory = MemorySaver()

# Lazy-initialized singleton graph (avoid rebuilding on every request)
_graph = None


def get_graph():
    """Return the singleton LangGraph instance (lazy init)."""
    global _graph
    if _graph is None:
        _graph = create_agent_graph()
    return _graph


# Lazy singleton agents - heavy models are loaded once, not per request
_rag_agent = None
_web_search_processor = None


def _get_rag_agent():
    """Return the singleton MedicalRAG instance (lazy init)."""
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = MedicalRAG(config)
    return _rag_agent


def _get_web_search_processor():
    """Return the singleton WebSearchProcessorAgent instance (lazy init)."""
    global _web_search_processor
    if _web_search_processor is None:
        _web_search_processor = WebSearchProcessorAgent(config)
    return _web_search_processor


# Agent that takes the decision of routing the request further to correct task specific agent
class AgentConfig:
    """Configuration settings for the agent decision system."""
    
    # Confidence threshold for routing decisions
    CONFIDENCE_THRESHOLD = 0.85
    
    # System instructions for the decision agent
    # DECISION_SYSTEM_PROMPT = """You are an intelligent medical triage system that routes user queries to
    # the appropriate specialized agent. Your job is to analyze the user's request and determine which agent
    # is best suited to handle it based on the query content, presence of images, and conversation context.
    #
    # Available agents:
    # 1. CONVERSATION_AGENT - For general chat, greetings, and non-medical questions.
    # 2. RAG_AGENT - For specific medical knowledge questions that can be answered from established medical literature. Currently ingested medical knowledge involves 'introduction to brain tumor', 'deep learning techniques to diagnose and detect brain tumors', 'deep learning techniques to diagnose and detect covid / covid-19 from chest x-ray'.
    # 3. WEB_SEARCH_PROCESSOR_AGENT - For questions about recent medical developments, current outbreaks, or time-sensitive medical information.
    # 4. BRAIN_TUMOR_AGENT - For analysis of brain MRI images to detect and segment tumors.
    # 5. CHEST_XRAY_AGENT - For analysis of chest X-ray images to detect abnormalities.
    # 6. SKIN_LESION_AGENT - For analysis of skin lesion images to classify them as benign or malignant.
    #
    # Make your decision based on these guidelines:
    # - If the user has not uploaded any image, always route to the conversation agent.
    # - If the user uploads a medical image, decide which medical vision agent is appropriate based on the image type and the user's query. If the image is uploaded without a query, always route to the correct medical vision agent based on the image type.
    # - If the user asks about recent medical developments or current health situations, use the web search pocessor agent.
    # - If the user asks specific medical knowledge questions, use the RAG agent.
    # - For general conversation, greetings, or non-medical questions, use the conversation agent. But if image is uploaded, always go to the medical vision agents first.
    #
    # You must provide your answer in JSON format with the following structure:
    # {{
    # "agent": "AGENT_NAME",
    # "reasoning": "Your step-by-step reasoning for selecting this agent",
    # "confidence": 0.95  // Value between 0.0 and 1.0 indicating your confidence in this decision
    # }}
    # """
    DECISION_SYSTEM_PROMPT = """你是一个智能医疗分诊系统，负责将用户查询路由到最合适的专业代理。你的任务是分析用户的请求，并根据查询内容、是否包含图像以及对话上下文，确定哪个代理最适合处理该请求。

    可用的代理：
    1.
    CONVERSATION_AGENT - 用于一般聊天、问候和非医疗问题。
    2.
    RAG_AGENT - 用于可以从已有医学文献中回答的具体医学知识问题。目前已收录的医学知识包括：“脑肿瘤简介”、“用于诊断和检测脑肿瘤的深度学习技术”、“用于从胸部X光片诊断和检测新冠 / COVID - 19
    的深度学习技术”。
    3.
    WEB_SEARCH_PROCESSOR_AGENT - 用于关于近期医学进展、当前疫情爆发或时间敏感的医疗信息的问题。
    4.
    BRAIN_TUMOR_AGENT - 用于分析脑部MRI图像以检测和分割肿瘤。
    5.
    CHEST_XRAY_AGENT - 用于分析胸部X光图像以检测异常。
    6.
    SKIN_LESION_AGENT - 用于分析皮肤病变图像以分类为良性或恶性。

    请根据以下指南做出决定：
    - 如果用户没有上传任何图像，始终路由到对话代理（CONVERSATION_AGENT）。
    - 如果用户上传了医学图像，请根据图像类型和用户查询决定哪个医学视觉代理是合适的。如果仅上传图像而没有附带查询，始终根据图像类型路由到正确的医学视觉代理。
    - 如果用户询问近期的医学发展或当前健康状况，请使用
    WEB_SEARCH_PROCESSOR_AGENT。
    - 如果用户询问具体的医学知识问题，请使用
    RAG_AGENT。
    - 对于一般对话、问候或非医疗问题，请使用对话代理（CONVERSATION_AGENT）。但如果上传了图像，始终优先路由到医学视觉代理。

    你必须以
    JSON
    格式提供答案，结构如下：
    {{
        "agent": "代理名称",
        "reasoning": "选择该代理的分步推理过程",
        "confidence": 0.95 // 介于 0.0 和 1.0 之间的数值，表示你对此决定的置信度
    }}
    """
    image_analyzer = ImageAnalysisAgent(config=config)


class AgentState(MessagesState):
    """State maintained across the workflow."""
    # messages: List[BaseMessage]  # Conversation history
    agent_name: Optional[str]  # Current active agent
    current_input: Optional[Union[str, Dict]]  # Input to be processed
    has_image: bool  # Whether the current input contains an image
    image_type: Optional[str]  # Type of medical image if present
    output: Optional[str]  # Final output to user
    needs_human_validation: bool  # Whether human validation is required
    retrieval_confidence: float  # Confidence in retrieval (for RAG agent)
    bypass_routing: bool  # Flag to bypass agent routing for guardrails
    insufficient_info: bool  # Flag indicating RAG response has insufficient information
    rag_sources: Optional[List[Dict[str, Any]]]  # Structured sources with citations from RAG agent
    session_id: Optional[str]  # Per-session id used for semantic cache isolation


class AgentDecision(TypedDict):
    """Output structure for the decision agent."""
    agent: str
    reasoning: str
    confidence: float


def create_agent_graph():
    """Create and configure the LangGraph for agent orchestration."""

    # Initialize guardrails with the same LLM used elsewhere
    guardrails = LocalGuardrails(config.rag.llm)

    # LLM
    decision_model = config.agent_decision.llm
    
    # Initialize the output parser
    json_parser = JsonOutputParser(pydantic_object=AgentDecision)
    
    # Create the decision prompt
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", AgentConfig.DECISION_SYSTEM_PROMPT),
        ("human", "{input}")
    ])
    
    # Create the decision chain
    decision_chain = decision_prompt | decision_model | json_parser
    
    # Define graph state transformations
    def analyze_input(state: AgentState) -> AgentState:
        """Analyze the input to detect images and determine input type."""
        current_input = state["current_input"]
        has_image = False
        image_type = None
        
        # Get the text from the input
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Check input through guardrails if text is present
        if input_text:
            is_allowed, message = guardrails.check_input(input_text)
            if not is_allowed:
                # If input is blocked, return early with guardrail message
                print(f"Selected agent: INPUT GUARDRAILS, Message: ", message)
                return {
                    **state,
                    "messages": message,
                    "agent_name": "INPUT_GUARDRAILS",
                    "has_image": False,
                    "image_type": None,
                    "bypass_routing": True  # flag to end flow
                }
        
        # Original image processing code
        if isinstance(current_input, dict) and "image" in current_input:
            has_image = True
            image_path = current_input.get("image", None)
            image_type_response = AgentConfig.image_analyzer.analyze_image(image_path)
            image_type = image_type_response['image_type']
            print("ANALYZED IMAGE TYPE: ", image_type)
        
        return {
            **state,
            "has_image": has_image,
            "image_type": image_type,
            "bypass_routing": False  # Explicitly set to False for normal flow
        }
    
    def check_if_bypassing(state: AgentState) -> str:
        """Check if we should bypass normal routing due to guardrails."""
        if state.get("bypass_routing", False):
            return "apply_guardrails"
        return "route_to_agent"
    
    def route_to_agent(state: AgentState) -> Dict:
        """Make decision about which agent should handle the query."""
        messages = state["messages"]
        current_input = state["current_input"]
        has_image = state["has_image"]
        image_type = state["image_type"]
        
        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history (last 3 messages)
        recent_context = ""
        for msg in messages[-6:]:  # Get last 3 exchanges (6 messages)  # Not provided control from config
            if isinstance(msg, HumanMessage):
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                recent_context += f"Assistant: {msg.content}\n"
        
        # Combine everything for the decision input
        decision_input = f"""
        User query: {input_text}

        Recent conversation context:
        {recent_context}

        Has image: {has_image}
        Image type: {image_type if has_image else 'None'}

        Based on this information, which agent should handle this query?
        """
        
        # Make the decision
        decision = decision_chain.invoke({"input": decision_input})

        # Decided agent
        print(f"Decision: {decision['agent']}")
        
        # Update state with decision
        updated_state = {
            **state,
            "agent_name": decision["agent"],
        }
        
        # Route based on agent name and confidence
        if decision["confidence"] < AgentConfig.CONFIDENCE_THRESHOLD:
            return {"agent_state": updated_state, "next": "needs_validation"}
        
        return {"agent_state": updated_state, "next": decision["agent"]}

    # Define agent execution functions (these will be implemented in their respective modules)
    def run_conversation_agent(state: AgentState) -> AgentState:
        """Handle general conversation."""

        print(f"Selected agent: CONVERSATION_AGENT")

        messages = state["messages"]
        current_input = state["current_input"]
        
        # Prepare input for decision model
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Create context from recent conversation history
        recent_context = ""
        for msg in messages:#[-20:]:  # Get last 10 exchanges (20 messages)  # currently considering complete history - limit control from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"
        
        # Combine everything for the decision input
        # conversation_prompt = f"""User query: {input_text}
        #
        # Recent conversation context: {recent_context}
        #
        # You are an AI-powered Medical Conversation Assistant. Your goal is to facilitate smooth and informative conversations with users, handling both casual and medical-related queries. You must respond naturally while ensuring medical accuracy and clarity.
        #
        # ### Role & Capabilities
        # - Engage in **general conversation** while maintaining professionalism.
        # - Answer **medical questions** using verified knowledge.
        # - Route **complex queries** to RAG (retrieval-augmented generation) or web search if needed.
        # - Handle **follow-up questions** while keeping track of conversation context.
        # - Redirect **medical images** to the appropriate AI analysis agent.
        #
        # ### Guidelines for Responding:
        # 1. **General Conversations:**
        # - If the user engages in casual talk (e.g., greetings, small talk), respond in a friendly, engaging manner.
        # - Keep responses **concise and engaging**, unless a detailed answer is needed.
        #
        # 2. **Medical Questions:**
        # - If you have **high confidence** in answering, provide a medically accurate response.
        # - Ensure responses are **clear, concise, and factual**.
        #
        # 3. **Follow-Up & Clarifications:**
        # - Maintain conversation history for better responses.
        # - If a query is unclear, ask **follow-up questions** before answering.
        #
        # 4. **Handling Medical Image Analysis:**
        # - Do **not** attempt to analyze images yourself.
        # - If user speaks about analyzing or processing or detecting or segmenting or classifying any disease from any image, ask the user to upload the image so that in the next turn it is routed to the appropriate medical vision agents.
        # - If an image was uploaded, it would have been routed to the medical computer vision agents. Read the history to know about the diagnosis results and continue conversation if user asks anything regarding the diagnosis.
        # - After processing, **help the user interpret the results**.
        #
        # 5. **Uncertainty & Ethical Considerations:**
        # - If unsure, **never assume** medical facts.
        # - Recommend consulting a **licensed healthcare professional** for serious medical concerns.
        # - Avoid providing **medical diagnoses** or **prescriptions**—stick to general knowledge.
        #
        # ### Response Format:
        # - Maintain a **conversational yet professional tone**.
        # - Use **bullet points or numbered lists** for clarity when needed.
        # - If pulling from external sources (RAG/Web Search), mention **where the information is from** (e.g., "According to Mayo Clinic...").
        # - If a user asks for a diagnosis, remind them to **seek medical consultation**.
        #
        # ### Example User Queries & Responses:
        #
        # **User:** "Hey, how's your day going?"
        # **You:** "I'm here and ready to help! How can I assist you today?"
        #
        # **User:** "I have a headache and fever. What should I do?"
        # **You:** "I'm not a doctor, but headaches and fever can have various causes, from infections to dehydration. If your symptoms persist, you should see a medical professional."
        #
        # Conversational LLM Response:"""
        conversation_prompt = f"""用户查询：{input_text}

                最近的对话上下文：{recent_context}

                你是一个基于人工智能的医疗对话助手。你的目标是与用户进行顺畅且信息丰富的交流，同时处理日常聊天和医疗相关的问题。你必须在确保医学准确性和清晰性的前提下自然作答。

                ### 角色与能力
                - 在保持专业性的同时进行**一般对话**。
                - 使用已验证的知识回答**医学问题**。
                - 如有需要，将**复杂查询**路由至 RAG（检索增强生成）或网络搜索。
                - 处理**后续追问**，同时跟踪对话上下文。
                - 将**医学图像**重定向到相应的 AI 分析代理。

                ### 回答准则：
                1. **一般对话：**
                   - 如果用户进行随意交谈（例如问候、闲聊），请以友好、引人入胜的方式回应。
                   - 除非需要详细回答，否则保持回复**简洁且富有吸引力**。

                2. **医学问题：**
                   - 如果你对答案有**高度把握**，请提供医学上准确的回答。
                   - 确保回复**清晰、简洁、基于事实**。

                3. **追问与澄清：**
                   - 保持对话历史，以便给出更好的回复。
                   - 如果查询不明确，请在回答前**提出追问**。

                4. **处理医学图像分析：**
                   - 请**不要**自行尝试分析图像。
                   - 如果用户提及要对任何图像进行分析、处理、检测、分割或分类某种疾病，请要求用户上传图像，以便在下一轮将其路由到合适的医学视觉代理。
                   - 如果用户已经上传了图像，那么它应该已被路由至医学计算机视觉代理。请阅读历史记录以了解诊断结果，并在用户就诊断结果提出询问时继续对话。
                   - 处理完成后，**帮助用户解读结果**。

                5. **不确定性与伦理考量：**
                   - 如果不确定，**绝不要**臆造医学事实。
                   - 建议用户就严重的健康问题咨询**持牌医疗专业人员**。
                   - 避免提供**医学诊断**或**处方**——仅提供一般性知识。

                ### 回复格式：
                - 保持**口语化但专业的语气**。
                - 需要时使用**项目符号或编号列表**来提高清晰度。
                - 如果引用了外部来源（RAG / 网络搜索），请说明**信息来源**（例如：“根据梅奥诊所……”）。
                - 如果用户要求诊断，请提醒他们**寻求医疗咨询**。

                ### 示例用户查询与回复：

                **用户：**“嘿，你今天过得怎么样？”
                **你：**“我在这里并准备提供帮助！今天我能为你做些什么？”

                **用户：**“我头疼，还发烧，该怎么办？”
                **你：**“我不是医生，但头痛和发烧可能有多种原因，从感染到脱水都有可能。如果症状持续，你应该去看医疗专业人员。”

                对话大语言模型回复："""
        # print("Conversation Prompt:", conversation_prompt)

        response = config.conversation.llm.invoke(conversation_prompt)

        # print("Conversation respone:", response)

        # response = AIMessage(content="This would be handled by the conversation agent.")

        return {
            **state,
            "output": response,
            "agent_name": "CONVERSATION_AGENT"
        }
    
    def run_rag_agent(state: AgentState) -> AgentState:
        """Handle medical knowledge queries using RAG."""
        # Initialize the RAG agent

        print(f"Selected agent: RAG_AGENT")

        rag_agent = _get_rag_agent()
        
        messages = state["messages"]
        query = state["current_input"]
        rag_context_limit = config.rag.context_limit

        recent_context = ""
        for msg in messages[-rag_context_limit:]:# limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        response = rag_agent.process_query(query, chat_history=recent_context, session_id=state.get("session_id"))
        retrieval_confidence = response.get("confidence", 0.0)  # Default to 0.0 if not provided

        print(f"Retrieval Confidence: {retrieval_confidence}")
        print(f"Sources: {len(response['sources'])}")

        # Check if response indicates insufficient information
        insufficient_info = False
        response_content = response["response"]
        
        # Extract the content properly based on type
        if isinstance(response_content, dict) and hasattr(response_content, 'content'):
            # If it's an AIMessage or similar object with a content attribute
            response_text = response_content.content
        else:
            # If it's already a string
            response_text = response_content
            
        print(f"Response text type: {type(response_text)}")
        print(f"Response text preview: {response_text[:100]}...")
        
        if isinstance(response_text, str) and (
            "I don't have enough information to answer this question based on the provided context" in response_text or 
            "I don't have enough information" in response_text or 
            "don't have enough information" in response_text.lower() or
            "not enough information" in response_text.lower() or
            "insufficient information" in response_text.lower() or
            "cannot answer" in response_text.lower() or
            "unable to answer" in response_text.lower()
            ):
            
            print("RAG response indicates insufficient information")
            print(f"Response text that triggered insufficient_info: {response_text[:100]}...")
            insufficient_info = True

        print(f"Insufficient info flag set to: {insufficient_info}")

        # Store RAG output ONLY if confidence is high
        if retrieval_confidence >= config.rag.min_retrieval_confidence:
            # response_output = response["response"]
            response_output = AIMessage(content=response_text)
        else:
            response_output = AIMessage(content="")
        
        return {
            **state,
            "output": response_output,
            "needs_human_validation": False,  # Assuming no validation needed for RAG responses
            "retrieval_confidence": retrieval_confidence,
            "agent_name": "RAG_AGENT",
            "insufficient_info": insufficient_info,
            "rag_sources": response.get("sources", [])
        }

    # Web Search Processor Node
    def run_web_search_processor_agent(state: AgentState) -> AgentState:
        """Handles web search results, processes them with LLM, and generates a refined response."""

        print(f"Selected agent: WEB_SEARCH_PROCESSOR_AGENT")
        print("[WEB_SEARCH_PROCESSOR_AGENT] Processing Web Search Results...")
        
        messages = state["messages"]
        web_search_context_limit = config.web_search.context_limit

        recent_context = ""
        for msg in messages[-web_search_context_limit:]: # limit controlled from config
            if isinstance(msg, HumanMessage):
                # print("######### DEBUG 1:", msg)
                recent_context += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                # print("######### DEBUG 2:", msg)
                recent_context += f"Assistant: {msg.content}\n"

        web_search_processor = _get_web_search_processor()

        processed_response = web_search_processor.process_web_search_results(query=state["current_input"], chat_history=recent_context)

        # print("######### DEBUG WEB SEARCH:", processed_response)
        
        if state['agent_name'] != None:
            involved_agents = f"{state['agent_name']}, WEB_SEARCH_PROCESSOR_AGENT"
        else:
            involved_agents = "WEB_SEARCH_PROCESSOR_AGENT"

        # Overwrite any previous output with the processed Web Search response
        return {
            **state,
            # "output": "This would be handled by the web search agent, finding the latest information.",
            "output": processed_response,
            "agent_name": involved_agents
        }

    # Define Routing Logic
    def confidence_based_routing(state: AgentState) -> Dict[str, str]:
        """Route based on RAG confidence score and response content."""
        # Debug prints
        print(f"Routing check - Retrieval confidence: {state.get('retrieval_confidence', 0.0)}")
        print(f"Routing check - Insufficient info flag: {state.get('insufficient_info', False)}")
        
        # Redirect if confidence is low or if response indicates insufficient info
        if (state.get("retrieval_confidence", 0.0) < config.rag.min_retrieval_confidence or 
            state.get("insufficient_info", False)):
            print("Re-routed to Web Search Agent due to low confidence or insufficient information...")
            return "WEB_SEARCH_PROCESSOR_AGENT"  # Correct format
        return "check_validation"  # No transition needed if confidence is high and info is sufficient
    
    def run_brain_tumor_agent(state: AgentState) -> AgentState:
        """Handle brain MRI image analysis."""

        print(f"Selected agent: BRAIN_TUMOR_AGENT")

        response = AIMessage(content="This would be handled by the brain tumor agent, analyzing the MRI image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "BRAIN_TUMOR_AGENT"
        }
    
    def run_chest_xray_agent(state: AgentState) -> AgentState:
        """Handle chest X-ray image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        print(f"Selected agent: CHEST_XRAY_AGENT")

        # classify chest x-ray into covid or normal
        predicted_class = AgentConfig.image_analyzer.classify_chest_xray(image_path)

        if predicted_class == "covid19":
            response = AIMessage(content="The analysis of the uploaded chest X-ray image indicates a **POSITIVE** result for **COVID-19**.")
        elif predicted_class == "normal":
            response = AIMessage(content="The analysis of the uploaded chest X-ray image indicates a **NEGATIVE** result for **COVID-19**, i.e., **NORMAL**.")
        else:
            response = AIMessage(content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image.")

        # response = AIMessage(content="This would be handled by the chest X-ray agent, analyzing the image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "CHEST_XRAY_AGENT"
        }
    
    def run_skin_lesion_agent(state: AgentState) -> AgentState:
        """Handle skin lesion image analysis."""

        current_input = state["current_input"]
        image_path = current_input.get("image", None)

        print(f"Selected agent: SKIN_LESION_AGENT")

        # classify chest x-ray into covid or normal
        predicted_mask = AgentConfig.image_analyzer.segment_skin_lesion(image_path)

        if predicted_mask:
            response = AIMessage(content="Following is the analyzed **segmented** output of the uploaded skin lesion image:")
        else:
            response = AIMessage(content="The uploaded image is not clear enough to make a diagnosis / the image is not a medical image.")

        # response = AIMessage(content="This would be handled by the skin lesion agent, analyzing the skin image.")

        return {
            **state,
            "output": response,
            "needs_human_validation": True,  # Medical diagnosis always needs validation
            "agent_name": "SKIN_LESION_AGENT"
        }
    
    def handle_human_validation(state: AgentState) -> Dict:
        """Prepare for human validation if needed."""
        if state.get("needs_human_validation", False):
            return {"agent_state": state, "next": "human_validation", "agent": "HUMAN_VALIDATION"}
        return {"agent_state": state, "next": END}
    
    def perform_human_validation(state: AgentState) -> AgentState:
        """Handle human validation process."""
        print(f"Selected agent: HUMAN_VALIDATION")

        # Append validation request to the existing output
        validation_prompt = f"{state['output'].content}\n\n**Human Validation Required:**\n- If you're a healthcare professional: Please validate the output. Select **Yes** or **No**. If No, provide comments.\n- If you're a patient: Simply click Yes to confirm."

        # Create an AI message with the validation prompt
        validation_message = AIMessage(content=validation_prompt)

        return {
            **state,
            "output": validation_message,
            "agent_name": f"{state['agent_name']}, HUMAN_VALIDATION"
        }

    # Check output through guardrails
    def apply_output_guardrails(state: AgentState) -> AgentState:
        """Apply output guardrails to the generated response."""
        output = state["output"]
        current_input = state["current_input"]

        # Check if output is valid
        if not output or not isinstance(output, (str, AIMessage)):
            return state

        output_text = output if isinstance(output, str) else output.content
        
        # If the last message was a human validation message
        if "Human Validation Required" in output_text:
            # Check if the current input is a human validation response
            validation_input = ""
            if isinstance(current_input, str):
                validation_input = current_input
            elif isinstance(current_input, dict):
                validation_input = current_input.get("text", "")
            
            # If validation input exists
            if validation_input.lower().startswith(('yes', 'no')):
                # Add the validation result to the conversation history
                validation_response = HumanMessage(content=f"Validation Result: {validation_input}")
                
                # If validation is 'No', modify the output
                if validation_input.lower().startswith('no'):
                    fallback_message = AIMessage(content="The previous medical analysis requires further review. A healthcare professional has flagged potential inaccuracies.")
                    return {
                        **state,
                        "messages": [validation_response, fallback_message],
                        "output": fallback_message
                    }
                
                return {
                    **state,
                    "messages": validation_response
                }
        
        # Get the original input text
        input_text = ""
        if isinstance(current_input, str):
            input_text = current_input
        elif isinstance(current_input, dict):
            input_text = current_input.get("text", "")
        
        # Apply output sanitization
        sanitized_output = guardrails.check_output(output_text, input_text)
        # sanitized_output = output_text
        
        # For non-validation cases, add the sanitized output to messages
        sanitized_message = AIMessage(content=sanitized_output) if isinstance(output, AIMessage) else sanitized_output
        
        return {
            **state,
            "messages": sanitized_message,
            "output": sanitized_message
        }

    
    # Create the workflow graph
    workflow = StateGraph(AgentState)
    
    # Add nodes for each step
    workflow.add_node("analyze_input", analyze_input)
    workflow.add_node("route_to_agent", route_to_agent)
    workflow.add_node("CONVERSATION_AGENT", run_conversation_agent)
    workflow.add_node("RAG_AGENT", run_rag_agent)
    workflow.add_node("WEB_SEARCH_PROCESSOR_AGENT", run_web_search_processor_agent)
    workflow.add_node("BRAIN_TUMOR_AGENT", run_brain_tumor_agent)
    workflow.add_node("CHEST_XRAY_AGENT", run_chest_xray_agent)
    workflow.add_node("SKIN_LESION_AGENT", run_skin_lesion_agent)
    workflow.add_node("check_validation", handle_human_validation)
    workflow.add_node("human_validation", perform_human_validation)
    workflow.add_node("apply_guardrails", apply_output_guardrails)
    
    # Define the edges (workflow connections)
    workflow.set_entry_point("analyze_input")
    # workflow.add_edge("analyze_input", "route_to_agent")
    # Add conditional routing for guardrails bypass
    workflow.add_conditional_edges(
        "analyze_input",
        check_if_bypassing,
        {
            "apply_guardrails": "apply_guardrails",
            "route_to_agent": "route_to_agent"
        }
    )
    
    # Connect decision router to agents
    workflow.add_conditional_edges(
        "route_to_agent",
        lambda x: x["next"],
        {
            "CONVERSATION_AGENT": "CONVERSATION_AGENT",
            "RAG_AGENT": "RAG_AGENT",
            "WEB_SEARCH_PROCESSOR_AGENT": "WEB_SEARCH_PROCESSOR_AGENT",
            "BRAIN_TUMOR_AGENT": "BRAIN_TUMOR_AGENT",
            "CHEST_XRAY_AGENT": "CHEST_XRAY_AGENT",
            "SKIN_LESION_AGENT": "SKIN_LESION_AGENT",
            "needs_validation": "RAG_AGENT"  # Default to RAG if confidence is low
        }
    )
    
    # Connect agent outputs to validation check
    workflow.add_edge("CONVERSATION_AGENT", "check_validation")
    # workflow.add_edge("RAG_AGENT", "check_validation")
    workflow.add_edge("WEB_SEARCH_PROCESSOR_AGENT", "check_validation")
    workflow.add_conditional_edges("RAG_AGENT", confidence_based_routing)
    workflow.add_edge("BRAIN_TUMOR_AGENT", "check_validation")
    workflow.add_edge("CHEST_XRAY_AGENT", "check_validation")
    workflow.add_edge("SKIN_LESION_AGENT", "check_validation")

    workflow.add_edge("human_validation", "apply_guardrails")
    workflow.add_edge("apply_guardrails", END)
    
    workflow.add_conditional_edges(
        "check_validation",
        lambda x: x["next"],
        {
            "human_validation": "human_validation",
            END: "apply_guardrails"  # Route to guardrails instead of END
        }
    )
    
    # workflow.add_edge("human_validation", END)
    
    # Compile the graph
    return workflow.compile(checkpointer=memory)


def init_agent_state() -> AgentState:
    """Initialize the agent state with default values."""
    return {
        "messages": [],
        "agent_name": None,
        "current_input": None,
        "has_image": False,
        "image_type": None,
        "output": None,
        "needs_human_validation": False,
        "retrieval_confidence": 0.0,
        "bypass_routing": False,
        "insufficient_info": False,
        "rag_sources": None,
        "session_id": None
    }


def process_query(
    query: Union[str, Dict],
    conversation_history: List[BaseMessage] = None,
    session_id: Optional[str] = None,
) -> str:
    """
    Process a user query through the agent decision system.

    Args:
        query: User input (text string or dict with text and image)
        conversation_history: Deprecated - state persists conversation history
        session_id: Optional unique id for per-user conversation isolation;
                    defaults to a fixed thread when not provided

    Returns:
        Response from the appropriate agent
    """
    # Reuse the singleton graph instead of rebuilding on every request
    graph = get_graph()

    # Per-session thread id so different users do not share history
    thread_config = {"configurable": {"thread_id": session_id or "default"}}

    # Initialize state
    state = init_agent_state()

    # Carry session id into state so RAG/web nodes can isolate per-session caches
    state["session_id"] = session_id or "default"

    # Add the current query
    state["current_input"] = query

    # To handle image upload case
    if isinstance(query, dict):
        query = query.get("text", "") + ", user uploaded an image for diagnosis."

    state["messages"] = [HumanMessage(content=query)]

    result = graph.invoke(state, thread_config)
    # print("######### DEBUG 4:", result)
    # state["messages"] = [result["messages"][-1].content]

    # Keep history to reasonable size (ANOTHER OPTION: summarize and store before truncating history)
    if len(result["messages"]) > config.max_conversation_history:  # Keep last config.max_conversation_history messages
        result["messages"] = result["messages"][-config.max_conversation_history:]

    # visualize conversation history in console
    for m in result["messages"]:
        m.pretty_print()
    
    # Add the response to conversation history
    return result