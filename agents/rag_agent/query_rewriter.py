"""
Multi-turn query rewriting for conversational RAG.

The RAG agent previously passed raw chat history to the LLM, so follow-up
questions containing pronouns or ellipsis (e.g. "How is it treated?") degraded
retrieval quality. This module rewrites the latest user question into a
standalone, searchable question using the recent conversation history.
"""

import logging
from typing import Optional

class QueryRewriter:
    """Rewrite the current user query into a standalone question using history."""

    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model = config.rag.llm

    def rewrite(self, query: str, chat_history: Optional[str] = None) -> str:
        """
        Rewrite query into a standalone question.

        Args:
            query: The latest user question
            chat_history: Raw conversation history text (User:/Assistant: lines)

        Returns:
            Standalone question string (original query if no history / on failure)
        """
        history = (chat_history or "").strip()
        if not history:
            return query

        prompt = f"""You are helping a medical knowledge retrieval system. The user is in a multi-turn conversation. Rewrite the user's LATEST question into a standalone question that contains all the context needed to be searched independently in a medical document database.

Rules:
- Resolve pronouns and ellipsis using the conversation history (e.g. "it" -> the disease/medicine mentioned earlier).
- Keep the same language as the user's latest question.
- Do NOT answer the question, only rewrite it.
- If the question is already standalone, return it unchanged.
- Output only the rewritten question, nothing else.

Conversation history:
{history[-2500:]}

Latest user question:
{query}

Rewritten standalone question:"""
        try:
            response = self.model.invoke(prompt)
            rewritten = (response.content or "").strip().strip('"')
            self.logger.info(f"Query rewritten: {query!r} -> {rewritten!r}")
            return rewritten if rewritten else query
        except Exception as exc:
            self.logger.warning(f"Query rewriting failed, fallback to original: {exc}")
            return query
