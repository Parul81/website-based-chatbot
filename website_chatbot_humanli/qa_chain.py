"""
Question Answering Chain Module
Handles question answering using retrieved chunks and LLM.
"""

from typing import List, Dict
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate


class QAChain:
    """Handles question answering with context retrieval."""

    def __init__(self, embedding_store, memory):
        """
        Initialize QA chain.

        Args:
            embedding_store: EmbeddingStore instance
            memory: ChatMemory instance
        """
        self.embedding_store = embedding_store
        self.memory = memory

        # Initialize open-source LLM via Hugging Face
        self.llm = HuggingFaceHub(
            repo_id="mistralai/Mistral-7B-Instruct-v0.1",
            model_kwargs={"temperature": 0.1}
        )

        # Strict prompt (NO hallucination)
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an AI assistant that answers questions strictly
based ONLY on the provided website content.

RULES (MUST FOLLOW):
1. Use ONLY the information in the context.
2. If the answer is not found in the context, respond EXACTLY with:
   "The answer is not available on the provided website."
3. Do NOT use external knowledge.
4. Be clear, concise, and accurate.

Website Context:
{context}

Question:
{question}
"""
        )

    def answer(self, question: str, k: int = 3) -> str:
        """
        Answer a question using retrieved context.

        Args:
            question: User's question
            k: Number of chunks to retrieve

        Returns:
            Answer string
        """

        # Retrieve relevant chunks
        relevant_chunks = self.embedding_store.search(question, k=k)

        if not relevant_chunks:
            return "The answer is not available on the provided website."

        # Combine chunks into context
        context_parts = []
        for chunk in relevant_chunks:
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            title = metadata.get("title", "Unknown")
            url = metadata.get("url", "")

            context_parts.append(
                f"[Source: {title} ({url})]\n{text}"
            )

        context = "\n\n---\n\n".join(context_parts)

        # Format prompt
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )

        try:
            response = self.llm.invoke(prompt)
            answer = response.strip()

            # Store in memory (session-based)
            self.memory.add_message(question, answer)

            return answer

        except Exception as e:
            print("LLM Error:", str(e))
            return "The answer is not available on the provided website."
