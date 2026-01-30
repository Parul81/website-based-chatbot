"""
Question Answering Chain Module
Handles question answering using retrieved chunks and LLM.
"""

import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


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
        
        # Initialize LLM (use OpenAI API key from environment)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
            print("Please create a .env file with: OPENAI_API_KEY=your_key_here")
        
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=api_key
        )
        
        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant that answers questions based ONLY on the provided context from a website.

IMPORTANT RULES:
1. Answer questions using ONLY the information provided in the context below.
2. If the answer is not in the context, say: "The answer is not available on the provided website."
3. Do NOT use any external knowledge or information from the internet.
4. Be concise and accurate.
5. If asked about something not in the context, politely state that the information is not available on the website.

Context from website:
{context}

Previous conversation:
{chat_history}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
    
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
            return "I couldn't find any relevant information on the website. Please make sure the website has been indexed first."
        
        # Combine chunks into context
        context_parts = []
        for chunk in relevant_chunks:
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {})
            title = metadata.get('title', 'Unknown')
            url = metadata.get('url', '')
            
            context_parts.append(f"[From: {title} ({url})]\n{text}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Get conversation history
        memory_vars = self.memory.get_memory_variables()
        chat_history = memory_vars.get('chat_history', [])
        
        # Format chat history for prompt
        formatted_history = []
        for msg in chat_history[-6:]:  # Keep last 6 messages
            if hasattr(msg, 'content'):
                if isinstance(msg, HumanMessage):
                    formatted_history.append(("human", msg.content))
                elif isinstance(msg, AIMessage):
                    formatted_history.append(("assistant", msg.content))
        
        # Create prompt
        prompt = self.prompt_template.format_messages(
            context=context,
            chat_history=formatted_history,
            question=question
        )
        
        # Get response from LLM
        try:
            response = self.llm.invoke(prompt)
            answer = response.content
            
            # Add to memory
            self.memory.add_message(question, answer)
            
            return answer
        except Exception as e:
            error_msg = f"Error generating answer: {str(e)}"
            print(error_msg)
            if "API key" in str(e) or "authentication" in str(e).lower():
                return "Error: Please configure your OpenAI API key in the .env file."
            return error_msg

