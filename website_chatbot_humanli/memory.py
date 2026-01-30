"""
Memory Module
Manages conversation memory for the chatbot.
"""

from langchain.memory import ConversationBufferMemory
from typing import Dict, List


class ChatMemory:
    """Manages conversation memory for the chatbot."""
    
    def __init__(self):
        """Initialize the memory."""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    def add_message(self, question: str, answer: str):
        """
        Add a question-answer pair to memory.
        
        Args:
            question: User's question
            answer: Bot's answer
        """
        self.memory.chat_memory.add_user_message(question)
        self.memory.chat_memory.add_ai_message(answer)
    
    def get_history(self) -> List[Dict]:
        """
        Get conversation history.
        
        Returns:
            List of message dictionaries
        """
        return self.memory.chat_memory.messages
    
    def clear(self):
        """Clear conversation memory."""
        self.memory.clear()
    
    def get_memory_variables(self) -> Dict:
        """
        Get memory variables for LangChain.
        
        Returns:
            Dictionary with memory variables
        """
        return self.memory.load_memory_variables({})

