from typing import List, Optional
from langgraph.graph import MessagesState
from langchain_core.documents import Document

class AgentState(MessagesState):
    """The state of the Self-RAG agent."""
    documents: List[Document] = []
    question: str = ""
    generation: str = ""
    generation_v_question_grade: str = ""
    generation_v_documents_grade: str = ""
