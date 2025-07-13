"""
BargainB Memory Agent State

Defines the state structure for the simplified memory agent following mem.md patterns.
"""

from typing import TypedDict, List, Optional, Dict, Any, Literal, Annotated
from langgraph.graph import MessagesState


class BargainBMemoryState(MessagesState):
    """
    Simplified state for BargainB memory agent following mem.md patterns.
    
    This includes:
    - Standard MessagesState for conversation handling
    - User identification for memory persistence
    - Memory context fields (compatible with existing formatters)
    """
    
    # User identification (required for memory persistence)
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    
    # Memory context (for backward compatibility with existing formatters)
    semantic_memory: Optional[Dict[str, Any]] = None  # Will map to profile
    episodic_memories: List[Dict[str, Any]] = []      # Will map to shopping
    procedural_memory: Optional[Dict[str, Any]] = None # Will map to instructions
    
    # Conversation summarization (optional feature)
    summary: Optional[str] = None
    
    # Product search context (for Scout Bee compatibility)
    products_discussed: List[str] = []
    price_sensitivity_detected: Optional[str] = None 