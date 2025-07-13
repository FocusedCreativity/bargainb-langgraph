"""
BargainB Memory Agent State

Defines the state structure for the memory agent that handles
user personalization and long-term memory management.
"""

from typing import TypedDict, List, Optional, Dict, Any, Literal, Annotated
from langgraph.graph import MessagesState


def update_dialog_stack(left: List[str], right: Optional[str]) -> List[str]:
    """
    Manage the dialog state stack for delegation between assistants.
    
    Args:
        left: Current dialog state stack
        right: Operation to perform ('pop' to return, assistant name to delegate, None to maintain)
    
    Returns:
        Updated dialog state stack
    """
    if right is None:
        return left
    if right == "pop":
        return left[:-1]
    return left + [right]


class BargainBMemoryState(MessagesState):
    """
    Extended state for BargainB memory agent that includes:
    - Standard MessagesState for conversation handling
    - Memory context for personalization
    - User identification for memory persistence
    - Conversation summarization for message management
    """
    
    # User identification
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Dialog state stack for delegation management (Beeb -> Product Search)
    dialog_state: Annotated[
        List[Literal["beeb_assistant", "product_search"]], 
        update_dialog_stack
    ] = []
    
    # Conversation summarization (like in the notebook)
    summary: Optional[str] = None
    
    # Current memory context (loaded from store)
    semantic_memory: Optional[Dict[str, Any]] = None
    episodic_memories: List[Dict[str, Any]] = []
    procedural_memory: Optional[Dict[str, Any]] = None
    
    # Interaction tracking
    current_interaction_type: Optional[str] = None
    products_discussed: List[str] = []
    price_sensitivity_detected: Optional[str] = None
    
    # Memory update flags
    needs_semantic_update: bool = False
    needs_episodic_update: bool = False  
    needs_procedural_update: bool = False
    
    # Message management
    messages_since_last_summary: int = 0
    should_summarize: bool = False 