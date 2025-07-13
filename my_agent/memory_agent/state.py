"""
BargainB Memory Agent State

Defines the state structure for the memory agent that handles
user personalization and long-term memory management with the bee hive architecture.
"""

from typing import TypedDict, List, Optional, Dict, Any, Literal, Annotated
from langgraph.graph import MessagesState


class BargainBMemoryState(MessagesState):
    """
    Extended state for BargainB memory agent that includes:
    - Standard MessagesState for conversation handling
    - Memory context for personalization
    - User identification for memory persistence
    - Conversation summarization for message management
    - Bee hive coordination state
    """
    
    # User identification
    user_id: Optional[str] = None
    thread_id: Optional[str] = None
    conversation_id: Optional[str] = None
    
    # Conversation summarization (like in the tutorial)
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
    
    # Bee hive coordination (for tracking which bee is currently active)
    current_bee: Optional[Literal["beeb", "scout_bee", "memory_bee", "scribe_bee"]] = "beeb"
    
    # Task delegation context
    delegation_context: Optional[Dict[str, Any]] = None
    
    # Worker bee results (for passing results back to Beeb)
    scout_results: Optional[str] = None
    memory_results: Optional[str] = None  # Updated field name
    scribe_results: Optional[str] = None
    
    # Delegation tracking to prevent infinite loops
    delegation_completed: bool = False
    iteration_count: int = 0 