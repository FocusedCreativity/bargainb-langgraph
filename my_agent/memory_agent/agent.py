"""
BargainB Memory Agent with Delegation Pattern

Main agent that orchestrates conversation flow between Beeb (main assistant)
and specialized agents using the delegation pattern, with memory management
and conversation summarization.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from .simple_persistence import SimpleMemoryStore

from .state import BargainBMemoryState
from .delegation_nodes import (
    entry_node,
    route_delegation,
    handle_delegation_tool_calls,
    exit_dialog_node,
    should_continue_dialog
)
from .nodes import (
    load_memories,
    update_semantic_memory,
    update_episodic_memory,
    update_procedural_memory,
    summarize_conversation
)


def create_bargainb_memory_agent():
    """
    Create the BargainB delegation-based agent with Beeb as main assistant.
    
    This agent uses the delegation pattern where:
    - Beeb is the main assistant who always responds to users
    - Product searches are delegated to specialized RAG agent
    - Memory management and summarization work in the background
    - Dialog state manages which assistant is currently active
    
    Returns:
        Compiled graph with delegation pattern and memory persistence
    """
    
    # Create the graph with delegation pattern
    builder = StateGraph(BargainBMemoryState)
    
    # Simple linear delegation flow (no loops)
    builder.add_node("load_memories", load_memories)  # Load user memory context
    builder.add_node("chat", entry_node)  # Main conversation node (Beeb or delegated agent)
    
    # Memory management nodes (background operations if needed)
    builder.add_node("semantic", update_semantic_memory)
    builder.add_node("episodic", update_episodic_memory)
    builder.add_node("procedural", update_procedural_memory)
    builder.add_node("summarize", summarize_conversation)
    
    # Simple flow: load memories then chat, then end
    builder.add_edge(START, "load_memories")
    builder.add_edge("load_memories", "chat")
    builder.add_edge("chat", END)  # End after each interaction - no loops!
    
    # Memory edges (if needed, but typically won't be used in this simple flow)
    builder.add_edge("semantic", END)
    builder.add_edge("episodic", END)
    builder.add_edge("procedural", END)
    builder.add_edge("summarize", END)
    
    # Create persistence layer
    checkpointer = MemorySaver()  # LangGraph Cloud will handle persistence
    memory_store = SimpleMemoryStore()  # Our custom memory store with Supabase backend
    
    # Compile the graph with delegation and persistence
    graph = builder.compile(
        checkpointer=checkpointer,
        store=memory_store
    )
    
    return graph


# Export the compiled graph
memory_agent = create_bargainb_memory_agent() 