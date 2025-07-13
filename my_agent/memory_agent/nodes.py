"""
BargainB Memory Agent Nodes

Node functions for the memory agent that handles personalized
grocery shopping recommendations with long-term memory.
"""

import uuid
from datetime import datetime
from typing import Literal, Dict, Any

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage, merge_message_runs
from langchain_openai import ChatOpenAI

from .state import BargainBMemoryState
from .tools import (
    MemoryToolkit, 
    detect_interaction_type, 
    detect_price_sensitivity,
    extract_products_mentioned,
    SEMANTIC_MEMORY_INSTRUCTION,
    EPISODIC_MEMORY_INSTRUCTION, 
    PROCEDURAL_MEMORY_INSTRUCTION
)
from .schemas import MemoryUpdate
from .database_integration import memory_db
from .simple_persistence import (
    save_conversation_summary,
    get_conversation_summary,
    log_message_truncation
)


# Initialize model and memory toolkit
model = ChatOpenAI(model="gpt-4o", temperature=0)
memory_toolkit = MemoryToolkit(model)

# System message for the main BargainB memory agent
BARGAINB_MEMORY_SYSTEM_MESSAGE = """
You are BargainB, a personalized grocery shopping assistant that learns from every interaction.

You help users find products, compare prices, and make shopping decisions while building a detailed understanding of their preferences and habits.

Current User Memory:

<semantic_memory>
{semantic_memory}
</semantic_memory>

<recent_interactions>
{recent_interactions}
</recent_interactions>

<behavior_instructions>
{procedural_memory}
</behavior_instructions>

Instructions for memory management:
1. Always analyze interactions for new preferences, dietary info, or shopping patterns
2. Record every meaningful interaction to build user patterns
3. Adapt your communication style based on user feedback
4. Prioritize products that match user preferences and budget sensitivity
5. Learn from user reactions to improve future recommendations

When you detect important information about the user, call the MemoryUpdate tool to save it.

Types of information to save:
- **Semantic**: Food preferences, dietary restrictions, budget sensitivity, shopping habits
- **Episodic**: This specific interaction, user requests, system responses, feedback
- **Procedural**: Changes to how you should communicate or behave with this user

Respond naturally while being aware of the user's preferences and past interactions.
"""


def load_memories(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Load user memories from the store to personalize the interaction.
    """
    user_id = config["configurable"]["user_id"]
    
    # Load semantic memory (user preferences)
    semantic_namespace = ("semantic", user_id)
    semantic_memories = store.search(semantic_namespace)
    semantic_memory = semantic_memories[0].value if semantic_memories else None
    
    # Load recent episodic memories (last 10 interactions)
    episodic_namespace = ("episodic", user_id)  
    episodic_memories = store.search(episodic_namespace)
    recent_interactions = [mem.value for mem in episodic_memories[-10:]] if episodic_memories else []
    
    # Load procedural memory (behavior instructions)
    procedural_namespace = ("procedural", user_id)
    procedural_memories = store.search(procedural_namespace)
    procedural_memory = procedural_memories[0].value if procedural_memories else None
    
    # Detect current interaction context
    current_interaction_type = detect_interaction_type(state["messages"])
    products_discussed = extract_products_mentioned(state["messages"])
    price_sensitivity_detected = detect_price_sensitivity(state["messages"])
    
    return {
        **state,
        "user_id": user_id,
        "semantic_memory": semantic_memory,
        "episodic_memories": recent_interactions,
        "procedural_memory": procedural_memory,
        "current_interaction_type": current_interaction_type,
        "products_discussed": products_discussed,
        "price_sensitivity_detected": price_sensitivity_detected
    }


def bargainb_chat(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Main chat function that provides personalized responses using loaded memories and product database.
    """
    # Check if this is a product search request
    last_message = state["messages"][-1].content.lower() if state["messages"] else ""
    is_product_search = any(keyword in last_message for keyword in [
        'find', 'search', 'recommend', 'options', 'suggest', 'what', 'show me', 'need', 'want'
    ])
    
    # Get product recommendations if it's a search request
    product_context = ""
    if is_product_search:
        try:
            # Extract search query from the message
            search_query = state["messages"][-1].content
            
            # Get user preferences for personalization
            user_preferences = state.get("semantic_memory")
            budget_sensitivity = "high" if state.get("price_sensitivity_detected") == "high" else "medium"
            
            # Search for personalized products  
            # For now, use regular semantic search until async issues are resolved
            from ..utils.database import semantic_search
            products = semantic_search(search_query, limit=3)
            
            if products:
                product_context = "\n\nAvailable Products:\n" + "\n".join([
                    f"- {product.page_content}" for product in products[:3]
                ])
            
        except Exception as e:
            print(f"Product search error: {e}")
            product_context = "\n\nNote: Product search temporarily unavailable."
    
    # Format memory context for the system message
    semantic_memory_text = str(state.get("semantic_memory", "No preferences recorded yet"))
    
    recent_interactions_text = "\n".join([
        f"- {interaction.get('date', 'Unknown date')}: {interaction.get('user_action', 'Unknown action')} → {interaction.get('outcome', 'unknown outcome')}"
        for interaction in state.get("episodic_memories", [])[-5:]  # Last 5 interactions
    ]) if state.get("episodic_memories") else "No recent interactions"
    
    procedural_memory_text = str(state.get("procedural_memory", "Default: Be friendly, helpful, and budget-aware"))
    
    system_msg = BARGAINB_MEMORY_SYSTEM_MESSAGE.format(
        semantic_memory=semantic_memory_text,
        recent_interactions=recent_interactions_text,
        procedural_memory=procedural_memory_text
    ) + product_context
    
    # Create decision tool for memory updates
    memory_decision_tool = memory_toolkit.create_memory_decision_tool()
    
    # Generate response with memory update capability
    response = memory_decision_tool.invoke([
        SystemMessage(content=system_msg)
    ] + state["messages"])
    
    return {"messages": [response]}


def update_semantic_memory(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Update semantic memory (user preferences, habits, dietary info).
    """
    user_id = config["configurable"]["user_id"]
    namespace = ("semantic", user_id)
    
    # Get existing semantic memory
    existing_memories = store.search(namespace)
    existing_memory = existing_memories[0] if existing_memories else None
    
    # Format for Trustcall
    tool_name = "SemanticMemory"
    existing_formatted = (
        [(existing_memory.key, tool_name, existing_memory.value)]
        if existing_memory
        else None
    )
    
    # Create instruction with current time
    instruction = SEMANTIC_MEMORY_INSTRUCTION.format(time=datetime.now().isoformat())
    messages = [SystemMessage(content=instruction)] + state["messages"][:-1]
    
    # Add spy for transparency
    extractor_with_spy = memory_toolkit.semantic_extractor.with_listeners(
        on_end=memory_toolkit.spy
    )
    
    # Extract semantic memories
    result = extractor_with_spy.invoke({
        "messages": list(merge_message_runs(messages)),
        "existing": existing_formatted
    })
    
    # Save to store
    for response, metadata in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            metadata.get("json_doc_id", str(uuid.uuid4())),
            response.model_dump(mode="json")
        )
    
    # Create tool response
    tool_calls = state["messages"][-1].tool_calls
    changes = memory_toolkit.spy.extract_memory_changes("SemanticMemory")
    
    return {
        "messages": [{
            "role": "tool",
            "content": f"Updated preferences: {changes}",
            "tool_call_id": tool_calls[0]["id"]
        }]
    }


def update_episodic_memory(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Update episodic memory (record this interaction).
    """
    user_id = config["configurable"]["user_id"]
    namespace = ("episodic", user_id)
    
    # Create instruction with context
    instruction = EPISODIC_MEMORY_INSTRUCTION.format(time=datetime.now().isoformat())
    messages = [SystemMessage(content=instruction)] + state["messages"][:-1]
    
    # Add spy for transparency
    extractor_with_spy = memory_toolkit.episodic_extractor.with_listeners(
        on_end=memory_toolkit.spy
    )
    
    # Extract episodic memory
    result = extractor_with_spy.invoke({
        "messages": list(merge_message_runs(messages))
    })
    
    # Save to store
    for response, metadata in zip(result["responses"], result["response_metadata"]):
        store.put(
            namespace,
            str(uuid.uuid4()),  # Each interaction gets unique ID
            response.model_dump(mode="json")
        )
    
    # Create tool response
    tool_calls = state["messages"][-1].tool_calls
    changes = memory_toolkit.spy.extract_memory_changes("EpisodicMemory")
    
    return {
        "messages": [{
            "role": "tool", 
            "content": f"Recorded interaction: {changes}",
            "tool_call_id": tool_calls[0]["id"]
        }]
    }


def update_procedural_memory(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Update procedural memory (behavior instructions).
    """
    user_id = config["configurable"]["user_id"]
    namespace = ("procedural", user_id)
    
    # Get existing procedural memory
    existing_memory = store.get(namespace, "behavior_instructions")
    current_procedural = existing_memory.value if existing_memory else None
    
    # Create instruction
    instruction = PROCEDURAL_MEMORY_INSTRUCTION.format(
        current_procedural=current_procedural,
        time=datetime.now().isoformat()
    )
    
    # Use simple model call for procedural updates (not Trustcall)
    new_instructions = model.invoke([
        SystemMessage(content=instruction)
    ] + state["messages"][:-1] + [
        HumanMessage(content="Please update my behavior instructions based on our conversation")
    ])
    
    # Save to store
    store.put(
        namespace,
        "behavior_instructions", 
        {
            "communication_style": "adaptive",
            "instructions": new_instructions.content,
            "updated_at": datetime.now().isoformat()
        }
    )
    
    # Create tool response
    tool_calls = state["messages"][-1].tool_calls
    
    return {
        "messages": [{
            "role": "tool",
            "content": f"Updated behavior instructions: {new_instructions.content[:200]}...",
            "tool_call_id": tool_calls[0]["id"]
        }]
    }


def route_memory_update(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> Literal["semantic", "episodic", "procedural", "__end__"]:
    """
    Route to appropriate memory update based on the tool call.
    """
    message = state["messages"][-1]
    
    if not hasattr(message, 'tool_calls') or len(message.tool_calls) == 0:
        return "__end__"
    
    tool_call = message.tool_calls[0]
    update_type = tool_call["args"]["update_type"]
    
    if update_type == "semantic":
        return "semantic"
    elif update_type == "episodic": 
        return "episodic"
    elif update_type == "procedural":
        return "procedural"
    else:
        return "__end__"


def summarize_conversation(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Create or extend a summary of the conversation when messages exceed threshold.
    Similar to the notebook example but adapted for BargainB.
    """
    # Get summary if it exists
    summary = state.get("summary", "")
    
    # Create our summarization prompt
    if summary:
        # A summary already exists
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    
    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Store summary in Supabase
    conversation_id = state.get("conversation_id")
    thread_id = state.get("thread_id") or config["configurable"]["thread_id"]
    
    if conversation_id:
        save_conversation_summary(
            conversation_id=conversation_id,
            thread_id=thread_id,
            summary_text=response.content,
            message_count=len(state["messages"]),
            tokens_used=response.usage_metadata.get('total_tokens', 0) if hasattr(response, 'usage_metadata') else 0
        )
    
    # Delete all but the 2 most recent messages (like in notebook)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    
    # Log the truncation event
    if conversation_id:
        log_message_truncation(
            conversation_id=conversation_id,
            thread_id=thread_id,
            messages_before=len(state["messages"]),
            messages_after=2,
            messages_removed=len(delete_messages),
            summary_tokens=response.usage_metadata.get('total_tokens', 0) if hasattr(response, 'usage_metadata') else 0
        )
    
    return {
        "summary": response.content, 
        "messages": delete_messages,
        "messages_since_last_summary": 0,
        "should_summarize": False
    }


def should_continue(state: BargainBMemoryState) -> Literal["summarize_conversation", "__end__"]:
    """
    Determine whether to end or summarize the conversation.
    Like in the notebook - summarize if more than 6 messages.
    """
    messages = state["messages"]
    
    # If there are more than 6 messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return "__end__"


def call_model_with_summary(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Enhanced model call that includes conversation summary context.
    Similar to call_model in the notebook but integrated with BargainB memory.
    """
    # Get summary if it exists  
    summary = state.get("summary", "")
    
    # If there is summary, then we add it
    if summary:
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"
        
        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
    
    response = model.invoke(messages)
    return {"messages": [response]} 