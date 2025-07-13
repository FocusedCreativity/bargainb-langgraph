"""
BargainB Memory Agent - Unified Bee Hive Architecture 🐝

Main agent that implements the supervisor pattern with Beeb as the Queen Bee
coordinating specialized worker bees for product search, memory management,
and conversation summarization.

Architecture:
- Beeb 🐝👑: Queen Bee supervisor (always responds to users)
- Scout Bee 🐝🔍: Product search specialist
- Memory Bee 🐝🧠: Memory management specialist  
- Scribe Bee 🐝📝: Conversation summarization specialist
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END

from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from my_agent.memory_agent.state import BargainBMemoryState
from my_agent.memory_agent.beeb_supervisor import create_beeb_supervisor, _format_semantic_memory, _format_episodic_memories, _format_procedural_memory
from my_agent.memory_agent.scout_bee import create_scout_bee
from my_agent.memory_agent.memory_bee import create_memory_bee
from my_agent.memory_agent.scribe_bee import create_scribe_bee

from langchain_openai import ChatOpenAI
from trustcall import create_extractor
from langchain_core.messages import merge_message_runs, SystemMessage, HumanMessage
from my_agent.memory_agent.schemas import UserProfile, ShoppingMemory, Instructions
import uuid


def load_memories(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Load user memories from the store to provide context for the conversation.
    
    Args:
        state: Current conversation state
        config: Runtime configuration with user_id
        store: Memory store for retrieval
        
    Returns:
        Updated state with loaded memories
    """
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return {
            **state,
            "iteration_count": 0,
            "delegation_completed": False
        }
    
    try:
        # Load user profile (replaces semantic memory)
        profile_items = store.search(("profile", user_id))
        user_profile = profile_items[0].value if profile_items else None
        
        # Load shopping memories (replaces episodic memories)
        shopping_items = store.search(("shopping", user_id))
        shopping_memories = [item.value for item in shopping_items[-5:]]  # Last 5 interactions
        
        # Load instructions (replaces procedural memory)
        instructions_items = store.search(("instructions", user_id))
        instructions = instructions_items[0].value if instructions_items else None
        
        return {
            **state,
            "user_id": user_id,
            "semantic_memory": user_profile,  # Keep compatible with existing formatting
            "episodic_memories": shopping_memories,  # Keep compatible with existing formatting
            "procedural_memory": instructions,  # Keep compatible with existing formatting
            "iteration_count": 0,
            "delegation_completed": False
        }
        
    except Exception as e:
        print(f"Error loading memories: {e}")
        return {
            **state, 
            "user_id": user_id,
            "iteration_count": 0,
            "delegation_completed": False
        }


def beeb_supervisor_node(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Beeb supervisor node that coordinates all worker bees and responds to users.
    
    Args:
        state: Current conversation state
        config: Runtime configuration
        store: Memory store for persistence
        
    Returns:
        Updated state with Beeb's response or delegation
    """
    # Create Beeb supervisor
    beeb = create_beeb_supervisor()
    
    # Check if this is a new user message (clear previous results)
    messages = state.get("messages", [])
    last_message = messages[-1] if messages else None
    is_new_user_message = last_message and hasattr(last_message, 'type') and last_message.type == "human"
    
    # Format the state for the prompt template
    formatted_state = {
        **state,
        "user_id": state.get("user_id", "unknown"),
        "thread_id": state.get("thread_id", "unknown"), 
        "summary": state.get("summary", "No previous conversation"),
        "semantic_memory": _format_semantic_memory(state.get("semantic_memory")),
        "episodic_memories": _format_episodic_memories(state.get("episodic_memories", [])),
        "procedural_memory": _format_procedural_memory(state.get("procedural_memory")),
        "scout_results": state.get("scout_results", ""),
        "memory_results": state.get("memory_results", ""),
        "scribe_results": state.get("scribe_results", "")
    }
    
    # Invoke Beeb with formatted state
    result = beeb.invoke(formatted_state)
    
    # If Beeb made tool calls, add tool response messages
    updated_messages = state["messages"] + [result]
    
    if hasattr(result, 'tool_calls') and result.tool_calls:
        # Add tool response messages for each tool call
        for tool_call in result.tool_calls:
            tool_response = {
                "role": "tool",
                "content": result.content if result.content else "Task delegated successfully",
                "name": tool_call["name"],
                "tool_call_id": tool_call["id"]
            }
            updated_messages.append(tool_response)
    
    # Increment iteration count
    iteration_count = state.get("iteration_count", 0) + 1
    
    # Prepare return state
    return_state = {
        **state,
        "messages": updated_messages,
        "current_bee": "beeb",
        "iteration_count": iteration_count
    }
    
    # If this is a response to user (no tool calls), clear worker bee results and mark delegation as completed
    if not (hasattr(result, 'tool_calls') and result.tool_calls):
        return_state.update({
            "scout_results": "",
            "memory_results": "", 
            "scribe_results": "",
            "delegation_completed": True
        })
    
    return return_state


def route_beeb_decisions(state: BargainBMemoryState) -> Literal["scout_bee", "memory_bee", "scribe_bee", "end"]:
    """
    Route Beeb's decisions to appropriate worker bees based on tool calls.
    
    Args:
        state: Current conversation state
        
    Returns:
        Next node to execute based on Beeb's tool calls
    """
    messages = state.get("messages", [])
    if not messages:
        return "end"
    
    # Check iteration count to prevent infinite loops
    iteration_count = state.get("iteration_count", 0)
    if iteration_count >= 10:  # Maximum 10 iterations
        return "end"
    
    # Check if delegation has been completed
    if state.get("delegation_completed", False):
        return "end"
    
    # Check if there are existing results from worker bees
    # If so, Beeb should respond to user instead of delegating again
    has_scout_results = bool(state.get("scout_results"))
    has_memory_results = bool(state.get("memory_results"))
    has_scribe_results = bool(state.get("scribe_results"))
    
    # If we have results from worker bees, don't delegate again unless explicitly needed
    if has_scout_results or has_memory_results or has_scribe_results:
        # Check if the last message has new tool calls that override existing results
        last_message = messages[-1] if messages else None
        if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Check if this is a new delegation (not a tool response)
            if hasattr(last_message, 'type') and last_message.type == "ai":
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call.get("name", "")
                    
                    if tool_name == "assign_to_scout_bee":
                        return "scout_bee"
                    elif tool_name == "assign_to_memory_bee":
                        return "memory_bee"
                    elif tool_name == "assign_to_scribe_bee":
                        return "scribe_bee"
        
        # If we have results but no new tool calls, end the conversation
        return "end"
    
    # Look for tool calls in the most recent AI message
    # If the last message is a tool response, check the second-to-last message
    for message in reversed(messages):
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.get("name", "")
                
                if tool_name == "assign_to_scout_bee":
                    return "scout_bee"
                elif tool_name == "assign_to_memory_bee":
                    return "memory_bee"
                elif tool_name == "assign_to_scribe_bee":
                    return "scribe_bee"
            break  # Only check the most recent message with tool calls
    
    # If no tool calls or unrecognized tool, end the conversation
    return "end"


def scout_bee_node(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Scout Bee node that handles product search and price comparison.
    
    Args:
        state: Current conversation state
        config: Runtime configuration
        store: Memory store for persistence
        
    Returns:
        Updated state with Scout Bee's search results
    """
    # Create Scout Bee processor
    scout_processor = create_scout_bee()
    
    # Extract task from Beeb's tool call
    messages = state.get("messages", [])
    task_description = "Search for products"  # Default task
    
    # Look for the most recent assign_to_scout_bee tool call
    for message in reversed(messages):
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tool_call in message.tool_calls:
                if tool_call.get("name") == "assign_to_scout_bee":
                    task_description = tool_call.get("args", {}).get("task_description", task_description)
                    break
        if task_description != "Search for products":
            break
    
    # Create task for Scout Bee processor
    scout_task = {"messages": [{"role": "user", "content": task_description}]}
    
    try:
        # Use direct database search for more reliable results
        from my_agent.utils.database import semantic_search
        
        # Extract search query from task description
        search_query = task_description.replace("Search for ", "").replace("Find ", "").replace("Compare prices for ", "")
        
        # Perform semantic search
        products = semantic_search(search_query, limit=5)
        
        if products:
            # Format results for Beeb to use
            scout_results = f"🔍 Found {len(products)} products for '{task_description}':\n\n"
            
            for i, product in enumerate(products, 1):
                name = product.get('title', 'Unknown Product')
                brand = product.get('brand', 'Unknown Brand')
                size = product.get('quantity', 'Unknown Size')
                price = product.get('price', 'Price not available')
                
                # Extract store from store_prices JSON
                import json
                store = 'Unknown Store'
                try:
                    store_prices = json.loads(product.get('store_prices', '[]'))
                    if store_prices:
                        store = store_prices[0].get('store', 'Unknown Store')
                except:
                    pass
                
                scout_results += f"{i}. **{name}** by {brand}\n"
                scout_results += f"   Product: {name}\n"
                scout_results += f"   Brand: {brand}\n"
                scout_results += f"   Size: {size}\n"
                scout_results += f"   Best price: {price} at {store}\n"
                
                # Add store comparison if available from store_prices JSON
                try:
                    store_prices = json.loads(product.get('store_prices', '[]'))
                    if len(store_prices) > 1:
                        stores_info = ", ".join([f"{s.get('store', 'Unknown')} {s.get('price', 'N/A')}" for s in store_prices[:3]])
                        scout_results += f"   Stores: {stores_info}\n"
                except:
                    pass
                scout_results += "\n"
        else:
            scout_results = f"🔍 No products found for '{task_description}'. Try different keywords or check spelling."
            
    except Exception as e:
        scout_results = f"🔍 Search encountered an error: {str(e)}"
    
    return {
        **state,
        "scout_results": scout_results,
        "current_bee": "scout_bee"
    }


def memory_bee_node(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Memory Bee node that handles memory management using Trustcall with proper store integration.
    
    Based on mem.md patterns for reliable memory updating and extraction.
    
    Args:
        state: Current conversation state
        config: Runtime configuration
        store: Memory store for persistence
        
    Returns:
        Updated state with Memory Bee's update results
    """
    # Get user ID from config
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        return {
            **state,
            "memory_results": "❌ No user ID provided for memory storage",
            "current_bee": "memory_bee"
        }
    
    # Extract memory task from Beeb's tool call
    messages = state.get("messages", [])
    memory_type = "profile"  # Default
    context = "Update user memory"  # Default
    
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            for tool_call in last_message.tool_calls:
                if tool_call.get("name") == "assign_to_memory_bee":
                    args = tool_call.get("args", {})
                    memory_type = args.get("memory_type", memory_type)
                    context = args.get("context", context)
    
    try:
        # Create Trustcall extractor and model
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Route to appropriate memory handler
        if memory_type == "profile":
            result = _handle_profile_memory(user_id, context, messages, model, store)
        elif memory_type == "shopping":
            result = _handle_shopping_memory(user_id, context, messages, model, store)
        elif memory_type == "instructions":
            result = _handle_instructions_memory(user_id, context, messages, model, store)
        else:
            result = f"❌ Unknown memory type: {memory_type}"
        
        return {
            **state,
            "memory_results": result,
            "current_bee": "memory_bee"
        }
        
    except Exception as e:
        return {
            **state,
            "memory_results": f"❌ Memory update failed: {str(e)}",
            "current_bee": "memory_bee"
        }


def _handle_profile_memory(user_id: str, context: str, messages: list, model, store: BaseStore) -> str:
    """Handle user profile memory updates using Trustcall."""
    # Create profile extractor
    profile_extractor = create_extractor(
        model,
        tools=[UserProfile],
        tool_choice="UserProfile",
        enable_inserts=True
    )
    
    # Get existing profile from store
    namespace = ("profile", user_id)
    existing_items = store.search(namespace)
    
    # Format existing memories for Trustcall
    existing_memories = None
    if existing_items:
        existing_memories = [(existing_item.key, "UserProfile", existing_item.value) for existing_item in existing_items]
    
    # Create instruction for Trustcall
    instruction = (
        "Extract and update user profile information from the following context. "
        "Focus on personal information, preferences, dietary restrictions, and shopping habits."
    )
    
    # Prepare messages for Trustcall
    trustcall_messages = [
        SystemMessage(content=instruction),
        HumanMessage(content=f"Context: {context}")
    ]
    
    # Invoke Trustcall extractor
    result = profile_extractor.invoke({
        "messages": trustcall_messages,
        "existing": existing_memories
    })
    
    # Save updated memories to store
    for response, metadata in zip(result["responses"], result["response_metadata"]):
        memory_id = metadata.get("json_doc_id", str(uuid.uuid4()))
        store.put(namespace, memory_id, response.model_dump(mode="json"))
    
    return f"✅ Updated user profile with: {context}"


def _handle_shopping_memory(user_id: str, context: str, messages: list, model, store: BaseStore) -> str:
    """Handle shopping memory updates using Trustcall."""
    # Create shopping extractor
    shopping_extractor = create_extractor(
        model,
        tools=[ShoppingMemory],
        tool_choice="ShoppingMemory",
        enable_inserts=True
    )
    
    # Get existing shopping memories from store
    namespace = ("shopping", user_id)
    existing_items = store.search(namespace)
    
    # Format existing memories for Trustcall
    existing_memories = None
    if existing_items:
        existing_memories = [(existing_item.key, "ShoppingMemory", existing_item.value) for existing_item in existing_items]
    
    # Create instruction for Trustcall
    instruction = (
        "Create a shopping memory entry for this interaction. "
        "Focus on products discussed, user feedback, and purchasing decisions."
    )
    
    # Prepare messages for Trustcall
    trustcall_messages = [
        SystemMessage(content=instruction),
        HumanMessage(content=f"Context: {context}")
    ]
    
    # Invoke Trustcall extractor
    result = shopping_extractor.invoke({
        "messages": trustcall_messages,
        "existing": existing_memories
    })
    
    # Save new shopping memory to store
    for response, metadata in zip(result["responses"], result["response_metadata"]):
        memory_id = metadata.get("json_doc_id", str(uuid.uuid4()))
        store.put(namespace, memory_id, response.model_dump(mode="json"))
    
    return f"✅ Recorded shopping memory: {context}"


def _handle_instructions_memory(user_id: str, context: str, messages: list, model, store: BaseStore) -> str:
    """Handle instructions memory updates using Trustcall."""
    # Create instructions extractor
    instructions_extractor = create_extractor(
        model,
        tools=[Instructions],
        tool_choice="Instructions",
        enable_inserts=True
    )
    
    # Get existing instructions from store
    namespace = ("instructions", user_id)
    existing_items = store.search(namespace)
    
    # Format existing memories for Trustcall
    existing_memories = None
    if existing_items:
        existing_memories = [(existing_item.key, "Instructions", existing_item.value) for existing_item in existing_items]
    
    # Create instruction for Trustcall
    instruction = (
        "Update system behavior instructions based on user preferences. "
        "Focus on communication style, recommendation approach, and personalization settings."
    )
    
    # Prepare messages for Trustcall
    trustcall_messages = [
        SystemMessage(content=instruction),
        HumanMessage(content=f"Context: {context}")
    ]
    
    # Invoke Trustcall extractor
    result = instructions_extractor.invoke({
        "messages": trustcall_messages,
        "existing": existing_memories
    })
    
    # Save updated instructions to store
    for response, metadata in zip(result["responses"], result["response_metadata"]):
        memory_id = metadata.get("json_doc_id", "user_instructions")
        store.put(namespace, memory_id, response.model_dump(mode="json"))
    
    return f"✅ Updated system instructions: {context}"


def scribe_bee_node(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Scribe Bee node that handles conversation summarization.
    
    Args:
        state: Current conversation state
        config: Runtime configuration
        store: Memory store for persistence
        
    Returns:
        Updated state with Scribe Bee's summarization results
    """
    # Create Scribe Bee
    scribe = create_scribe_bee()
    
    # Create summarization task for Scribe Bee
    summarization_task = {
        "messages": [{"role": "user", "content": "Summarize this conversation"}]
    }
    
    # Invoke Scribe Bee with task
    result = scribe.invoke(summarization_task)
    
    # Extract the final response for Beeb
    final_message = result["messages"][-1] if result["messages"] else None
    summary_result = final_message.content if final_message else "Summarization failed"
    
    # Update the summary in state
    return {
        **state,
        "scribe_results": summary_result,
        "summary": summary_result,  # Update the main summary
        "current_bee": "scribe_bee"
    }


def check_conversation_length(state: BargainBMemoryState) -> bool:
    """
    Check if conversation needs summarization based on message count.
    
    Args:
        state: Current conversation state
        
    Returns:
        True if summarization is needed
    """
    return len(state.get("messages", [])) > 8


def update_memories_node(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Update memories in the background after conversations.
    
    Args:
        state: Current conversation state
        config: Runtime configuration
        store: Memory store for persistence
        
    Returns:
        Updated state with memory update flags reset
    """
    # This is a placeholder for background memory updates
    # In practice, this would be triggered by specific conditions
    return {
        **state,
        "needs_semantic_update": False,
        "needs_episodic_update": False,
        "needs_procedural_update": False
    }


def create_bargainb_memory_agent():
    """
    Create the unified BargainB memory agent with bee hive architecture.
    
    This agent implements the supervisor pattern where:
    - Beeb 🐝👑 is the Queen Bee supervisor who always responds to users
    - Scout Bee 🐝🔍 handles product searches and price comparisons
    - Memory Bee 🐝🧠 manages user memories using Trustcall
    - Scribe Bee 🐝📝 handles conversation summarization
    
    The flow follows the tutorial supervisor pattern:
    - All conversations start with loading memories
    - Beeb coordinates with worker bees via handoff tools
    - Worker bees return results to Beeb automatically
    - Beeb always provides the final response to users
    
    Returns:
        Compiled graph with supervisor pattern and memory persistence
    """
    
    # Create the graph with supervisor pattern
    builder = StateGraph(BargainBMemoryState)
    
    # Add all nodes
    builder.add_node("load_memories", load_memories)
    builder.add_node("beeb", beeb_supervisor_node)
    builder.add_node("scout_bee", scout_bee_node)
    builder.add_node("memory_bee", memory_bee_node)
    builder.add_node("scribe_bee", scribe_bee_node)
    builder.add_node("update_memories", update_memories_node)
    
    # Define the flow
    builder.add_edge(START, "load_memories")
    builder.add_edge("load_memories", "beeb")
    
    # Conditional routing from Beeb to worker bees
    builder.add_conditional_edges(
        "beeb",
        route_beeb_decisions,
        {
            "scout_bee": "scout_bee",
            "memory_bee": "memory_bee",
            "scribe_bee": "scribe_bee",
            "end": END
        }
    )
    
    # Worker bees return to Beeb (supervisor pattern)
    builder.add_edge("scout_bee", "beeb")
    builder.add_edge("memory_bee", "beeb")  
    builder.add_edge("scribe_bee", "beeb")
    
    # Background memory updates
    builder.add_edge("update_memories", END)
    
    # Import required persistence classes
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.store.memory import InMemoryStore
    
    # Store for long-term (across-thread) memory
    across_thread_memory = InMemoryStore()
    
    # Checkpointer for short-term (within-thread) memory
    within_thread_memory = MemorySaver()
    
    # Compile the graph with checkpointer and store (following mem.md patterns)
    graph = builder.compile(
        checkpointer=within_thread_memory,
        store=across_thread_memory,
        debug=True  # Enable debug mode for better error tracking
    )
    
    return graph


# Export the compiled graph
memory_agent = create_bargainb_memory_agent() 