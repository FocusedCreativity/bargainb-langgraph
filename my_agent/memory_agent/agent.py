"""
BargainB Memory Agent - Bee Delegation System with Memory & Summarization

This agent uses the bee delegation system with:
- Beeb Supervisor 🐝👑: Main interface and coordination
- Scout Bee 🐝🔍: Product search and price comparison
- Memory Bee 🐝🧠: Memory management and personalization
- Scribe Bee 🐝📝: Conversation summarization
- Database integration for product data and memory persistence
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs, HumanMessage, SystemMessage, RemoveMessage
from langchain_core.tools import tool

from langchain_openai import ChatOpenAI
from trustcall import create_extractor
from datetime import datetime
import uuid

# Import state and schemas
from my_agent.memory_agent.state import BargainBMemoryState
from my_agent.memory_agent.schemas import UserProfile, ShoppingMemory, Instructions

# Import bee system components
from my_agent.memory_agent.beeb_supervisor import (
    create_beeb_supervisor,
    _format_semantic_memory,
    _format_episodic_memories,
    _format_procedural_memory
)
from my_agent.memory_agent.scout_bee import create_scout_bee
from my_agent.memory_agent.memory_bee import create_memory_bee
from my_agent.memory_agent.scribe_bee import create_scribe_bee

# Import database utilities
try:
    from my_agent.utils.database import log_message_truncation
except ImportError:
    def log_message_truncation(user_id: str, thread_id: str, original_count: int, truncated_count: int, summary: str):
        print(f"📝 Mock log: Truncated {original_count} to {truncated_count} messages for {user_id}")

# Import persistence layer for conversation summaries
from my_agent.memory_agent.simple_persistence import (
    save_conversation_summary,
    get_conversation_summary,
    log_message_truncation as log_truncation_db
)

# Missing constants from mem.md patterns
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 

Use the provided tools to retain any necessary memories about the user. 

Use parallel tool calling to handle updates and insertions simultaneously.

System Time: {time}"""

CREATE_INSTRUCTIONS = """Reflect on the following interaction.

Based on this interaction, update your instructions for how to update ToDo list items. 

Use any feedback from the user to update how they like to have items added, etc.

Your current instructions are:

<current_instructions>
{current_instructions}
</current_instructions>"""

# Initialize the language model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Create extractors for each memory type following mem.md patterns
user_profile_extractor = create_extractor(
    model,
    tools=[UserProfile],
    tool_choice="UserProfile",
    enable_inserts=True,
)

shopping_memory_extractor = create_extractor(
    model,
    tools=[ShoppingMemory],
    tool_choice="ShoppingMemory",
    enable_inserts=True,
)

instructions_extractor = create_extractor(
    model,
    tools=[Instructions],
    tool_choice="Instructions",
    enable_inserts=True,
)

# Spy class from mem.md for visibility into Trustcall updates
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        # Collect information about the tool calls made by the extractor.
        q = [run]
        while q:
            run = q.pop()
            if hasattr(run, 'tool_calls') and run.tool_calls:
                self.called_tools.extend(
                    [
                        {
                            "name": tool_call["name"],
                            "args": tool_call["args"],
                        }
                        for tool_call in run.tool_calls
                    ]
                )
            if hasattr(run, 'steps'):
                for step in run.steps:
                    q.append(step)

def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract tool information from Trustcall runs"""
    return [
        {
            "tool_name": tool_call.get("name", "Unknown"),
            "args": tool_call.get("args", {}),
            "schema": schema_name
        }
        for tool_call in tool_calls
    ]

# Initialize bee system components
beeb_supervisor = create_beeb_supervisor()
scout_bee = create_scout_bee()
memory_bee = create_memory_bee()
scribe_bee = create_scribe_bee()

def beeb_main_node(state: BargainBMemoryState, config: RunnableConfig):
    """
    Main Beeb supervisor node that coordinates all worker bees.
    
    This node:
    - Loads conversation summary from database
    - Uses Beeb supervisor to coordinate worker bees
    - Handles delegation to Scout, Memory, and Scribe bees
    - Maintains conversation continuity
    """
    
    # Get the user ID from the config
    user_id = config["configurable"].get("user_id", "default")
    thread_id = config["configurable"].get("thread_id", "default")
    conversation_id = f"{user_id}_{thread_id}"

    # Load conversation summary from database (following external DB memory pattern)
    summary = state.get("summary", "")
    if not summary:
        # Try to load from database if not in state
        try:
            db_summary = get_conversation_summary(conversation_id)
            if db_summary:
                summary = db_summary
                print(f"📝 Beeb: Loaded conversation summary from database for {conversation_id}")
        except Exception as e:
            print(f"❌ Beeb: Failed to load conversation summary: {e}")
    
    # For now, use empty memories for Trustcall system (separate from conversation summaries)
    # TODO: Implement proper Trustcall memory loading when needed
    semantic_memory = None
    episodic_memories = []
    procedural_memory = None
    
    # Initialize worker bee results as empty
    scout_results = ""
    memory_results = ""
    scribe_results = ""
    
    # Check if we have previous worker bee results in the state
    # (This would be populated by worker bee nodes)
    scout_results = state.get("scout_results", "")
    memory_results = state.get("memory_results", "")
    scribe_results = state.get("scribe_results", "")

    # Format memory context for Beeb
    formatted_semantic = _format_semantic_memory(semantic_memory)
    formatted_episodic = _format_episodic_memories(episodic_memories)
    formatted_procedural = _format_procedural_memory(procedural_memory)

    # Filter messages to only include user and assistant messages without tool calls
    # This prevents OpenAI API errors about unresponded tool calls
    clean_messages = []
    for msg in state["messages"]:
        if hasattr(msg, 'role'):
            if msg.role == "user":
                clean_messages.append(msg)
            elif msg.role == "assistant" and not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                clean_messages.append(msg)
        else:
            # Handle message dict format
            if isinstance(msg, dict):
                if msg.get('role') == 'user':
                    clean_messages.append(msg)
                elif msg.get('role') == 'assistant' and not msg.get('tool_calls'):
                    clean_messages.append(msg)

    # Create context for Beeb supervisor
    context = {
        "user_id": user_id,
        "thread_id": thread_id,
        "summary": summary,
        "semantic_memory": formatted_semantic,
        "episodic_memories": formatted_episodic,
        "procedural_memory": formatted_procedural,
        "scout_results": scout_results,
        "memory_results": memory_results,
        "scribe_results": scribe_results,
        "messages": clean_messages
    }

    # Use Beeb supervisor to coordinate and respond
    response = beeb_supervisor.invoke(context)
    
    # Check if Beeb made tool calls for delegation
    if hasattr(response, 'tool_calls') and response.tool_calls:
        # Return the response with tool calls for routing
        return {"messages": [response]}
    else:
        # Beeb responded directly without delegation
        return {"messages": [response]}

def scout_bee_node(state: BargainBMemoryState, config: RunnableConfig):
    """
    Scout Bee node for product search and price comparison.
    
    This node:
    - Receives product search tasks from Beeb
    - Uses Scout Bee to search for products
    - Returns search results to Beeb
    """
    
    # Get the last message which should contain the search task
    last_message = state["messages"][-1]
    
    # Check if this is a delegation to Scout Bee
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "assign_to_scout_bee":
                search_task = tool_call["args"]["task_description"]
                
                # Use Scout Bee to process the search
                search_state = {"messages": [{"role": "user", "content": search_task}]}
                result = scout_bee(search_state)
                
                # Extract the search results
                search_results = result["messages"][-1]["content"]
                
                # Return tool response and update state with results
                tool_response = {
                    "role": "tool",
                    "content": search_results,
                    "tool_call_id": tool_call["id"]
                }
                
                return {
                    "messages": [tool_response],
                    "scout_results": search_results
                }
    
    return {"messages": []}

def memory_bee_node(state: BargainBMemoryState, config: RunnableConfig):
    """
    Memory Bee node for memory management and updates.
    
    This node:
    - Receives memory update tasks from Beeb
    - Uses Memory Bee and Trustcall to update memories
    - Saves memories to the store
    - Returns confirmation to Beeb
    """
    
    # Get the last message which should contain the memory task
    last_message = state["messages"][-1]
    
    # Check if this is a delegation to Memory Bee
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "assign_to_memory_bee":
                memory_type = tool_call["args"]["memory_type"]
                context = tool_call["args"]["context"]
                
                # Route to appropriate memory update based on type
                if memory_type == "profile":
                    result = update_profile_memory(state, config, store)
                elif memory_type == "shopping":
                    result = update_shopping_memory(state, config, store)
                elif memory_type == "instructions":
                    result = update_instructions_memory(state, config, store)
                else:
                    result = {"messages": [{"role": "tool", "content": f"Unknown memory type: {memory_type}", "tool_call_id": tool_call["id"]}]}
                
                # Add memory results to state
                memory_results = f"Updated {memory_type} memory with: {context}"
                result["memory_results"] = memory_results
                
                return result
    
    return {"messages": []}

def scribe_bee_node(state: BargainBMemoryState, config: RunnableConfig):
    """
    Scribe Bee node for conversation summarization.
    
    This node:
    - Receives summarization tasks from Beeb
    - Uses Scribe Bee to summarize conversations
    - Returns summarization results to Beeb
    """
    
    # Get the last message which should contain the summarization task
    last_message = state["messages"][-1]
    
    # Check if this is a delegation to Scribe Bee
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "assign_to_scribe_bee":
                # Use the summarization logic
                result = summarize_conversation(state, config)
                
                # Return tool response with summarization results
                tool_response = {
                    "role": "tool",
                    "content": f"Conversation summarized and saved to database: {result['summary'][:100]}...",
                    "tool_call_id": tool_call["id"]
                }
                
                return {
                    "messages": [tool_response],
                    "scribe_results": f"Summarized and saved conversation: {result['summary'][:100]}...",
                    "summary": result["summary"]
                }
    
    return {"messages": []}

# Memory management functions (adapted from mem.md patterns)
def update_profile_memory(state: BargainBMemoryState, config: RunnableConfig):
    """Update user profile memory using Trustcall (like update_profile in mem.md)."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]
    
    # Define the namespace for the memories
    namespace = ("profile", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "UserProfile"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None)

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Create the extractor
    profile_extractor = create_extractor(
        model,
        tools=[UserProfile],
        tool_choice="UserProfile",
        enable_inserts=True
    )

    # Invoke the extractor
    result = profile_extractor.invoke({"messages": updated_messages, 
                                     "existing": existing_memories})

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"))
    
    # Return tool response
    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated profile", "tool_call_id": tool_calls[0]['id']}]}

def update_shopping_memory(state: BargainBMemoryState, config: RunnableConfig):
    """Update shopping history memory using Trustcall (like update_todos in mem.md)."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]
    
    # Define the namespace for the memories
    namespace = ("shopping", user_id)

    # Retrieve the most recent memories for context
    existing_items = store.search(namespace)

    # Format the existing memories for the Trustcall extractor
    tool_name = "ShoppingMemory"
    existing_memories = ([(existing_item.key, tool_name, existing_item.value)
                          for existing_item in existing_items]
                          if existing_items
                          else None)

    # Merge the chat history and the instruction
    TRUSTCALL_INSTRUCTION_FORMATTED = TRUSTCALL_INSTRUCTION.format(time=datetime.now().isoformat())
    updated_messages = list(merge_message_runs(messages=[SystemMessage(content=TRUSTCALL_INSTRUCTION_FORMATTED)] + state["messages"][:-1]))

    # Initialize the spy for visibility into the tool calls made by Trustcall
    spy = Spy()
    
    # Create the Trustcall extractor
    shopping_extractor = create_extractor(
        model,
        tools=[ShoppingMemory],
        tool_choice=tool_name,
        enable_inserts=True
    ).with_listeners(on_end=spy)

    # Invoke the extractor
    result = shopping_extractor.invoke({"messages": updated_messages, 
                                      "existing": existing_memories})

    # Save the memories from Trustcall to the store
    for r, rmeta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace,
                  rmeta.get("json_doc_id", str(uuid.uuid4())),
                  r.model_dump(mode="json"))
        
    # Respond to the tool call with visibility into changes
    tool_calls = state['messages'][-1].tool_calls
    shopping_update_msg = extract_tool_info(spy.called_tools, tool_name)
    
    return {"messages": [{"role": "tool", "content": shopping_update_msg, "tool_call_id": tool_calls[0]['id']}]}

def update_instructions_memory(state: BargainBMemoryState, config: RunnableConfig):
    """Update instructions memory (like update_instructions in mem.md)."""
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]
    
    namespace = ("instructions", user_id)

    existing_memory = store.get(namespace, "user_instructions")
        
    # Format the memory in the system prompt
    system_msg = CREATE_INSTRUCTIONS.format(
        current_instructions=existing_memory.value if existing_memory else None
    )
    
    new_memory = model.invoke([SystemMessage(content=system_msg)] + state['messages'][:-1] + 
                            [HumanMessage(content="Please update the instructions based on the conversation")])

    # Overwrite the existing memory in the store 
    key = "user_instructions"
    store.put(namespace, key, {"memory": new_memory.content})
    
    tool_calls = state['messages'][-1].tool_calls
    return {"messages": [{"role": "tool", "content": "updated instructions", "tool_call_id": tool_calls[0]['id']}]}

# Conversation summarization function following the document pattern
def summarize_conversation(state: BargainBMemoryState, config: RunnableConfig):
    """
    Summarize the conversation and truncate messages following the external DB memory pattern.
    
    This function:
    1. Creates a summary of the conversation
    2. Saves the summary to the database
    3. Truncates messages to keep only the most recent ones
    4. Logs the truncation event for tracking
    """
    
    # Get thread/user IDs for database storage
    thread_id = config["configurable"].get("thread_id", "default")
    user_id = config["configurable"].get("user_id", thread_id)
    conversation_id = f"{user_id}_{thread_id}"
    
    # Get existing summary if it exists
    summary = state.get("summary", "")
    
    # Create our summarization prompt
    if summary:
        # A summary already exists - extend it
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    
    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Save the conversation summary to database
    try:
        save_conversation_summary(
            conversation_id=conversation_id,
            thread_id=thread_id,
            summary_text=response.content,
            message_count=len(state["messages"]),
            tokens_used=getattr(response, 'usage_metadata', {}).get('total_tokens', 0)
        )
        print(f"📝 Scribe Bee: Saved conversation summary to database for {conversation_id}")
    except Exception as e:
        print(f"❌ Scribe Bee: Failed to save conversation summary: {e}")
    
    # Delete all but the 2 most recent messages (following external DB pattern)
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    
    # Log the truncation event to database
    try:
        log_truncation_db(
            conversation_id=conversation_id,
            thread_id=thread_id,
            messages_before=len(state["messages"]),
            messages_after=2,  # Keep only 2 most recent
            messages_removed=len(delete_messages),
            summary_tokens=getattr(response, 'usage_metadata', {}).get('total_tokens', 0)
        )
        print(f"📝 Scribe Bee: Logged message truncation for {conversation_id}")
    except Exception as e:
        print(f"❌ Scribe Bee: Failed to log message truncation: {e}")
    
    return {"summary": response.content, "messages": delete_messages}

def route_decisions(state: BargainBMemoryState, config: RunnableConfig) -> Literal[END, "scout_bee_node", "memory_bee_node", "scribe_bee_node", "summarize_conversation"]:
    """
    Route decisions based on Beeb's tool calls and message count.
    
    This function:
    1. First checks if summarization is needed (message count > 10)
    2. Then checks for bee delegation based on tool calls
    3. Otherwise ends the conversation
    """
    
    # First, check if summarization is needed following document pattern
    messages = state["messages"]
    if len(messages) > 10:
        return "summarize_conversation"
    
    # Check the last AI message for tool calls (bee delegation)
    last_message = state["messages"][-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call["name"] == "assign_to_scout_bee":
                return "scout_bee_node"
            elif tool_call["name"] == "assign_to_memory_bee":
                return "memory_bee_node"
            elif tool_call["name"] == "assign_to_scribe_bee":
                return "scribe_bee_node"
    
    # If no tool calls or summarization needed, end the conversation
    return END

def create_bargainb_memory_agent():
    """
    Create the BargainB memory agent with bee delegation system.
    
    This agent uses:
    - Beeb Supervisor as the main interface
    - Worker bees for specialized tasks (Scout, Memory, Scribe)
    - Memory management with Trustcall
    - Conversation summarization
    - Database integration for products and memory
    - LangGraph platform built-in persistence
    """
    
    # Create the graph with bee delegation system
    builder = StateGraph(BargainBMemoryState)
    
    # Define nodes
    builder.add_node("beeb_main_node", beeb_main_node)
    builder.add_node("scout_bee_node", scout_bee_node)
    builder.add_node("memory_bee_node", memory_bee_node)
    builder.add_node("scribe_bee_node", scribe_bee_node)
    builder.add_node("summarize_conversation", summarize_conversation)
    
    # Define the flow - start with Beeb, then route based on decisions
    builder.add_edge(START, "beeb_main_node")
    builder.add_conditional_edges("beeb_main_node", route_decisions)
    
    # Worker bees return to Beeb for coordination
    builder.add_edge("scout_bee_node", "beeb_main_node")
    builder.add_edge("memory_bee_node", "beeb_main_node")
    builder.add_edge("scribe_bee_node", "beeb_main_node")
    
    # Summarization ends the conversation
    builder.add_edge("summarize_conversation", END)
    
    # Compile the graph - LangGraph platform handles persistence automatically
    graph = builder.compile()
    
    return graph

# Keep the old function for backward compatibility
def create_bargainb_memory_agent_legacy():
    """Legacy function - use create_bargainb_memory_agent() instead."""
    return create_bargainb_memory_agent()

# Export the graph instance for LangGraph deployment
memory_agent = create_bargainb_memory_agent() 