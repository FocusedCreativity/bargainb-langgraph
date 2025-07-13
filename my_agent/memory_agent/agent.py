"""
BargainB Memory Agent - Simplified Architecture Following mem.md Patterns

Based on the mem.md tutorial, this agent implements a simple architecture where:
- Main agent decides what memory type to update
- Separate nodes handle each memory type (profile, shopping, instructions)
- Trustcall with Spy visibility for memory operations
- Direct store integration without fallbacks

This replaces the complex bee hierarchy with a simpler, more reliable approach.
"""

from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import merge_message_runs, HumanMessage, SystemMessage
from langgraph.store.base import BaseStore

from my_agent.memory_agent.state import BargainBMemoryState
from my_agent.memory_agent.schemas import UserProfile, ShoppingMemory, Instructions

from langchain_openai import ChatOpenAI
from trustcall import create_extractor
from datetime import datetime
import uuid

# Import the spy class from mem.md
class Spy:
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        # Collect information about the tool calls made by the extractor.
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

def extract_tool_info(tool_calls, schema_name="Memory"):
    """Extract information from tool calls for both patches and new memories."""
    changes = []
    
    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                changes.append({
                    'type': 'update',
                    'doc_id': call['args']['json_doc_id'],
                    'planned_edits': call['args']['planned_edits'],
                    'value': call['args']['patches'][0]['value']
                })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })

    # Format results as a single string
    result_parts = []
    for change in changes:
        if change['type'] == 'update':
            result_parts.append(
                f"Document {change['doc_id']} updated:\n"
                f"Plan: {change['planned_edits']}\n"
                f"Updated content: {change['value']}"
            )
        else:
            result_parts.append(
                f"New {schema_name} created:\n"
                f"Content: {change['value']}"
            )
    
    return "\n\n".join(result_parts) if result_parts else f"No {schema_name} changes detected"

# Memory type decision tool (like UpdateMemory in mem.md)
from typing import TypedDict
from pydantic import BaseModel

class UpdateMemory(BaseModel):
    """Decision on what memory type to update"""
    update_type: Literal['profile', 'shopping', 'instructions']

# Initialize the model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# System message for the main agent (like MODEL_SYSTEM_MESSAGE in mem.md)
BARGAINB_SYSTEM_MESSAGE = """You are BargainB 💰, a helpful grocery shopping assistant with long-term memory.

You help users find the best deals on groceries while remembering their preferences and shopping history.

You have long-term memory that keeps track of three things:
1. User profile (personal info, preferences, dietary restrictions)
2. Shopping history (past interactions, products discussed, feedback)
3. Instructions (how you should behave and communicate)

Here is the current User Profile:
<user_profile>
{user_profile}
</user_profile>

Here is the current Shopping History:
<shopping_history>
{shopping_history}
</shopping_history>

Here are the current Instructions:
<instructions>
{instructions}
</instructions>

## Your Instructions for Memory Updates:

1. Reason carefully about the user's messages.

2. Decide whether any of your long-term memory should be updated:
- If personal information, preferences, or dietary info was provided, update the user profile by calling UpdateMemory with type `profile`
- If shopping interactions, product discussions, or feedback occurred, update shopping history by calling UpdateMemory with type `shopping`
- If the user specified preferences for how you should behave, update instructions by calling UpdateMemory with type `instructions`

3. Tell the user when you update memories:
- Don't mention profile updates (personal info is private)
- Do mention when you remember shopping preferences or feedback
- Don't mention instruction updates (system behavior is internal)

4. Always prioritize helping with grocery shopping and finding good deals.

5. Respond naturally after updating memories or if no updates are needed.

Current time: {current_time}
"""

# Trustcall instruction (like mem.md)
TRUSTCALL_INSTRUCTION = """Reflect on the following interaction and extract relevant information for BargainB's memory system.

Use the provided tools to retain necessary information about the user's grocery shopping needs.

Use parallel tool calling to handle updates and insertions simultaneously.

Focus on actionable information that will help personalize future shopping assistance.

System Time: {time}"""

# Instructions for updating system behavior (like mem.md)
CREATE_INSTRUCTIONS = """Reflect on the following interaction and update your instructions for how to help this user with grocery shopping.

Based on this interaction, update your behavior preferences, communication style, and personalization approach.

Your current instructions are:
<current_instructions>
{current_instructions}
</current_instructions>

Focus on how to better assist with grocery shopping, deal finding, and product recommendations."""

def bargainb_main_agent(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore):
    """
    Main BargainB agent that makes decisions about memory updates.
    
    Like task_mAIstro in mem.md, this agent:
    - Loads memories from store for context
    - Makes decisions about what memory types to update
    - Routes to appropriate memory handlers
    - Responds to users naturally
    """
    
    # Get the user ID from the config
    user_id = config["configurable"]["user_id"]

    # Retrieve profile memory from the store
    namespace = ("profile", user_id)
    memories = store.search(namespace)
    user_profile = memories[0].value if memories else None

    # Retrieve shopping memory from the store
    namespace = ("shopping", user_id)
    memories = store.search(namespace)
    shopping_history = "\n".join(f"- {mem.value}" for mem in memories[-5:])  # Last 5 interactions

    # Retrieve instructions
    namespace = ("instructions", user_id)
    memories = store.search(namespace)
    instructions = memories[0].value if memories else ""
    
    # Format system message with current memories
    system_msg = BARGAINB_SYSTEM_MESSAGE.format(
        user_profile=user_profile,
        shopping_history=shopping_history,
        instructions=instructions,
        current_time=datetime.now().isoformat()
    )

    # Respond using memory context and chat history
    response = model.bind_tools([UpdateMemory], parallel_tool_calls=False).invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )

    return {"messages": [response]}

def update_profile_memory(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore):
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

def update_shopping_memory(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore):
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

def update_instructions_memory(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore):
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

def route_memory_decisions(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> Literal[END, "update_shopping_memory", "update_instructions_memory", "update_profile_memory"]:
    """Route memory decisions to appropriate handlers (like route_message in mem.md)."""
    
    message = state['messages'][-1]
    if len(message.tool_calls) == 0:
        return END
    else:
        tool_call = message.tool_calls[0]
        if tool_call['args']['update_type'] == "profile":
            return "update_profile_memory"
        elif tool_call['args']['update_type'] == "shopping":
            return "update_shopping_memory"
        elif tool_call['args']['update_type'] == "instructions":
            return "update_instructions_memory"
        else:
            return END

def create_bargainb_memory_agent():
    """
    Create the simplified BargainB memory agent following mem.md patterns.
    
    This agent follows the exact structure from mem.md:
    - Main agent makes memory type decisions
    - Separate nodes handle each memory type
    - Direct store integration
    - Trustcall with Spy visibility
    
    Returns:
        Compiled graph with memory persistence
    """
    
    # Create the graph following mem.md structure
    builder = StateGraph(BargainBMemoryState)
    
    # Define nodes (like mem.md)
    builder.add_node("bargainb_main_agent", bargainb_main_agent)
    builder.add_node("update_profile_memory", update_profile_memory)
    builder.add_node("update_shopping_memory", update_shopping_memory)
    builder.add_node("update_instructions_memory", update_instructions_memory)
    
    # Define the flow (like mem.md)
    builder.add_edge(START, "bargainb_main_agent")
    builder.add_conditional_edges("bargainb_main_agent", route_memory_decisions)
    builder.add_edge("update_profile_memory", "bargainb_main_agent")
    builder.add_edge("update_shopping_memory", "bargainb_main_agent")
    builder.add_edge("update_instructions_memory", "bargainb_main_agent")
    
    # Store for long-term (across-thread) memory
    across_thread_memory = InMemoryStore()
    
    # Checkpointer for short-term (within-thread) memory
    within_thread_memory = MemorySaver()
    
    # Compile the graph with store integration (like mem.md)
    graph = builder.compile(checkpointer=within_thread_memory, store=across_thread_memory)
    
    return graph

# Keep the old function for backward compatibility, but use the new simplified version
def create_bargainb_memory_agent_legacy():
    """Legacy function - use create_bargainb_memory_agent() instead."""
    return create_bargainb_memory_agent()

# Export the graph instance for LangGraph deployment
# This is what langgraph.json expects to find
memory_agent = create_bargainb_memory_agent() 