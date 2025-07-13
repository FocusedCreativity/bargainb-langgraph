"""
Delegation Routing Nodes

Handles routing between Beeb (main assistant) and specialized agents
based on the dialog_state stack, following the tutorial patterns.
"""

from typing import Literal
from langchain_core.runnables import RunnableConfig
from langgraph.store.base import BaseStore

from .state import BargainBMemoryState
from .beeb_assistant import BeebAssistant, create_beeb_assistant
from .product_search_agent import ProductSearchAgent, create_product_search_agent, format_search_results_for_beeb


# Create agent instances
beeb_assistant = BeebAssistant(create_beeb_assistant())
product_search_agent = ProductSearchAgent(create_product_search_agent())


def entry_node(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Entry point for all conversations. Handles delegation between Beeb and specialized agents.
    
    This node implements the full delegation flow:
    1. Start with Beeb for user interaction
    2. If Beeb calls ToProductSearch, delegate to product search agent
    3. Product search agent uses CompleteOrEscalate to return results to Beeb
    4. Beeb incorporates results and responds to user
    
    Args:
        state: Current conversation state
        config: Runtime configuration
        store: Memory store for persistence
        
    Returns:
        Updated state with final response to user
    """
    messages = state.get("messages", [])
    
    # Step 1: Start with Beeb (main assistant)
    beeb_result = beeb_assistant(state, config)
    current_messages = beeb_result["messages"]
    last_message = current_messages[-1] if current_messages else None
    
    # Step 2: Check if Beeb wants to delegate
    if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            if tool_call.get("name") == "ToProductSearch":
                # Prepare clean state for product search agent (remove Beeb's tool calls)
                # The product search agent should only see user messages
                clean_messages = []
                for msg in current_messages[:-1]:  # Exclude the last message with tool calls
                    if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                        clean_messages.append(msg)
                
                # Add the original user query as a new message for the search agent
                from langchain_core.messages import HumanMessage
                search_query = tool_call.get("args", {}).get("query", "product search")
                clean_messages.append(HumanMessage(content=search_query))
                
                delegation_state = {
                    **beeb_result,
                    "messages": clean_messages,
                    "dialog_state": ["product_search"]  # Push to delegation stack
                }
                
                search_result = product_search_agent(delegation_state, config)
                search_messages = search_result["messages"]
                search_last_message = search_messages[-1] if search_messages else None
                
                # Step 3: Process results from product search agent
                if (search_last_message and hasattr(search_last_message, 'tool_calls') and 
                    search_last_message.tool_calls):
                    for search_tool_call in search_last_message.tool_calls:
                        if search_tool_call.get("name") == "CompleteOrEscalate":
                            # Get results from product search
                            search_results = search_tool_call.get("args", {}).get("reason", "No results found")
                            
                            # Step 4: Give results back to Beeb for final response
                            from langchain_core.messages import SystemMessage, ToolMessage
                            
                            # Create proper tool response message
                            tool_response = ToolMessage(
                                content=search_results,
                                tool_call_id=tool_call.get("id", "unknown")
                            )
                            
                            # Add tool response to conversation
                            final_messages = current_messages + [tool_response]
                            
                            # Have Beeb respond with the search results incorporated
                            final_state = {
                                **state,
                                "messages": final_messages,
                                "dialog_state": []  # Reset dialog state
                            }
                            
                            # Get Beeb's final response with search results
                            final_result = beeb_assistant(final_state, config)
                            return final_result
    
    # No delegation needed - return Beeb's direct response
    return beeb_result


def route_delegation(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> Literal["continue_dialog", "pop_dialog", "__end__"]:
    """
    Route conversation flow based on tool calls and dialog state.
    
    Determines whether to:
    - Continue with current assistant
    - Delegate to specialized agent (push to stack)
    - Return to previous assistant (pop from stack)
    - End conversation
    
    Args:
        state: Current conversation state
        config: Runtime configuration  
        store: Memory store for persistence
        
    Returns:
        Next node to execute
    """
    dialog_state = state.get("dialog_state", [])
    last_message = state["messages"][-1] if state["messages"] else None
    
    # Check for conversation ending conditions first
    if last_message:
        content = getattr(last_message, 'content', '')
        if isinstance(content, str):
            # Look for conversation ending phrases
            end_phrases = ["goodbye", "bye", "thanks, that's all", "end conversation"]
            if any(phrase in content.lower() for phrase in end_phrases):
                return "__end__"
    
    # Check if there are tool calls to process
    if last_message and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name")
            
            if tool_name == "ToProductSearch":
                # Beeb wants to delegate to product search
                return "continue_dialog"  # Will push "product_search" to stack
            
            elif tool_name == "CompleteOrEscalate":
                # Specialized agent wants to return control
                return "pop_dialog"  # Will pop from stack and return to previous
    
    # If this is the first message (user just started), continue
    if len(state.get("messages", [])) <= 1:
        return "continue_dialog"
    
    # If no tool calls and we have a response from Beeb, we can end the turn
    # This prevents infinite loops - each user input gets one response cycle
    if last_message and hasattr(last_message, 'content') and not hasattr(last_message, 'tool_calls'):
        return "__end__"
    
    # Default to continue
    return "continue_dialog"


def handle_delegation_tool_calls(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Process delegation tool calls and update dialog_state accordingly.
    
    Handles:
    - ToProductSearch: Delegate from Beeb to product search agent
    - CompleteOrEscalate: Return from specialized agent to previous assistant
    
    Args:
        state: Current conversation state
        config: Runtime configuration
        store: Memory store for persistence
        
    Returns:
        Updated state with processed tool calls and updated dialog_state
    """
    dialog_state = state.get("dialog_state", [])
    last_message = state["messages"][-1]
    updated_state = dict(state)
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            
            if tool_name == "ToProductSearch":
                # Delegate to product search agent
                updated_state["dialog_state"] = "product_search"
                
                # Remove the delegation tool call message from the conversation
                # The product search agent will handle the actual search
                # This prevents the tool call from appearing in user-facing conversation
                
            elif tool_name == "CompleteOrEscalate":
                # Return from specialized agent
                reason = tool_args.get("reason", "Task completed")
                
                # Pop from dialog stack (return to previous assistant)
                updated_state["dialog_state"] = "pop"
                
                # Process the results from specialized agent
                if dialog_state and dialog_state[-1] == "product_search":
                    # Format search results for Beeb to incorporate into response
                    updated_state = _handle_product_search_results(updated_state, reason)
    
    return updated_state


def _handle_product_search_results(state: dict, search_results: str) -> dict:
    """
    Handle results returned from product search agent.
    
    Args:
        state: Current conversation state
        search_results: Formatted results from product search agent
        
    Returns:
        Updated state with search results for Beeb to use
    """
    # Store search results in state for Beeb to access
    state["product_search_results"] = search_results
    
    # Add a system message with the search results for Beeb
    from langchain_core.messages import SystemMessage
    
    search_system_message = SystemMessage(
        content=f"Product search completed. Results: {search_results}"
    )
    
    # Insert the system message for Beeb to see
    state["messages"] = state["messages"] + [search_system_message]
    
    return state


def exit_dialog_node(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> BargainBMemoryState:
    """
    Handle exiting from specialized agent back to previous assistant.
    
    This node processes the results from specialized agents and
    ensures smooth transition back to the main conversation flow.
    
    Args:
        state: Current conversation state
        config: Runtime configuration
        store: Memory store for persistence
        
    Returns:
        Updated state ready for previous assistant
    """
    dialog_state = state.get("dialog_state", [])
    
    if not dialog_state:
        # Already at root level - nothing to pop
        return state
    
    # The dialog_state will be popped by the graph's update mechanism
    # We just need to ensure the state is clean for the previous assistant
    
    # Remove any internal agent messages that shouldn't be shown to users
    cleaned_messages = []
    for message in state["messages"]:
        # Keep user messages and assistant responses, filter internal tool messages
        if not (hasattr(message, 'tool_calls') and message.tool_calls and 
                any(tc.get("name") in ["ToProductSearch", "CompleteOrEscalate"] 
                    for tc in message.tool_calls)):
            cleaned_messages.append(message)
    
    return {
        **state,
        "messages": cleaned_messages
    }


def should_continue_dialog(state: BargainBMemoryState, config: RunnableConfig, store: BaseStore) -> Literal["entry", "__end__"]:
    """
    Determine if the conversation should continue or end.
    
    Args:
        state: Current conversation state
        config: Runtime configuration
        store: Memory store for persistence
        
    Returns:
        Next action: continue with "entry" or "__end__" conversation
    """
    last_message = state["messages"][-1] if state["messages"] else None
    
    # Continue if there are no messages (shouldn't happen) or if we have unprocessed tool calls
    if not last_message:
        return "entry"
    
    # Check if there are pending tool calls that need processing
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "entry"  # Continue to process tool calls
    
    # Check for explicit end signals
    content = getattr(last_message, 'content', '')
    if isinstance(content, str):
        # Look for conversation ending phrases
        end_phrases = ["goodbye", "bye", "thanks, that's all", "end conversation"]
        if any(phrase in content.lower() for phrase in end_phrases):
            return "__end__"
    
    # Default to continuing the conversation
    return "entry" 