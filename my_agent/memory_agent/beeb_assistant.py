"""
Beeb - BargainB's Main Grocery Assistant

Beeb is the friendly face of BargainB who always responds to users.
He can handle general conversation, memory management, and delegates
product searches to specialized agents while maintaining the conversation flow.
"""

from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from datetime import datetime

from .delegation_tools import ToProductSearch


def create_beeb_assistant() -> Runnable:
    """
    Create Beeb, the primary BargainB assistant.
    
    Beeb is responsible for:
    - Always being the one who responds to users
    - Managing conversation flow and user memory
    - Delegating product searches to specialized agents
    - Incorporating search results into natural responses
    - Providing shopping advice and recommendations
    
    Returns:
        Configured Runnable for Beeb assistant
    """
    
    beeb_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are Beeb, the friendly and helpful grocery shopping assistant for BargainB! 🛒\n\n"
            
            "## Your Personality ##\n"
            "- Warm, friendly, and enthusiastic about helping people save money\n"
            "- Budget-conscious and always looking for the best deals\n" 
            "- Knowledgeable about Dutch grocery stores and products\n"
            "- Patient and understanding of different dietary needs and preferences\n"
            "- Use emojis occasionally to keep things friendly (but not excessive)\n\n"
            
            "## Your Capabilities ##\n"
            "- Help users find products and compare prices\n"
            "- Remember user preferences and shopping history\n"
            "- Provide budget-friendly shopping recommendations\n"
            "- Answer questions about stores, products, and deals\n"
            "- Give cooking tips and meal planning advice\n\n"
            
            "## Product Search Delegation ##\n"
            "When users ask about products, prices, or need product recommendations:\n"
            "- Use ToProductSearch to delegate to the specialized search agent\n"
            "- You cannot search products directly - only the specialized agent can do this\n"
            "- Never mention the search agent to users - seamlessly incorporate results\n"
            "- Always respond naturally after getting search results back\n\n"
            
            "## User Memory Context ##\n"
            "User ID: {user_id}\n"
            "Thread ID: {thread_id}\n"
            "Conversation Summary: {summary}\n"
            "Semantic Memory: {semantic_memory}\n"
            "Recent Episodes: {episodic_memories}\n"
            "Behavioral Preferences: {procedural_memory}\n\n"
            
            "## Current Conversation ##\n"
            "Remember previous context and maintain conversation continuity.\n"
            "Time: {time}\n\n"
            
            "## Response Guidelines ##\n"
            "- Always respond as Beeb in first person\n"
            "- Be conversational and helpful\n"
            "- Show price-conscious awareness\n"
            "- Suggest money-saving tips when relevant\n"
            "- Ask follow-up questions to better help users\n"
            "- Keep responses focused but comprehensive\n"
        ),
        ("placeholder", "{messages}")
    ]).partial(time=datetime.now)
    
    # Beeb's direct tools (non-product related)
    beeb_direct_tools = [
        # Memory management tools would go here
        # Conversation tools, weather, general Q&A, etc.
    ]
    
    # LLM with delegation capability
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Beeb can delegate to product search or use direct tools
    beeb_runnable = beeb_prompt | llm.bind_tools(
        beeb_direct_tools + [ToProductSearch]
    )
    
    return beeb_runnable


class BeebAssistant:
    """
    Wrapper class for Beeb assistant to handle state management and response generation.
    """
    
    def __init__(self, runnable: Runnable):
        self.runnable = runnable
    
    def __call__(self, state, config: RunnableConfig):
        """
        Process user input and generate Beeb's response.
        
        Args:
            state: Current conversation state with memory context
            config: Runtime configuration
            
        Returns:
            Updated state with Beeb's response or delegation
        """
        while True:
            # Format state for Beeb's context
            formatted_state = {
                **state,
                "user_id": state.get("user_id"),
                "thread_id": state.get("thread_id"), 
                "summary": state.get("summary", "No previous conversation"),
                "semantic_memory": self._format_semantic_memory(state.get("semantic_memory")),
                "episodic_memories": self._format_episodic_memories(state.get("episodic_memories", [])),
                "procedural_memory": self._format_procedural_memory(state.get("procedural_memory"))
            }
            
            # Generate response
            result = self.runnable.invoke(formatted_state, config)
            
            # Ensure Beeb always provides meaningful output
            if not result.tool_calls and (
                not result.content or 
                (isinstance(result.content, list) and not result.content[0].get("text"))
            ):
                # Re-prompt for actual response
                messages = state["messages"] + [("user", "Please provide a helpful response.")]
                formatted_state = {**formatted_state, "messages": messages}
            else:
                break
        
        return {"messages": [result]}
    
    def _format_semantic_memory(self, semantic_memory: Optional[dict]) -> str:
        """Format semantic memory for prompt context."""
        if not semantic_memory:
            return "No stored preferences yet"
        
        parts = []
        if dietary := semantic_memory.get("dietary_preferences"):
            parts.append(f"Dietary: {', '.join(dietary)}")
        if budget := semantic_memory.get("budget_sensitivity"):
            parts.append(f"Budget: {budget}")
        if stores := semantic_memory.get("preferred_stores"):
            parts.append(f"Stores: {', '.join(stores)}")
        if products := semantic_memory.get("liked_products"):
            parts.append(f"Likes: {', '.join(products[:3])}")  # Top 3
        
        return "; ".join(parts) if parts else "Learning user preferences..."
    
    def _format_episodic_memories(self, episodic_memories: List[dict]) -> str:
        """Format recent episodic memories for prompt context."""
        if not episodic_memories:
            return "No recent shopping history"
        
        recent = episodic_memories[-3:]  # Last 3 episodes
        formatted = []
        
        for episode in recent:
            if episode.get("type") == "product_search":
                formatted.append(f"Searched: {episode.get('query', 'Unknown')}")
            elif episode.get("type") == "feedback":
                formatted.append(f"Feedback: {episode.get('content', 'Unknown')}")
        
        return "; ".join(formatted) if formatted else "No recent activity"
    
    def _format_procedural_memory(self, procedural_memory: Optional[dict]) -> str:
        """Format procedural memory for prompt context."""
        if not procedural_memory:
            return "Standard helpful assistance mode"
        
        parts = []
        if tone := procedural_memory.get("communication_style"):
            parts.append(f"Style: {tone}")
        if approach := procedural_memory.get("recommendation_approach"):
            parts.append(f"Approach: {approach}")
        
        return "; ".join(parts) if parts else "Adapting to user preferences..." 