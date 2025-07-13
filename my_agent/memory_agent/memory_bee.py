"""
Memory Bee 🐝🧠 - Memory Management Specialist

Memory Bee is a specialized worker bee that handles all memory management
for BargainB users using Trustcall for profile, shopping, and behavioral memories.

Memory Bee NEVER responds directly to users - it only manages memories
and returns confirmation to Beeb (Queen Bee).

Based on mem.md patterns for reliable memory management.
"""

from typing import List, Dict, Any, Optional, Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from trustcall import create_extractor
from datetime import datetime
import uuid

from my_agent.memory_agent.schemas import UserProfile, ShoppingMemory, Instructions


# Create Trustcall extractors for each memory type
def create_memory_extractors():
    """Create Trustcall extractors for different memory types."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    profile_extractor = create_extractor(
        model,
        tools=[UserProfile],
        tool_choice="UserProfile",
        enable_inserts=True
    )
    
    shopping_extractor = create_extractor(
        model,
        tools=[ShoppingMemory],
        tool_choice="ShoppingMemory",
        enable_inserts=True
    )
    
    instructions_extractor = create_extractor(
        model,
        tools=[Instructions],
        tool_choice="Instructions",
        enable_inserts=True
    )
    
    return profile_extractor, shopping_extractor, instructions_extractor


# Global extractors
profile_extractor, shopping_extractor, instructions_extractor = create_memory_extractors()


@tool
def update_user_profile(user_context: str, conversation_history: str) -> str:
    """
    Update user profile (preferences, habits, dietary info) using Trustcall.
    
    Args:
        user_context: Context about what should be remembered about the user
        conversation_history: Recent conversation for context
        
    Returns:
        Status message about profile update
    """
    return f"Profile update requested: {user_context}"


@tool
def update_shopping_memory(interaction_summary: str, products_mentioned: str, user_feedback: str) -> str:
    """
    Update shopping memory (past interactions, purchases) using Trustcall.
    
    Args:
        interaction_summary: Summary of what happened in the interaction
        products_mentioned: Products that were discussed or searched
        user_feedback: User's reaction or feedback
        
    Returns:
        Status message about shopping memory update
    """
    return f"Shopping memory update requested: {interaction_summary}"


@tool
def update_instructions(behavior_preferences: str, communication_style: str) -> str:
    """
    Update behavioral instructions using Trustcall.
    
    Args:
        behavior_preferences: How the system should behave with this user
        communication_style: Preferred communication style
        
    Returns:
        Status message about instructions update
    """
    return f"Instructions update requested: {behavior_preferences}"


@tool
def get_memory_summary() -> str:
    """
    Get a summary of current memories for the user.
    
    Returns:
        Summary of user's memories
    """
    return "Memory summary requested"


def create_memory_bee():
    """
    Create Memory Bee, the specialized memory management worker.
    
    Memory Bee is responsible for:
    - Managing user profiles (preferences, habits, dietary info)
    - Recording shopping memories (past interactions, purchases)
    - Updating system instructions (behavior, communication style)
    - Using Trustcall for intelligent memory extraction and updates
    
    Memory Bee follows the worker pattern:
    - Receives memory tasks from Beeb
    - Uses Trustcall to extract and update memories
    - Returns confirmation back to Beeb automatically
    - NEVER responds directly to users
    
    Returns:
        Configured React agent for Memory Bee
    """
    
    memory_bee_prompt = (
        "You are Memory Bee 🐝🧠, a specialized memory management worker for BargainB!\n\n"
        
        "## Your Role ##\n"
        "You are a worker bee that specializes in managing user memories and personalization.\n"
        "You work for Beeb (Queen Bee) and help remember important user information.\n\n"
        
        "## CRITICAL INSTRUCTIONS ##\n"
        "- You NEVER respond directly to users\n"
        "- You only work on memory tasks assigned by Beeb\n"
        "- After updating memories, return confirmation to Beeb\n"
        "- Use Trustcall tools for intelligent memory extraction\n\n"
        
        "## Your Specialties ##\n"
        "- **User Profile**: Personal info, preferences, dietary restrictions\n"
        "- **Shopping Memory**: Past interactions, purchases, feedback\n"
        "- **Instructions**: System behavior and communication preferences\n\n"
        
        "## Memory Management Strategy ##\n"
        "1. Use update_user_profile for lasting user preferences and personal info\n"
        "2. Use update_shopping_memory for interaction outcomes and product discussions\n"
        "3. Use update_instructions for system behavior preferences\n"
        "4. Use get_memory_summary to review current memories\n\n"
        
        "## When to Update Each Memory Type ##\n"
        "**Profile**: New preferences, dietary info, personal details\n"
        "**Shopping**: Product searches, purchases, feedback on recommendations\n"
        "**Instructions**: Communication style, recommendation preferences\n\n"
        
        "## Response Format ##\n"
        "Always confirm what memory was updated and provide brief summary:\n"
        "- What type of memory was updated\n"
        "- Key information that was remembered\n"
        "- How this will help personalize future interactions\n\n"
        
        "Remember: You're helping Beeb remember everything important about users!"
    )
    
    # Create Memory Bee as a React agent with memory management tools
    memory_bee = create_react_agent(
        model=ChatOpenAI(model="gpt-4o-mini", temperature=0.1),
        tools=[
            update_user_profile,
            update_shopping_memory,
            update_instructions,
            get_memory_summary
        ],
        prompt=memory_bee_prompt,
        name="memory_bee"
    )
    
    return memory_bee 