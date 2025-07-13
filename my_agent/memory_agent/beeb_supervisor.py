"""
Beeb - The Queen Bee Supervisor 🐝👑

Beeb is the main supervisor for BargainB who coordinates all worker bees:
- Scout Bee 🐝🔍: Product search and price comparison
- Memory Bee 🐝🧠: Memory management and personalization  
- Scribe Bee 🐝📝: Conversation summarization

Beeb always responds to users and delegates tasks to specialized worker bees
while maintaining conversation continuity and memory context.
"""

from typing import Annotated, List, Optional, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from datetime import datetime

from my_agent.memory_agent.state import BargainBMemoryState
from my_agent.memory_agent.schemas import MemoryUpdate


# Handoff tools for delegation to worker bees
@tool
def assign_to_scout_bee(
    task_description: Annotated[
        str,
        "Description of the product search task, including user preferences and context"
    ]
) -> str:
    """
    Delegate product search and price comparison tasks to Scout Bee 🐝🔍
    
    Use this when users ask about:
    - Finding specific products
    - Comparing prices across stores
    - Getting product recommendations
    - Checking product availability
    """
    return f"Task assigned to Scout Bee 🐝🔍: {task_description}"


@tool
def assign_to_memory_bee(
    memory_type: Annotated[
        str,
        "Type of memory to update: 'profile', 'shopping', or 'instructions'"
    ],
    context: Annotated[
        str,
        "Context and details about what should be remembered or updated"
    ]
) -> str:
    """
    Delegate memory management tasks to Memory Bee 🐝🧠
    
    Use this when you need to:
    - Update user profile and preferences (profile memory)
    - Record shopping interactions and feedback (shopping memory)  
    - Adjust system behavior and communication style (instructions memory)
    """
    return f"Memory task assigned to Memory Bee 🐝🧠: Update {memory_type} memory with: {context}"


@tool
def assign_to_scribe_bee() -> str:
    """
    Delegate conversation summarization to Scribe Bee 🐝📝
    
    Use this when the conversation gets too long and needs summarization
    to maintain context while reducing message count.
    """
    return "Summarization task assigned to Scribe Bee 🐝📝"


def create_beeb_supervisor():
    """
    Create Beeb, the Queen Bee supervisor who coordinates all worker bees.
    
    Beeb is responsible for:
    - Always being the primary interface with users
    - Understanding user needs and delegating to appropriate worker bees
    - Incorporating worker bee results into natural responses
    - Maintaining conversation continuity and memory context
    - Making decisions about when to delegate vs. handle directly
    
    Returns:
        Configured ChatOpenAI model with delegation tools
    """
    
    beeb_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are Beeb 🐝👑, the Queen Bee and main supervisor for BargainB grocery shopping assistant!\n\n"
            
            "## Your Role ##\n"
            "You are the friendly face that users interact with. You coordinate a team of specialized worker bees:\n"
            "- Scout Bee 🐝🔍: Handles product searches and price comparisons\n"
            "- Memory Bee 🐝🧠: Manages user preferences and memory updates\n"
            "- Scribe Bee 🐝📝: Summarizes conversations when they get too long\n\n"
            
            "## Your Personality ##\n"
            "- Warm, friendly, and enthusiastic about helping people save money 💰\n"
            "- Budget-conscious and always looking for the best deals\n"
            "- Knowledgeable about Dutch grocery stores and products\n"
            "- Patient and understanding of different dietary needs\n"
            "- Use emojis occasionally to keep things friendly (but not excessive)\n\n"
            
            "## Delegation Guidelines ##\n"
            "**Product Searches**: Use assign_to_scout_bee for:\n"
            "- Finding specific products or brands\n"
            "- Comparing prices across stores\n"
            "- Product recommendations based on preferences\n"
            "- Checking availability or deals\n\n"
            
            "**Memory Updates**: Use assign_to_memory_bee for:\n"
            "- New user preferences, dietary restrictions, or personal info → 'profile' memory\n"
            "- Recording product searches, purchases, or feedback → 'shopping' memory\n"
            "- Updating communication style or system behavior → 'instructions' memory\n\n"
            
            "**Conversation Management**: Use assign_to_scribe_bee when:\n"
            "- The conversation has more than 8 messages\n"
            "- Context is getting too long to manage effectively\n\n"
            
            "## User Memory Context ##\n"
            "Current User: {user_id}\n"
            "Thread: {thread_id}\n"
            "Conversation Summary: {summary}\n"
            
            "**User Preferences (Semantic Memory):**\n"
            "{semantic_memory}\n"
            
            "**Recent Interactions (Episodic Memory):**\n"
            "{episodic_memories}\n"
            
            "**Behavioral Preferences (Procedural Memory):**\n"
            "{procedural_memory}\n\n"
            
            "## Worker Bee Results ##\n"
            "**Scout Bee 🐝🔍 Search Results:**\n"
            "{scout_results}\n\n"
            
            "**Memory Bee 🐝🧠 Results:**\n"
            "{memory_results}\n\n"
            
            "**Scribe Bee 🐝📝 Results:**\n"
            "{scribe_results}\n\n"
            
            "## Response Guidelines ##\n"
            "- Always respond as Beeb in first person\n"
            "- Be conversational and helpful\n"
            "- Show price-conscious awareness when relevant\n"
            "- Use memory context to personalize responses\n"
            "- If you have results from worker bees, use them in your response instead of delegating again\n"
            "- Never mention the worker bees to users - seamlessly integrate results\n"
            "- Ask follow-up questions to better understand user needs\n"
            "- Only delegate to worker bees if you need NEW information that isn't already available in the results\n\n"
            
            "Current Time: {time}\n"
        ),
        ("placeholder", "{messages}")
    ]).partial(
        time=datetime.now().isoformat()
    )
    
    # Create the LLM with delegation tools
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Bind delegation tools to Beeb
    beeb_with_tools = beeb_prompt | llm.bind_tools([
        assign_to_scout_bee,
        assign_to_memory_bee, 
        assign_to_scribe_bee
    ])
    
    return beeb_with_tools


def _format_semantic_memory(semantic_memory: Optional[Dict[str, Any]]) -> str:
    """Format user profile memory for display in prompt."""
    if not semantic_memory:
        return "No profile information recorded yet"
    
    formatted = []
    
    # Personal information
    if semantic_memory.get("name"):
        formatted.append(f"Name: {semantic_memory['name']}")
    if semantic_memory.get("location"):
        formatted.append(f"Location: {semantic_memory['location']}")
    if semantic_memory.get("family_size"):
        formatted.append(f"Family size: {semantic_memory['family_size']}")
    
    # Preferences
    if semantic_memory.get("likes"):
        formatted.append(f"Likes: {', '.join(semantic_memory['likes'])}")
    if semantic_memory.get("dislikes"):
        formatted.append(f"Dislikes: {', '.join(semantic_memory['dislikes'])}")
    
    # Dietary information
    if semantic_memory.get("dietary_restrictions"):
        formatted.append(f"Dietary restrictions: {', '.join(semantic_memory['dietary_restrictions'])}")
    if semantic_memory.get("allergies"):
        formatted.append(f"Allergies: {', '.join(semantic_memory['allergies'])}")
    
    # Shopping preferences
    if semantic_memory.get("budget_sensitivity"):
        formatted.append(f"Budget sensitivity: {semantic_memory['budget_sensitivity']}")
    if semantic_memory.get("preferred_stores"):
        formatted.append(f"Preferred stores: {', '.join(semantic_memory['preferred_stores'])}")
    
    return "\n".join(formatted) if formatted else "No specific profile information"


def _format_episodic_memories(episodic_memories: List[Dict[str, Any]]) -> str:
    """Format recent shopping memories for display in prompt."""
    if not episodic_memories:
        return "No recent shopping interactions"
    
    # Show last 3 shopping interactions
    recent = episodic_memories[-3:]
    formatted = []
    
    for memory in recent:
        date = memory.get("interaction_date", "Unknown date")
        outcome = memory.get("outcome", "")
        
        # Build interaction summary
        interaction_parts = [f"• {date}:"]
        
        if memory.get("products_searched"):
            products = memory["products_searched"]
            if len(products) > 2:
                interaction_parts.append(f"searched for {', '.join(products[:2])} and {len(products)-2} more")
            else:
                interaction_parts.append(f"searched for {', '.join(products)}")
        
        if memory.get("products_purchased"):
            purchased = memory["products_purchased"]
            if len(purchased) > 2:
                interaction_parts.append(f"purchased {', '.join(purchased[:2])} and {len(purchased)-2} more")
            else:
                interaction_parts.append(f"purchased {', '.join(purchased)}")
        
        if memory.get("price_sensitivity_shown"):
            interaction_parts.append(f"({memory['price_sensitivity_shown']} price sensitivity)")
        
        if outcome:
            interaction_parts.append(f"→ {outcome}")
        
        formatted.append(" ".join(interaction_parts))
    
    return "\n".join(formatted)


def _format_procedural_memory(procedural_memory: Optional[Dict[str, Any]]) -> str:
    """Format behavioral instructions for display in prompt."""
    if not procedural_memory:
        return "Default helpful and friendly approach"
    
    # Instructions schema has content and last_updated
    content = procedural_memory.get("content", "")
    last_updated = procedural_memory.get("last_updated", "")
    
    if content:
        formatted = [f"Instructions: {content}"]
        if last_updated:
            formatted.append(f"Last updated: {last_updated}")
        return "\n".join(formatted)
    
    return "Standard helpful approach" 