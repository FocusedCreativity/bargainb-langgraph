"""
BargainB Memory Agent Tools

Tools for extracting and managing user memories using Trustcall
and integrating with the BargainB product database.
"""

import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from trustcall import create_extractor
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage

from .schemas import SemanticMemory, EpisodicMemory, ProceduralMemory, MemoryUpdate


class MemoryToolkit:
    """Toolkit for managing BargainB user memories with Trustcall integration."""
    
    def __init__(self, model: ChatOpenAI):
        self.model = model
        
        # Create Trustcall extractors for each memory type
        self.semantic_extractor = create_extractor(
            model,
            tools=[SemanticMemory],
            tool_choice="SemanticMemory",
            enable_inserts=True
        )
        
        self.episodic_extractor = create_extractor(
            model,
            tools=[EpisodicMemory], 
            tool_choice="EpisodicMemory",
            enable_inserts=True
        )
        
        self.procedural_extractor = create_extractor(
            model,
            tools=[ProceduralMemory],
            tool_choice="ProceduralMemory",
            enable_inserts=True
        )
        
        # Spy for visibility into Trustcall operations
        self.spy = MemorySpy()
    
    def create_memory_decision_tool(self):
        """Create a tool for deciding which memory type to update."""
        return self.model.bind_tools([MemoryUpdate], parallel_tool_calls=False)


class MemorySpy:
    """Spy class to track Trustcall memory operations for transparency."""
    
    def __init__(self):
        self.called_tools = []
        self.last_operation = None
    
    def __call__(self, run):
        """Collect information about tool calls made by Trustcall extractors."""
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                tool_calls = r.outputs.get("generations", [[]])[0]
                if tool_calls and len(tool_calls) > 0:
                    message = tool_calls[0].get("message", {})
                    kwargs = message.get("kwargs", {})
                    if "tool_calls" in kwargs:
                        self.called_tools.append(kwargs["tool_calls"])
    
    def extract_memory_changes(self, schema_name: str) -> str:
        """Extract and format memory changes from Trustcall operations."""
        changes = []
        
        for call_group in self.called_tools:
            for call in call_group:
                if call['name'] == 'PatchDoc':
                    changes.append({
                        'type': 'update',
                        'doc_id': call['args']['json_doc_id'],
                        'planned_edits': call['args']['planned_edits'],
                        'patches': call['args']['patches']
                    })
                elif call['name'] == schema_name:
                    changes.append({
                        'type': 'new',
                        'content': call['args']
                    })
        
        # Format results
        result_parts = []
        for change in changes:
            if change['type'] == 'update':
                patches_info = []
                for patch in change['patches']:
                    patches_info.append(f"Updated {patch['path']}: {patch['value']}")
                
                result_parts.append(
                    f"Memory {change['doc_id']} updated:\n"
                    f"Plan: {change['planned_edits']}\n"
                    f"Changes: {'; '.join(patches_info)}"
                )
            else:
                result_parts.append(
                    f"New {schema_name} created:\n"
                    f"Content: {change['content']}"
                )
        
        return "\n\n".join(result_parts) if result_parts else f"No {schema_name} changes detected"


# System prompts for memory extraction
SEMANTIC_MEMORY_INSTRUCTION = """
Analyze the conversation to extract and update semantic memory about the user's food preferences, dietary habits, and shopping behavior.

Focus on:
- Food likes and dislikes mentioned
- Dietary restrictions or allergies
- Budget sensitivity indicators  
- Preferred stores mentioned
- Household size or cooking frequency clues
- Meal type preferences

System Time: {time}
"""

EPISODIC_MEMORY_INSTRUCTION = """
Create an episodic memory entry for this interaction, capturing:
- What the user requested or searched for
- How the system responded
- Any feedback or reactions from the user
- Products that were discussed
- Signs of price sensitivity

This helps track user behavior patterns and improve future recommendations.

System Time: {time}
"""

PROCEDURAL_MEMORY_INSTRUCTION = """
Based on user feedback and interaction patterns, update the instructions for how to communicate and behave with this user.

Consider:
- Communication style preferences shown
- Budget behavior patterns
- Types of recommendations they prefer
- Level of detail they want
- Any specific instructions they've given

Current procedural memory:
{current_procedural}

System Time: {time}
"""


def detect_interaction_type(messages: List[Any]) -> str:
    """Detect the type of interaction from conversation messages."""
    if not messages:
        return "general"
    
    last_message = messages[-1].content.lower() if hasattr(messages[-1], 'content') else ""
    
    # Keywords to detect interaction types
    if any(word in last_message for word in ['search', 'find', 'looking for', 'need']):
        return "product_search"
    elif any(word in last_message for word in ['price', 'cost', 'cheap', 'expensive', 'compare']):
        return "price_comparison"
    elif any(word in last_message for word in ['meal', 'recipe', 'cook', 'dinner', 'lunch']):
        return "meal_planning"
    elif any(word in last_message for word in ['love', 'hate', 'like', 'dislike', 'good', 'bad']):
        return "feedback"
    elif any(word in last_message for word in ['list', 'shopping', 'buy', 'purchase']):
        return "shopping_list"
    else:
        return "general"


def detect_price_sensitivity(messages: List[Any]) -> Optional[str]:
    """Detect price sensitivity level from user messages."""
    if not messages:
        return None
    
    recent_content = " ".join([
        msg.content.lower() for msg in messages[-3:] 
        if hasattr(msg, 'content')
    ])
    
    # High sensitivity indicators
    if any(phrase in recent_content for phrase in [
        'too expensive', 'cheaper', 'budget', 'save money', 'deal', 'discount'
    ]):
        return "high"
    
    # Low sensitivity indicators  
    elif any(phrase in recent_content for phrase in [
        'quality', 'organic', 'premium', 'don\'t mind paying', 'worth it'
    ]):
        return "low"
    
    return "medium"


def extract_products_mentioned(messages: List[Any]) -> List[str]:
    """Extract product names or types mentioned in the conversation."""
    products = []
    
    for message in messages[-5:]:  # Look at recent messages
        if hasattr(message, 'content'):
            content = message.content.lower()
            
            # Common product keywords to look for
            product_keywords = [
                'milk', 'cheese', 'bread', 'pasta', 'rice', 'chicken', 'beef',
                'vegetables', 'fruit', 'yogurt', 'eggs', 'butter', 'oil',
                'cereal', 'coffee', 'tea', 'juice', 'water', 'wine', 'beer'
            ]
            
            for keyword in product_keywords:
                if keyword in content:
                    products.append(keyword)
    
    return list(set(products))  # Remove duplicates 