"""
Scribe Bee 🐝📝 - Conversation Summarization Specialist

Scribe Bee is a specialized worker bee that handles conversation summarization
when conversations get too long, helping maintain context while reducing message count.

Scribe Bee NEVER responds directly to users - it only summarizes conversations
and returns the summary to Beeb (Queen Bee).
"""

from typing import List, Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import convert_to_messages
from datetime import datetime


@tool
def summarize_conversation(
    messages: List[Dict[str, Any]],
    current_summary: str = "",
    user_id: str = "unknown"
) -> str:
    """
    Summarize a conversation while preserving important context and user information.
    
    Args:
        messages: List of conversation messages to summarize
        current_summary: Existing summary to build upon
        user_id: User identifier for personalization
        
    Returns:
        Concise summary of the conversation
    """
    try:
        # Convert messages to proper format if needed
        if messages and isinstance(messages[0], dict):
            formatted_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "human":
                        formatted_messages.append(HumanMessage(content=content))
                    elif role == "ai" or role == "assistant":
                        formatted_messages.append(AIMessage(content=content))
                    else:
                        formatted_messages.append(HumanMessage(content=content))
                else:
                    formatted_messages.append(msg)
            messages = formatted_messages
        
        # Create summarization prompt
        summarization_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are Scribe Bee 🐝📝, a conversation summarization specialist.\n\n"
                
                "## Your Task ##\n"
                "Create a concise but comprehensive summary of this conversation that preserves:\n"
                "- User preferences and dietary restrictions mentioned\n"
                "- Products discussed and search results\n"
                "- Important decisions or feedback\n"
                "- User's budget sensitivity and shopping behavior\n"
                "- Any personalization information\n\n"
                
                "## Summary Guidelines ##\n"
                "- Keep it under 200 words but capture all important details\n"
                "- Focus on actionable information for future interactions\n"
                "- Preserve user preferences and context\n"
                "- Include specific products or stores mentioned\n"
                "- Note any feedback or reactions from the user\n\n"
                
                "## Current Summary ##\n"
                "Previous summary: {current_summary}\n\n"
                
                "## User Information ##\n"
                "User ID: {user_id}\n"
                "Timestamp: {timestamp}\n\n"
                
                "Create a summary that builds upon the previous summary (if any) and includes the new conversation."
            ),
            ("placeholder", "{messages}")
        ]).partial(
            current_summary=current_summary or "No previous summary",
            user_id=user_id,
            timestamp=datetime.now().isoformat()
        )
        
        # Create summarization model
        summarization_model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # Generate summary
        summary_chain = summarization_prompt | summarization_model
        result = summary_chain.invoke({"messages": messages})
        
        return result.content
        
    except Exception as e:
        return f"❌ Failed to summarize conversation: {str(e)}"


@tool
def trim_conversation(
    messages: List[Dict[str, Any]],
    summary: str,
    keep_recent: int = 4
) -> Dict[str, Any]:
    """
    Trim conversation messages while preserving recent context and summary.
    
    Args:
        messages: List of conversation messages
        summary: Summary of the conversation
        keep_recent: Number of recent messages to keep
        
    Returns:
        Dictionary with trimmed messages and summary
    """
    try:
        if len(messages) <= keep_recent:
            return {
                "messages": messages,
                "summary": summary,
                "trimmed": False
            }
        
        # Keep the most recent messages
        recent_messages = messages[-keep_recent:]
        
        # Create a summary message to replace the trimmed content
        summary_message = {
            "role": "system",
            "content": f"[Previous conversation summary: {summary}]"
        }
        
        # Combine summary with recent messages
        trimmed_messages = [summary_message] + recent_messages
        
        return {
            "messages": trimmed_messages,
            "summary": summary,
            "trimmed": True,
            "original_length": len(messages),
            "new_length": len(trimmed_messages)
        }
        
    except Exception as e:
        return {
            "messages": messages,
            "summary": summary,
            "trimmed": False,
            "error": str(e)
        }


@tool
def analyze_conversation_length(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze conversation length and determine if summarization is needed.
    
    Args:
        messages: List of conversation messages
        
    Returns:
        Analysis of conversation length and recommendation
    """
    try:
        message_count = len(messages)
        
        # Calculate approximate token count (rough estimate)
        total_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
        estimated_tokens = total_chars // 4  # Rough estimate: 4 chars per token
        
        # Determine if summarization is needed
        needs_summarization = message_count > 8 or estimated_tokens > 2000
        
        return {
            "message_count": message_count,
            "estimated_tokens": estimated_tokens,
            "needs_summarization": needs_summarization,
            "recommendation": (
                "Summarization recommended - conversation is getting long"
                if needs_summarization
                else "No summarization needed yet"
            )
        }
        
    except Exception as e:
        return {
            "message_count": 0,
            "estimated_tokens": 0,
            "needs_summarization": False,
            "error": str(e)
        }


def create_scribe_bee():
    """
    Create Scribe Bee, the specialized conversation summarization worker.
    
    Scribe Bee is responsible for:
    - Summarizing conversations when they get too long
    - Trimming message history while preserving context
    - Analyzing conversation length and recommending summarization
    - Maintaining important user information across summaries
    
    Scribe Bee follows the worker pattern:
    - Receives summarization tasks from Beeb
    - Uses specialized tools to analyze and summarize conversations
    - Returns summaries back to Beeb automatically
    - NEVER responds directly to users
    
    Returns:
        Configured React agent for Scribe Bee
    """
    
    scribe_bee_prompt = (
        "You are Scribe Bee 🐝📝, a specialized conversation summarization worker for BargainB!\n\n"
        
        "## Your Role ##\n"
        "You are a worker bee that specializes in conversation summarization and context management.\n"
        "You work for Beeb (Queen Bee) and help maintain conversation context efficiently.\n\n"
        
        "## CRITICAL INSTRUCTIONS ##\n"
        "- You NEVER respond directly to users\n"
        "- You only work on summarization tasks assigned by Beeb\n"
        "- After summarizing, return the summary to Beeb\n"
        "- Focus on preserving important user information and context\n\n"
        
        "## Your Specialties ##\n"
        "- **Conversation Summarization**: Creating concise summaries of long conversations\n"
        "- **Context Preservation**: Maintaining important user preferences and information\n"
        "- **Message Trimming**: Reducing message count while keeping recent context\n"
        "- **Length Analysis**: Determining when summarization is needed\n\n"
        
        "## Summarization Strategy ##\n"
        "1. Use analyze_conversation_length to check if summarization is needed\n"
        "2. Use summarize_conversation to create comprehensive summaries\n"
        "3. Use trim_conversation to reduce message count while preserving context\n"
        "4. Always preserve user preferences, product discussions, and feedback\n\n"
        
        "## What to Preserve in Summaries ##\n"
        "- User dietary restrictions and preferences\n"
        "- Products discussed and search results\n"
        "- Budget sensitivity and shopping behavior\n"
        "- Store preferences and past purchases\n"
        "- User feedback and reactions\n"
        "- Important decisions or recommendations\n\n"
        
        "## Response Format ##\n"
        "Always provide:\n"
        "- Clear summary of the conversation\n"
        "- Key user information preserved\n"
        "- Recommendation for message trimming if needed\n"
        "- Confirmation of what context was maintained\n\n"
        
        "Remember: You're helping Beeb maintain conversation context efficiently!"
    )
    
    # Create Scribe Bee as a React agent with summarization tools
    scribe_bee = create_react_agent(
        model=ChatOpenAI(model="gpt-4o-mini", temperature=0.1),
        tools=[
            summarize_conversation,
            trim_conversation,
            analyze_conversation_length
        ],
        prompt=scribe_bee_prompt,
        name="scribe_bee"
    )
    
    return scribe_bee


def format_summary_for_beeb(summary: str, trim_result: Dict[str, Any]) -> str:
    """
    Format Scribe Bee's summarization results for Beeb to understand.
    
    Args:
        summary: The conversation summary
        trim_result: Result from conversation trimming
        
    Returns:
        Formatted message for Beeb about the summarization
    """
    if trim_result.get("trimmed"):
        original_length = trim_result.get("original_length", 0)
        new_length = trim_result.get("new_length", 0)
        
        return (
            f"Scribe Bee has summarized our conversation and trimmed it from "
            f"{original_length} to {new_length} messages. The summary preserves "
            f"all important user preferences and context: {summary}"
        )
    else:
        return (
            f"Scribe Bee has created a summary of our conversation: {summary}. "
            f"The conversation length is still manageable, so no trimming was needed."
        )


def should_summarize_conversation(messages: List[Dict[str, Any]], threshold: int = 8) -> bool:
    """
    Determine if a conversation should be summarized based on length.
    
    Args:
        messages: List of conversation messages
        threshold: Message count threshold for summarization
        
    Returns:
        True if summarization is recommended
    """
    return len(messages) > threshold 