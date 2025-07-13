"""
Simplified persistence layer for BargainB Memory Agent

A simpler approach to external memory storage that focuses on core functionality.
For now, this provides in-memory storage with logging to show where Supabase 
integration would happen.
"""

import json
from typing import Any, Dict, Optional
from datetime import datetime

# In-memory storage for development/testing
# In production, this would connect to Supabase
conversation_summaries = {}
memory_store_data = {}
truncation_logs = []


def save_conversation_summary(
    conversation_id: str,
    thread_id: str,
    summary_text: str,
    message_count: int,
    tokens_used: int = 0
) -> str:
    """Save a conversation summary (in-memory for now, would be Supabase in production)."""
    print(f"📝 [DB] Saving conversation summary for {conversation_id} (thread: {thread_id})")
    print(f"    📊 Summary: {summary_text[:100]}...")
    print(f"    📊 Message count: {message_count}, Tokens: {tokens_used}")
    
    # Store in memory (would be Supabase in production)
    if conversation_id not in conversation_summaries:
        conversation_summaries[conversation_id] = []
    
    conversation_summaries[conversation_id].append({
        'summary': summary_text,
        'version': len(conversation_summaries[conversation_id]) + 1,
        'message_count': message_count,
        'tokens_used': tokens_used,
        'created_at': datetime.now().isoformat()
    })
    
    return summary_text


def get_conversation_summary(conversation_id: str) -> Optional[str]:
    """Get the latest conversation summary (in-memory for now)."""
    if conversation_id in conversation_summaries and conversation_summaries[conversation_id]:
        latest = conversation_summaries[conversation_id][-1]
        print(f"📖 [DB] Retrieved summary for {conversation_id}: {latest['summary'][:50]}...")
        return latest['summary']
    return None


def log_message_truncation(
    conversation_id: str,
    thread_id: str,
    messages_before: int,
    messages_after: int,
    messages_removed: int,
    summary_tokens: int = 0
) -> None:
    """Log message truncation event (in-memory for now)."""
    print(f"✂️ [DB] Logging message truncation for {conversation_id}")
    print(f"    📊 Before: {messages_before}, After: {messages_after}, Removed: {messages_removed}")
    
    truncation_logs.append({
        'conversation_id': conversation_id,
        'thread_id': thread_id,
        'messages_before': messages_before,
        'messages_after': messages_after,
        'messages_removed': messages_removed,
        'summary_tokens': summary_tokens,
        'timestamp': datetime.now().isoformat()
    })


def save_memory_data(namespace: str, key: str, value: Any) -> None:
    """Save memory data (in-memory for now)."""
    print(f"💾 [DB] Saving memory data: {namespace}/{key}")
    
    if namespace not in memory_store_data:
        memory_store_data[namespace] = {}
    
    memory_store_data[namespace][key] = {
        'value': value,
        'updated_at': datetime.now().isoformat()
    }


def get_memory_data(namespace: str, key: str, default: Any = None) -> Any:
    """Get memory data (in-memory for now)."""
    if namespace in memory_store_data and key in memory_store_data[namespace]:
        value = memory_store_data[namespace][key]['value']
        print(f"📖 [DB] Retrieved memory data: {namespace}/{key}")
        return value
    return default


def list_memory_keys(namespace: str, prefix: str = "") -> list:
    """List all keys in a namespace with optional prefix filter."""
    if namespace not in memory_store_data:
        return []
    
    keys = list(memory_store_data[namespace].keys())
    
    if prefix:
        keys = [k for k in keys if k.startswith(prefix)]
    
    print(f"📋 [DB] Listed {len(keys)} keys in {namespace} (prefix: {prefix})")
    return keys


class SimpleMemoryStore:
    """
    Simplified memory store that uses Supabase for persistence.
    Compatible with LangGraph's store interface but simplified.
    """
    
    def __init__(self, namespace: str = "default"):
        """Initialize the memory store with a namespace."""
        self.namespace = namespace
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key."""
        return get_memory_data(self.namespace, key, default)
    
    def put(self, key: str, value: Any) -> None:
        """Store a value by key."""
        save_memory_data(self.namespace, key, value)
    
    def delete(self, key: str) -> None:
        """Delete a value by key."""
        try:
            query = "DELETE FROM memory_store WHERE namespace = %s AND key = %s"
            db.execute(query, (self.namespace, key))
        except Exception as e:
            print(f"Error deleting memory data: {e}")
    
    def search(self, namespace_tuple, limit: int = 10) -> list:
        """Search for memories in a namespace (simplified version)."""
        # Convert tuple namespace to string
        namespace_str = f"{namespace_tuple[0]}_{namespace_tuple[1]}" if isinstance(namespace_tuple, tuple) else str(namespace_tuple)
        
        print(f"🔍 [DB] Searching memories in {namespace_str} (limit: {limit})")
        
        if namespace_str not in memory_store_data:
            return []
        
        # Return in format expected by memory nodes
        results = []
        items = list(memory_store_data[namespace_str].items())[:limit]
        
        for key, data in items:
            results.append(type('Memory', (), {
                'key': key,
                'value': data['value']
            }))
        
        print(f"📋 [DB] Found {len(results)} memories in {namespace_str}")
        return results 