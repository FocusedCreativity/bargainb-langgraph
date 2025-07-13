"""
Supabase-based persistence layer for LangGraph

Custom implementation of checkpointer and memory store using Supabase
to replace SQLite and InMemoryStore from the original notebook example.
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Iterator
from datetime import datetime

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple, Checkpoint
from langgraph.store.base import BaseStore
import asyncio
from ..utils.database import Database

# Create a shared database instance
db = Database()


class SupabaseCheckpointSaver(BaseCheckpointSaver):
    """
    Custom checkpointer that stores conversation state in Supabase
    instead of SQLite, following the same patterns as the notebook.
    """
    
    def __init__(self):
        """Initialize the Supabase checkpointer."""
        pass
    
    def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Get a checkpoint tuple by thread_id and checkpoint_id."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_id = config["configurable"].get("checkpoint_id")
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        
        if checkpoint_id:
            # Get specific checkpoint
            query = """
                SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, created_at
                FROM langgraph_checkpoints 
                WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
            """
            rows = db.query(query, (thread_id, checkpoint_ns, checkpoint_id))
            row = rows[0] if rows else None
        else:
            # Get latest checkpoint
            query = """
                SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, created_at
                FROM langgraph_checkpoints 
                WHERE thread_id = %s AND checkpoint_ns = %s 
                ORDER BY created_at DESC 
                LIMIT 1
            """
            rows = db.query(query, (thread_id, checkpoint_ns))
            row = rows[0] if rows else None
        
        if not row:
            return None
            
        # Get associated writes
        writes_query = """
            SELECT task_id, idx, channel, type, value
            FROM langgraph_writes
            WHERE thread_id = %s AND checkpoint_ns = %s AND checkpoint_id = %s
            ORDER BY idx
        """
        writes_rows = db.query(writes_query, (thread_id, checkpoint_ns, row['checkpoint_id']))
        
        writes = []
        for write_row in writes_rows:
            writes.append((
                write_row['task_id'],
                write_row['channel'],
                write_row['value']
            ))
        
        config_with_checkpoint = {
            **config,
            "configurable": {
                **config["configurable"],
                "checkpoint_id": row['checkpoint_id']
            }
        }
        
        return CheckpointTuple(
            config=config_with_checkpoint,
            checkpoint=row['checkpoint'],
            metadata=row['metadata'],
            pending_writes=writes,
            parent_config={
                **config,
                "configurable": {
                    **config["configurable"],
                    "checkpoint_id": row['parent_checkpoint_id']
                }
            } if row['parent_checkpoint_id'] else None
        )
    
    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Sync version - not implemented as we use async."""
        raise NotImplementedError("Use aget_tuple instead")
    
    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: Dict[str, Any],
        new_versions: Dict[str, Any] = None
    ) -> RunnableConfig:
        """Save a checkpoint and return updated config."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        parent_checkpoint_id = config["configurable"].get("checkpoint_id")
        
        # Generate new checkpoint ID
        checkpoint_id = str(uuid.uuid4())
        
        conn = get_database_connection()
        
        # Insert checkpoint
        checkpoint_query = """
            INSERT INTO langgraph_checkpoints 
            (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, checkpoint, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
        """
        await conn.execute(
            checkpoint_query,
            thread_id,
            checkpoint_ns, 
            checkpoint_id,
            parent_checkpoint_id,
            json.dumps(checkpoint),
            json.dumps(metadata)
        )
        
        # Return updated config
        return {
            **config,
            "configurable": {
                **config["configurable"],
                "checkpoint_id": checkpoint_id
            }
        }
    
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: Dict[str, Any],
        new_versions: Dict[str, Any] = None
    ) -> RunnableConfig:
        """Sync version - not implemented."""
        raise NotImplementedError("Use aput instead")
    
    async def alist(
        self,
        config: RunnableConfig,
        *,
        filter: Dict[str, Any] = None,
        before: RunnableConfig = None,
        limit: int = 10
    ) -> Iterator[CheckpointTuple]:
        """List checkpoints for a thread."""
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        
        conn = get_database_connection()
        
        query = """
            SELECT checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata, created_at
            FROM langgraph_checkpoints 
            WHERE thread_id = $1 AND checkpoint_ns = $2 
            ORDER BY created_at DESC 
            LIMIT $3
        """
        
        rows = await conn.fetch(query, thread_id, checkpoint_ns, limit)
        
        for row in rows:
            config_with_checkpoint = {
                **config,
                "configurable": {
                    **config["configurable"],
                    "checkpoint_id": row['checkpoint_id']
                }
            }
            
            yield CheckpointTuple(
                config=config_with_checkpoint,
                checkpoint=row['checkpoint'],
                metadata=row['metadata'],
                pending_writes=[],  # Can be loaded separately if needed
                parent_config={
                    **config,
                    "configurable": {
                        **config["configurable"],
                        "checkpoint_id": row['parent_checkpoint_id']
                    }
                } if row['parent_checkpoint_id'] else None
            )
    
    def list(
        self,
        config: RunnableConfig,
        *,
        filter: Dict[str, Any] = None,
        before: RunnableConfig = None,
        limit: int = 10
    ) -> Iterator[CheckpointTuple]:
        """Sync version - not implemented."""
        raise NotImplementedError("Use alist instead")


class SupabaseMemoryStore(BaseStore):
    """
    Custom memory store that uses Supabase instead of InMemoryStore
    for persistent memory across conversations.
    """
    
    def __init__(self, namespace: str = "default"):
        """Initialize the Supabase memory store."""
        self.namespace = namespace
    
    async def aget(self, key: str, default: Any = None) -> Any:
        """Get a value by key."""
        conn = get_database_connection()
        
        query = "SELECT value FROM memory_store WHERE namespace = $1 AND key = $2"
        row = await conn.fetchrow(query, self.namespace, key)
        
        if row:
            return row['value']
        return default
    
    def get(self, key: str, default: Any = None) -> Any:
        """Sync version - not implemented."""
        raise NotImplementedError("Use aget instead")
    
    async def aput(self, key: str, value: Any) -> None:
        """Store a value by key."""
        conn = get_database_connection()
        
        query = """
            INSERT INTO memory_store (namespace, key, value)
            VALUES ($1, $2, $3)
            ON CONFLICT (namespace, key) 
            DO UPDATE SET value = $3, updated_at = NOW()
        """
        await conn.execute(query, self.namespace, key, json.dumps(value))
    
    def put(self, key: str, value: Any) -> None:
        """Sync version - not implemented."""
        raise NotImplementedError("Use aput instead")
    
    async adelete(self, key: str) -> None:
        """Delete a value by key."""
        conn = get_database_connection()
        
        query = "DELETE FROM memory_store WHERE namespace = $1 AND key = $2"
        await conn.execute(query, self.namespace, key)
    
    def delete(self, key: str) -> None:
        """Sync version - not implemented."""
        raise NotImplementedError("Use adelete instead")
    
    async alist_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix filter."""
        conn = get_database_connection()
        
        if prefix:
            query = "SELECT key FROM memory_store WHERE namespace = $1 AND key LIKE $2"
            rows = await conn.fetch(query, self.namespace, f"{prefix}%")
        else:
            query = "SELECT key FROM memory_store WHERE namespace = $1"
            rows = await conn.fetch(query, self.namespace)
        
        return [row['key'] for row in rows]
    
    def list_keys(self, prefix: str = "") -> List[str]:
        """Sync version - not implemented."""
        raise NotImplementedError("Use alist_keys instead")


async def create_conversation_summary(
    conversation_id: str,
    thread_id: str,
    summary_text: str,
    message_count: int,
    tokens_used: int = 0
) -> str:
    """Create or update a conversation summary in the database."""
    conn = get_database_connection()
    
    # Check if summary already exists
    check_query = """
        SELECT id, summary_version FROM conversation_summaries 
        WHERE conversation_id = $1 
        ORDER BY summary_version DESC 
        LIMIT 1
    """
    existing = await conn.fetchrow(check_query, conversation_id)
    
    if existing:
        # Update existing summary
        new_version = existing['summary_version'] + 1
        update_query = """
            INSERT INTO conversation_summaries 
            (conversation_id, summary, summary_version, message_count_at_summary, tokens_used)
            VALUES ($1, $2, $3, $4, $5)
        """
        await conn.execute(
            update_query,
            conversation_id,
            summary_text,
            new_version,
            message_count,
            tokens_used
        )
    else:
        # Create new summary
        insert_query = """
            INSERT INTO conversation_summaries 
            (conversation_id, summary, summary_version, message_count_at_summary, tokens_used)
            VALUES ($1, $2, 1, $3, $4)
        """
        await conn.execute(
            insert_query,
            conversation_id,
            summary_text,
            message_count,
            tokens_used
        )
    
    return summary_text


async def get_latest_conversation_summary(conversation_id: str) -> Optional[str]:
    """Get the latest conversation summary from the database."""
    conn = get_database_connection()
    
    query = """
        SELECT summary FROM conversation_summaries 
        WHERE conversation_id = $1 
        ORDER BY summary_version DESC 
        LIMIT 1
    """
    row = await conn.fetchrow(query, conversation_id)
    
    return row['summary'] if row else None


async def log_message_truncation(
    conversation_id: str,
    thread_id: str,
    messages_before: int,
    messages_after: int,
    messages_removed: int,
    summary_tokens: int = 0
) -> None:
    """Log message truncation event."""
    conn = get_database_connection()
    
    query = """
        INSERT INTO message_truncation_log 
        (conversation_id, thread_id, messages_before_truncation, 
         messages_after_truncation, messages_removed, summary_tokens_used)
        VALUES ($1, $2, $3, $4, $5, $6)
    """
    await conn.execute(
        query,
        conversation_id,
        thread_id,
        messages_before,
        messages_after,
        messages_removed,
        summary_tokens
    ) 