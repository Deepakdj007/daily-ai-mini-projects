# src/memory.py
# ─────────────────────────────────────────────────────────────
# Thin wrapper around mem0's Memory class.
# Exposes three operations used by the agent:
#   - add_memory: store facts from a conversation turn
#   - search_memory: retrieve relevant facts for a query
#   - list_memories: inspect everything stored for a user
# ─────────────────────────────────────────────────────────────

from mem0 import Memory
from src.config import get_config

def create_memory() -> Memory:
    """
    Initialize and return a configured mem0 Memory instance.

    On first call, this downloads the HuggingFace embedding model
    (~90MB) and creates the Qdrant collection at /tmp/qdrant.
    Subsequent calls reuse the existing collection.

    Returns:
        Memory: A fully configured mem0 Memory instance.
    """
    config = get_config()
    return Memory.from_config(config)

def add_memory(memory: Memory, messages: list[dict], user_id: str) -> None:
    """
    Extract facts from a conversation turn and store them.

    mem0 reads 'messages' (same format as OpenAI chat messages),
    calls the LLM to extract meaningful facts, and saves them
    to Qdrant under the given user_id.

    Args:
        memory: The initialized Memory instance.
        messages: List of dicts with 'role' and 'content' keys.
                  Example: [{"role": "user", "content": "I love hiking"}]
        user_id: Unique identifier for this user's memory space.
                 Memories are isolated per user_id.
    """
    memory.add(messages, user_id=user_id)

def search_memory(memory: Memory, query: str, user_id: str) -> list[dict]:
    """
    Retrieve the most relevant stored memories for a query.

    mem0 embeds the query, searches the Qdrant collection for
    semantically similar memories, and returns the top results
    scoped to the given user_id.

    Args:
        memory: The initialized Memory instance.
        query: The current user message to search against.
        user_id: Only return memories belonging to this user.

    Returns:
        List of memory dicts. Each dict has a 'memory' key
        containing the stored fact as a plain string.
        Example: [{"memory": "User is vegetarian", "score": 0.91}]
    """
    results = memory.search(query, filters={"user_id": user_id})
    return results.get("results", [])

def list_memories(memory: Memory, user_id: str) -> list[dict]:
    """
    Return all stored memories for a user (for debugging/inspection).

    Useful for understanding what the agent has learned about
    a user without needing to query for a specific topic.

    Args:
        memory: The initialized Memory instance.
        user_id: The user whose memories to retrieve.

    Returns:
        List of all memory dicts for this user.
    """
    results = memory.get_all(filters={"user_id": user_id})
    return results.get("results", [])