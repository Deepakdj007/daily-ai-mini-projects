# src/agent.py
# ─────────────────────────────────────────────────────────────
# The memory-aware chat agent.
# On every turn: retrieves relevant memories → builds prompt →
# calls Groq → stores the new turn back to mem0.
# ─────────────────────────────────────────────────────────────

import os
import threading
from groq import Groq
from mem0 import Memory
from src.memory import add_memory, search_memory, list_memories

def build_system_prompt(relevant_memories: list[dict]) -> str:
    """
    Build the system prompt by injecting retrieved memories.

    If the user has no stored memories yet, returns a generic
    assistant prompt. Otherwise, prepends a 'What you know
    about this user' section so the LLM has personal context.

    Args:
        relevant_memories: List of memory dicts from search_memory().

    Returns:
        str: A complete system prompt string.
    """
    base_prompt = (
        "You are a helpful personal assistant. "
        "You remember things about the user across conversations. "
        "Be warm, concise, and use what you know to personalize your answers."
    )

    if not relevant_memories:
        return base_prompt

    # Format each memory as a bullet point
    memory_lines = "\n".join(
        f"- {m['memory']}" for m in relevant_memories
    )

    return (
        f"{base_prompt}\n\n"
        f"What you know about this user:\n{memory_lines}"
    )

def call_llm(client: Groq, system_prompt: str, conversation: list[dict]) -> str:
    """
    Send the conversation to Groq and return the reply text.

    Args:
        client: An initialized Groq client.
        system_prompt: The system prompt (with injected memories).
        conversation: Full conversation history for this session.
                      List of role/content dicts.

    Returns:
        str: The assistant's reply text.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        *conversation,
    ]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,    # Higher temp for replies — we want natural conversation
        max_tokens=1024,
    )

    return response.choices[0].message.content or ""

def run_agent(memory: Memory, user_id: str) -> None:
    """
    Run the interactive chat loop for a given user.

    On each turn:
    1. Reads user input
    2. Searches mem0 for relevant memories
    3. Builds a personalized system prompt
    4. Calls Groq to generate a reply
    5. Stores the turn back to mem0

    Special commands:
    - 'quit' or 'exit' — end the session
    - 'memories' — show everything stored for this user

    Args:
        memory: The initialized Memory instance.
        user_id: The user identifier for this session.
    """
    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
    conversation_history = []   # In-session turn history

    print(f"\n🧠 Memory Agent ready — chatting as '{user_id}'")
    print("Type 'memories' to see what I remember. Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print("Goodbye! Your memories are saved for next time. 👋")
            break

        # Show stored memories on request
        if user_input.lower() == "memories":
            all_memories = list_memories(memory, user_id)
            if not all_memories:
                print("🧠 No memories stored yet.\n")
            else:
                print(f"\n🧠 What I remember about you ({len(all_memories)} facts):")
                for m in all_memories:
                    print(f"  - {m['memory']}")
                print()
            continue

        # Step 1: Search for relevant memories
        relevant = search_memory(memory, user_input, user_id)

        # Step 2: Build the personalized system prompt
        system_prompt = build_system_prompt(relevant)

        # Step 3: Add this turn to in-session history
        conversation_history.append({"role": "user", "content": user_input})

        # Step 4: Call Groq for a reply
        reply = call_llm(groq_client, system_prompt, conversation_history)
        print(f"\nAssistant: {reply}\n")

        # Step 5: Track the reply in session history
        conversation_history.append({"role": "assistant", "content": reply})

        # Step 6: Store this turn's facts to mem0
        turn = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": reply},
        ]
        threading.Thread(target=add_memory, args=(memory, turn, user_id), daemon=True).start()