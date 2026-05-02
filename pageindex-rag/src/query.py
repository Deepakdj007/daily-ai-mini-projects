import os
from typing import Generator
from dotenv import load_dotenv
from pageindex import PageIndexClient

load_dotenv()

client = PageIndexClient(api_key=os.getenv("PAGEINDEX_API_KEY"))

class DocumentChat:
    """
    All-in-one query engine for a PageIndex document.

    Three ways to get an answer:
      chat.ask(question)              → returns the full answer as a string
      chat.ask_stream(question)       → streams the answer token by token
      chat.ask_stream_with_trace(q)  → streams answer + shows retrieval steps

    All three methods automatically include conversation history,
    so follow-up questions work correctly without any extra setup.
    """

    def __init__(self, doc_id: str):
        self.doc_id = doc_id
        self.history: list[dict] = []

    def ask(self, question: str) -> str:
        """Ask a question and wait for the full answer."""
        self.history.append({"role": "user", "content": question})

        response = client.chat_completions(
            messages=self.history,
            doc_id=self.doc_id,
            enable_citations=True   # adds page refs like <doc=file.pdf;page=5>
        )

        answer = response["choices"][0]["message"]["content"]

        # Save the assistant's reply so the next question has context
        self.history.append({"role": "assistant", "content": answer})
        return answer
    
    def ask_stream(self, question: str) -> Generator[str, None, None]:
        """Ask a question and stream the answer token by token."""
        self.history.append({"role": "user", "content": question})

        full_answer = ""
        for chunk in client.chat_completions(
            messages=self.history,
            doc_id=self.doc_id,
            stream=True,
            enable_citations=True
        ):
            full_answer += chunk
            yield chunk

        # Save the complete answer to history once streaming finishes
        self.history.append({"role": "assistant", "content": full_answer})

    def ask_stream_with_trace(self, question: str) -> Generator[dict, None, None]:
        """
        Stream the answer with retrieval trace.

        Yields dicts with two shapes:
          {"type": "trace",  "content": "Searching: retrieve_node"}
          {"type": "answer", "content": "Revenue grew 34%..."}

        Print trace lines differently from answer lines in your UI.
        """
        self.history.append({"role": "user", "content": question})

        full_answer = ""
        for chunk in client.chat_completions(
            messages=self.history,
            doc_id=self.doc_id,
            stream=True,
            stream_metadata=True,
            enable_citations=True
        ):
            # Check for retrieval metadata
            metadata = chunk.get("block_metadata", {})
            block_type = metadata.get("type", "")

            if block_type == "mcp_tool_use_start":
                yield {"type": "trace", "content": f"Searching: {metadata.get('tool_name', '')}"}

            elif block_type == "mcp_tool_result_start":
                yield {"type": "trace", "content": "Retrieved relevant section"}

            # Extract the answer text chunk (empty string if this was a metadata-only chunk)
            content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                full_answer += content
                yield {"type": "answer", "content": content}

        self.history.append({"role": "assistant", "content": full_answer})

    def reset(self):
        """Clear conversation history to start a new topic."""
        self.history = []