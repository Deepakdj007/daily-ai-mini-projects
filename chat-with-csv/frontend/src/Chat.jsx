// frontend/src/Chat.jsx

import { useState, useRef, useEffect } from "react";

// Props:
//   sessionId — the UUID returned from /upload, identifies the CSV in memory
export default function Chat({ sessionId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  const bottomRef = useRef(null);

  // Auto-scroll to the bottom whenever a new message or chunk arrives
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage() {
    if (!input.trim() || streaming) return;

    const userText = input.trim();
    setInput("");

    // Add the user's message to the list immediately (feels instant)
    setMessages((prev) => [...prev, { role: "user", content: userText }]);

    // Add an empty assistant placeholder that we fill in as tokens stream in
    setMessages((prev) => [...prev, { role: "assistant", content: "" }]);
    setStreaming(true);

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: userText }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Chat request failed.");
      }

      // res.body is a ReadableStream.
      // The backend sends chunks as Gemini generates them.
      // getReader() lets us pull each Uint8Array chunk one by one.
      // TextDecoder converts each Uint8Array into a UTF-8 string.
      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        // Append the chunk to the last message (the assistant placeholder).
        // We spread prev into a new array so React detects the state change.
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          updated[updated.length - 1] = {
            ...last,
            content: last.content + chunk,
          };
          return updated;
        });
      }
    } catch (err) {
      // Replace the empty placeholder with the error text
      setMessages((prev) => {
        const updated = [...prev];
        updated[updated.length - 1] = {
          role: "assistant",
          content: `Error: ${err.message}`,
        };
        return updated;
      });
    } finally {
      setStreaming(false);
    }
  }

  // Enter sends; Shift+Enter inserts a newline
  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  return (
    <div>
      {/* Message list */}
      <div
        style={{
          height: "420px",
          overflowY: "auto",
          border: "1px solid #ddd",
          borderRadius: "8px",
          padding: "1rem",
          marginBottom: "1rem",
          display: "flex",
          flexDirection: "column",
          gap: "0.75rem",
          backgroundColor: "#fafafa",
        }}
      >
        {messages.length === 0 && (
          <p style={{ color: "#aaa", fontStyle: "italic" }}>
            Ask anything about your CSV data...
          </p>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
              background: msg.role === "user" ? "#1a73e8" : "#e8f0fe",
              color: msg.role === "user" ? "white" : "#1a1a1a",
              borderRadius: "12px",
              padding: "0.6rem 1rem",
              maxWidth: "78%",
              whiteSpace: "pre-wrap",
              lineHeight: "1.5",
            }}
          >
            {msg.content}
            {/* Blinking cursor on the last assistant message while streaming */}
            {streaming &&
              i === messages.length - 1 &&
              msg.role === "assistant" && (
                <span style={{ animation: "blink 1s step-end infinite" }}>
                  ▌
                </span>
              )}
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* Input row */}
      <div style={{ display: "flex", gap: "0.5rem" }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your data... (Enter to send, Shift+Enter for newline)"
          rows={2}
          disabled={streaming}
          style={{
            flex: 1,
            padding: "0.6rem 0.75rem",
            borderRadius: "6px",
            border: "1px solid #ddd",
            fontFamily: "inherit",
            fontSize: "14px",
            resize: "none",
          }}
        />
        <button
          onClick={sendMessage}
          disabled={streaming || !input.trim()}
          style={{
            padding: "0 1.25rem",
            borderRadius: "6px",
            border: "none",
            background: streaming || !input.trim() ? "#ccc" : "#1a73e8",
            color: "white",
            cursor: streaming || !input.trim() ? "not-allowed" : "pointer",
            fontWeight: 500,
            fontSize: "14px",
          }}
        >
          {streaming ? "..." : "Send"}
        </button>
      </div>

      <style>{`
        @keyframes blink {
          50% { opacity: 0; }
        }
      `}</style>
    </div>
  );
}