import { useState, useRef, useEffect } from "react";

export default function ChatPanel({ onQueryResult }) {
  // messages is the chat history array: [{role: "user"|"assistant", content: "..."}]
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Knowledge graph loaded. Ask me anything about the documents you ingested."
    }
  ]);
  
  // input is the current value of the text field
  const [input, setInput] = useState("");
  
  // loading is true while waiting for the API response
  const [loading, setLoading] = useState(false);
  
  // messagesEndRef is attached to a div at the bottom of the chat list.
  // We call scrollIntoView() on it to auto-scroll to new messages.
  const messagesEndRef = useRef(null);

// Auto-scroll to the bottom whenever a new message appears
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

async function sendMessage() {
    const question = input.trim();
    if (!question || loading) return;
    
    // Add the user's message to the chat immediately (optimistic update)
    setMessages(prev => [...prev, { role: "user", content: question }]);
    setInput("");
    setLoading(true);
    
    try {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question })
      });
      
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Query failed");
      }
      
      const data = await res.json();
      
      // Add the assistant's answer to the chat
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: data.answer }
      ]);
      
      // Pass the subgraph back to the parent (App.jsx)
      // so it can highlight the relevant nodes in GraphView
      if (onQueryResult && data.subgraph) {
        onQueryResult(data.subgraph);
      }
      
    } catch (err) {
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: `Error: ${err.message}` }
      ]);
    } finally {
      setLoading(false);
    }
  }

// Allow sending with Enter key (Shift+Enter for newline)
  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();  // Prevent default newline insertion
      sendMessage();
    }
  }

return (
    <div style={styles.panel}>
      <div style={styles.header}>
        <h2 style={styles.title}>Chat with Your Graph</h2>
      </div>
      
      {/* Message list */}
      <div style={styles.messageList}>
        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              ...styles.message,
              ...(msg.role === "user" ? styles.userMessage : styles.assistantMessage)
            }}
          >
            <div style={styles.roleBadge}>
              {msg.role === "user" ? "You" : "Graph"}
            </div>
            <p style={styles.messageContent}>{msg.content}</p>
          </div>
        ))}
        
        {/* Loading indicator */}
        {loading && (
          <div style={{ ...styles.message, ...styles.assistantMessage }}>
            <div style={styles.roleBadge}>Graph</div>
            <p style={{ ...styles.messageContent, color: "#94a3b8" }}>
              Searching knowledge graph...
            </p>
          </div>
        )}
        
        {/* Invisible div at the bottom — we scroll to this */}
        <div ref={messagesEndRef} />
      </div>
      
      {/* Input row */}
      <div style={styles.inputRow}>
        <textarea
          style={styles.input}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a question about your documents..."
          rows={2}
          disabled={loading}
        />
        <button
          style={{
            ...styles.sendButton,
            opacity: loading || !input.trim() ? 0.5 : 1,
            cursor: loading || !input.trim() ? "not-allowed" : "pointer"
          }}
          onClick={sendMessage}
          disabled={loading || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}

const styles = {
  panel: {
    display: "flex",
    flexDirection: "column",
    height: "100%",
    background: "#1e293b",
    borderRadius: "8px",
    overflow: "hidden"
  },
  header: {
    padding: "16px 20px",
    borderBottom: "1px solid #334155"
  },
  title: {
    margin: 0,
    fontSize: 16,
    fontWeight: 600,
    color: "#f1f5f9"
  },
  messageList: {
    flex: 1,
    overflowY: "auto",
    padding: "16px",
    display: "flex",
    flexDirection: "column",
    gap: 12
  },
  message: {
    borderRadius: "8px",
    padding: "12px 14px",
    maxWidth: "90%"
  },
  userMessage: {
    background: "#3b82f6",
    alignSelf: "flex-end"
  },
  assistantMessage: {
    background: "#0f172a",
    border: "1px solid #334155",
    alignSelf: "flex-start"
  },
  roleBadge: {
    fontSize: 11,
    fontWeight: 600,
    color: "#94a3b8",
    marginBottom: 4,
    textTransform: "uppercase",
    letterSpacing: "0.05em"
  },
  messageContent: {
    margin: 0,
    color: "#e2e8f0",
    fontSize: 14,
    lineHeight: 1.6,
    whiteSpace: "pre-wrap"  // Preserve line breaks in LLM responses
  },
  inputRow: {
    display: "flex",
    gap: 8,
    padding: "12px 16px",
    borderTop: "1px solid #334155"
  },
  input: {
    flex: 1,
    background: "#0f172a",
    border: "1px solid #334155",
    borderRadius: "6px",
    color: "#f1f5f9",
    padding: "10px 12px",
    fontSize: 14,
    resize: "none",
    outline: "none",
    fontFamily: "inherit"
  },
  sendButton: {
    background: "#3b82f6",
    color: "white",
    border: "none",
    borderRadius: "6px",
    padding: "0 20px",
    fontSize: 14,
    fontWeight: 600,
    alignSelf: "flex-end",
    height: 40
  }
};

