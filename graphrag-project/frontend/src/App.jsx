import { useState } from "react";
import GraphView from "./GraphView";
import ChatPanel from "./ChatPanel";

export default function App() {
  // highlightNodes stores the nodes from the last query's subgraph.
  // These get passed to GraphView to highlight relevant nodes.
  const [highlightNodes, setHighlightNodes] = useState([]);
  
  // ingesting tracks whether the document ingestion pipeline is running
  const [ingesting, setIngesting] = useState(false);
  const [ingestStatus, setIngestStatus] = useState("");

async function runIngestion() {
    setIngesting(true);
    setIngestStatus("Processing documents...");
    
    try {
      const res = await fetch("/api/ingest", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ folder_path: "documents", clear_first: true })
      });
      
      const data = await res.json();
      
      if (!res.ok) {
        throw new Error(data.detail || "Ingestion failed");
      }
      
      setIngestStatus(
        `✅ Ingested ${data.entities_count} entities, ${data.relationships_count} relationships`
      );
    } catch (err) {
      setIngestStatus(`❌ ${err.message}`);
    } finally {
      setIngesting(false);
    }
  }
// Called by ChatPanel when a query returns a subgraph.
  // We update highlightNodes so GraphView can visually highlight
  // the nodes and edges that were used to answer the question.
  function handleQueryResult(subgraph) {
    setHighlightNodes(subgraph.nodes || []);
  }

return (
    <div style={styles.app}>
      {/* Top bar */}
      <div style={styles.topBar}>
        <div style={styles.brand}>
          <span style={styles.brandIcon}>◈</span>
          <span style={styles.brandName}>GraphRAG Explorer</span>
        </div>
        
        <div style={styles.ingestSection}>
          {ingestStatus && (
            <span style={styles.ingestStatus}>{ingestStatus}</span>
          )}
          <button
            style={{
              ...styles.ingestButton,
              opacity: ingesting ? 0.6 : 1,
              cursor: ingesting ? "not-allowed" : "pointer"
            }}
            onClick={runIngestion}
            disabled={ingesting}
          >
            {ingesting ? "Ingesting..." : "⬆ Ingest Documents"}
          </button>
        </div>
      </div>
      
      {/* Main content: graph on left, chat on right */}
      <div style={styles.content}>
        <div style={styles.graphPane}>
          <GraphView highlightNodes={highlightNodes} />
        </div>
        <div style={styles.chatPane}>
          <ChatPanel onQueryResult={handleQueryResult} />
        </div>
      </div>
    </div>
  );
}
const styles = {
  app: {
    display: "flex",
    flexDirection: "column",
    height: "100vh",
    background: "#0f172a",
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"
  },
  topBar: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "12px 20px",
    borderBottom: "1px solid #334155",
    background: "#1e293b",
    flexShrink: 0
  },
  brand: {
    display: "flex",
    alignItems: "center",
    gap: 10
  },
  brandIcon: {
    fontSize: 20,
    color: "#60a5fa"
  },
  brandName: {
    fontSize: 16,
    fontWeight: 700,
    color: "#f1f5f9"
  },
  ingestSection: {
    display: "flex",
    alignItems: "center",
    gap: 12
  },
  ingestStatus: {
    fontSize: 13,
    color: "#94a3b8"
  },
  ingestButton: {
    background: "#3b82f6",
    color: "white",
    border: "none",
    borderRadius: "6px",
    padding: "8px 16px",
    fontSize: 13,
    fontWeight: 600
  },
  content: {
    display: "flex",
    flex: 1,
    overflow: "hidden",
    padding: 16,
    gap: 16
  },
  graphPane: {
    flex: 2,         // Graph takes 2/3 of the horizontal space
    minHeight: 0    // Required for flex children to respect overflow
  },
  chatPane: {
    flex: 1,         // Chat takes 1/3 of the horizontal space
    minHeight: 0
  }
};



