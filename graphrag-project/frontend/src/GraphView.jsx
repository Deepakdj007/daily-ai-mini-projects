import { useEffect, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";

// Color mapping for different entity types
// This makes the graph visually meaningful — you can see at a glance
// which nodes are people, organizations, products, etc.
const TYPE_COLORS = {
  Person: "#4ade80",        // Green
  Organization: "#60a5fa",  // Blue
  Product: "#f97316",       // Orange
  Concept: "#c084fc",       // Purple
  Place: "#fbbf24",         // Yellow
  Event: "#f87171",         // Red
  Unknown: "#94a3b8",       // Gray
};

export default function GraphView({ highlightNodes = [] }) {
  // graphData holds the nodes and links fetched from the backend
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  
  // selectedNode is set when the user clicks a node — shows info panel
  const [selectedNode, setSelectedNode] = useState(null);
  
  // loading/error state for the fetch
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // containerRef lets us read the actual width of the container div
  // so the canvas fills it exactly
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

// Fetch the full graph when the component mounts
  useEffect(() => {
    fetchGraph();
  }, []);

// Measure the container size and update when the window resizes
  // This makes the graph canvas responsive
  useEffect(() => {
    if (!containerRef.current) return;
    
    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        setDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height
        });
      }
    });
    
    observer.observe(containerRef.current);
    return () => observer.disconnect(); // Clean up when component unmounts
  }, []);

async function fetchGraph() {
    try {
      setLoading(true);
      setError(null);
      const res = await fetch("/api/graph");
      
      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }
      
      const data = await res.json();
      setGraphData(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

// Compute a set of highlighted node IDs for O(1) lookup
  // highlightNodes is passed in from the parent when a query returns a subgraph
  const highlightSet = new Set(highlightNodes.map(n => n.name));

if (loading) {
    return (
      <div style={styles.loadingContainer}>
        <p style={styles.loadingText}>Loading knowledge graph...</p>
      </div>
    );
  }
  
  if (error) {
    return (
      <div style={styles.errorContainer}>
        <p style={styles.errorText}>Error: {error}</p>
        <button onClick={fetchGraph} style={styles.retryButton}>Retry</button>
      </div>
    );
  }

return (
    <div style={styles.wrapper}>
      {/* Stats bar */}
      <div style={styles.statsBar}>
        <span>{graphData.nodes.length} nodes</span>
        <span style={{ margin: "0 12px" }}>·</span>
        <span>{graphData.links.length} relationships</span>
        <button onClick={fetchGraph} style={styles.refreshButton}>↻ Refresh</button>
      </div>
      
      {/* The graph canvas container — ForceGraph2D fills this */}
      <div ref={containerRef} style={styles.graphContainer}>
        <ForceGraph2D
          graphData={graphData}
          width={dimensions.width}
          height={dimensions.height}
          
          // Node appearance — nodeLabel is tooltip only (canvas handles display)
          nodeLabel={node => `${node.name} (${node.type})`}
          nodeColor={node => {
            if (highlightSet.has(node.id)) return "#ffffff";
            return TYPE_COLORS[node.type] || TYPE_COLORS.Unknown;
          }}
          nodeRelSize={6}
          
          // Custom node rendering: replace default — draw circle + label
          nodeCanvasObjectMode={() => "replace"}
          nodeCanvasObject={(node, ctx, globalScale) => {
            const label = node.name;
            const fontSize = 4 / globalScale;
            const r = 6;

            // Highlighted nodes get a white ring
            if (highlightSet.has(node.id)) {
              ctx.beginPath();
              ctx.arc(node.x, node.y, r + 3, 0, 2 * Math.PI);
              ctx.fillStyle = "rgba(255,255,255,0.25)";
              ctx.fill();
            }

            // Draw the circle
            ctx.beginPath();
            ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
            ctx.fillStyle = highlightSet.has(node.id)
              ? "#ffffff"
              : (TYPE_COLORS[node.type] || TYPE_COLORS.Unknown);
            ctx.fill();

            // Always draw the label below the circle
            ctx.font = `${fontSize}px Sans-Serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "top";
            ctx.fillStyle = "#e2e8f0";
            ctx.fillText(label, node.x, node.y + r + 2);
          }}
          
          // Link appearance
          linkLabel="relation"
          linkColor={() => "#475569"}
          linkWidth={1.5}
          linkDirectionalArrowLength={4}
          linkDirectionalArrowRelPos={1}
          
          // Show relationship type as a label on the edge
          linkCanvasObjectMode={() => "after"}
          linkCanvasObject={(link, ctx, globalScale) => {
            if (globalScale < 2) return;  // Only show at high zoom
            
            const start = link.source;
            const end = link.target;
            
            // Calculate the midpoint of the link
            const textPos = {
              x: start.x + (end.x - start.x) / 2,
              y: start.y + (end.y - start.y) / 2
            };
            
            const label = link.relation || "";
            const fontSize = 8 / globalScale;
            ctx.font = `${fontSize}px Sans-Serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillStyle = "#94a3b8";
            ctx.fillText(label, textPos.x, textPos.y);
          }}
          
          // Click handler: show node details in sidebar
          onNodeClick={(node) => setSelectedNode(node)}
          
          // Physics settings: how the force simulation behaves
          d3AlphaDecay={0.02}     // How fast the simulation cools down (lower = longer simulation)
          d3VelocityDecay={0.3}   // Node damping (higher = slower movement)
          cooldownTicks={100}      // Stop simulation after 100 ticks
          backgroundColor="#0f172a"
        />
      </div>
      
      {/* Node detail panel — appears when a node is clicked */}
      {selectedNode && (
        <div style={styles.nodePanel}>
          <button 
            onClick={() => setSelectedNode(null)} 
            style={styles.closeButton}
          >✕</button>
          <h3 style={{ color: "#f1f5f9", margin: "0 0 8px" }}>{selectedNode.name}</h3>
          <p style={{ color: "#94a3b8", margin: 0 }}>Type: {selectedNode.type}</p>
        </div>
      )}
      
      {/* Legend — only show types present in the current graph */}
      <div style={styles.legend}>
        {[...new Set(graphData.nodes.map(n => n.type))].sort().map(type => (
          <div key={type} style={styles.legendItem}>
            <div style={{ ...styles.legendDot, backgroundColor: TYPE_COLORS[type] || TYPE_COLORS.Unknown }} />
            <span style={{ color: "#94a3b8", fontSize: 11 }}>{type}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// Inline styles — no CSS file needed for this component
const styles = {
  wrapper: {
    position: "relative",
    width: "100%",
    height: "100%",
    background: "#0f172a",
    borderRadius: "8px",
    overflow: "hidden"
  },
  statsBar: {
    position: "absolute",
    top: 12,
    left: 12,
    zIndex: 10,
    background: "rgba(15,23,42,0.8)",
    color: "#94a3b8",
    padding: "6px 12px",
    borderRadius: "6px",
    fontSize: 13,
    display: "flex",
    alignItems: "center"
  },
  graphContainer: {
    width: "100%",
    height: "100%"
  },
  loadingContainer: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    background: "#0f172a"
  },
  loadingText: {
    color: "#94a3b8",
    fontSize: 16
  },
  errorContainer: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    height: "100%",
    background: "#0f172a",
    gap: 12
  },
  errorText: {
    color: "#f87171",
    fontSize: 14
  },
  retryButton: {
    background: "#3b82f6",
    color: "white",
    border: "none",
    padding: "8px 16px",
    borderRadius: "6px",
    cursor: "pointer"
  },
  refreshButton: {
    marginLeft: 12,
    background: "none",
    border: "1px solid #334155",
    color: "#94a3b8",
    padding: "2px 8px",
    borderRadius: "4px",
    cursor: "pointer",
    fontSize: 13
  },
  nodePanel: {
    position: "absolute",
    top: 12,
    right: 12,
    background: "#1e293b",
    border: "1px solid #334155",
    borderRadius: "8px",
    padding: 16,
    minWidth: 200,
    zIndex: 20
  },
  closeButton: {
    position: "absolute",
    top: 8,
    right: 8,
    background: "none",
    border: "none",
    color: "#94a3b8",
    cursor: "pointer",
    fontSize: 16
  },
  legend: {
    position: "absolute",
    bottom: 12,
    left: 12,
    display: "flex",
    flexWrap: "wrap",
    gap: "6px 12px",
    maxWidth: 400
  },
  legendItem: {
    display: "flex",
    alignItems: "center",
    gap: 4
  },
  legendDot: {
    width: 8,
    height: 8,
    borderRadius: "50%"
  }
};

