// frontend/src/App.jsx

import { useState } from "react";
import Upload from "./Upload";
import Chat from "./Chat";

export default function App() {
  // sessionId is null until the user uploads a CSV.
  // Once set, Chat is rendered with it.
  const [sessionId, setSessionId] = useState(null);

  return (
    <div
      style={{
        maxWidth: "740px",
        margin: "2rem auto",
        padding: "0 1rem",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <h1 style={{ marginBottom: "0.25rem" }}>CSV Analyst AI</h1>
      <p style={{ color: "#555", marginBottom: "1.75rem" }}>
        Upload a CSV file, then ask questions about your data in plain English.
        Powered by Gemini 2.5 Flash — free tier, no credit card required.
      </p>

      {/* Upload always shown so the user can swap CSV files */}
      <Upload onUpload={setSessionId} />

      {sessionId ? (
        <Chat sessionId={sessionId} />
      ) : (
        <p style={{ color: "#bbb", fontStyle: "italic" }}>
          Upload a CSV to start chatting.
        </p>
      )}
    </div>
  );
}