// frontend/src/Upload.jsx

import { useState } from "react";

// Props:
//   onUpload(sessionId) — called after a successful upload
//                         so the parent (App.jsx) can store the session_id
export default function Upload({ onUpload }) {
  const [status, setStatus] = useState("idle"); // idle | uploading | done | error
  const [filename, setFilename] = useState("");
  const [errorMsg, setErrorMsg] = useState("");

  async function handleFileChange(e) {
    const file = e.target.files[0];
    if (!file) return;

    setFilename(file.name);
    setStatus("uploading");
    setErrorMsg("");

    // FormData is required for multipart/form-data uploads.
    // Do NOT set Content-Type manually — the browser sets it automatically
    // with the correct multipart boundary when you use FormData.
    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed.");
      }

      const data = await res.json();
      setStatus("done");
      onUpload(data.session_id); // bubble session_id up to App.jsx
    } catch (err) {
      setStatus("error");
      setErrorMsg(err.message);
    }
  }

  return (
    <div style={{ marginBottom: "1.5rem" }}>
      <label style={{ display: "block", marginBottom: "0.5rem", fontWeight: 500 }}>
        Upload a CSV file:
      </label>
      <input
        type="file"
        accept=".csv"
        onChange={handleFileChange}
        disabled={status === "uploading"}
      />

      {status === "uploading" && (
        <p style={{ color: "#666", marginTop: "0.5rem" }}>
          Uploading {filename}...
        </p>
      )}
      {status === "done" && (
        <p style={{ color: "green", marginTop: "0.5rem" }}>
          ✓ {filename} is ready. Ask a question below.
        </p>
      )}
      {status === "error" && (
        <p style={{ color: "red", marginTop: "0.5rem" }}>
          Error: {errorMsg}
        </p>
      )}
    </div>
  );
}