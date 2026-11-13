"use client";

import React, { useState, useRef, useEffect } from "react";

import { UploadIcon, CloseIcon, CheckCircleIcon, AlertCircleIcon, FileTextIcon } from "../ui/Icons";
import { uploadDocument } from "@/lib/api/client";
import { DocumentUploadResponse } from "@/lib/api/types";

interface DocumentUploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUploadSuccess?: (res: DocumentUploadResponse) => void;
}

export function DocumentUploadModal({
  isOpen,
  onClose,
  onUploadSuccess,
}: DocumentUploadModalProps) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DocumentUploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen && !loading) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, loading, onClose]);

  if (!isOpen) return null;

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      if (file.name.toLowerCase().endsWith(".pdf")) {
        setSelectedFile(file);
        setError(null);
      } else {
        setError("Only PDF files are supported.");
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (file.name.toLowerCase().endsWith(".pdf")) {
        setSelectedFile(file);
        setError(null);
      } else {
        setError("Only PDF files are supported.");
      }
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await uploadDocument(selectedFile);
      setResult(res);
      if (onUploadSuccess) onUploadSuccess(res);
    } catch (err: any) {
      setError(err.message || "Failed to upload document.");
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="modal-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div style={{ display: "flex", alignItems: "center", gap: "8px", fontWeight: 600 }}>
            <UploadIcon size={18} />
            <span>Upload Document to Knowledge Base</span>
          </div>
          <button className="btn btn-icon" onClick={onClose} aria-label="Close modal" disabled={loading}>
            <CloseIcon />
          </button>
        </div>

        <div className="modal-body">
          {!result ? (
            <>
              <div
                className={`dropzone ${dragActive ? "active" : ""}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf"
                  style={{ display: "none" }}
                  onChange={handleFileChange}
                />
                <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: "8px" }}>
                  <FileTextIcon size={32} className="icon" />
                  <div style={{ fontWeight: 500, color: "var(--text-primary)" }}>
                    {selectedFile ? selectedFile.name : "Click to browse or drop PDF here"}
                  </div>
                  <div style={{ fontSize: "12px", color: "var(--text-muted)" }}>
                    {selectedFile
                      ? `${(selectedFile.size / 1024).toFixed(1)} KB selected`
                      : "Supports bilingual English and Telugu academic / institutional PDFs"}
                  </div>
                </div>
              </div>

              {error && (
                <div
                  style={{
                    marginTop: "12px",
                    padding: "10px",
                    backgroundColor: "var(--error-subtle)",
                    border: "1px solid var(--error)",
                    borderRadius: "var(--radius-sm)",
                    color: "var(--error)",
                    fontSize: "12.5px",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                  }}
                >
                  <AlertCircleIcon size={16} />
                  <span>{error}</span>
                </div>
              )}

              {loading && (
                <div
                  style={{
                    marginTop: "16px",
                    padding: "12px",
                    backgroundColor: "var(--bg-surface)",
                    borderRadius: "var(--radius-sm)",
                    textAlign: "center",
                    fontSize: "13px",
                    color: "var(--text-secondary)",
                  }}
                >
                  Uploading, parsing pages, embedding, and indexing into Qdrant & BM25...
                </div>
              )}
            </>
          ) : (
            <div style={{ textAlign: "center", padding: "16px 0" }}>
              {result.status === "warning_no_text" ? (
                <>
                  <div style={{ color: "var(--warning)", display: "flex", justifyContent: "center", marginBottom: "8px" }}>
                    <AlertCircleIcon size={36} />
                  </div>
                  <h3 style={{ fontSize: "16px", fontWeight: 600, color: "var(--warning)", marginBottom: "4px" }}>
                    No Extractable Text Found
                  </h3>
                  <p style={{ color: "var(--text-secondary)", fontSize: "13px", marginBottom: "16px" }}>
                    <strong>{result.filename}</strong> contains {result.total_pages} page(s), but no selectable text was found. If this is a scanned document, please run OCR first.
                  </p>
                </>
              ) : (
                <>
                  <div style={{ color: "var(--success)", display: "flex", justifyContent: "center", marginBottom: "8px" }}>
                    <CheckCircleIcon size={36} />
                  </div>
                  <h3 style={{ fontSize: "16px", fontWeight: 600, color: "var(--text-primary)", marginBottom: "4px" }}>
                    Document Successfully Indexed!
                  </h3>
                  <p style={{ color: "var(--text-secondary)", fontSize: "13px", marginBottom: "16px" }}>
                    <strong>{result.filename}</strong> has been embedded and indexed into the vector & sparse knowledge base.
                  </p>
                </>
              )}

              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr",
                  gap: "10px",
                  backgroundColor: "var(--bg-app)",
                  padding: "12px",
                  borderRadius: "var(--radius-md)",
                  border: "1px solid var(--border-subtle)",
                  fontSize: "12.5px",
                  textAlign: "left",
                }}
              >
                <div>
                  <span style={{ color: "var(--text-muted)" }}>Total Pages:</span>{" "}
                  <strong>{result.total_pages}</strong>
                </div>
                <div>
                  <span style={{ color: "var(--text-muted)" }}>Indexed Chunks:</span>{" "}
                  <strong>{result.total_chunks}</strong>
                </div>
                <div>
                  <span style={{ color: "var(--text-muted)" }}>Processing Time:</span>{" "}
                  <strong>{result.processing_time_ms} ms</strong>
                </div>
                <div>
                  <span style={{ color: "var(--text-muted)" }}>Collection:</span>{" "}
                  <strong style={{ fontSize: "11px" }}>{result.collection_name}</strong>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="modal-footer">
          {!result ? (
            <>
              <button className="btn btn-secondary" onClick={onClose} disabled={loading}>
                Cancel
              </button>
              <button
                className="btn btn-primary"
                onClick={handleUpload}
                disabled={!selectedFile || loading}
                id="btn-confirm-upload"
              >
                {loading ? "Indexing..." : "Upload & Index"}
              </button>
            </>
          ) : (
            <>
              <button className="btn btn-secondary" onClick={handleReset}>
                Upload Another
              </button>
              <button className="btn btn-primary" onClick={onClose}>
                Done
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

