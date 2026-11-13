"use client";

import React from "react";
import {
  PlusIcon,
  UploadIcon,
  MessageSquareIcon,
  CloseIcon,
  BotIcon,
  CheckCircleIcon,
  AlertCircleIcon,
} from "../ui/Icons";
import { ConversationSummary, HealthResponse } from "@/lib/api/types";

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  conversations: ConversationSummary[];
  activeId: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onOpenUpload: () => void;
  health: HealthResponse | null;
}

export function Sidebar({
  isOpen,
  onClose,
  conversations,
  activeId,
  onSelectConversation,
  onNewConversation,
  onOpenUpload,
  health,
}: SidebarProps) {
  React.useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        onClose();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  return (
    <>
      {/* Mobile Backdrop Overlay */}
      {isOpen && <div className="mobile-overlay" onClick={onClose} />}

      <aside className={`sidebar ${isOpen ? "open" : ""}`}>
        {/* Header */}
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <BotIcon />
            <span>Telugu AI</span>
            <span className="brand-badge">v0.1.0</span>
          </div>
          <button
            className="btn btn-icon"
            onClick={onClose}
            aria-label="Close Sidebar"
            style={{ display: isOpen ? "flex" : "none" }}
          >
            <CloseIcon />
          </button>
        </div>

        {/* Top Actions: New Chat & Upload PDF */}
        <div className="sidebar-actions">
          <button
            className="btn btn-primary"
            onClick={() => {
              onNewConversation();
              onClose();
            }}
            id="btn-new-chat"
            style={{ width: "100%" }}
          >
            <PlusIcon />
            <span>New Conversation</span>
          </button>

          <button
            className="btn btn-secondary"
            onClick={() => {
              onOpenUpload();
              onClose();
            }}
            id="btn-open-upload"
            style={{ width: "100%" }}
          >
            <UploadIcon />
            <span>Upload Document</span>
          </button>
        </div>

        {/* Conversations History List */}
        <div className="sidebar-section-title">Conversations</div>
        <div className="conversation-list">
          {conversations.length === 0 ? (
            <div style={{ padding: "12px", color: "var(--text-muted)", fontSize: "12.5px" }}>
              No active conversations yet. Start a new session above.
            </div>
          ) : (
            conversations.map((c) => {
              const isActive = c.id === activeId;
              return (
                <button
                  key={c.id}
                  className={`conversation-item ${isActive ? "active" : ""}`}
                  onClick={() => {
                    onSelectConversation(c.id);
                    onClose();
                  }}
                  title={c.title}
                >
                  <MessageSquareIcon />
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                    {c.title || `Session ${c.id.slice(0, 8)}`}
                  </span>
                </button>
              );
            })
          )}
        </div>

        {/* Footer: Live Infrastructure Status */}
        <div className="sidebar-footer">
          <div style={{ fontWeight: 600, color: "var(--text-secondary)", marginBottom: "8px", fontSize: "11px", textTransform: "uppercase" }}>
            Infrastructure
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "5px", fontSize: "12px" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>Qdrant Vector DB</span>
              <span style={{ color: health?.qdrant_connected ? "var(--success)" : "var(--error)", display: "flex", alignItems: "center", gap: "4px" }}>
                {health?.qdrant_connected ? <CheckCircleIcon size={13} /> : <AlertCircleIcon size={13} />}
                {health?.qdrant_connected ? "Connected" : "Offline"}
              </span>
            </div>

            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>PostgreSQL Memory</span>
              <span style={{ color: health?.postgres_connected ? "var(--success)" : "var(--error)", display: "flex", alignItems: "center", gap: "4px" }}>
                {health?.postgres_connected ? <CheckCircleIcon size={13} /> : <AlertCircleIcon size={13} />}
                {health?.postgres_connected ? "Ready" : "Offline"}
              </span>
            </div>

            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span style={{ color: "var(--text-muted)" }}>LangGraph Agent</span>
              <span style={{ color: health?.langgraph_active ? "var(--success)" : "var(--error)", display: "flex", alignItems: "center", gap: "4px" }}>
                {health?.langgraph_active ? <CheckCircleIcon size={13} /> : <AlertCircleIcon size={13} />}
                {health?.langgraph_active ? "Active" : "Inactive"}
              </span>
            </div>

            {health?.tts_engine && (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
                <span style={{ color: "var(--text-muted)" }}>TTS Engine</span>
                <span style={{ color: health.tts_adapter_active ? "var(--primary)" : "var(--text-secondary)", fontSize: "11px", fontWeight: 500 }}>
                  {health.tts_adapter_active ? "IndicF5 (LoRA)" : health.tts_engine}
                </span>
              </div>
            )}
          </div>
        </div>
      </aside>
    </>
  );
}

