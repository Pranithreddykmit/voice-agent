"use client";

import React from "react";
import { MenuIcon, BotIcon } from "../ui/Icons";
import { HealthResponse } from "@/lib/api/types";

interface AppHeaderProps {
  onToggleSidebar: () => void;
  health: HealthResponse | null;
  activeConversationId: string | null;
  retrievalMode: string;
}

export function AppHeader({
  onToggleSidebar,
  health,
  activeConversationId,
  retrievalMode,
}: AppHeaderProps) {
  const isHealthy = health?.status === "healthy";

  return (
    <header className="app-header">
      <div className="header-left">
        <button
          className="btn btn-icon"
          onClick={onToggleSidebar}
          aria-label="Toggle Navigation Sidebar"
          id="btn-toggle-sidebar"
        >
          <MenuIcon />
        </button>

        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ fontWeight: 600, fontSize: "15px", color: "var(--text-primary)" }}>
            Telugu Conversational AI
          </span>
          {activeConversationId && (
            <span style={{ fontSize: "11px", color: "var(--text-muted)", fontFamily: "monospace" }}>
              #{activeConversationId.slice(0, 8)}
            </span>
          )}
        </div>
      </div>

      <div className="header-right">
        {/* TTS & LoRA Engine Badge */}
        {health?.tts_engine && (
          <span
            style={{
              fontSize: "11px",
              padding: "3px 8px",
              borderRadius: "var(--radius-sm)",
              backgroundColor: "var(--bg-surface)",
              border: "1px solid var(--border-subtle)",
              color: health.tts_adapter_active ? "var(--primary)" : "var(--text-muted)",
              display: "none",
            }}
            className="tts-header-badge"
            title={`TTS Engine: ${health.tts_engine} | LoRA Adapter: ${health.tts_adapter || "none"}`}
          >
            TTS: <strong>{health.tts_adapter_active ? "IndicF5 + LoRA" : health.tts_engine}</strong>
          </span>
        )}

        {/* Retrieval Mode Badge */}
        <span
          style={{
            fontSize: "11.5px",
            padding: "3px 8px",
            borderRadius: "var(--radius-sm)",
            backgroundColor: "var(--bg-surface)",
            border: "1px solid var(--border-subtle)",
            color: "var(--text-secondary)",
          }}
          title={`Active Retrieval Mode: ${retrievalMode}`}
        >
          Mode: <strong style={{ color: "var(--text-primary)" }}>{retrievalMode}</strong>
        </span>

        {/* System Health Status Indicator */}
        <div
          className="status-pill"
          title={
            health
              ? `Qdrant: ${health.qdrant_connected ? "Connected" : "Disconnected"} | PostgreSQL: ${
                  health.postgres_connected ? "Connected" : "Disconnected"
                } | LangGraph: ${health.langgraph_active ? "Active" : "Inactive"} | TTS: ${health.tts_engine || "unknown"}`
              : "Connecting to AI Service..."
          }
        >
          <span className={`status-dot ${isHealthy ? "healthy" : "degraded"}`} />
          <span style={{ color: isHealthy ? "var(--text-secondary)" : "var(--warning)", fontSize: "11.5px" }}>
            {health ? (isHealthy ? "System Ready" : "Degraded") : "Connecting..."}
          </span>
        </div>
      </div>

    </header>
  );
}
