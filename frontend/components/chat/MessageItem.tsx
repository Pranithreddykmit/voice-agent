"use client";

import React from "react";
import { UserIcon, BotIcon, CalculatorIcon, CheckCircleIcon, AlertCircleIcon } from "../ui/Icons";
import { SourceCard } from "../sources/SourceCard";
import { AudioPlayer } from "./AudioPlayer";
import { MessageSchema } from "@/lib/api/types";

interface MessageItemProps {
  message: MessageSchema;
}

export function MessageItem({ message }: MessageItemProps) {
  const isUser = message.role === "user";

  const renderIntentBadge = (intent?: string | null) => {
    if (!intent) return null;
    const labelMap: Record<string, string> = {
      knowledge_query: "Document Knowledge",
      tool_query: "Calculator Tool",
      casual_query: "Casual Dialogue",
      ambiguous_query: "Clarification Needed",
    };
    return <span className={`intent-pill ${intent}`}>{labelMap[intent] || intent}</span>;
  };

  return (
    <div className={`message-row ${isUser ? "user" : "assistant"}`}>
      {/* Avatar */}
      <div className={`message-avatar ${isUser ? "user" : "assistant"}`}>
        {isUser ? <UserIcon size={16} /> : <BotIcon size={16} />}
      </div>

      {/* Bubble */}
      <div className={`message-bubble ${isUser ? "user" : "assistant"}`}>
        {!isUser && (
          <div className="message-meta">
            {renderIntentBadge(message.intent)}
            {message.latency && (
              <span style={{ color: "var(--text-muted)", fontSize: "11px", marginLeft: "auto" }}>
                {message.latency.total_ms ? `${message.latency.total_ms} ms` : ""}
              </span>
            )}
          </div>
        )}

        {/* Text Content */}
        <div style={{ whiteSpace: "pre-wrap" }}>{message.content}</div>

        {/* Audio Player for Voice Responses */}
        {!isUser && message.audio_base64 && (
          <AudioPlayer audioBase64={message.audio_base64} autoPlay={false} />
        )}

        {/* Structured Tool Result Box */}
        {message.tool_result && message.tool_result.status === "success" && (
          <div className="tool-result-box">
            <div className="tool-result-header">
              <CalculatorIcon size={14} />
              <span>Calculated Result</span>
            </div>
            <div style={{ color: "var(--text-secondary)" }}>
              Expression: <code>{message.tool_result.expression}</code>
            </div>
            <div style={{ fontSize: "15px", fontWeight: 600, color: "var(--text-primary)", marginTop: "4px" }}>
              = {message.tool_result.formatted_result}
            </div>
          </div>
        )}

        {/* Source Citations */}
        {message.sources && message.sources.length > 0 && (
          <div className="sources-section">
            <div className="sources-header">
              <span>Verified Document Sources ({message.sources.length})</span>
            </div>
            <div className="sources-grid">
              {message.sources.map((src, idx) => (
                <SourceCard key={idx} source={src} />
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
