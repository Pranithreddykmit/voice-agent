"use client";

import React, { useState, useRef, useEffect } from "react";
import { SendIcon, MicIcon } from "../ui/Icons";

interface MessageComposerProps {
  onSendMessage: (query: string, mode: "dense" | "bm25" | "hybrid" | "hybrid_rerank") => void;
  onToggleVoice: () => void;
  isVoiceActive: boolean;
  isLoading: boolean;
  selectedMode: "dense" | "bm25" | "hybrid" | "hybrid_rerank";
  onModeChange: (mode: "dense" | "bm25" | "hybrid" | "hybrid_rerank") => void;
}

export function MessageComposer({
  onSendMessage,
  onToggleVoice,
  isVoiceActive,
  isLoading,
  selectedMode,
  onModeChange,
}: MessageComposerProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 180)}px`;
    }
  }, [input]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    onSendMessage(input.trim(), selectedMode);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "48px";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  return (
    <div className="composer-wrapper">
      <div className="composer-container">
        <form onSubmit={handleSubmit}>
          <div className="composer-box">
            <textarea
              ref={textareaRef}
              className="composer-textarea"
              placeholder="Ask a question in Telugu, English, or mixed text (e.g. కనీస హాజరు ఎంత శాతం ఉండాలి?)..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              rows={1}
              id="input-query-composer"
            />

            <div className="composer-toolbar">
              <div className="composer-tools-left">
                {/* Voice Input Button */}
                <button
                  type="button"
                  className={`btn btn-icon ${isVoiceActive ? "active" : ""}`}
                  onClick={onToggleVoice}
                  aria-label="Toggle Voice Input"
                  id="btn-toggle-voice"
                  style={{
                    color: isVoiceActive ? "var(--error)" : "var(--text-secondary)",
                    backgroundColor: isVoiceActive ? "var(--error-subtle)" : "transparent",
                  }}
                  title="Speak query using microphone"
                >
                  <MicIcon />
                </button>

                {/* Retrieval Mode Selector */}
                <select
                  value={selectedMode}
                  onChange={(e) => onModeChange(e.target.value as any)}
                  style={{
                    backgroundColor: "var(--bg-surface-elevated)",
                    color: "var(--text-secondary)",
                    border: "1px solid var(--border-subtle)",
                    borderRadius: "var(--radius-sm)",
                    padding: "3px 8px",
                    fontSize: "11.5px",
                    outline: "none",
                    cursor: "pointer",
                  }}
                  title="Select Retrieval Mode"
                  id="select-retrieval-mode"
                >
                  <option value="hybrid">Hybrid (Dense + BM25 + RRF)</option>
                  <option value="hybrid_rerank">Hybrid + Reranker</option>
                  <option value="dense">Dense (E5 Multilingual)</option>
                  <option value="bm25">Sparse (BM25Plus)</option>
                </select>
              </div>

              <div className="composer-tools-right">
                <button
                  type="submit"
                  className="btn btn-primary"
                  disabled={!input.trim() || isLoading}
                  id="btn-send-message"
                  style={{ padding: "6px 12px" }}
                >
                  <SendIcon size={14} />
                  <span>Send</span>
                </button>
              </div>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
