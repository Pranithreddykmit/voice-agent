import React, { useRef, useEffect, useState } from "react";
import { MessageItem } from "./MessageItem";
import { EmptyState } from "./EmptyState";
import { MessageComposer } from "./MessageComposer";
import { VoiceRecorder } from "./VoiceRecorder";
import { MessageSchema } from "@/lib/api/types";
import { AlertCircleIcon, RefreshIcon, BotIcon } from "../ui/Icons";

interface ChatContainerProps {
  messages: MessageSchema[];
  isLoading: boolean;
  isVoiceProcessing: boolean;
  error: string | null;
  onSendMessage: (query: string, mode: "dense" | "bm25" | "hybrid" | "hybrid_rerank") => void;
  onSendVoice: (audioBlob: Blob, mode: "dense" | "bm25" | "hybrid" | "hybrid_rerank") => void;
  onRetry: () => void;
  selectedMode: "dense" | "bm25" | "hybrid" | "hybrid_rerank";
  onModeChange: (mode: "dense" | "bm25" | "hybrid" | "hybrid_rerank") => void;
}

export function ChatContainer({
  messages,
  isLoading,
  isVoiceProcessing,
  error,
  onSendMessage,
  onSendVoice,
  onRetry,
  selectedMode,
  onModeChange,
}: ChatContainerProps) {
  const [voiceActive, setVoiceActive] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, isVoiceProcessing, error]);

  const handleRecordingComplete = (audioBlob: Blob) => {
    setVoiceActive(false);
    onSendVoice(audioBlob, selectedMode);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
      {/* Scrollable Messages Viewport */}
      <div className="messages-viewport">
        <div className="messages-container">
          {messages.length === 0 ? (
            <EmptyState onSelectPrompt={(prompt) => onSendMessage(prompt, selectedMode)} />
          ) : (
            messages.map((msg) => <MessageItem key={msg.id} message={msg} />)
          )}

          {/* Loading Indicator */}
          {(isLoading || isVoiceProcessing) && (
            <div className="message-row assistant">
              <div className="message-avatar assistant">
                <BotIcon size={16} />
              </div>
              <div className="message-bubble assistant" style={{ color: "var(--text-secondary)" }}>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <RefreshIcon className="spin" size={15} />
                  <span>
                    {isVoiceProcessing
                      ? "Transcribing voice audio, consulting knowledge base & synthesizing Telugu speech..."
                      : "Understanding query, searching indexed context & synthesizing answer..."}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Error Banner */}
          {error && (
            <div
              style={{
                padding: "12px 16px",
                backgroundColor: "var(--error-subtle)",
                border: "1px solid var(--error)",
                borderRadius: "var(--radius-md)",
                color: "var(--error)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                fontSize: "13px",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <AlertCircleIcon size={16} />
                <span>{error}</span>
              </div>
              <button
                className="btn btn-secondary"
                onClick={onRetry}
                style={{ fontSize: "12px", padding: "4px 8px" }}
              >
                <RefreshIcon size={12} />
                <span>Retry</span>
              </button>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Voice Recorder Overlay / Toolbar */}
      {voiceActive && (
        <div style={{ padding: "0 20px" }}>
          <div style={{ maxWidth: "820px", margin: "0 auto" }}>
            <VoiceRecorder
              isOpen={voiceActive}
              onClose={() => setVoiceActive(false)}
              onRecordingComplete={handleRecordingComplete}
              isProcessing={isVoiceProcessing}
            />
          </div>
        </div>
      )}

      {/* Composer Toolbar */}
      <MessageComposer
        onSendMessage={onSendMessage}
        onToggleVoice={() => setVoiceActive(!voiceActive)}
        isVoiceActive={voiceActive}
        isLoading={isLoading || isVoiceProcessing}
        selectedMode={selectedMode}
        onModeChange={onModeChange}
      />
    </div>
  );
}
