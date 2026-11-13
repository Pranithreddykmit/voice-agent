"use client";

import React, { useState, useEffect, useCallback, useRef } from "react";
import { AppHeader } from "@/components/layout/AppHeader";
import { Sidebar } from "@/components/layout/Sidebar";
import { ChatContainer } from "@/components/chat/ChatContainer";
import { DocumentUploadModal } from "@/components/documents/DocumentUploadModal";
import {
  fetchHealth,
  sendQuery,
  streamQuery,
  sendVoiceChat,
  streamVoiceChat,
  fetchConversationHistory,
} from "@/lib/api/client";
import {
  HealthResponse,
  MessageSchema,
  ConversationSummary,
  DocumentUploadResponse,
  SSEEvent,
} from "@/lib/api/types";

const STORAGE_CONVERSATIONS_KEY = "telugu_ai_conversations";

export default function Home() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [uploadModalOpen, setUploadModalOpen] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<MessageSchema[]>([]);

  const [isLoading, setIsLoading] = useState(false);
  const [isVoiceProcessing, setIsVoiceProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastQuery, setLastQuery] = useState<string | null>(null);
  const [retrievalMode, setRetrievalMode] = useState<"dense" | "bm25" | "hybrid" | "hybrid_rerank">("hybrid");

  const abortControllerRef = useRef<AbortController | null>(null);

  // Load conversations list from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_CONVERSATIONS_KEY);
      if (stored) {
        const parsed: ConversationSummary[] = JSON.parse(stored);
        setConversations(parsed);
      }
    } catch {
      // Ignore localStorage read errors
    }
  }, []);

  // Sync conversations list to localStorage
  const saveConversations = (updated: ConversationSummary[]) => {
    setConversations(updated);
    try {
      localStorage.setItem(STORAGE_CONVERSATIONS_KEY, JSON.stringify(updated));
    } catch {
      // Ignore localStorage write errors
    }
  };

  // Cleanup in-flight stream on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  // Fetch health status
  const checkHealth = useCallback(async () => {
    try {
      const h = await fetchHealth();
      setHealth(h);
    } catch {
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  // Load conversation messages when activeConversationId changes
  const loadConversation = useCallback(async (id: string) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setIsLoading(true);
    setError(null);
    try {
      const history = await fetchConversationHistory(id);
      setMessages(history.messages);
      setActiveConversationId(id);
    } catch (err: any) {
      setError(`Failed to load conversation: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleSelectConversation = (id: string) => {
    if (id === activeConversationId) return;
    loadConversation(id);
  };

  const handleNewConversation = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    setActiveConversationId(null);
    setMessages([]);
    setError(null);
  };

  // Text message handler with progressive SSE streaming
  const handleSendMessage = async (
    queryText: string,
    mode: "dense" | "bm25" | "hybrid" | "hybrid_rerank"
  ) => {
    // Abort previous stream if still in-flight
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);
    setError(null);
    setLastQuery(queryText);

    const userMsgId = `user_${Date.now()}`;
    const asstMsgId = `asst_${Date.now()}`;

    // Optimistic user message
    const tempUserMessage: MessageSchema = {
      id: userMsgId,
      conversation_id: activeConversationId || "temp",
      role: "user",
      content: queryText,
      created_at: new Date().toISOString(),
    };

    // Placeholder assistant message for progressive chunk accumulation
    const initialAssistantMessage: MessageSchema = {
      id: asstMsgId,
      conversation_id: activeConversationId || "temp",
      role: "assistant",
      content: "",
      intent: "knowledge_query",
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, tempUserMessage, initialAssistantMessage]);

    let accumulatedContent = "";
    let resolvedConvId = activeConversationId;

    try {
      await streamQuery(
        {
          query: queryText,
          conversation_id: activeConversationId,
          retrieval_mode: mode,
        },
        (event: SSEEvent) => {
          if (event.type === "meta") {
            resolvedConvId = event.conversation_id;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === asstMsgId
                  ? {
                      ...msg,
                      conversation_id: event.conversation_id,
                      intent: event.intent,
                      sources: event.sources,
                      tool_result: event.tool_result,
                    }
                  : msg.id === userMsgId
                  ? { ...msg, conversation_id: event.conversation_id }
                  : msg
              )
            );
          } else if (event.type === "chunk") {
            accumulatedContent += event.text;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === asstMsgId
                  ? { ...msg, content: accumulatedContent }
                  : msg
              )
            );
          } else if (event.type === "done") {
            resolvedConvId = event.conversation_id;
            const finalAnswer = event.answer || accumulatedContent;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === asstMsgId
                  ? {
                      ...msg,
                      conversation_id: event.conversation_id,
                      content: finalAnswer,
                      intent: event.intent,
                      sources: event.sources,
                      tool_result: event.tool_result,
                      latency: event.latency,
                    }
                  : msg.id === userMsgId
                  ? { ...msg, conversation_id: event.conversation_id }
                  : msg
              )
            );

            if (!activeConversationId && resolvedConvId) {
              setActiveConversationId(resolvedConvId);
            }

            // Update conversation list in sidebar
            const summaryTitle =
              queryText.length > 35 ? `${queryText.slice(0, 35)}...` : queryText;
            const updatedList = [
              {
                id: resolvedConvId || "conv",
                title: summaryTitle,
                lastMessageAt: new Date().toISOString(),
                preview: finalAnswer.slice(0, 50),
              },
              ...conversations.filter((c) => c.id !== (resolvedConvId || "conv")),
            ];
            saveConversations(updatedList);
          } else if (event.type === "error") {
            setError(event.detail || "Streaming error encountered.");
          }
        },
        controller.signal
      );
    } catch (err: any) {
      if (err.name !== "AbortError") {
        setError(err.message || "Failed to execute query.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Voice message handler with progressive SSE streaming
  const handleSendVoice = async (
    audioBlob: Blob,
    mode: "dense" | "bm25" | "hybrid" | "hybrid_rerank"
  ) => {
    // Abort previous stream if still in-flight
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsVoiceProcessing(true);
    setError(null);

    const userMsgId = `user_${Date.now()}`;
    const asstMsgId = `asst_${Date.now()}`;

    // Optimistic user transcribed message placeholder
    const tempUserMessage: MessageSchema = {
      id: userMsgId,
      conversation_id: activeConversationId || "temp",
      role: "user",
      content: "🎙 Transcribing speech audio...",
      created_at: new Date().toISOString(),
    };

    // Placeholder assistant message
    const initialAssistantMessage: MessageSchema = {
      id: asstMsgId,
      conversation_id: activeConversationId || "temp",
      role: "assistant",
      content: "",
      intent: "knowledge_query",
      created_at: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, tempUserMessage, initialAssistantMessage]);

    let accumulatedContent = "";
    let resolvedConvId = activeConversationId;
    let transcriptText = "";

    try {
      await streamVoiceChat(
        audioBlob,
        activeConversationId,
        mode,
        (event: SSEEvent) => {
          if (event.type === "stt") {
            transcriptText = event.transcript;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === userMsgId
                  ? { ...msg, content: `🎙 ${event.transcript}` }
                  : msg
              )
            );
          } else if (event.type === "meta") {
            resolvedConvId = event.conversation_id;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === asstMsgId
                  ? {
                      ...msg,
                      conversation_id: event.conversation_id,
                      intent: event.intent,
                      sources: event.sources,
                      tool_result: event.tool_result,
                    }
                  : msg.id === userMsgId
                  ? { ...msg, conversation_id: event.conversation_id }
                  : msg
              )
            );
          } else if (event.type === "chunk") {
            accumulatedContent += event.text;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === asstMsgId
                  ? { ...msg, content: accumulatedContent }
                  : msg
              )
            );
          } else if (event.type === "done") {
            resolvedConvId = event.conversation_id;
            const finalTranscript = event.query_transcript || transcriptText;
            const finalAnswer = event.answer || accumulatedContent;

            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === asstMsgId
                  ? {
                      ...msg,
                      conversation_id: event.conversation_id,
                      content: finalAnswer,
                      intent: event.intent,
                      sources: event.sources,
                      tool_result: event.tool_result,
                      audio_base64: event.audio_base64,
                      latency: event.latency,
                    }
                  : msg.id === userMsgId
                  ? {
                      ...msg,
                      conversation_id: event.conversation_id,
                      content: `🎙 ${finalTranscript}`,
                    }
                  : msg
              )
            );

            if (!activeConversationId && resolvedConvId) {
              setActiveConversationId(resolvedConvId);
            }

            const summaryTitle =
              finalTranscript.length > 35
                ? `🎙 ${finalTranscript.slice(0, 35)}...`
                : `🎙 ${finalTranscript}`;
            const updatedList = [
              {
                id: resolvedConvId || "conv",
                title: summaryTitle,
                lastMessageAt: new Date().toISOString(),
                preview: finalAnswer.slice(0, 50),
              },
              ...conversations.filter((c) => c.id !== (resolvedConvId || "conv")),
            ];
            saveConversations(updatedList);
          } else if (event.type === "error") {
            setError(event.detail || "Voice streaming error encountered.");
          }
        },
        controller.signal
      );
    } catch (err: any) {
      if (err.name !== "AbortError") {
        setError(err.message || "Voice chat processing failed.");
      }
    } finally {
      setIsVoiceProcessing(false);
    }
  };

  const handleRetry = () => {
    if (lastQuery) {
      handleSendMessage(lastQuery, retrievalMode);
    }
  };

  const handleUploadSuccess = (res: DocumentUploadResponse) => {
    checkHealth();
  };

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <Sidebar
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        conversations={conversations}
        activeId={activeConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
        onOpenUpload={() => setUploadModalOpen(true)}
        health={health}
      />

      {/* Main App Area */}
      <div className="main-wrapper">
        <AppHeader
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          health={health}
          activeConversationId={activeConversationId}
          retrievalMode={retrievalMode}
        />

        <ChatContainer
          messages={messages}
          isLoading={isLoading}
          isVoiceProcessing={isVoiceProcessing}
          error={error}
          onSendMessage={handleSendMessage}
          onSendVoice={handleSendVoice}
          onRetry={handleRetry}
          selectedMode={retrievalMode}
          onModeChange={setRetrievalMode}
        />
      </div>

      {/* Document Upload Modal */}
      <DocumentUploadModal
        isOpen={uploadModalOpen}
        onClose={() => setUploadModalOpen(false)}
        onUploadSuccess={handleUploadSuccess}
      />
    </div>
  );
}
