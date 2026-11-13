"use client";

import React, { useState, useRef, useEffect } from "react";
import { MicIcon, CloseIcon, SendIcon } from "../ui/Icons";

interface VoiceRecorderProps {
  isOpen: boolean;
  onClose: () => void;
  onRecordingComplete: (blob: Blob) => void;
  isProcessing: boolean;
}

export function VoiceRecorder({
  isOpen,
  onClose,
  onRecordingComplete,
  isProcessing,
}: VoiceRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startRecording = async () => {
    setError(null);
    audioChunksRef.current = [];
    setSeconds(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const mimeType = MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : MediaRecorder.isTypeSupported("audio/mp4")
        ? "audio/mp4"
        : "";

      const options = mimeType ? { mimeType } : undefined;
      const mediaRecorder = new MediaRecorder(stream, options);
      mediaRecorderRef.current = mediaRecorder;

      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, {
          type: mimeType || "audio/wav",
        });
        if (audioBlob.size > 0) {
          onRecordingComplete(audioBlob);
        }
        cleanupStream();
      };

      mediaRecorder.start(200); // 200ms chunk slices
      setIsRecording(true);

      timerRef.current = setInterval(() => {
        setSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err: any) {
      setError(
        err.name === "NotAllowedError" || err.name === "PermissionDeniedError"
          ? "Microphone access denied. Please enable microphone permissions in your browser."
          : `Failed to access microphone: ${err.message}`
      );
      setIsRecording(false);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  const cancelRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.onstop = null; // Ignore chunks
      mediaRecorderRef.current.stop();
    }
    cleanupStream();
    setIsRecording(false);
    if (timerRef.current) clearInterval(timerRef.current);
    onClose();
  };

  const cleanupStream = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  };

  useEffect(() => {
    if (isOpen && !isRecording && !isProcessing) {
      startRecording();
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      cleanupStream();
    };
  }, [isOpen]);

  if (!isOpen) return null;

  const formatTime = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const remaining = secs % 60;
    return `${mins.toString().padStart(2, "0")}:${remaining.toString().padStart(2, "0")}`;
  };

  return (
    <div
      style={{
        padding: "14px 16px",
        backgroundColor: "var(--bg-surface-elevated)",
        border: "1px solid var(--border-strong)",
        borderRadius: "var(--radius-lg)",
        marginBottom: "10px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: "12px",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
        {/* Pulsing Recording Dot */}
        <div
          style={{
            width: "12px",
            height: "12px",
            borderRadius: "50%",
            backgroundColor: isRecording ? "var(--error)" : "var(--text-muted)",
            animation: isRecording ? "pulse 1.5s infinite" : "none",
          }}
        />

        <div>
          <div style={{ fontWeight: 600, fontSize: "13px", color: "var(--text-primary)" }}>
            {isProcessing ? "Processing Speech & Reasoning..." : isRecording ? "Listening to Telugu / English..." : "Ready"}
          </div>
          <div style={{ fontSize: "11.5px", color: "var(--text-muted)" }}>
            {error ? (
              <span style={{ color: "var(--error)" }}>{error}</span>
            ) : isProcessing ? (
              "Transcribing speech, consulting knowledge base, and generating voice..."
            ) : (
              `Duration: ${formatTime(seconds)} (Click 'Done' to send voice query)`
            )}
          </div>
        </div>
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <button
          className="btn btn-secondary"
          onClick={cancelRecording}
          disabled={isProcessing}
          style={{ padding: "6px 10px", fontSize: "12px" }}
        >
          <CloseIcon size={14} />
          <span>Cancel</span>
        </button>

        {isRecording && (
          <button
            className="btn btn-primary"
            onClick={stopRecording}
            disabled={isProcessing || seconds === 0}
            id="btn-finish-recording"
            style={{ padding: "6px 12px", fontSize: "12px" }}
          >
            <SendIcon size={13} />
            <span>Done</span>
          </button>
        )}
      </div>
    </div>
  );
}
