"use client";

import React from "react";
import { BotIcon, FileTextIcon, CalculatorIcon, MessageSquareIcon } from "../ui/Icons";

interface EmptyStateProps {
  onSelectPrompt: (prompt: string) => void;
}

export function EmptyState({ onSelectPrompt }: EmptyStateProps) {
  const promptExamples = [
    {
      category: "English Document Query",
      icon: <FileTextIcon size={14} />,
      prompt: "What is the minimum attendance requirement for semester examinations?",
    },
    {
      category: "Telugu Document Query",
      icon: <FileTextIcon size={14} />,
      prompt: "సెమిస్టర్ పరీక్షలకు హాజరు కావడానికి కనీసం ఎంత శాతం హాజరు తప్పనిసరి?",
    },
    {
      category: "Code-Mixed Document Query",
      icon: <FileTextIcon size={14} />,
      prompt: "revaluation fee మరియు application deadline ఏమిటి?",
    },
    {
      category: "Safe Calculator Tool",
      icon: <CalculatorIcon size={14} />,
      prompt: "What is 15% of 240?",
    },
  ];

  return (
    <div style={{ padding: "40px 10px", textAlign: "center", maxWidth: "680px", margin: "auto" }}>
      <div
        style={{
          width: "48px",
          height: "48px",
          borderRadius: "var(--radius-md)",
          backgroundColor: "var(--primary-subtle)",
          color: "var(--primary)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          margin: "0 auto 16px",
          border: "1px solid rgba(59, 130, 246, 0.25)",
        }}
      >
        <BotIcon size={24} />
      </div>

      <h2 style={{ fontSize: "19px", fontWeight: 600, color: "var(--text-primary)", marginBottom: "8px" }}>
        Telugu Conversational AI
      </h2>
      <p style={{ color: "var(--text-secondary)", fontSize: "14px", lineHeight: "1.6", marginBottom: "28px" }}>
        Ask questions in <strong>Telugu</strong>, <strong>English</strong>, or <strong>code-mixed</strong> text.
        Answers are grounded with verifiable document citations and multi-agent reasoning.
      </p>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: "10px",
          textAlign: "left",
        }}
      >
        {promptExamples.map((item, idx) => (
          <button
            key={idx}
            onClick={() => onSelectPrompt(item.prompt)}
            className="btn btn-secondary"
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "flex-start",
              padding: "12px 14px",
              height: "auto",
              textAlign: "left",
              backgroundColor: "var(--bg-surface)",
              borderColor: "var(--border-subtle)",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: "6px",
                fontSize: "11px",
                color: "var(--primary)",
                fontWeight: 600,
                marginBottom: "4px",
              }}
            >
              {item.icon}
              <span>{item.category}</span>
            </div>
            <div style={{ fontSize: "13px", color: "var(--text-primary)", lineHeight: "1.4" }}>
              {item.prompt}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
