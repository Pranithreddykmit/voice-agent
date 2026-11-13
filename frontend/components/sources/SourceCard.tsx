"use client";

import React, { useState } from "react";
import { FileTextIcon, ChevronDownIcon, ChevronRightIcon } from "../ui/Icons";
import { SourceCitation } from "@/lib/api/types";

interface SourceCardProps {
  source: SourceCitation;
}

export function SourceCard({ source }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className="source-card"
      onClick={() => setExpanded(!expanded)}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          setExpanded(!expanded);
        }
      }}
      aria-expanded={expanded}
    >
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div className="source-title">
          <FileTextIcon size={14} />
          <span>{source.document}</span>
        </div>
        {expanded ? <ChevronDownIcon size={14} /> : <ChevronRightIcon size={14} />}
      </div>

      <div className="source-meta">
        <span>Page {source.page}</span>
        <span>•</span>
        <span>Score: {source.score}</span>
        <span>•</span>
        <span style={{ textTransform: "uppercase", fontSize: "10.5px" }}>{source.retrieval_method}</span>
      </div>

      {expanded && (
        <div className="source-snippet-drawer" onClick={(e) => e.stopPropagation()}>
          {source.snippet}
        </div>
      )}
    </div>
  );
}
