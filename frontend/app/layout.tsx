import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Telugu Conversational AI — Multi-Agent RAG Platform",
  description: "Enterprise and research conversational AI platform supporting Telugu & English document knowledge bases, hybrid RRF retrieval, cross-encoder reranking, and multi-agent orchestration.",
  keywords: ["Telugu AI", "RAG", "LangGraph", "Multilingual E5", "Conversational AI"],
  authors: [{ name: "Telugu AI Research Team" }],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Telugu:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
