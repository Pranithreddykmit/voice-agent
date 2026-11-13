"""
Pydantic Data Models and API Schemas.
Extended for Phase 3 Multi-Agent LangGraph System and Conversational Memory.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall service status ('healthy' or 'degraded')")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Runtime environment")
    qdrant_connected: bool = Field(..., description="Whether Qdrant vector database is reachable")
    postgres_connected: bool = Field(..., description="Whether PostgreSQL relational database is reachable")
    langgraph_active: bool = Field(True, description="Whether the LangGraph multi-agent orchestrator is initialized")
    embedding_provider: str = Field(..., description="Active embedding provider")
    embedding_model: str = Field(..., description="Active embedding model")
    retrieval_mode: str = Field(..., description="Default active retrieval mode ('dense', 'bm25', 'hybrid', 'hybrid_rerank')")
    reranker_model: str = Field(..., description="Active cross-encoder reranker model")
    llm_provider: str = Field(..., description="Active LLM provider")
    llm_model: str = Field(..., description="Active LLM model")
    tts_engine: str = Field("indicf5", description="Active TTS engine ('indicf5' or 'fallback')")
    tts_adapter: str = Field("none", description="Active TTS LoRA adapter ('none' or adapter path)")
    tts_adapter_active: bool = Field(False, description="Whether the fine-tuned LoRA adapter is actively loaded")


class DocumentUploadResponse(BaseModel):
    document_id: str = Field(..., description="Unique deterministic or generated identifier for the document")
    filename: str = Field(..., description="Original filename of the uploaded document")
    total_pages: int = Field(..., description="Total extracted pages from the PDF")
    total_chunks: int = Field(..., description="Total structured chunks created and indexed")
    collection_name: str = Field(..., description="Target Qdrant collection name")
    status: str = Field("success", description="Status of the ingestion operation")
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="User query in Telugu, English, or mixed text")
    conversation_id: Optional[str] = Field(None, description="Optional conversation ID to maintain session memory")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of top context chunks to retrieve")
    retrieval_mode: Optional[str] = Field(None, description="Retrieval mode: 'dense', 'bm25', 'hybrid', or 'hybrid_rerank'")
    document_id: Optional[str] = Field(None, description="Optional document ID filter to scope search to a single document")
    min_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score threshold")


class SourceCitation(BaseModel):
    document: str = Field(..., description="Source document filename")
    page: int = Field(..., description="1-indexed page number in the source document")
    score: float = Field(..., description="Relevance / similarity / RRF / reranker score of the retrieved chunk")
    chunk_id: str = Field(..., description="Unique identifier of the chunk")
    snippet: str = Field(..., description="Excerpt from the retrieved chunk for verification")
    retrieval_method: str = Field("dense", description="Method that retrieved or ranked this chunk ('dense', 'bm25', 'rrf', 'rerank')")
    rank: int = Field(1, description="Rank in the final retrieved result list (1-indexed)")


class LatencyMetrics(BaseModel):
    understanding_ms: Optional[float] = Field(None, description="Time spent on query understanding and intent routing (ms)")
    dense_ms: Optional[float] = Field(None, description="Time spent on dense vector search (ms)")
    bm25_ms: Optional[float] = Field(None, description="Time spent on BM25 sparse search (ms)")
    rrf_ms: Optional[float] = Field(None, description="Time spent on Reciprocal Rank Fusion (ms)")
    rerank_ms: Optional[float] = Field(None, description="Time spent on cross-encoder reranking (ms)")
    retrieval_ms: float = Field(0.0, description="Total time spent across all retrieval stages (ms)")
    tool_ms: Optional[float] = Field(None, description="Time spent on tool execution (ms)")
    generation_ms: float = Field(..., description="Time spent executing LLM generation / synthesis (ms)")
    validation_ms: Optional[float] = Field(None, description="Time spent validating response (ms)")
    total_ms: float = Field(..., description="Total end-to-end request processing time in milliseconds")


class ValidationInfo(BaseModel):
    valid: bool = Field(True, description="Whether the response passed grounding and citation validation")
    reason: Optional[str] = Field(None, description="Detailed explanation of validation outcome")
    retries_used: int = Field(0, description="Number of retry attempts executed")


class QueryResponse(BaseModel):
    query: str = Field(..., description="Original input query")
    conversation_id: str = Field("conv_default", description="Session identifier for multi-turn conversational memory")
    intent: str = Field("knowledge_query", description="Detected intent: 'knowledge_query', 'casual_query', 'tool_query', 'ambiguous_query'")
    retrieval_mode: str = Field("hybrid", description="The retrieval mode executed for this query")
    answer: str = Field(..., description="Grounded response generated strictly from retrieved context or tool output")
    sources: List[SourceCitation] = Field(default_factory=list, description="List of source citations supporting the answer")
    tool_result: Optional[Dict[str, Any]] = Field(None, description="Structured output from tool execution if a tool was invoked")
    validation: ValidationInfo = Field(default_factory=ValidationInfo, description="Validation status and audit trail")
    latency: LatencyMetrics = Field(..., description="Performance latency breakdown in milliseconds")


class DocumentChunk(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    page: int
    chunk_index: int
    text: str
    char_count: int


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    page: int
    chunk_index: int
    text: str
    score: float
    retrieval_method: str = "dense"
    rank: int = 1


class MessageSchema(BaseModel):
    id: str = Field(..., description="Unique message identifier")
    conversation_id: str = Field(..., description="Associated conversation identifier")
    role: str = Field(..., description="Role of the speaker ('user' or 'assistant')")
    content: str = Field(..., description="Text content of the message")
    intent: Optional[str] = Field(None, description="Detected intent of the query/response")
    created_at: str = Field(..., description="ISO 8601 timestamp of message creation")


class ConversationHistoryResponse(BaseModel):
    conversation_id: str = Field(..., description="Conversation identifier")
    messages: List[MessageSchema] = Field(default_factory=list, description="Ordered list of historical messages")
    total_messages: int = Field(0, description="Total number of messages in the conversation")


class STTResponse(BaseModel):
    transcript: str = Field(..., description="Transcribed text from speech audio")
    language: str = Field("te", description="Detected or configured language code (e.g. 'te', 'en')")
    engine_used: str = Field(..., description="The ASR engine that processed the audio")
    is_fallback: bool = Field(False, description="Whether a mock/fallback STT was used")
    duration_sec: float = Field(..., description="Duration of processed audio in seconds")
    latency_ms: float = Field(..., description="ASR transcription latency in milliseconds")


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to synthesize to speech audio")
    voice_id: str = Field("default", description="Reference voice identifier")
    language: str = Field("te", description="Target language code (default 'te')")


class VoiceLatencyMetrics(BaseModel):
    stt_ms: float = Field(..., description="Time spent on speech-to-text transcription (ms)")
    agent_ms: float = Field(..., description="Time spent on LangGraph retrieval and reasoning (ms)")
    tts_ms: float = Field(..., description="Time spent on text-to-speech synthesis (ms)")
    total_ms: float = Field(..., description="Total end-to-end voice roundtrip latency (ms)")


class VoiceChatResponse(BaseModel):
    query_transcript: str = Field(..., description="Transcribed query from voice audio")
    conversation_id: str = Field(..., description="Session identifier for conversational memory continuity")
    intent: str = Field(..., description="Detected intent of the query")
    answer: str = Field(..., description="Synthesized text response")
    sources: List[SourceCitation] = Field(default_factory=list, description="Verified document sources")
    tool_result: Optional[Dict[str, Any]] = Field(None, description="Structured tool calculation output")
    validation: ValidationInfo = Field(default_factory=ValidationInfo, description="Validation status and audit")
    audio_base64: str = Field(..., description="Base64 encoded 16kHz PCM WAV audio of the assistant response")
    stt_engine: str = Field("whisper_turbo", description="Speech-to-text engine used")
    tts_engine: str = Field("fallback_acoustic", description="Text-to-speech engine used")
    tts_adapter: str = Field("none", description="Active TTS LoRA adapter used")
    is_fallback: bool = Field(False, description="Whether fallback TTS was used")
    latency: VoiceLatencyMetrics = Field(..., description="End-to-end voice latency breakdown")
