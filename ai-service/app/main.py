"""
FastAPI Application Entrypoint for the AI Backend Service.
Phase 3: Multi-Agent Conversational AI with LangGraph and PostgreSQL Conversational Memory.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.encoders import jsonable_encoder
import base64

from app import __version__
from app.config import settings
from app.schemas import (
    HealthResponse,
    DocumentUploadResponse,
    QueryRequest,
    QueryResponse,
    ValidationInfo,
    LatencyMetrics,
    ConversationHistoryResponse,
    STTResponse,
    TTSRequest,
    VoiceLatencyMetrics,
    VoiceChatResponse
)
from app.ingestion.pdf_loader import PDFExtractionError
from app.storage.qdrant import QdrantVectorStore, QdrantStorageError
from app.storage.postgres import PostgresManager
from app.generation.llm import LLMGenerationError
from app.rag import RAGPipeline
from app.agents.graph import AgentOrchestrator
from app.speech.audio import decode_and_normalize_audio, AudioProcessingError
from app.speech.stt import get_stt_provider, BaseSTTProvider, STTTranscriptionError
from app.speech.tts import get_tts_provider, BaseTTSProvider, TTSSynthesisError

# Configure Logging & Security Filters
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
# Silence third-party HTTP transport loggers to avoid exposing URLs/headers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger("ai_service")

# Global instances
rag_pipeline: Optional[RAGPipeline] = None
postgres_manager: Optional[PostgresManager] = None
agent_orchestrator: Optional[AgentOrchestrator] = None
stt_provider: Optional[BaseSTTProvider] = None
tts_provider: Optional[BaseTTSProvider] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown routines."""
    global rag_pipeline, postgres_manager, agent_orchestrator, stt_provider, tts_provider
    logger.info(f"Starting {settings.APP_NAME} v{__version__} ({settings.APP_ENV})...")
    
    # Initialize Core Pipelines, Storage & Speech Models
    rag_pipeline = RAGPipeline()
    postgres_manager = PostgresManager()
    agent_orchestrator = AgentOrchestrator(rag_pipeline=rag_pipeline)
    stt_provider = get_stt_provider()
    tts_provider = get_tts_provider()

    qdrant_status = rag_pipeline.vector_store.is_connected()
    pg_status = postgres_manager.is_connected()

    # Log comprehensive development diagnostics
    logger.info("=" * 70)
    logger.info("SYSTEM MODEL INVENTORY & INITIALIZATION STATUS")
    logger.info("=" * 70)
    logger.info(f"1. Primary LLM:      Provider={settings.LLM_PROVIDER} | Model={settings.LLM_MODEL} | Auth=Header (x-goog-api-key)")
    logger.info(f"2. Dense Embeddings: Provider={settings.EMBEDDING_PROVIDER} | Model={settings.EMBEDDING_MODEL} | Dim={rag_pipeline.embedding_provider.dimension}")
    logger.info(f"3. Speech-to-Text:   Engine={settings.STT_ENGINE} | Model={settings.STT_MODEL_NAME} | Device={settings.STT_DEVICE} | LangMode={settings.STT_LANGUAGE or 'auto (LID)'}")
    logger.info(f"4. Text-to-Speech:   Engine={tts_provider.engine_name} | Adapter={tts_provider.adapter_name} | IsFallback={tts_provider.is_fallback}")
    logger.info(f"5. Vector DB:        Engine=Qdrant (:6333) | Status={'ONLINE' if qdrant_status else 'OFFLINE'}")
    logger.info(f"6. Relational DB:    Engine=PostgreSQL (:5432) | Status={'ONLINE' if pg_status else 'OFFLINE'}")
    logger.info("=" * 70)

    yield

    logger.info("Shutting down AI service...")



# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=__version__,
    description="Multi-Agent Telugu Conversational AI - Phase 5 Speech Interface & Multi-Agent RAG",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_pipeline() -> RAGPipeline:
    """Dependency injector for RAG pipeline."""
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = RAGPipeline()
    return rag_pipeline


def get_db() -> PostgresManager:
    """Dependency injector for PostgreSQL manager."""
    global postgres_manager
    if postgres_manager is None:
        postgres_manager = PostgresManager()
    return postgres_manager


def get_orchestrator() -> AgentOrchestrator:
    """Dependency injector for LangGraph orchestrator."""
    global agent_orchestrator, rag_pipeline
    if agent_orchestrator is None:
        if rag_pipeline is None:
            rag_pipeline = RAGPipeline()
        agent_orchestrator = AgentOrchestrator(rag_pipeline=rag_pipeline)
    return agent_orchestrator


def get_stt() -> BaseSTTProvider:
    """Dependency injector for STT provider."""
    global stt_provider
    if stt_provider is None:
        stt_provider = get_stt_provider()
    return stt_provider


def get_tts() -> BaseTTSProvider:
    """Dependency injector for TTS provider."""
    global tts_provider
    if tts_provider is None:
        tts_provider = get_tts_provider()
    return tts_provider


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health and Readiness Check",
    tags=["System"]
)
async def health_check(
    pipeline: RAGPipeline = Depends(get_pipeline),
    db: PostgresManager = Depends(get_db),
    tts: BaseTTSProvider = Depends(get_tts)
):
    """
    Returns operational status of the service, Qdrant & PostgreSQL connectivity,
    active retrieval mode, and model configurations.
    """
    qdrant_healthy = pipeline.vector_store.is_connected()
    postgres_healthy = db.is_connected()
    status_str = "healthy" if (qdrant_healthy and postgres_healthy) else "degraded"

    return HealthResponse(
        status=status_str,
        version=__version__,
        environment=settings.APP_ENV,
        qdrant_connected=qdrant_healthy,
        postgres_connected=postgres_healthy,
        langgraph_active=True,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        embedding_model=settings.EMBEDDING_MODEL,
        retrieval_mode=settings.RETRIEVAL_MODE,
        reranker_model=settings.RERANKER_MODEL,
        llm_provider=settings.LLM_PROVIDER,
        llm_model=settings.LLM_MODEL,
        tts_engine=tts.engine_name,
        tts_adapter=tts.adapter_name,
        tts_adapter_active=tts.is_adapter_active
    )



@app.post(
    "/documents/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest and Index PDF Document",
    tags=["RAG Ingestion"]
)
async def upload_document(
    file: UploadFile = File(..., description="PDF document to parse, chunk, and index into Qdrant & BM25"),
    pipeline: RAGPipeline = Depends(get_pipeline)
):
    """
    Accepts a PDF document, extracts text page-by-page, generates structure-aware chunks,
    computes multilingual dense vector embeddings, and stores them in Qdrant & BM25.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format. Only PDF files are accepted (received '{file.filename}')."
        )

    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{file.filename}' is empty."
            )

        response = pipeline.ingest_pdf(file_bytes=file_bytes, filename=file.filename)
        return response

    except PDFExtractionError as e:
        logger.warning(f"PDF extraction error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except QdrantStorageError as e:
        logger.error(f"Qdrant storage error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Vector storage unavailable: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected ingestion error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Ingestion failed: {str(e)}")


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Query Conversational Agent with LangGraph Orchestration",
    tags=["Multi-Agent Query"]
)
async def query_knowledge_base(
    request: QueryRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    db: PostgresManager = Depends(get_db)
):
    """
    Executes the multi-agent LangGraph workflow:
    - Resolves or creates conversation session.
    - Loads historical context.
    - Routes dynamically across knowledge retrieval, safe calculator tool, casual conversation, or clarification.
    - Validates response grounding and records conversational memory.
    """
    try:
        # 1. Resolve conversation ID and load context
        conv_id = db.get_or_create_conversation(request.conversation_id)
        recent_messages = db.get_recent_messages(conv_id, limit=settings.CONVERSATION_HISTORY_LIMIT)

        # 2. Persist user message
        db.add_message(conversation_id=conv_id, role="user", content=request.query)

        # 3. Execute LangGraph workflow
        agent_state = orchestrator.run(
            query=request.query,
            conversation_id=conv_id,
            messages=recent_messages,
            retrieval_mode=request.retrieval_mode,
            top_k=request.top_k,
            document_id=request.document_id,
            min_score=request.min_score
        )

        final_answer = agent_state.get("final_answer", "")
        intent = agent_state.get("intent", "knowledge_query")

        # 4. Persist assistant response
        db.add_message(
            conversation_id=conv_id,
            role="assistant",
            content=final_answer,
            intent=intent
        )

        timings = agent_state.get("timings", {})
        return QueryResponse(
            query=request.query,
            conversation_id=conv_id,
            intent=intent,
            retrieval_mode=agent_state.get("retrieval_mode", "hybrid"),
            answer=final_answer,
            sources=agent_state.get("citations", []),
            tool_result=agent_state.get("tool_results"),
            validation=ValidationInfo(
                valid=agent_state.get("validation_result", True),
                reason=agent_state.get("validation_reason"),
                retries_used=agent_state.get("retry_count", 0)
            ),
            latency=LatencyMetrics(
                understanding_ms=timings.get("understanding_ms"),
                dense_ms=timings.get("dense_ms"),
                bm25_ms=timings.get("bm25_ms"),
                rrf_ms=timings.get("rrf_ms"),
                rerank_ms=timings.get("rerank_ms"),
                retrieval_ms=timings.get("retrieval_ms", 0.0),
                tool_ms=timings.get("tool_ms"),
                generation_ms=timings.get("generation_ms", 0.0),
                validation_ms=timings.get("validation_ms"),
                total_ms=timings.get("total_ms", 0.0)
            )
        )

    except QdrantStorageError as e:
        logger.error(f"Qdrant retrieval error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Vector retrieval unavailable: {str(e)}")
    except LLMGenerationError as e:
        logger.error(f"LLM generation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"LLM generation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Multi-agent query processing error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Agent workflow failed: {str(e)}")


@app.post(
    "/query/stream",
    summary="Query Conversational Agent with SSE Streaming",
    tags=["Multi-Agent Query"]
)
async def query_knowledge_base_stream(
    request: QueryRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    db: PostgresManager = Depends(get_db)
):
    """
    Streams multi-agent conversational reasoning tokens progressively via Server-Sent Events (SSE).
    Emits 'meta', 'chunk', and 'done' events.
    """
    async def event_generator():
        try:
            # 1. Resolve conversation ID and load context
            conv_id = db.get_or_create_conversation(request.conversation_id)
            recent_messages = db.get_recent_messages(conv_id, limit=settings.CONVERSATION_HISTORY_LIMIT)

            # 2. Persist user message
            db.add_message(conversation_id=conv_id, role="user", content=request.query)

            # 3. Stream agent workflow
            final_answer = ""
            intent = "knowledge_query"

            for event in orchestrator.run_stream(
                query=request.query,
                conversation_id=conv_id,
                messages=recent_messages,
                retrieval_mode=request.retrieval_mode,
                top_k=request.top_k,
                document_id=request.document_id,
                min_score=request.min_score
            ):
                if event["type"] == "done":
                    final_answer = event.get("answer", "")
                    intent = event.get("intent", "knowledge_query")

                yield f"data: {json.dumps(jsonable_encoder(event), ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)

            # 4. Persist assistant message
            if final_answer:
                db.add_message(
                    conversation_id=conv_id,
                    role="assistant",
                    content=final_answer,
                    intent=intent
                )
        except Exception as e:
            logger.error(f"Streaming query error: {str(e)}", exc_info=True)
            err_event = {"type": "error", "detail": f"Streaming failed: {str(e)}"}
            yield f"data: {json.dumps(jsonable_encoder(err_event), ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get(
    "/conversations/{conversation_id}",
    response_model=ConversationHistoryResponse,
    summary="Get Conversation Message History",
    tags=["Conversational Memory"]
)
async def get_conversation_history(
    conversation_id: str,
    db: PostgresManager = Depends(get_db)
):
    """
    Returns the chronologically ordered message history for a given conversation session.
    """
    messages = db.get_conversation_history(conversation_id)
    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        messages=messages,
        total_messages=len(messages)
    )


# ------------------------------------------------------------------------------
# Phase 5 Speech Endpoints (STT, TTS, and Composite Voice-Chat)
# ------------------------------------------------------------------------------

@app.post(
    "/api/stt",
    response_model=STTResponse,
    summary="Transcribe Speech Audio to Text (STT)",
    tags=["Voice Subsystem"]
)
async def speech_to_text(
    file: UploadFile = File(..., description="Audio recording file (WAV, WebM, OGG, FLAC)"),
    stt: BaseSTTProvider = Depends(get_stt)
):
    """
    Accepts raw audio recorded in the browser, downmixes to mono, resamples to 16kHz,
    and executes speech-to-text transcription.
    """
    try:
        audio_bytes = await file.read()
        audio_array, sample_rate, duration_sec = decode_and_normalize_audio(audio_bytes, target_sr=16000)

        transcript, language, latency_ms = await asyncio.to_thread(
            stt.transcribe, audio_array, sample_rate=sample_rate
        )

        return STTResponse(
            transcript=transcript,
            language=language,
            engine_used=settings.STT_ENGINE,
            is_fallback=False,
            duration_sec=round(duration_sec, 2),
            latency_ms=latency_ms
        )
    except AudioProcessingError as e:
        logger.warning(f"Audio processing error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except STTTranscriptionError as e:
        logger.error(f"STT transcription error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected STT error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Speech transcription failed: {str(e)}")


@app.post(
    "/api/tts",
    summary="Synthesize Text to Speech Audio (TTS)",
    tags=["Voice Subsystem"]
)
async def text_to_speech(
    request: TTSRequest,
    tts: BaseTTSProvider = Depends(get_tts)
):
    """
    Synthesizes input text to 16kHz PCM WAV audio stream.
    """
    try:
        wav_bytes, latency_ms = await asyncio.to_thread(
            tts.synthesize,
            text=request.text,
            voice_id=request.voice_id,
            language=request.language
        )

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=speech.wav",
                "X-TTS-Latency-Ms": str(latency_ms),
                "X-TTS-Engine": tts.engine_name,
                "X-TTS-Adapter": tts.adapter_name,
                "X-TTS-Is-Fallback": str(tts.is_fallback)
            }
        )
    except TTSSynthesisError as e:
        logger.warning(f"TTS synthesis error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected TTS error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Text-to-speech synthesis failed: {str(e)}")


@app.post(
    "/api/voice-chat",
    response_model=VoiceChatResponse,
    summary="End-to-End Voice Conversation (STT -> LangGraph Agent -> TTS)",
    tags=["Voice Subsystem"]
)
async def voice_chat(
    file: UploadFile = File(..., description="Audio recording from user"),
    conversation_id: Optional[str] = None,
    retrieval_mode: Optional[str] = None,
    stt: BaseSTTProvider = Depends(get_stt),
    tts: BaseTTSProvider = Depends(get_tts),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    db: PostgresManager = Depends(get_db)
):
    """
    Executes single-roundtrip end-to-end voice conversational flow:
    1. Normalizes and transcribes user audio via STT.
    2. Executes multi-agent LangGraph workflow with PostgreSQL conversation memory context.
    3. Synthesizes assistant answer to 16kHz PCM WAV audio.
    4. Returns structured JSON containing transcript, grounded answer, citations, tool outputs, and base64 audio.
    """
    import time
    total_start = time.perf_counter()

    try:
        # 1. Decode & Transcribe Audio
        audio_bytes = await file.read()
        audio_array, sample_rate, _ = decode_and_normalize_audio(audio_bytes, target_sr=16000)
        query_text, _, stt_ms = await asyncio.to_thread(
            stt.transcribe, audio_array, sample_rate=sample_rate
        )

        if not query_text or not query_text.strip():
            query_text = "..."

        # 2. Resolve Conversation Session & Context
        conv_id = db.get_or_create_conversation(conversation_id)
        recent_messages = db.get_recent_messages(conv_id, limit=settings.CONVERSATION_HISTORY_LIMIT)

        # Persist user voice transcript
        db.add_message(conversation_id=conv_id, role="user", content=query_text)

        # 3. LangGraph Multi-Agent Orchestration
        agent_start = time.perf_counter()
        agent_state = await asyncio.to_thread(
            orchestrator.run,
            query=query_text,
            conversation_id=conv_id,
            messages=recent_messages,
            retrieval_mode=retrieval_mode
        )
        agent_ms = (time.perf_counter() - agent_start) * 1000.0

        final_answer = agent_state.get("final_answer", "")
        intent = agent_state.get("intent", "knowledge_query")

        # Persist assistant response
        db.add_message(
            conversation_id=conv_id,
            role="assistant",
            content=final_answer,
            intent=intent
        )

        # 4. Text-to-Speech Synthesis
        wav_bytes, tts_ms = await asyncio.to_thread(
            tts.synthesize,
            text=final_answer,
            voice_id=settings.TTS_DEFAULT_VOICE
        )
        audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")

        total_ms = (time.perf_counter() - total_start) * 1000.0

        return VoiceChatResponse(
            query_transcript=query_text,
            conversation_id=conv_id,
            intent=intent,
            answer=final_answer,
            sources=agent_state.get("citations", []),
            tool_result=agent_state.get("tool_results"),
            validation=ValidationInfo(
                valid=agent_state.get("validation_result", True),
                reason=agent_state.get("validation_reason"),
                retries_used=agent_state.get("retry_count", 0)
            ),
            audio_base64=audio_base64,
            stt_engine=settings.STT_ENGINE,
            tts_engine=tts.engine_name,
            tts_adapter=tts.adapter_name,
            is_fallback=tts.is_fallback,
            latency=VoiceLatencyMetrics(
                stt_ms=round(stt_ms, 2),
                agent_ms=round(agent_ms, 2),
                tts_ms=round(tts_ms, 2),
                total_ms=round(total_ms, 2)
            )
        )


    except AudioProcessingError as e:
        logger.warning(f"Voice chat audio error: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Voice chat pipeline failure: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Voice chat error: {str(e)}")


@app.post(
    "/api/voice-chat/stream",
    summary="End-to-End Voice Conversation with SSE Streaming",
    tags=["Voice Subsystem"]
)
async def voice_chat_stream(
    file: UploadFile = File(..., description="Audio recording from user"),
    conversation_id: Optional[str] = None,
    retrieval_mode: Optional[str] = None,
    stt: BaseSTTProvider = Depends(get_stt),
    tts: BaseTTSProvider = Depends(get_tts),
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    db: PostgresManager = Depends(get_db)
):
    """
    Executes voice conversational flow with SSE progressive text streaming:
    1. Transcribes user audio and emits STT transcript event.
    2. Streams agent draft text tokens progressively.
    3. Synthesizes complete answer audio using IndicF5 TTS on completion.
    4. Emits 'done' event with final text, audio base64, and latency breakdown.
    """
    import time
    total_start = time.perf_counter()

    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to read audio: {str(e)}")

    async def event_generator():
        try:
            # 1. STT Transcription
            audio_array, sample_rate, _ = decode_and_normalize_audio(audio_bytes, target_sr=16000)
            query_text, lang, stt_ms = await asyncio.to_thread(
                stt.transcribe, audio_array, sample_rate=sample_rate
            )
            if not query_text or not query_text.strip():
                query_text = "..."

            # Emit STT transcript event immediately
            stt_event = {
                "type": "stt",
                "transcript": query_text,
                "language": lang,
                "latency_ms": round(stt_ms, 2)
            }
            yield f"data: {json.dumps(jsonable_encoder(stt_event), ensure_ascii=False)}\n\n"
            await asyncio.sleep(0)

            # 2. Resolve Conversation Session & Persist user voice transcript
            conv_id = db.get_or_create_conversation(conversation_id)
            recent_messages = db.get_recent_messages(conv_id, limit=settings.CONVERSATION_HISTORY_LIMIT)
            db.add_message(conversation_id=conv_id, role="user", content=query_text)

            # 3. Stream agent workflow
            final_answer = ""
            intent = "knowledge_query"
            done_payload = None

            agent_start = time.perf_counter()
            for event in orchestrator.run_stream(
                query=query_text,
                conversation_id=conv_id,
                messages=recent_messages,
                retrieval_mode=retrieval_mode
            ):
                if event["type"] == "done":
                    final_answer = event.get("answer", "")
                    intent = event.get("intent", "knowledge_query")
                    done_payload = event
                else:
                    yield f"data: {json.dumps(jsonable_encoder(event), ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0)

            agent_ms = (time.perf_counter() - agent_start) * 1000.0

            # Persist assistant message
            if final_answer:
                db.add_message(
                    conversation_id=conv_id,
                    role="assistant",
                    content=final_answer,
                    intent=intent
                )

            # 4. Text-to-Speech Synthesis of final complete text (unchanged IndicF5 TTS behavior)
            tts_start = time.perf_counter()
            wav_bytes, tts_ms = await asyncio.to_thread(
                tts.synthesize,
                text=final_answer,
                voice_id=settings.TTS_DEFAULT_VOICE
            )
            audio_base64 = base64.b64encode(wav_bytes).decode("utf-8")
            total_ms = (time.perf_counter() - total_start) * 1000.0

            # 5. Emit final done event
            final_done_event = {
                "type": "done",
                "query_transcript": query_text,
                "conversation_id": conv_id,
                "intent": intent,
                "answer": final_answer,
                "sources": done_payload.get("sources", []) if done_payload else [],
                "tool_result": done_payload.get("tool_result") if done_payload else None,
                "validation": done_payload.get("validation") if done_payload else {"valid": True, "reason": None, "retries_used": 0},
                "audio_base64": audio_base64,
                "stt_engine": settings.STT_ENGINE,
                "tts_engine": tts.engine_name,
                "tts_adapter": tts.adapter_name,
                "is_fallback": tts.is_fallback,
                "latency": {
                    "stt_ms": round(stt_ms, 2),
                    "agent_ms": round(agent_ms, 2),
                    "tts_ms": round(tts_ms, 2),
                    "total_ms": round(total_ms, 2)
                }
            }
            yield f"data: {json.dumps(jsonable_encoder(final_done_event), ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error(f"Voice chat streaming error: {str(e)}", exc_info=True)
            err_event = {"type": "error", "detail": f"Voice streaming failed: {str(e)}"}
            yield f"data: {json.dumps(jsonable_encoder(err_event), ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
