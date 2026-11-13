# System Architecture Specification

**Project:** Multi-Agent Telugu Conversational AI with Retrieval-Augmented Generation and Adaptive Speech Synthesis  
**Architecture Version:** 2.2.0 (Phase 8 Final Demo & Research Release)  
**Execution Environment:** Local-Only (FastAPI + Next.js + PostgreSQL + Qdrant + Local PyTorch Models)

---

## 1. Overall System Architecture

```mermaid
flowchart TD
    User([User / Web Browser]) <-->|HTTP / JSON / Audio WAV| NextJS[Next.js Frontend :3000\nChat UI & VoiceRecorder]
    NextJS <-->|REST API| FastAPI[FastAPI Gateway :8000\nSpeech & Agent Endpoints]

    subgraph Backend [FastAPI Multi-Agent Core]
        FastAPI --> LG[LangGraph Orchestrator]
        
        subgraph Workflow [LangGraph State Workflow]
            LG --> QU[Query Understanding & Intent Classifier]
            QU --> Route{Intent Router}
            
            Route -->|Knowledge Query| Ret[Hybrid Retrieval Engine]
            Route -->|Calculation| Calc[AST Safe Calculator Tool]
            Route -->|Clarification / Chit-Chat| GenDirect[Direct Generation Node]
            
            subgraph Retrieval [Phase 2 Advanced Hybrid Retrieval]
                Ret --> Dense[Dense Retrieval\nMultilingual E5]
                Ret --> Sparse[Sparse Retrieval\nBM25Plus]
                Dense --> RRF[Reciprocal Rank Fusion\nk=60]
                Sparse --> RRF
                RRF --> Rerank[Cross-Encoder Reranker\nbge-reranker-base]
            end
            
            Rerank --> Synth[Grounded Response Synthesis]
            Calc --> Synth
            GenDirect --> Val[Grounding & Citation Validator]
            Synth --> Val
            
            Val -->|Valid Grounding| Res[Final Structured Answer]
            Val -->|Ungrounded / Low Score| Retry[Bounded Retry / Fallback]
            Retry --> Res
        end
        
        Res --> TTS[TTS Synthesis Engine\nIndicF5 + Telugu LoRA]
    end

    subgraph Storage [Local Storage Layer]
        QU <-->|Session History| PG[(PostgreSQL :5432\nConversational Memory)]
        Synth <-->|Persist Messages| PG
        Dense <-->|Vector Cosine Search| Qdrant[(Qdrant Vector DB :6333\ntelugu_documents)]
    end
```

---

## 2. End-to-End Voice Pipeline

```mermaid
flowchart LR
    Mic([Microphone]) --> Prep[Audio Preprocessing\nMono 16kHz PCM]
    Prep --> STT[Whisper STT Provider\nopenai/whisper-large-v3-turbo]
    STT --> Agent[LangGraph Multi-Agent\nHybrid RAG & Memory]
    Agent --> SynthText[Validated Response Text]
    SynthText --> TTS[IndicF5 Neural Synthesizer\nai4bharat/IndicF5]
    LoRA[LoRA Adapter\nmodels/indicf5_telugu_lora\nr=8, a=16, 2.52M params] -.->|PEFT Attach| TTS
    TTS --> AudioOut[24kHz / 16kHz PCM WAV Audio]
    AudioOut --> Speaker([Audio Playback])
```

---

## 3. Ingestion & RAG Data Flow

```mermaid
flowchart TD
    PDF[Institutional PDF Document] --> PyMuPDF[PyMuPDF Page Parser]
    PyMuPDF --> Chunker[Structure-Aware Telugu Chunker\n600 chars, 120 overlap]
    Chunker --> Chunks[Document Chunks]
    
    Chunks --> E5[Multilingual E5 Embedder\nintfloat/multilingual-e5-large]
    Chunks --> BM25Index[BM25Plus Sparse Inverted Index]
    
    E5 --> Vectors[1024-dim Dense Vectors]
    Vectors --> Qdrant[(Qdrant Collection\ntelugu_documents\nDistance: Cosine)]
    BM25Index --> DiskBM25[(In-Memory / Serialized BM25 Corpus)]
```

---

## 4. LoRA Training & Live Evaluation Pipeline

```mermaid
flowchart TD
    subgraph DatasetPreparation [Phase 6B Multi-Source Dataset]
        IndicTTS[IndicTTS Female\n60 samples, 7.04 min] --> Assembly[Hybrid Dataset Pipeline]
        SYSPINF[SYSPIN Female\n40 samples, 5.76 min] --> Assembly
        SYSPINM[SYSPIN Male\n40 samples, 5.07 min] --> Assembly
        Assembly --> Split[Stratified Leakage-Free Split\nTrain: 112, Val: 14, Test: 14]
    end

    subgraph ColabTraining [Colab Tesla T4 CUDA Acceleration]
        Split -->|Train & Val Splits| Train[LoRA Fine-Tuning\nAdamW, Cosine Warmup, 50 Epochs]
        BaseIndicF5[Base IndicF5 335.9M DiT\n100% Frozen] --> Train
        LoRATargets[LoRA Projections: to_q, to_k, to_v,\nto_out.0, ff.ff.0.0, ff.ff.2] --> Train
        Train --> BestCkpt[Epoch 44 Best Checkpoint\nVal Loss: 0.284200]
        BestCkpt --> Adapter[Export Adapter\nmodels/indicf5_telugu_lora\n10.1 MB, 2.52M weights (0.7511%)]
    end


    subgraph EvaluationAudit [Phase 6D & 7 Live Whisper Benchmarks]
        Adapter --> LocalTTS[Local FastAPI Integration\nTTS_ADAPTER=models/indicf5_telugu_lora]
        Manifest[120-Sentence Manifest & 14-sample Test Set] --> GenWAV[Live Physical WAV Synthesis]
        GenWAV --> LiveWhisper[Live Whisper Decoding\nwhisper-large-v3-turbo]
        LiveWhisper --> MetricEval[CER / WER / Terminology Evaluation]
    end
```

---

## 5. Relational Database Schema (PostgreSQL)

The relational database maintains persistent multi-turn conversational memory and session tracking.

### 5.1 Tables & Relationships

```sql
-- Conversations Table
CREATE TABLE conversations (
    id VARCHAR(36) PRIMARY KEY,                  -- UUID v4 string
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, -- UTC creation timestamp
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL  -- UTC last update timestamp
);

-- Messages Table
CREATE TABLE messages (
    id VARCHAR(36) PRIMARY KEY,                  -- UUID v4 string
    conversation_id VARCHAR(36) NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,                   -- 'user' | 'assistant'
    content TEXT NOT NULL,                       -- Message text in Telugu / English
    intent VARCHAR(50) NULL,                     -- Classified query intent
    created_at TIMESTAMP WITH TIME ZONE NOT NULL -- UTC creation timestamp
);

-- Indexes
CREATE INDEX ix_messages_conversation_id ON messages (conversation_id);
CREATE INDEX ix_messages_created_at ON messages (created_at);
```

### 5.2 Storage Fallback
If the PostgreSQL Docker container is offline or unreachable, `PostgresManager` initializes an in-memory SQLite database (`sqlite:///:memory:`) dynamically to allow uninterrupted development and local test execution.

---

## 6. Vector Database Configuration (Qdrant)

* **Collection Name:** `telugu_documents` (configurable via `QDRANT_COLLECTION_NAME`)
* **Vector Dimension:** `1024` (matches `intfloat/multilingual-e5-large`)
* **Distance Metric:** `Cosine`
* **Payload Fields:**
  * `document_id` (string)
  * `chunk_id` (string)
  * `source` (string, filename)
  * `page` (integer, 1-indexed)
  * `chunk_index` (integer, 0-indexed)
  * `text` (string, chunk text)
  * `char_count` (integer)
