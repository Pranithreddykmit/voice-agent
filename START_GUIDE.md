# Quick Start & User Guide

**A practical guide on how to launch the system, which models/modes to choose for specific tasks, and example test workflows.**

---

## 1. Quick Model & Mode Selection Matrix

### A. Retrieval Modes (in the Top-Right Dropdown of the UI)

| Retrieval Mode | When to Use This Mode | How It Works | Avg Latency |
| :--- | :--- | :--- | :---: |
| **Hybrid (Dense + BM25 + RRF)** *(Recommended / Default)* | **Standard Document Q&A (Telugu, English, & Code-Mixed).** Best overall mode for balancing speed and accuracy across institutional regulations, syllabus documents, and exam queries. | Searches both 1024-dim dense vectors (FastEmbed Multilingual E5) and sparse BM25Plus index, fusing top ranks via Reciprocal Rank Fusion ($k=60$). | **~28 ms** |
| **Hybrid + Reranker** | **High-Precision / Ambiguous Regulatory Queries.** Use when questions have complex conditions or subtle phrasing differences that require deep cross-attention re-scoring. | Runs Hybrid search, then feeds top candidate chunks through `BAAI/bge-reranker-base` cross-encoder to re-order by contextual relevance. | ~900 ms |
| **Dense (E5 Multilingual)** | **Semantic / Conceptual & Paraphrased Questions.** Use when the user's query uses different wording or synonyms in Telugu from the exact words in the PDF. | 1024-dimensional Cosine vector search in Qdrant using `intfloat/multilingual-e5-large`. | ~30 ms |
| **Sparse (BM25Plus)** | **Exact Keyword, Number, Acronym, or Code Queries.** Use when searching for specific section numbers (e.g. *“Section 4.2”*), course codes, fee amounts (e.g. *“Rs. 1000”*), percentages (e.g. *“75%”*), or exact names. | BM25Plus inverted index with Telugu and alphanumeric tokenization. | **< 1 ms** |

---

### B. LLM Provider & Model Selection (`.env`)

| LLM Model | Configuration in `.env` | Best Use Case | Requirements |
| :--- | :--- | :--- | :--- |
| **Google Gemini 1.5 Flash** *(Recommended / Default)* | `LLM_PROVIDER=gemini`<br>`LLM_MODEL=gemini-1.5-flash` | **Primary production reasoning model.** Fast response times, large context window, accurate Telugu translation and structured citation formatting. | `GEMINI_API_KEY=your_key` |
| **Google Gemini 2.0 Flash** | `LLM_PROVIDER=gemini`<br>`LLM_MODEL=gemini-2.0-flash` | Latest high-speed multimodal reasoning. | `GEMINI_API_KEY=your_key` |
| **Google Gemini 1.5 Pro** | `LLM_PROVIDER=gemini`<br>`LLM_MODEL=gemini-1.5-pro` | Deep reasoning on highly complex, multi-page institutional policy documents. | `GEMINI_API_KEY=your_key` |
| **Local Ollama (Llama 3.1 / Gemma 2)** | `LLM_PROVIDER=ollama`<br>`LLM_MODEL=llama3.1` | **100% Offline Execution.** Completely eliminates cloud API calls for air-gapped local environments. | Local Ollama running on `:11434` |
| **Mock LLM Provider** | `LLM_PROVIDER=mock` | Deterministic unit tests and offline testing without any API keys or network connection. | None |

---

### C. Speech Subsystem Modes (`.env`)

| Component & Mode | Configuration in `.env` | How It Works & When to Use |
| :--- | :--- | :--- |
| **Speech-to-Text (STT)** | `STT_ENGINE=whisper_turbo`<br>`STT_MODEL_NAME=openai/whisper-large-v3-turbo` | Runs local Whisper Large v3 Turbo with Telugu language conditioning. Automatically transcribes user voice in Telugu, English, or mixed speech. |
| **TTS: Neural Diffusion + LoRA** *(Best Quality / Demo)* | `TTS_ENGINE=indicf5`<br>`TTS_ADAPTER=models/indicf5_telugu_lora` | Uses genuine `ai4bharat/IndicF5` flow-matching diffusion model with the Epoch 44 Telugu LoRA adapter. Delivers authentic pronunciation of technical loanwords and numbers. *(Note: ~15-20s on Mac CPU; ~0.5s on GPU)* |
| **TTS: Fast Fallback Synthesizer** *(Instant / Dev Mode)* | `TTS_ENGINE=fallback` | Delivers instant 16kHz PCM audio in **~2 milliseconds**. Ideal for rapid UI testing and continuous development without waiting for CPU diffusion. |

---

## 2. Launch Sequence (Step-by-Step)

### Step 1: Configure `.env`
Ensure your local `.env` file exists in the repository root:
```env
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
GEMINI_API_KEY=your_gemini_api_key_here

TTS_ENGINE=indicf5
TTS_ADAPTER=models/indicf5_telugu_lora
```

---

### Step 2: Start Services in 3 Terminals

#### Terminal 1 — Start Local Databases (Docker)
```bash
docker compose up -d
```
* **PostgreSQL:** `localhost:5432` (Conversational memory)
* **Qdrant Vector DB:** `localhost:6333` (Dashboard: [http://localhost:6333/dashboard](http://localhost:6333/dashboard))

#### Terminal 2 — Start FastAPI AI Backend
```bash
# Activate virtual environment
source .venv/bin/activate

# Start server
PYTHONPATH=ai-service uvicorn app.main:app --app-dir ai-service --host 0.0.0.0 --port 8000 --reload
```
* **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

#### Terminal 3 — Start Next.js Frontend
```bash
cd frontend
npm run dev
```
* **Web App:** Open **[http://localhost:3000](http://localhost:3000)**

---

## 3. Example Workflows & Test Prompts

### Task 1: Upload a Knowledge Document
1. Open [http://localhost:3000](http://localhost:3000).
2. Click **"Upload Documents"** in the sidebar.
3. Select and upload [sample_data/academic_regulations.pdf](sample_data/academic_regulations.pdf).
4. Verify that the ingestion badge shows chunks indexed into Qdrant & BM25.

---

### Task 2: Telugu & Code-Mixed Regulatory QA
Test questions grounded in the uploaded document:
* **Telugu Query:**
  > `"సెమిస్టర్ పరీక్షలకు హాజరు కావడానికి కనీసం ఎంత శాతం హాజరు తప్పనిసరి?"`
  * *Expected Answer:* 75% attendance rule with citation referencing `academic_regulations.pdf, Page 1`.
* **Code-Mixed Query:**
  > `"Revaluation fee ఎంత మరియు deadline ఎప్పుడు?"`
  * *Expected Answer:* Rs. 1000 per theory subject and 10-day application deadline from `Page 3`.
* **English Query:**
  > `"What is the grading criteria for Grade A+ and Grade O?"`
  * *Expected Answer:* 10-point scale breakdown from `Page 2`.

---

### Task 3: Arithmetic & Percentage Calculations (Safe AST Tool)
The LangGraph agent detects calculation intent, bypasses document retrieval to save latency, and executes safe AST arithmetic:
* Query: `"What is 15% of 240?"` $\to$ Returns `36`
* Query: `"Calculate (450 * 0.75) + 120"` $\to$ Returns `457.5`
* Notice in the response metadata that zero document chunks were searched.

---

### Task 4: Voice Interaction (Microphone $\to$ STT $\to$ Agent $\to$ TTS)
1. Click the **Microphone** icon in the chat input bar.
2. Speak: *"హాజరు నిబంధనలు ఏమిటి?"* (or any English/Telugu question).
3. Click the **Stop** button.
4. The system transcribes your speech via Whisper STT, reasons with Gemini + Hybrid RAG, synthesizes speech with IndicF5 + LoRA, and automatically plays back the spoken answer.

---

### Task 5: Out-of-Context Refusal (Grounding & Safety)
Ask a question unsupported by the uploaded document:
* Query: `"Who won the 2024 ICC Cricket World Cup?"`
* *Expected Behavior:* The agent detects that retrieved chunks have insufficient relevance score and returns a polite grounded refusal in your query's language rather than hallucinating.

---

## 4. Running Verification & Tests

```bash
# 1. Run full automated test suite (100 unit & integration tests)
.venv/bin/pytest tests/ -v

# 2. Run Full 120-Sentence TTS Domain Benchmark
PYTHONPATH=ai-service python evaluation/evaluate_phase7.py

# 3. Run Hybrid Retrieval Evaluation (Dense vs BM25 vs RRF vs Rerank)
PYTHONPATH=ai-service python evaluation/evaluate.py

# 4. Run LangGraph Multi-Agent Routing Benchmark
PYTHONPATH=ai-service python evaluation/evaluate_agent.py
```
