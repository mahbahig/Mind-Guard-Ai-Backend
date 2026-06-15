# NEURA — Mental Health Companion

An agentic RAG-powered mental health chatbot built with LangGraph, DSM-5-TR, and a dual LLM backend (Google Gemini + fine-tuned Llama 3). Designed as a warm, supportive companion — not a clinical tool.

---

## Architecture

```
User message
     │
     ▼
  Guard Node          <- blocks harmful/abusive input
     │
  Language Node       <- detects English / Arabic
     │
  Crisis Node         <- intercepts suicidal/self-harm signals
     │
  History Node        <- summarises past conversation turns
     │
  Conversational Node <- handles greetings, positive messages
     │
  Router Node         <- classifies: emotional / advice / factual / off_topic
     │
  Rewrite Node        <- rewrites query to DSM-5 search terms (factual only)
     │
  Retrieve Node       <- hybrid BM25 + vector search over DSM-5-TR (factual only)
     │
  Generate Node       <- Llama (Vertex AI) + Gemini fallback
     │
  Postprocess Node    <- strips banned openers, saves to MongoDB
     │
     ▼
  Response + Title
```

## Features

- **LangGraph pipeline** -- 10-node agentic graph with conditional routing
- **RAG over DSM-5-TR** -- hybrid BM25 + vector retrieval, parent-document chunking, relevance guard
- **Dual LLM backend** -- fine-tuned Llama 3 on Vertex AI with Gemini 2.5 Flash fallback
- **Crisis detection** -- dedicated node with Egyptian hotlines, Arabic-aware
- **Off-topic redirect** -- politely declines non-mental-health queries
- **Title generation** -- from-scratch encoder-decoder Transformer trained on 10k pairs
- **Arabic support** -- full pipeline support including crisis response and title generation
- **Evaluation suite** -- 23 test cases, LLM-as-Judge scoring (23/23 passing, avg 0.975)
- **Conversation history** -- MongoDB persistence with session summaries

---

## Project Structure

```
NEURA/
├── src/
│   ├── app.py                  # Streamlit UI
│   ├── api.py                  # FastAPI REST endpoint
│   ├── graph.py                # LangGraph pipeline
│   ├── rag2.py                 # RAG engine + LLM helpers
│   ├── crisis.py               # Crisis detection logic
│   ├── title_generator.py      # Title inference wrapper
│   ├── title_model.py          # Encoder-decoder Transformer
│   ├── hybrid_retriever.py     # BM25 + vector retrieval
│   ├── feedback_scorer.py      # Thumbs up/down logging
│   └── prompts/                # Prompt templates (5 files)
│
├── scripts/
│   ├── evaluate.py             # Evaluation pipeline (LLM-as-Judge)
│   ├── train_title_model.py    # Train the title Transformer
│   └── generate_title_data.py  # Generate training pairs via Gemini
│
├── data/
│   ├── DSM-5.pdf               # Source knowledge base
│   ├── knowledge_base/         # 333 structured markdown files (29 categories)
│   └── training_data/          # Title model training data
│
├── models/title_model/         # Trained weights + vocab
├── tests/                      # Phase test suites
├── evaluation/                 # Eval reports (JSON)
├── chroma_db/                  # ChromaDB vector store
├── parent_store.pkl            # Parent document store
└── logs/                       # Runtime logs (gitignored)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Fill in your API keys — see .env.example for all required variables
```

### 3. Run the app
```bash
streamlit run src/app.py
```

---

## Evaluation

```bash
python scripts/evaluate.py
```

Results are saved to `evaluation/eval_report_<timestamp>.json`.

| Category  | Score |
|-----------|-------|
| emotional | 1.00  |
| crisis    | 1.00  |
| off_topic | 1.00  |
| edge      | 1.00  |
| advice    | 0.95  |
| factual   | 0.92  |
| **avg**   | **0.975** |

---

## Title Model

Train from scratch:
```bash
python scripts/generate_title_data.py   # generate 10k training pairs
python scripts/train_title_model.py     # train the Transformer (80 epochs)
```

Architecture: encoder-decoder Transformer · d=64 · 2 layers · 4 heads · word-level tokenizer · 1.08M parameters

---

## Environment Variables

See [`.env.example`](.env.example) for all required variables:

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Gemini API key (get one at aistudio.google.com) |
| `VERTEX_PROJECT_ID` | GCP project ID |
| `VERTEX_ENDPOINT_ID` | Vertex AI Llama endpoint ID |
| `VERTEX_DEDICATED_URL` | Full Vertex AI prediction URL |
| `VERTEX_LOCATION` | GCP region (e.g. us-central1) |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON |
