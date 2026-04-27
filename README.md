# D&D 5e Character Creation Guide

An interactive CLI chatbot that guides users through the creation of a Dungeons & Dragons 5e character. It combines a **LangGraph state machine**, a **hybrid RAG pipeline**, and **Google Gemini** models to provide lore-accurate, conversational guidance drawn directly from the D&D 5e manual.

---

## Features

- Step-by-step character creation: name → race → class → final summary
- RAG pipeline grounded on a local D&D 5e PDF manual
- Hybrid retrieval: BM25 keyword search + Chroma vector search, reranked by FlashRank
- Persistent conversation memory across turns (LangGraph `MemorySaver`)
- Structured slot extraction to track user choices reliably
- Guards against re-running once a character is already complete

---

## Architecture

```
User input
    │
    ▼
[analyzer]  ← extracts name / race / class from conversation history
    │
    ▼  conditional router
┌───────────────────────────────────────────────┐
│  pg_name           → asks for character name  │
│  pg_race           → guides race choice (RAG) │
│  pg_class          → guides class choice (RAG)│
│  pg_summary        → generates final sheet    │
│  already_completed → politely blocks replay   │
└───────────────────────────────────────────────┘
```

| Component | Detail |
|-----------|--------|
| **LLM (guide nodes)** | `gemini-2.5-flash` — full reasoning for RAG-enriched responses |
| **LLM (analyzer / summary)** | `gemini-2.5-flash-lite` — lightweight slot extraction and short replies |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace |
| **Vector store** | Chroma (persisted to `./chroma_db`) |
| **Keyword retriever** | BM25 (top-5 results) |
| **Ensemble weights** | BM25 40% + Chroma 60% |
| **Reranker** | FlashRank contextual compression |
| **Graph framework** | LangGraph `StateGraph` with `MemorySaver` checkpointing |

---

## Usage

The chatbot opens an interactive session in the terminal. It will ask for your character's **name**, **race**, and **class** in sequence, drawing context from the D&D 5e manual at each step. Once all three slots are filled, it generates a rich character summary and ends the session.

```
============================================================
 🎲 Welcome! I am your D&D 5e character creation assistant! 🎲
 (Type 'exit' to end the session)
============================================================

You: I want to be a wizard
DM: Great choice! What would you like to name your character?

You: Aldric
DM: Wonderful name. Now let's choose your race...
```

Type `exit` at any prompt to quit early.

> **Note:** The Chroma vector store is built on the first run and persisted to `./chroma_db`. Subsequent runs reuse it, so startup is significantly faster after the first launch.

---

## Project Structure

```
.
├── Dnd_guide.py        # Main application
├── guida_dnd_5e.pdf    # D&D 5e source manual
├── chroma_db/          # Persisted vector store (auto-generated)
├── .env                # API keys (not tracked by git)
└── README.md
```

---

## License

This project is for personal and educational use. D&D 5e content is property of Wizards of the Coast.
