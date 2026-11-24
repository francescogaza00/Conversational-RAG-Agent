# Conversational RAG Agent: D&D 5e Character Creation Guide

An interactive CLI chatbot that guides users through the creation of a Dungeons & Dragons 5e character. It combines a **LangGraph state machine**, a **hybrid RAG pipeline**, and **Groq**-hosted `gpt-oss` models (free plan) to provide conversational guidance, in Italian, grounded on the official Italian **System Reference Document 5.1**.

---

## Features

- One step at a time: general idea and name → race → class → ability scores → equipment → final sheet
- The player's general idea (role, play style, experience) is saved and used in every later step
- Final sheet rendered in the terminal (Markdown via `rich`) and saved to `schede/<name>.md`
- RAG pipeline grounded on the SRD 5.1 (Italian, 453 pages); options limited to its 9 races and 12 classes
- Hybrid retrieval: BM25 keyword search + Chroma vector search, reranked by FlashRank
- Facts the models got wrong in live tests are cut straight from the manual instead of searched: race and class overviews, each class's hit points, proficiencies, starting equipment and 1st-level features (plus the SRD's only subclass for cleric, sorcerer and warlock), and the armour, weapon and pack tables
- The player picks the ability score method (standard array, point buy, 4d6) and places the values; racial increases, modifiers and hit points are computed in code
- Structured slot extraction from the last exchange: a choice counts only if the player states it or accepts a DM proposal

---

## Architecture

```
User input
    │
    ▼
[analyzer]  ← updates the saved sheet from the last exchange
    │
    ▼  conditional router (first empty step)
┌──────────────────────────────────────────────────────┐
│  pg_name       → general idea and character name     │
│  pg_race       → guides race choice (RAG)            │
│  pg_class      → guides class choice (RAG)           │
│  pg_abilities  → asks the method, then the placement │
│  pg_equipment  → picks the class starting equipment  │
│  pg_summary    → writes the final sheet to schede/   │
└──────────────────────────────────────────────────────┘
```

| Component | Detail |
|-----------|--------|
| **LLM (race, class, scores, equipment)** | `openai/gpt-oss-120b` on Groq, low reasoning |
| **LLM (name step, final sheet)** | `openai/gpt-oss-20b` on Groq, low reasoning |
| **LLM (analyzer)** | `openai/gpt-oss-20b` on Groq, medium reasoning, JSON-schema structured output |
| **Embeddings** | `intfloat/multilingual-e5-small` via HuggingFace (local, multilingual) |
| **Vector store** | Chroma (persisted to `./chroma_db/<fingerprint>`) |
| **Keyword retriever** | BM25 (top-10 results) |
| **Ensemble weights** | BM25 40% + Chroma 60% |
| **Reranker** | FlashRank `ms-marco-MultiBERT-L-12` (local, multilingual), top 6 |
| **Graph framework** | LangGraph `StateGraph` with `MemorySaver` checkpointing |

---

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
curl -LO https://media.dndbeyond.com/compendium-images/srd/5.1/SRD_CC_v5.1_IT.pdf
echo "GROQ_API_KEY=..." > .env         # free key from console.groq.com/keys
.venv/bin/python Dnd_guide.py
.venv/bin/python test_grafo.py          # graph check with stub LLMs, no API calls
```

## Usage

The chatbot opens an interactive session in the terminal. It first asks what you would like to play and how well you know D&D, then walks through name, race, class, ability scores and equipment, one step per message. Once every step is done it writes the final sheet, saves it to `schede/<name>.md` and ends the session.

```
============================================================
 🎲 Benvenuto! Sono il tuo assistente per creare un personaggio di D&D 5e! 🎲
 (Scrivi 'exit' per chiudere la sessione)
============================================================

DM: Hai già un'idea del personaggio? Raccontami cosa ti piacerebbe fare (combattere in prima linea,
    usare la magia, muoverti di nascosto...), se conosci già D&D e, se ce l'hai, il nome.

Tu: conosco poco D&D, mi piacerebbe lanciare magie potenti da lontano
DM: Ottimo! Come si chiama il tuo personaggio?
```

Type `exit` at any prompt to quit early. On the free Groq plan (8,000 tokens per minute per model) a fast exchange can wait 10-20 seconds for the limit to reset.

> **Note:** The Chroma vector store is built on the first run (about 5 minutes on CPU) and persisted under `./chroma_db/<fingerprint>`, where the fingerprint covers the chunk texts and the embedding model: changing the PDF, the chunking or the model builds a new index automatically. Folders of old fingerprints can be deleted by hand.

---

## Project Structure

```
.
├── Dnd_guide.py          # Main application
├── test_grafo.py         # Graph check with stub LLMs
├── requirements.txt
├── SRD_CC_v5.1_IT.pdf    # SRD 5.1, Italian (downloaded, see Setup)
├── chroma_db/            # Persisted vector store (auto-generated)
├── schede/               # Final character sheets (auto-generated, not tracked by git)
├── .env                  # API keys (not tracked by git)
└── README.md
```

---

## License

This project is for personal and educational use.

This work includes material from the System Reference Document 5.1 ("SRD 5.1") by Wizards of the Coast LLC, available at https://dnd.wizards.com/it/resources/systems-reference-document. The SRD 5.1 is licensed under the Creative Commons Attribution 4.0 International License, available at https://creativecommons.org/licenses/by/4.0/legalcode.
