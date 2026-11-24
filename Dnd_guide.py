import os
import re
import hashlib
import warnings
from pathlib import Path

# Startup noise that the player doesn't need to see. Must come before the imports below.
# ponytail: silenced, not fixed; the real fix is moving PyMuPDFLoader, BM25Retriever and
# FlashrankRerank to standalone packages before langchain-community stops working.
warnings.filterwarnings("ignore", message=r".*langchain-community. is being sunset")
from transformers.utils import logging as transformers_logging
transformers_logging.disable_progress_bar()  # the "Loading weights" bar of the embedding model

from typing import Optional
from typing_extensions import TypedDict, Annotated
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from flashrank import Ranker
from rich.console import Console
from rich.markdown import Markdown
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent  # files are found wherever the script is launched from

load_dotenv(BASE_DIR / ".env")  # reads .env file from the project root

key = os.getenv("GROQ_API_KEY")
if not key:
    raise ValueError("GROQ_API_KEY not found. Check your .env file.")

# Groq free plan: gpt-oss models only, each with its own quota (8,000 tokens/minute),
# so two different models split the load. The reasoning comes back in a separate
# field, never in the reply text. Low effort for the models that write text: at medium,
# the small one used its whole token budget reasoning in 1 run out of 4 and answered nothing.
# Full model for RAG-enriched responses
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=key,
    temperature=0,
    reasoning_effort="low"
)

# Lightweight model for slot extraction and short replies
llm_lite = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=key,
    temperature=0,
    reasoning_effort="low"
)

# Slot extraction only: short output, so medium effort is affordable here, and it
# matters (on the test exchanges for the analyzer, low got 5 right out of 8, medium 6)
extractor_llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=key,
    temperature=0,
    reasoning_effort="medium"
)

# The only races and classes in SRD 5.1: without this list the model proposes
# options from its own memory (subclasses, races from other books) that the manual doesn't have.
SRD_RACES   = "Dragonide, Elfo, Gnomo, Halfling, Mezzelfo, Mezzorco, Nano, Tiefling, Umano"
SRD_CLASSES = "Barbaro, Bardo, Chierico, Druido, Guerriero, Ladro, Mago, Monaco, Paladino, Ranger, Stregone, Warlock"


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]
    pg_preferences: Optional[str]  # general idea: role in the party, play style, experience
    pg_name:      Optional[str]
    pg_race:      Optional[str]
    pg_class:     Optional[str]
    pg_abilities: Optional[str]
    pg_equipment: Optional[str]
    completed: bool  # tells the CLI loop the sheet is done


# The creation steps, in order: the router asks for the first one still empty.
STEPS = [
    ("pg_name",      "Nome"),
    ("pg_race",      "Razza"),
    ("pg_class",     "Classe"),
    ("pg_abilities", "Punteggi di caratteristica"),
    ("pg_equipment", "Equipaggiamento"),
]


def sheet_text(state, with_preferences: bool = True) -> str:
    """The choices saved so far, one per line: this, not the whole chat, is what
    carries the character from one step to the next."""
    lines = [f"- {label}: {state.get(slot) or 'da scegliere'}" for slot, label in STEPS]
    if with_preferences:
        lines.append(f"- Preferenze del giocatore: {state.get('pg_preferences') or 'non ancora espresse'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structured output schema for slot extraction
# ---------------------------------------------------------------------------

# Optional: the model must be able to answer null. With plain `str` a null raised
# ValidationError, and the model tended to fill the gap with "Unknown" instead.
class Pg_Extraction(BaseModel):
    pg_preferences: Optional[str] = Field(default=None, description="Solo preferenze NUOVE espresse in questa risposta: cosa il giocatore vorrebbe fare o essere (ruolo nel gruppo, stile di gioco) e, se la dice, la sua esperienza con D&D (per esempio 'conosce poco D&D'), in una frase. Un dubbio fra opzioni o una domanda non sono preferenze, e nemmeno la scelta di nome, razza, classe o equipaggiamento: quella va nel suo campo. null se non ce ne sono.")
    pg_name:  Optional[str] = Field(default=None, description="Nome del personaggio scelto dal giocatore; null se non l'ha scelto.")
    pg_race:  Optional[str] = Field(default=None, description=f"Nome italiano esatto della razza, uno fra: {SRD_RACES}. Solo se il giocatore la sceglie o accetta la proposta del DM; null altrimenti.")
    pg_class: Optional[str] = Field(default=None, description=f"Nome italiano esatto della classe, uno fra: {SRD_CLASSES}. Solo se il giocatore la nomina come scelta o accetta la proposta del DM; un desiderio generico ('combattere in prima linea') non è la scelta di una classe. null altrimenti.")
    pg_equipment: Optional[str] = Field(default=None, description="Equipaggiamento iniziale scelto dal giocatore fra le alternative del DM, come elenco breve; null se non l'ha scelto.")


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"  # Italian-capable, reads up to 512 tokens per chunk

# Legal notice + title + page number printed on every SRD page: left in, it made
# hundreds of chunks look alike and pushed the real sections out of the results.
SRD_FOOTER = re.compile(r"Rivendita vietata\..*?personale\.\s*Systems Reference Document 5\.1\s*\d+\s*", re.S)


def load_documents(pdf_paths: list[str]):
    """Loads the PDFs one document per page, without the SRD footer."""
    documents = []
    for path in pdf_paths:
        try:
            loader = PyMuPDFLoader(path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {path}: {e}")
    for doc in documents:
        doc.page_content = SRD_FOOTER.sub("", doc.page_content)
    return documents


def advanced_local_pdf_retriever(documents, persist_root: Path = BASE_DIR / "chroma_db"):
    """
    Builds a hybrid retriever (BM25 + Chroma) with contextual reranking via
    Flashrank. Skips rebuilding the vectorstore when a complete copy built from
    the same chunks and embedding model exists on disk.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No chunks generated. Verify the input PDFs.")

    # e5 models expect these prefixes to tell stored passages from user queries
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"prompt": "passage: ", "normalize_embeddings": True},
        query_encode_kwargs={"prompt": "query: ", "normalize_embeddings": True},
    )

    # One folder per (chunk texts, embedding model): changing the PDF, the cleaning,
    # the splitting or the model builds a new index instead of silently reusing old vectors.
    fingerprint = hashlib.md5(EMBEDDING_MODEL.encode())
    for chunk in chunks:
        fingerprint.update(chunk.page_content.encode())
    persist_dir = persist_root / fingerprint.hexdigest()[:12]

    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings
    )
    # Rebuild when empty or left incomplete by an interrupted first run
    if len(vectorstore.get(include=[])["ids"]) != len(chunks):
        print("Building the vector index (first run only, a few minutes)...")
        vectorstore.reset_collection()
        vectorstore.add_documents(chunks)

    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    # Combine keyword (BM25) and semantic (Chroma) search
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6]
    )

    # default 3 chunks are too few for a 450-page manual. Own Ranker with WARNING:
    # the default one raises the whole program's logging to INFO, so every HTTP
    # request to the LLM got printed in the chat.
    compressor = FlashrankRerank(
        client=Ranker(model_name="ms-marco-MultiBERT-L-12", log_level="WARNING"),
        top_n=6
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    return compression_retriever


dnd_manuals = [str(BASE_DIR / "SRD_CC_v5.1_IT.pdf")]
srd_documents = load_documents(dnd_manuals)
SRD_TEXT = "".join(doc.page_content for doc in srd_documents)
my_retriever = advanced_local_pdf_retriever(srd_documents)


def search(query: str) -> str:
    return "\n\n".join(doc.page_content for doc in my_retriever.invoke(query))


def class_basics(pg_class: str) -> str:
    """Hit points, proficiencies and starting equipment of a class, cut straight
    from the SRD: every class opens with "Un <classe> ottiene i seguenti privilegi
    di classe" and these three blocks follow (checked on all 12). Search found the
    equipment for only 4 classes out of 5, and the wording varies ("Un barbaro
    inizia con..."), so a fixed phrase search would mix up classes."""
    match = re.search(
        rf"\bUno? {re.escape(pg_class.lower())} ottiene i seguenti privilegi di classe.*?\nEquipaggiamento.*?\n\s*\n",
        SRD_TEXT, re.S
    )
    return match.group(0) if match else search(f"equipaggiamento iniziale del {pg_class.lower()}")


def race_overview() -> str:
    """One line per SRD race with size, speed and ability increases, cut from the
    manual. In a live run the race step, given only search results for "chiamalo
    Amlach", answered from memory: dwarf +2 Strength, elf 30 m. Sections are found by
    their "Tratti dei/degli ..." heading, in SRD order. Only the number is kept for the
    speed: the Italian SRD names the wrong race in the halfling and human sentences."""
    heads = [m.start() for m in re.finditer(r"\nTratti de(?:i|gli) ", SRD_TEXT)][:9]
    order = ["Elfo", "Halfling", "Nano", "Umano", "Dragonide", "Gnomo", "Mezzelfo", "Mezzorco", "Tiefling"]
    lines = []
    for race, start, end in zip(order, heads, heads[1:] + [heads[-1] + 4000]):
        section = SRD_TEXT[start:end]
        size = re.search(r"taglia (Media|Piccola)", section)
        speed = re.search(r"Velocità\..*?(\d+(?:,\d+)?) metri", section, re.S)
        increases = re.findall(r"Incremento dei punteggi di caratteristica\.\s*(.*?\.)", section, re.S)
        lines.append(f"- {race}: taglia {size.group(1) if size else '?'}, velocità {speed.group(1) if speed else '?'} m; "
                     + " ".join(" ".join(s.split()) for s in increases))
    return "\n".join(lines)


def class_overview() -> str:
    """One line per SRD class with hit die, saving throws and armour, from class_basics."""
    lines = []
    for name in SRD_CLASSES.split(", "):
        block = class_basics(name)
        hit_die = re.search(r"Dadi [Vv]ita: (\S+)", block)
        saves = re.search(r"Tiri Salvezza: ([^\n]+)", block)
        armour = re.search(r"Armature: (.*?)\nArmi:", block, re.S)
        lines.append(f"- {name}: dado vita {hit_die.group(1) if hit_die else '?'}, "
                     f"tiri salvezza {saves.group(1).strip() if saves else '?'}, "
                     f"armature {' '.join(armour.group(1).split()) if armour else '?'}")
    return "\n".join(lines)


HEADING = r"(?:\n|\.|\d)([A-ZÀ-Ü][^\n.]{2,40}?)\s*\n"  # pages are joined without a newline: "1Druidico"
LEVEL_2 = re.compile(r"(?:\n\s*|\.\s*)(?:Al|A partire dal|Quando [^\n.]{0,30}?(?:al|il)) 2° livello")


def class_features(pg_class: str) -> str:
    """The class's 1st-level features, cut from the SRD. They follow the class table
    (which ends at the 20th-level row): from the first heading followed by prose to
    the heading of the first feature that opens with "Al 2° livello" or the like.
    The search on the class name returned other parts of the class, so the final sheet
    made up fighting styles ("Doppio attacco") and added the 3rd-level archetype.
    The text has every sub-heading at the same level ("Recuperare Energie" read like a
    fighting style), so the feature names from the table's 1st-level row go on top:
    the cells of that row without the numeric columns (dice, slots, "—")."""
    basics = class_basics(pg_class)
    table_start = SRD_TEXT.find(basics) + len(basics)
    row = re.search(r"\n1°\s*\n\+2(.*?)\n2°\s*\n\+2", SRD_TEXT[table_start:], re.S).group(1)
    names = " ".join(c.strip() for c in row.splitlines() if c.strip() and not re.match(r"[\d+—–-]", c.strip()))
    table_end = SRD_TEXT.find("20°", table_start)
    start = re.compile(HEADING + r"(?=[A-ZÀ-Ü][^\n]{30,})").search(SRD_TEXT, table_end).start()
    level_2 = LEVEL_2.search(SRD_TEXT, start + 1).start()
    end = start + list(re.finditer(HEADING, SRD_TEXT[start:level_2 + 1]))[-1].start()
    features = f"Privilegi di 1° livello: {names}. Gli altri titoli qui sotto sono parti o opzioni di questi.\n\n" + SRD_TEXT[start + 1:end + 1]
    if pg_class in SUBCLASSES:
        features += (f"\n\nL'SRD ha una sola opzione per la sottoclasse di 1° livello, {SUBCLASSES[pg_class]}; "
                     f"i suoi privilegi di 1° livello:\n" + subclass_features(pg_class))
    return features


# Cleric, sorcerer and warlock pick their subclass at 1st level; the SRD has one each,
# described at the end of the class, so class_features alone left its features out
SUBCLASSES = {"Chierico": "Dominio della Vita", "Stregone": "Discendenza draconica", "Warlock": "L'Immondo"}
HIGHER_LEVEL = re.compile(r"(?:\n\s*|\.\s*)(?:Al |All'|A partire dal |A partire dall')(?:[2-9]|1\d|20)° livello")


def subclass_features(pg_class: str) -> str:
    """From the subclass heading to the heading of its first feature above 1st level.
    Searched after the class: "Discendenza draconica" is also a dragonborn heading."""
    pattern = re.compile(rf"(?:\n|\.){re.escape(SUBCLASSES[pg_class])}\s*\n(?=[A-ZÀ-Ü])")
    start = pattern.search(SRD_TEXT, SRD_TEXT.find(class_basics(pg_class))).start()
    higher = HIGHER_LEVEL.search(SRD_TEXT, start + 1).start()
    end = start + list(re.finditer(HEADING, SRD_TEXT[start:higher + 1]))[-1].start()
    return SRD_TEXT[start + 1:end + 1]


def table_rows(start: str, end: str, columns: int) -> list[str]:
    """Rows of an SRD table between two headings. PyMuPDF gives one cell per line,
    empty cells as blank lines: after dropping those, a category heading ("Armature
    medie") is a lone line and each item is its name followed by `columns` cells."""
    block = re.search(rf"{start}(.*?){end}", SRD_TEXT, re.S).group(1)
    cells = [c.strip() for c in block.splitlines() if c.strip()]
    rows, i = [], 0
    while i < len(cells):
        if cells[i].startswith(("Armature ", "Armi ", "Scudi")):
            rows.append(f"{cells[i]}:")
            i += 1
        else:
            rows.append(cells[i:i + columns + 1])
            i += columns + 1
    return rows


def equipment_tables() -> str:
    """Armour, weapons and packs from the SRD equipment chapter. In live runs the
    equipment step, with only the class's (a)/(b) list, made up the details: chain
    mail as a medium armour adding Dexterity, the explorer's pack contents."""
    lines = ["Armature (Classe Armatura, Forza minima, Furtività):"]
    for row in table_rows(r"\nArmature\s*\nArmatura\s*\nCosto\s*\nClasse Armatura \(CA\)\s*\nForza\s*\nFurtività\s*\nPeso",
                          r"Indossare e togliere un'armatura", 5):
        lines.append(row if isinstance(row, str) else f"- {row[0]}: CA {row[2]}, Forza minima {row[3].removeprefix('For ')}, Furtività {row[4]}")
    # Both models read "due asce" (two handaxes) as two battleaxes, in the step and in the sheet
    lines.append("\nArmi (danni; proprietà). Un'arma nominata da sola nelle alternative della classe è quella con "
                 "quel nome esatto: 'due asce' sono due Ascia, non Ascia da battaglia; 'una mazza' è una Mazza, "
                 "non una Mazza chiodata.")
    for row in table_rows(r"Armi\s*\nNome\s*\nCosto\s*\nDanni\s*\nPeso\s*\nProprietà",
                          r"Equipaggiamento d'avventura", 4):  # pages are joined without a newline
        lines.append(row if isinstance(row, str) else f"- {row[0]}: {row[2]}; {row[4]}")
    packs = re.search(r"\n(Dotazione da avventuriero.*?)\nStrumenti\s*\n", SRD_TEXT, re.S).group(1)
    lines.append("\nDotazioni:\n" + "\n".join(" ".join(p.split()) for p in packs.split("\n \n")))
    return "\n".join(lines)


RACE_OVERVIEW = race_overview()
CLASS_OVERVIEW = class_overview()
EQUIPMENT_TABLES = equipment_tables()


def race_line(race: str) -> str:
    return next((line for line in RACE_OVERVIEW.splitlines() if line.startswith(f"- {race}:")), "")


# ---------------------------------------------------------------------------
# Ability scores and hit points, computed in code: in live runs the model forgot the
# racial increase once and applied it twice another time (COS 14 -> 16 -> 18)
# ---------------------------------------------------------------------------

ABILITIES = ["Forza", "Destrezza", "Costituzione", "Intelligenza", "Saggezza", "Carisma"]


def parse_scores(text: str) -> Optional[dict]:
    """{"Forza": 15, ...} when the text names all six abilities, each with its number
    right after ("Forza 15", "FOR: 15", "forza a 15") or right before ("15 in forza");
    otherwise None."""
    scores = {}
    for name in ABILITIES:
        word = rf"\b(?:{name}|{name[:3]})\b"
        match = (re.search(rf"{word}[^\d\n,;]{{0,6}}(\d{{1,2}})", text, re.I)
                 or re.search(rf"(\d{{1,2}})[^\d\n,;]{{0,6}}{word}", text, re.I))
        if not match:
            return None
        scores[name] = int(match.group(1))
    return scores


def race_increases(race: str) -> dict:
    """Racial increases read from the SRD sentences in RACE_OVERVIEW, e.g. Nano ->
    {"Costituzione": 2, "Saggezza": 1}. The half-elf's two free +1 are not included."""
    text = race_line(race).partition(";")[2]
    if "Ogni punteggio" in text:  # Umano
        return {name: 1 for name in ABILITIES}
    increases = {}
    for name, amount in re.findall(rf"({'|'.join(ABILITIES)})\D*?(\d)", text):
        increases[name] = increases.get(name, 0) + int(amount)
    return increases


def final_scores(state) -> Optional[str]:
    """Base + racial increase = final score and modifier, then level-1 hit points;
    None when the saved scores can't be read."""
    base = parse_scores(state.get("pg_abilities") or "")
    if not base:
        return None
    increases = race_increases(state["pg_race"])
    final = {name: base[name] + increases.get(name, 0) for name in ABILITIES}
    lines = [f"- {name}: {base[name]} + {increases.get(name, 0)} (razza) = {final[name]}, "
             f"modificatore {(final[name] - 10) // 2:+d}" for name in ABILITIES]
    if state["pg_race"] == "Mezzelfo":
        lines.append("- Il mezzelfo aggiunge +1 a due caratteristiche a scelta, non incluse sopra: "
                     "se il giocatore non le ha indicate, segnalale come scelta da fare.")

    hit_points = re.search(r"Punti [Ff]erita al 1° [Ll]ivello: (\d+)", class_basics(state["pg_class"]))
    if hit_points:
        con = (final["Costituzione"] - 10) // 2
        # +1 hit point per level from the SRD's only dwarf (hill dwarf) and only sorcerer origin
        extras = [label for label, applies in [("Robustezza nanica", state["pg_race"] == "Nano"),
                                               ("Resilienza draconica", state["pg_class"] == "Stregone")] if applies]
        total = int(hit_points.group(1)) + con + len(extras)
        lines.append(f"- Punti ferita al 1° livello: {hit_points.group(1)} + {con} (Costituzione)"
                     + "".join(f" + 1 ({label})" for label in extras) + f" = {total}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

# json_schema instead of the default tool call: in a live run gpt-oss called the tool
# "functions.Pg_Extraction" and the turn failed. One retry covers a bad answer.
structured_llm = extractor_llm.with_structured_output(Pg_Extraction, method="json_schema").with_retry(stop_after_attempt=2)


def ask(model, messages):
    """gpt-oss sometimes spends its whole token budget reasoning and returns an empty
    answer (measured: 1 run in 4 on the small model at medium effort): ask once more."""
    response = model.invoke(messages)
    return response if response.content else model.invoke(messages)


def analyzer(state: State):
    """Updates the saved choices from the last exchange: the DM's last message and the user's reply."""
    messages = state.get("messages", [])
    if not messages:
        return {}

    # Last exchange only, with roles spelled out: the saved sheet already holds the
    # earlier choices, and the DM's message is needed to understand a "sì, va bene"
    # (reading only the user's words, an accepted proposal was never saved).
    user_reply = messages[-1].content
    dm_message = next((m.content for m in reversed(messages[:-1]) if m.type == "ai"), "(nessuno)")

    prompt = [
        {"role": "system", "content": """Sei un assistente per la creazione di personaggi di D&D 5e.
            Aggiorna la scheda del personaggio in base all'ultima risposta del giocatore.
            Una scelta vale solo se il giocatore la dichiara o accetta esplicitamente una proposta del DM ("sì", "va bene, prendo quella").
            Un dubbio ("forse elfo o nano"), una domanda o una proposta del DM non accettata non sono scelte.
            Se il giocatore cambia idea, riporta il nuovo valore. Per tutto ciò di cui non parla, restituisci null.
            Non aggiungere niente di tuo."""},
        # Without the saved preferences: shown them, the model copied them back
        # (measured), and merging made them grow at every turn
        {"role": "user", "content": f"""Scheda attuale:
{sheet_text(state, with_preferences=False)}

Ultimo messaggio del DM:
{dm_message}

Risposta del giocatore:
{user_reply}"""},
    ]
    extraction = structured_llm.invoke(prompt)

    # null means "nothing new": only real values overwrite the saved ones. A value equal
    # to the saved one is not news either (the model often echoes the whole sheet back).
    update = {slot: value for slot, value in extraction.model_dump().items()
              if value and value != state.get(slot)}

    # Race and class only as SRD names: in a live run the model saved "Fighter" from
    # "combattere in prima linea", and class_basics found nothing for an English name
    for slot, names in (("pg_race", SRD_RACES), ("pg_class", SRD_CLASSES)):
        if slot in update:
            update[slot] = srd_name(update[slot], names)
            if not update[slot]:
                del update[slot]

    # Scores straight from the player's words whenever they name all six, never from the
    # model: in a live run it added the racial increase on its own, and the final sheet
    # added it again (COS 14 -> 16 -> 18)
    scores = parse_scores(user_reply)
    if scores:
        update["pg_abilities"] = ", ".join(f"{name[:3].upper()} {value}" for name, value in scores.items())

    # The model returns only new preferences; merging them is done here, not by the model
    if "pg_preferences" in update and state.get("pg_preferences"):
        update["pg_preferences"] = f"{state['pg_preferences']}; {update['pg_preferences']}"
    return update


def srd_name(value: str, names: str) -> Optional[str]:
    """The SRD name contained in the model's answer ("Nano (colline)" -> "Nano"), or None.
    Word boundaries keep "Elfo" from matching inside "Mezzelfo"."""
    return next((name for name in names.split(", ") if re.search(rf"\b{name}\b", value, re.I)), None)


# How every step answers. Big answers mixing race, class and equipment were hard to
# follow, and the model filled gaps from memory (English names, wrong numbers).
STEP_STYLE = """Rispondi in italiano, in Markdown, e in modo breve.
Parla solo del passo corrente: non anticipare i passi successivi.
Al massimo 3-4 opzioni, ognuna in una o due righe; niente tabelle lunghe.
Usa solo le informazioni del manuale riportate qui sotto e i suoi termini italiani: se un dettaglio non c'è, non inventarlo.
Chiudi con una domanda."""

# The saved sheet carries the earlier steps, so the model sees only the last few
# messages: the full chat would soon exceed Groq's free quota of 8,000 tokens/minute.
RECENT_MESSAGES = 6


def guide_step(model, state: State, task: str, manual: str = ""):
    prompt = [{"role": "system", "content": f"""Sei un assistente di D&D 5e che guida un giocatore nella creazione del personaggio, un passo alla volta.
{STEP_STYLE}

Scheda finora:
{sheet_text(state)}

Compito di questo passo: {task}

Informazioni dal manuale (SRD 5.1):
{manual or "(nessuna per questo passo)"}"""}]
    # The task again after the chat: in a live run the recent messages of the scores step
    # ("Passo 1", "Passo 2") won over the task at the top, and the equipment step went on
    # with "Passo 3 – bonus razziali" and its own (wrong) hit points
    reminder = {"role": "system", "content": f"Ricorda il compito di questo passo: {task}"}
    return {"messages": [ask(model, prompt + state["messages"][-RECENT_MESSAGES:] + [reminder])]}


def last_user_message(state: State) -> str:
    return next(m.content for m in reversed(state["messages"]) if m.type == "human")


def ask_pg_name(state: State):
    return guide_step(llm_lite, state,
        "Il personaggio non ha ancora un nome. Se nella scheda ci sono già le preferenze del giocatore, non chiederle di nuovo: "
        "riprendile in una frase e chiedi solo come vuole chiamarlo. Se non ci sono, chiedigli che idea ha del personaggio "
        "(ruolo nel gruppo, stile di gioco, esperienza con D&D) e come vuole chiamarlo.")


def pg_race_guide(state: State):
    # The overview cut from the manual gives exact numbers for every option; search on
    # the last user message adds detail when they ask about something specific.
    # A topic prefix on the query ("razza ...") was measured to drag in generic pages.
    return guide_step(llm, state,
        f"Aiuta il giocatore a scegliere la razza fra: {SRD_RACES}, tenendo conto delle sue preferenze.",
        "Le razze del manuale:\n" + RACE_OVERVIEW + "\n\n" + search(last_user_message(state)))


def pg_class_guide(state: State):
    return guide_step(llm, state,
        f"Aiuta il giocatore a scegliere la classe fra: {SRD_CLASSES}, tenendo conto delle sue preferenze e della razza.",
        "Le classi del manuale:\n" + CLASS_OVERVIEW + "\n\n" + search(last_user_message(state)))


def pg_abilities_guide(state: State):
    # The SRD has no rules for generating scores: the three usual methods are given here.
    # The player decides: the first version proposed a full assignment on its own.
    return guide_step(llm, state,
        "Accompagna il giocatore nell'assegnare i sei punteggi di caratteristica (Forza, Destrezza, Costituzione, "
        "Intelligenza, Saggezza, Carisma). Le decisioni sono sue: non scegliere tu il metodo e non assegnare tu i punteggi. "
        "Metodi, dal Manuale del Giocatore perché l'SRD non li riporta: "
        "(1) serie standard 15, 14, 13, 12, 10, 8; "
        "(2) acquisto a punti: 27 punti, ogni punteggio fra 8 e 15, costi 8=0, 9=1, 10=2, 11=3, 12=4, 13=5, 14=7, 15=9; "
        "(3) tiro dei dadi: sei volte 4d6 sommando i tre dadi migliori, con dadi veri tirati dal giocatore, che ti dice i risultati. "
        "Procedi così, un punto per messaggio: "
        "se il giocatore non ha ancora scelto il metodo, spiega brevemente i tre metodi e chiedigli quale preferisce, senza proporre numeri; "
        "poi chiedigli come vuole disporre i valori fra le sei caratteristiche, indicando in una riga quali contano di più per la sua classe "
        "e quali incrementi gli darà la razza (se la razza ha incrementi a scelta, chiedi anche quelli); "
        "non proporre disposizioni già pronte, a meno che il giocatore non te lo chieda; "
        "chiedigli di scriverli nominando tutte e sei le caratteristiche, per esempio "
        "'Forza 15, Destrezza 13, Costituzione 14, Intelligenza 8, Saggezza 12, Carisma 10', senza l'incremento della razza; "
        "con l'acquisto a punti, controlla che non superi i 27 punti. "
        "L'incremento della razza verrà applicato nella scheda finale.",
        "La razza del giocatore:\n" + race_line(state["pg_race"]) + "\n\n" + class_basics(state["pg_class"]))


def pg_equipment_guide(state: State):
    # The class block says "in aggiunta all'equipaggiamento conferito dal suo background": in live
    # runs, with a softer task, the model moved on to the background or the subclass instead
    return guide_step(llm, state,
        "Ora tocca all'equipaggiamento iniziale: presenta le alternative (a), (b) della classe dal manuale, con un consiglio "
        "breve legato ai punteggi e alle preferenze, e chiedi quali sceglie. Se un'alternativa è generica ('un'arma da guerra'), "
        "aiutalo a sceglierne una precisa dalle tabelle. Parla solo di questo: background, sottoclasse, abilità e incantesimi "
        "non sono passi di questa guida.",
        # The computed scores too: the model recaps them before the equipment, and on its own it gave 8 hit points instead of 9
        "Punteggi finali, già calcolati:\n" + (final_scores(state) or "") + "\n\n"
        + class_basics(state["pg_class"]) + "\n\n" + EQUIPMENT_TABLES)


def pg_summary(state: State):
    # One short query for the race ("Elfo"): bare names find their SRD section.
    # The class blocks are cut from the manual (see class_basics and class_features).
    manual = (race_line(state["pg_race"]) + "\n\n" + search(state["pg_race"]) + "\n\n"
              + class_basics(state["pg_class"]) + "\n\nPrivilegi di classe di 1° livello:\n" + class_features(state["pg_class"])
              + "\n\n" + EQUIPMENT_TABLES)

    computed = final_scores(state)
    if computed:
        scores_rule = f"Punteggi finali e punti ferita, già calcolati: riportali esattamente così, senza ricalcolarli.\n{computed}"
        scores_item = "2. Punteggi di caratteristica, come calcolati sopra (base, incremento della razza, finale, modificatore)."
    else:  # the player gave the scores in a form parse_scores can't read
        scores_rule = "I punteggi in scheda sono quelli disposti dal giocatore, prima dell'incremento della razza."
        scores_item = "2. Punteggi di caratteristica: valore disposto, incremento della razza (dalla prima riga del manuale qui sotto), valore finale e modificatore."

    prompt = [{"role": "system", "content": f"""Scrivi in italiano, in Markdown, la scheda finale di un personaggio di D&D 5e di 1° livello.

Scheda:
{sheet_text(state)}

{scores_rule}

Contenuto, in questo ordine:
1. Titolo con il nome, poi razza e classe.
{scores_item}
3. Punti ferita al 1° livello e competenze.
4. Equipaggiamento, con la Classe Armatura e i danni delle armi presi dalle tabelle; per le dotazioni, il contenuto.
5. Tratti razziali e privilegi di classe di 1° livello, una riga ciascuno. I privilegi sono tutti e soli quelli della sezione "Privilegi di classe di 1° livello" qui sotto: riportali tutti, senza aggiungerne di livello superiore; quando un privilegio offre opzioni (per esempio gli stili di combattimento), elencale sotto di lui.
6. Due o tre righe che descrivono il personaggio, legate alle preferenze del giocatore.

Usa solo le informazioni del manuale qui sotto e i suoi termini italiani; non inventare dettagli.
Per le scelte che il giocatore non ha fatto (per esempio le abilità, lo stile di combattimento o quale arma da guerra), elenca le opzioni del manuale senza scegliere al suo posto.

Informazioni dal manuale (SRD 5.1):
{manual}"""}]

    response = ask(llm_lite, prompt)
    return {"messages": [response], "completed": True}  # tells the CLI loop to end the session


def save_sheet(state, folder: Path = BASE_DIR / "schede") -> Path:
    """Writes the final sheet (the summary message) to schede/<name>.md."""
    folder.mkdir(exist_ok=True)
    slug = re.sub(r"\W+", "_", state.get("pg_name") or "").strip("_").lower() or "personaggio"
    path = folder / f"{slug}.md"
    path.write_text(state["messages"][-1].content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

def build_graph():

    def router_logic(state: State) -> str:
        # Drive the flow slot by slot, in STEPS order, then the summary
        return next((slot for slot, _ in STEPS if not state.get(slot)), "pg_summary")

    nodes = {
        "pg_name":      ask_pg_name,
        "pg_race":      pg_race_guide,
        "pg_class":     pg_class_guide,
        "pg_abilities": pg_abilities_guide,
        "pg_equipment": pg_equipment_guide,
        "pg_summary":   pg_summary,
    }

    g = StateGraph(State)
    g.add_node("analyzer", analyzer)
    g.add_edge(START, "analyzer")
    g.add_conditional_edges("analyzer", router_logic, {name: name for name in nodes})

    # Each guide node ends the turn; the next user message re-enters at START
    for name, node in nodes.items():
        g.add_node(name, node)
        g.add_edge(name, END)

    memory = MemorySaver()
    return g.compile(checkpointer=memory)


def discard_failed_turn(guide, config):
    """The checkpointer saves the user's message before any node runs: if the turn
    then fails (rate limit, network), the message stays with no reply and the next
    turn sends two user messages in a row. Remove it, so a retry starts clean."""
    messages = guide.get_state(config).values.get("messages", [])
    if messages and messages[-1].type == "human":
        guide.update_state(config, {"messages": [RemoveMessage(id=messages[-1].id)]})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

console = Console()


def start_dnd_guide():
    print("\n" + "="*60)
    print(" 🎲 Benvenuto! Sono il tuo assistente per creare un personaggio di D&D 5e! 🎲 ")
    print(" (Scrivi 'exit' per chiudere la sessione) ")
    print("="*60 + "\n")
    print("DM: Hai già un'idea del personaggio? Raccontami cosa ti piacerebbe fare")
    print("    (combattere in prima linea, usare la magia, muoverti di nascosto...),")
    print("    se conosci già D&D e, se ce l'hai, il nome.")

    guide = build_graph()
    thread_config = {"configurable": {"thread_id": "tester"}}

    while True:
        user_input = input("\nTu: ")

        if user_input.lower() == "exit":
            print("DM: Alla prossima avventura!")
            break

        try:
            state = guide.invoke(
                {"messages": [("user", user_input)]},
                config=thread_config
            )

            # The model answers in Markdown: rich renders it (bold, lists, tables)
            console.print("\n[bold]DM:[/bold]")
            console.print(Markdown(state["messages"][-1].content))

            # End the session once the summary has been delivered
            if state.get("completed"):
                path = save_sheet(state)
                console.print(f"\n[bold]DM:[/bold] Il tuo personaggio è pronto, la scheda è in {path}. Buona fortuna, avventuriero!")
                break

        except Exception as e:
            discard_failed_turn(guide, thread_config)
            print(f"\nSi è verificato un errore: {e}\nRiscrivi il messaggio per riprovare.")


def main():
    start_dnd_guide()


if __name__ == "__main__":
    main()
