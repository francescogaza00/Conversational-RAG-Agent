"""Checks the graph logic without calling Groq: the LLMs are replaced by stubs,
the SRD retrieval is the real one. Run: python test_grafo.py"""
import os
import tempfile
from pathlib import Path
os.environ.setdefault("GROQ_API_KEY", "stub")  # never used: every LLM call below is stubbed

from langchain_core.messages import AIMessage, HumanMessage
import Dnd_guide as g


class Stub:
    """Fake chat model: records what it receives, answers with a fixed text."""
    def __init__(self, reply):
        self.reply, self.seen = reply, []

    def invoke(self, messages):
        self.seen.append(messages)
        if callable(self.reply):
            return self.reply(messages)
        # A fresh copy each time, like a real model: the same object would keep the
        # same id, and LangGraph would overwrite the earlier reply instead of appending
        return self.reply.model_copy() if isinstance(self.reply, AIMessage) else self.reply


# 1. A null from the model must be accepted, not raise ValidationError
assert g.Pg_Extraction(pg_name=None, pg_race=None, pg_class=None).pg_name is None

# 2. class_basics cuts the right block for every class (Barbaro and Ranger word it differently)
for cls in g.SRD_CLASSES.split(", "):
    block = g.class_basics(cls)
    assert block.lower().startswith(f"un {cls.lower()} ottiene") or block.lower().startswith(f"uno {cls.lower()} ottiene"), cls
    assert "Equipaggiamento" in block and "•" in block, cls
assert "grandascia" in g.class_basics("Barbaro") and "stocco" not in g.class_basics("Barbaro")

# 3. Overviews cut from the manual: right numbers, including the two races whose
#    speed sentence names the wrong race in the Italian SRD
races = g.RACE_OVERVIEW.splitlines()
assert len(races) == 9 and len(g.CLASS_OVERVIEW.splitlines()) == 12
assert "velocità 7,5 m" in g.race_line("Halfling") and "velocità 9 m" in g.race_line("Umano")
assert "Costituzione di un nano aumenta di 2" in g.race_line("Nano")
assert "?" not in g.RACE_OVERVIEW + g.CLASS_OVERVIEW
assert "dado vita 1d10" in g.CLASS_OVERVIEW.splitlines()[4]  # Guerriero

# 1st-level features cut from the manual: first and last heading of each class, checked by hand on the SRD
for cls, first, last in [("Barbaro", "Ira", "Difesa Senza Armatura"), ("Bardo", "Incantesimi", "Ispirazione Bardica"),
                         ("Chierico", "Incantesimi", "Incantesimi di dominio"), ("Druido", "Druidico", "Focus da incantatore"),
                         ("Guerriero", "Stile di Combattimento", "Recuperare Energie"), ("Ladro", "Maestria", "Gergo Ladresco"),
                         ("Mago", "Incantesimi", "Recupero Arcano"), ("Monaco", "Difesa senza armatura", "Arti Marziali"),
                         ("Paladino", "Percezione del Divino", "Imposizione delle Mani"), ("Ranger", "Nemico Prescelto", "Esploratore Nato"),
                         ("Stregone", "Incantesimi", "Origine Stregonesca"), ("Warlock", "Patrono Ultraterreno", "Focus da incantatore")]:
    features = g.class_features(cls)
    assert f"\n\n{first}" in features and f"{last} \n" in features, cls
assert g.class_features("Guerriero").startswith("Privilegi di 1° livello: Recuperare Energie, Stile di Combattimento.")
assert g.class_features("Monaco").startswith("Privilegi di 1° livello: Difesa senza armatura, Arti Marziali.")  # dice columns dropped
assert "Duellare" in g.class_features("Guerriero") and "Azione Impetuosa" not in g.class_features("Guerriero")
assert "Tradizione Arcana" not in g.class_features("Mago")
# Cleric, sorcerer and warlock get the SRD's only subclass, cut after the class (the
# dragonborn also has a "Discendenza draconica" heading) and stopping before 2nd+ level
for cls, first, last, later in [("Chierico", "Dominio della Vita", "Discepolo della Vita", "Preservare Vita"),
                                ("Stregone", "Discendenza draconica", "Resilienza draconica", "Affinità elementale"),
                                ("Warlock", "L'Immondo", "Benedizione dell'Oscuro", "Fortuna dell'Oscuro")]:
    sub = g.subclass_features(cls)
    assert sub.startswith(first) and last in sub and later not in sub and sub in g.class_features(cls), cls
assert "Antenato draconico" in g.subclass_features("Stregone")

# Equipment tables cut from the manual: every row parsed, values as in the SRD
assert "- Cotta di maglia: CA 16, Forza minima 13, Furtività Svantaggio" in g.EQUIPMENT_TABLES
assert "- Ascia bipenne: 1d12 taglienti; Due mani, Pesante" in g.EQUIPMENT_TABLES
assert "- Rete: —; Lancio (gittata 1,5/4,5), Speciale" in g.EQUIPMENT_TABLES        # last weapon row
assert g.EQUIPMENT_TABLES.count("\n- ") == 13 + 37 and g.EQUIPMENT_TABLES.count("Dotazione da ") == 7
assert "Dotazione da esploratore (10 mo). Include uno zaino, un giaciglio, una gavetta" in g.EQUIPMENT_TABLES

# 4. Race and class are kept only as SRD names
assert g.srd_name("Nano (colline)", g.SRD_RACES) == "Nano"
assert g.srd_name("Mezzelfo", g.SRD_RACES) == "Mezzelfo"   # not "Elfo"
assert g.srd_name("Fighter", g.SRD_CLASSES) is None

# 5. Analyzer post-processing: echoed values dropped, English class dropped, preferences merged
g.structured_llm = Stub(g.Pg_Extraction(pg_name="Amlach", pg_class="Fighter", pg_preferences="ama le asce"))
update = g.analyzer({"messages": [AIMessage("ciao"), HumanMessage("x")],
                     "pg_name": "Amlach", "pg_preferences": "prima linea"})
assert update == {"pg_preferences": "prima linea; ama le asce"}, update

# 6. Ability scores: read from the player's words, increased by the race in code
assert g.parse_scores("Forza 15, Costituzione 14, Destrezza 13, Saggezza 12, Carisma 10, Intelligenza 8") == \
    {"Forza": 15, "Destrezza": 13, "Costituzione": 14, "Intelligenza": 8, "Saggezza": 12, "Carisma": 10}
assert g.parse_scores("15 in forza, 14 in costituzione, 13 in destrezza, 12 in saggezza, 10 in carisma, 8 in intelligenza")["Costituzione"] == 14
assert g.parse_scores("FOR 15 DES 13 COS 14 INT 8 SAG 12 CAR 10")["Carisma"] == 10
assert g.parse_scores("come funzionano i punteggi?") is None
assert g.race_increases("Nano") == {"Costituzione": 2, "Saggezza": 1}
assert g.race_increases("Tiefling") == {"Intelligenza": 1, "Carisma": 2}
assert g.race_increases("Umano") == dict.fromkeys(g.ABILITIES, 1)
assert g.race_increases("Mezzelfo") == {"Carisma": 2}          # the two free +1 are the player's
assert all(g.race_increases(r) for r in g.SRD_RACES.split(", "))
sheet = g.final_scores({"pg_abilities": "FOR 15, DES 13, COS 14, INT 8, SAG 12, CAR 10",
                        "pg_race": "Nano", "pg_class": "Guerriero"})
assert "Costituzione: 14 + 2 (razza) = 16, modificatore +3" in sheet
assert "Punti ferita al 1° livello: 10 + 3 (Costituzione) + 1 (Robustezza nanica) = 14" in sheet
assert "6 + 2 (Costituzione) + 1 (Resilienza draconica) = 9" in g.final_scores(
    {"pg_abilities": "FOR 8, DES 14, COS 14, INT 10, SAG 12, CAR 15", "pg_race": "Tiefling", "pg_class": "Stregone"})

# 7. Scripted extractions, one per turn; the chat models answer with a fixed line
turns = iter([
    g.Pg_Extraction(pg_preferences="combattere in prima linea"),       # idea, no name yet
    g.Pg_Extraction(pg_name="Amlach"),
    g.Pg_Extraction(pg_race="Nano"),
    g.Pg_Extraction(pg_class="Guerriero"),
    g.Pg_Extraction(),                                                  # scores come from the reply
    g.Pg_Extraction(pg_equipment="cotta di maglia, arma da guerra e scudo"),
])
g.structured_llm = Stub(lambda _: next(turns))
g.llm = Stub(AIMessage("Ti propongo un nano, va bene?"))
g.llm_lite = Stub(AIMessage("# Amlach\nNano guerriero"))

guide = g.build_graph()
cfg = {"configurable": {"thread_id": "test"}}
replies = ["voglio combattere in prima linea", "Amlach", "sì, va bene il nano",
           "guerriero", "Forza 15, Destrezza 13, Costituzione 14, Intelligenza 8, Saggezza 12, Carisma 10", "prendo la (a)"]
states = [guide.invoke({"messages": [("user", text)]}, config=cfg) for text in replies]

# The router walks name -> race -> class -> abilities -> equipment -> summary, one per turn,
# and a slot saved earlier is never lost (only real values overwrite)
assert [s.get("pg_name") for s in states] == [None] + ["Amlach"] * 5
assert [s.get("completed") for s in states] == [None] * 5 + [True]
assert states[-1]["pg_preferences"] == "combattere in prima linea"

# The extractor sees the last exchange with roles, so an accepted DM proposal can be saved,
# and the saved sheet instead of the whole chat
race_turn = g.structured_llm.seen[2][-1]["content"]
assert "Ultimo messaggio del DM:" in race_turn and "sì, va bene il nano" in race_turn
assert "voglio combattere" not in race_turn          # older user messages are not resent
assert "combattere in prima linea" not in race_turn   # nor the saved preferences (the model echoed them)

# Each step prompt carries the saved preferences and only the recent messages
class_prompt = g.llm.seen[1]
assert "combattere in prima linea" in class_prompt[0]["content"]
assert len(class_prompt) - 1 <= g.RECENT_MESSAGES

# The equipment step gets the warrior's starting equipment straight from the manual
equipment_prompt = g.llm.seen[-1][0]["content"]
assert "cotta di maglia" in equipment_prompt
assert "Cotta di maglia: CA 16" in equipment_prompt and "Dotazione da esploratore" in equipment_prompt

# The summary prompt carries the SRD sections of both the race and the class
summary_prompt = g.llm_lite.seen[-1][0]["content"].lower()
assert "tratti dei nani" in summary_prompt
assert "un guerriero ottiene i seguenti privilegi di classe" in summary_prompt
assert "cotta di maglia: ca 16" in summary_prompt
assert "recuperare energie" in summary_prompt and "azione impetuosa" not in summary_prompt

# The final sheet is written to <folder>/<name>.md
with tempfile.TemporaryDirectory() as tmp:
    path = g.save_sheet(states[-1], Path(tmp))
    assert path.name == "amlach.md" and path.read_text(encoding="utf-8").startswith("# Amlach")

# 8. A failed turn leaves no orphan user message behind
def failing(_):
    raise RuntimeError("simulated 429")

g.structured_llm = Stub(failing)
cfg = {"configurable": {"thread_id": "test-failure"}}
try:
    guide.invoke({"messages": [("user", "turno che fallisce")]}, config=cfg)
    raise AssertionError("the stub should have failed")
except RuntimeError:
    g.discard_failed_turn(guide, cfg)
assert guide.get_state(cfg).values["messages"] == []

g.structured_llm = Stub(g.Pg_Extraction())
state = guide.invoke({"messages": [("user", "riprovo")]}, config=cfg)
assert [(m.type, m.content) for m in state["messages"]][0] == ("human", "riprovo")
assert len(state["messages"]) == 2  # retry + reply, nothing left from the failure

# 9. An empty answer (token budget spent reasoning) is asked once more
answers = iter([AIMessage(""), AIMessage("eccomi")])
assert g.ask(Stub(lambda _: next(answers)), []).content == "eccomi"

print("OK: tutte le verifiche passate")
