import os
from typing import Optional
from typing_extensions import TypedDict, Annotated
from pydantic import Field, BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()  # reads .env file from the project root

key = os.getenv("GOOGLE_API_KEY")
if not key:
    raise ValueError("GOOGLE_API_KEY not found. Check your .env file.")

# Full model for RAG-enriched responses
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=key,
    temperature=0
)

# Lightweight model for slot extraction and short replies
llm_lite = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=key,
    temperature=0
)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class State(TypedDict):
    messages: Annotated[list, add_messages]
    pg_name:  Optional[str]
    pg_race:  Optional[str]
    pg_class: Optional[str]
    completed: bool  # prevents further cycles once the character is done


# ---------------------------------------------------------------------------
# Structured output schema for slot extraction
# ---------------------------------------------------------------------------

class Pg_Extraction(BaseModel):
    pg_name:  str = Field(default=None, description="Name of the pg choosen by the user.")
    pg_race:  str = Field(default=None, description="Race of the pg choosen by the user.")
    pg_class: str = Field(default=None, description="Class of the pg choosen by the user.")


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

def advanced_local_pdf_retriever(pdf_paths: list[str], persist_dir: str = "./chroma_db"):
    """
    Loads one or more PDFs and builds a hybrid retriever (BM25 + Chroma)
    with contextual reranking via Flashrank. Skips rebuilding the vectorstore
    when a persisted copy already exists on disk.
    """
    documents = []
    for path in pdf_paths:
        try:
            loader = PyMuPDFLoader(path)
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {path}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        raise ValueError("No chunks generated. Verify the input PDFs.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Reuse an existing vectorstore to avoid expensive re-embedding on every run
    if os.path.exists(persist_dir):
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir
        )

    chroma_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    # Combine keyword (BM25) and semantic (Chroma) search
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, chroma_retriever],
        weights=[0.4, 0.6]
    )

    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    return compression_retriever


dnd_manuals = ["guida_dnd_5e.pdf"]
my_retriever = advanced_local_pdf_retriever(dnd_manuals)


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

structured_llm = llm_lite.with_structured_output(Pg_Extraction)


def analyzer(state: State):
    """Reads the conversation history and extracts name, race, and class slots."""
    prompt = [{"role": "system", "content": """You are an assistant for D&D 5e character creation.
                Analyze the conversation and extract the following information if present: the character's name, race, and class.
                Return only the information you can extract, without adding anything of your own."""}]

    messages = state.get("messages", [])
    if not messages:
        return {}

    extraction = structured_llm.invoke(prompt + messages)

    update = {}
    if extraction.pg_name:  update["pg_name"]  = extraction.pg_name
    if extraction.pg_race:  update["pg_race"]  = extraction.pg_race
    if extraction.pg_class: update["pg_class"] = extraction.pg_class

    return update


def ask_pg_name(state: State):
    prompt = [{"role": "system", "content": "Ask the user what they want to name their character."}]
    response = llm_lite.invoke(prompt + state["messages"])
    return {"messages": [response]}


def pg_race_guide(state: State):
    # Use the last human message as the RAG query; fall back to a generic query
    last_user_msg = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"),
        "D&D 5e races"
    )
    docs = my_retriever.invoke(last_user_msg)
    rag_context = "\n\n".join([doc.page_content for doc in docs])

    prompt = [{"role": "system", "content": f"""You are a D&D 5e assistant.
                Guide the user in choosing their character's race.
                Provide options and ask questions to understand their preferences.

                Use the following information from the manual to answer accurately:
                {rag_context}"""}]

    response = llm.invoke(prompt + state["messages"])
    return {"messages": [response]}


def pg_class_guide(state: State):
    # Use the last human message as the RAG query; fall back to a generic query
    last_user_msg = next(
        (m.content for m in reversed(state["messages"]) if m.type == "human"),
        "D&D 5e classes"
    )
    docs = my_retriever.invoke(last_user_msg)
    rag_context = "\n\n".join([doc.page_content for doc in docs])

    prompt = [{"role": "system", "content": f"""You are a D&D 5e assistant.
                Guide the user in choosing their character's class.
                Provide options and ask questions to understand their preferences.

                Use the following information from the manual to answer accurately:
                {rag_context}"""}]

    response = llm.invoke(prompt + state["messages"])
    return {"messages": [response]}


def pg_summary(state: State):
    query = f"class {state.get('pg_class')} race {state.get('pg_race')} abilities traits features"
    docs = my_retriever.invoke(query)
    rag_context = "\n\n".join([doc.page_content for doc in docs])

    prompt = [{"role": "system", "content": f"""Summarize the character creation process and the choices made by the user.
                The character details are:
                - Name:  {state.get('pg_name')}
                - Race:  {state.get('pg_race')}
                - Class: {state.get('pg_class')}

                Use the following information from the D&D 5e manual to enrich the summary
                with details about class abilities, racial traits, and features:

                {rag_context}

                Provide an engaging description of the character, highlighting their main abilities and traits."""}]

    response = llm_lite.invoke(prompt + state["messages"])
    return {"messages": [response], "completed": True}  # lock the graph after summary


def already_completed(state: State):
    """Politely informs the user their character is already done and suggests a new session."""
    prompt = [{"role": "system", "content": f"""The user's character has already been successfully created:
                - Name:  {state.get('pg_name')}
                - Race:  {state.get('pg_race')}
                - Class: {state.get('pg_class')}

                Politely inform the user that their character is already complete.
                If they want to create a new one, suggest starting a new session."""}]

    response = llm_lite.invoke(prompt + state["messages"])
    return {"messages": [response]}


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

def build_graph():

    def router_logic(state: State) -> str:
        # Short-circuit if the character is already complete
        if state.get("completed"):
            return "already_completed"
        # Drive the flow slot by slot: name → race → class → summary
        if not state.get("pg_name"):
            return "pg_name"
        if not state.get("pg_race"):
            return "pg_race"
        if not state.get("pg_class"):
            return "pg_class"
        return "pg_summary"

    g = StateGraph(State)

    g.add_node("analyzer",          analyzer)
    g.add_node("pg_name",           ask_pg_name)
    g.add_node("pg_race",           pg_race_guide)
    g.add_node("pg_class",          pg_class_guide)
    g.add_node("pg_summary",        pg_summary)
    g.add_node("already_completed", already_completed)

    g.add_edge(START, "analyzer")

    g.add_conditional_edges(
        "analyzer",
        router_logic,
        {
            "pg_name":           "pg_name",
            "pg_race":           "pg_race",
            "pg_class":          "pg_class",
            "pg_summary":        "pg_summary",
            "already_completed": "already_completed",
        }
    )

    # Each guide node ends the turn; the next user message re-enters at START
    g.add_edge("pg_name",           END)
    g.add_edge("pg_race",           END)
    g.add_edge("pg_class",          END)
    g.add_edge("pg_summary",        END)
    g.add_edge("already_completed", END)

    memory = MemorySaver()
    return g.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def start_dnd_guide():
    print("\n" + "="*60)
    print(" 🎲 Welcome! I am your D&D 5e character creation assistant! 🎲 ")
    print(" (Type 'exit' to end the session) ")
    print("="*60 + "\n")

    guide = build_graph()
    thread_config = {"configurable": {"thread_id": "tester"}}

    while True:
        user_input = input("\nTu: ")

        if user_input.lower() == "exit":
            print("DM: Until the next adventure!")
            break

        try:
            state = guide.invoke(
                {"messages": [("user", user_input)]},
                config=thread_config
            )

            ai_response = state["messages"][-1].content
            print(f"\nDM: {ai_response}")

            # Debug: show extracted slots after each turn
            print(f"\n  Name: {state.get('pg_name')} | Race: {state.get('pg_race')} | Class: {state.get('pg_class')} | Completed: {state.get('completed', False)}")

            # End the session once the summary has been delivered
            current_state = guide.get_state(thread_config)
            if state.get("completed") and current_state.next == ():
                print("\nDM: Your character is ready. Good luck, adventurer!")
                break

        except Exception as e:
            print(f"\nAn error occurred: {e}")


def main():
    start_dnd_guide()


if __name__ == "__main__":
    main()
