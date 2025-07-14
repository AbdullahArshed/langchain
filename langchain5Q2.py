import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Load .env variables
load_dotenv()

PGHOST = os.getenv("PGHOST", "127.0.0.1")
PGPORT = os.getenv("PGPORT", "5432")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")
PGDATABASE = os.getenv("PGDATABASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATABASE_URL = f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

engine = create_engine(DATABASE_URL, echo=False, future=True)

def create_tables():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_facts (
                id SERIAL PRIMARY KEY,
                session_id TEXT NOT NULL,
                fact_key TEXT NOT NULL,
                fact_value TEXT NOT NULL,
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """))

create_tables()

def save_message(session_id: str, role: str, content: str):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO conversations (session_id, role, content)
            VALUES (:session_id, :role, :content)
        """), {"session_id": session_id, "role": role, "content": content})

def get_last_messages(session_id: str, limit: int = 5):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT role, content FROM conversations
            WHERE session_id = :session_id
            ORDER BY created_at DESC
            LIMIT :limit
        """), {"session_id": session_id, "limit": limit}).fetchall()
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

def upsert_user_fact(session_id: str, key: str, value: str):
    with engine.begin() as conn:
        existing = conn.execute(text("""
            SELECT id FROM user_facts
            WHERE session_id = :session_id AND fact_key = :key
        """), {"session_id": session_id, "key": key}).fetchone()
        if existing:
            conn.execute(text("""
                UPDATE user_facts SET fact_value = :value, updated_at = NOW()
                WHERE id = :id
            """), {"value": value, "id": existing[0]})
        else:
            conn.execute(text("""
                INSERT INTO user_facts (session_id, fact_key, fact_value)
                VALUES (:session_id, :key, :value)
            """), {"session_id": session_id, "key": key, "value": value})

def get_user_facts(session_id: str) -> dict:
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT fact_key, fact_value FROM user_facts
            WHERE session_id = :session_id
        """), {"session_id": session_id}).fetchall()
    return {row[0]: row[1] for row in rows}

# Initialize chat model for both chat and fact extraction
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

SESSION_ID = "default_session"

def extract_facts_with_llm(session_id: str, user_input: str):
    """
    Use LLM to extract key user facts in JSON style from user input, then upsert facts.
    """
    # Construct prompt for fact extraction
    prompt = [
        SystemMessage(content=(
            "Extract user facts from the following input in JSON format. "
            "Only extract facts like name, profession, location, age, etc. "
            "If no facts found, return empty JSON {}."
        )),
        HumanMessage(content=user_input)
    ]

    response = chat.invoke(prompt)
    # Expected to return JSON string, e.g. {"name":"Abdullah", "profession":"backend engineer"}

    import json
    try:
        facts = json.loads(response.content)
        if isinstance(facts, dict):
            for key, value in facts.items():
                if isinstance(value, str) and value.strip():
                    upsert_user_fact(session_id, key.lower(), value.strip())
    except json.JSONDecodeError:
        # fallback: no facts extracted
        pass

def chat_with_memory(user_input: str) -> str:
    # Extract facts flexibly using LLM prompt
    extract_facts_with_llm(SESSION_ID, user_input)
    save_message(SESSION_ID, "user", user_input)

    facts = get_user_facts(SESSION_ID)
    facts_str = ", ".join(f"{k.capitalize()} = {v}" for k, v in facts.items())
    system_prompt = (
        "You are a helpful assistant. Use the following user facts to answer all queries. "
        f"User facts: {facts_str if facts_str else 'No facts known yet.'} "
        "If user asks unrelated questions, still respond politely and helpfully."
    )

    messages = [SystemMessage(content=system_prompt)]

    history = get_last_messages(SESSION_ID, limit=5)
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_input))

    response = chat.invoke(messages)
    save_message(SESSION_ID, "assistant", response.content)
    return response.content

if __name__ == "__main__":
    print("🤖 Chatbot with flexible memory and fact extraction. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() == "exit":
            break
        reply = chat_with_memory(user_input)
        print(f"Bot: {reply}")
