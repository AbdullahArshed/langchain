import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Load .env variables
load_dotenv()

# Load DB and API config
PGHOST = os.getenv("PGHOST", "127.0.0.1")
PGPORT = os.getenv("PGPORT", "5432")
PGUSER = os.getenv("PGUSER")
PGPASSWORD = os.getenv("PGPASSWORD")
PGDATABASE = os.getenv("PGDATABASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Construct DATABASE URL
DATABASE_URL = f"postgresql://{PGUSER}:{PGPASSWORD}@{PGHOST}:{PGPORT}/{PGDATABASE}"

# Set OpenAI key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Init database engine
engine = create_engine(DATABASE_URL, echo=False, future=True)

# Create tables
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

# Save message in conversation table
def save_message(session_id: str, role: str, content: str):
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO conversations (session_id, role, content)
            VALUES (:session_id, :role, :content)
        """), {"session_id": session_id, "role": role, "content": content})

# Get last N messages
def get_last_messages(session_id: str, limit: int = 5):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT role, content FROM conversations
            WHERE session_id = :session_id
            ORDER BY created_at DESC
            LIMIT :limit
        """), {"session_id": session_id, "limit": limit}).fetchall()
    return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

# Insert or update a user fact
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

# Retrieve all facts as a dictionary
def get_user_facts(session_id: str) -> dict:
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT fact_key, fact_value FROM user_facts
            WHERE session_id = :session_id
        """), {"session_id": session_id}).fetchall()
    return {row[0]: row[1] for row in rows}

# Extract facts from user input
def extract_and_store_facts(session_id: str, user_input: str):
    lower = user_input.lower()
    if "my name is" in lower:
        name = user_input.split("my name is")[-1].strip().split()[0]
        upsert_user_fact(session_id, "name", name)
    if "i am a" in lower:
        profession = user_input.split("i am a")[-1].strip().split('.')[0]
        upsert_user_fact(session_id, "profession", profession)
    if "i live in" in lower:
        location = user_input.split("i live in")[-1].strip().split('.')[0]
        upsert_user_fact(session_id, "location", location)
    if "moved to" in lower:
        new_location = user_input.split("moved to")[-1].strip().split('.')[0]
        upsert_user_fact(session_id, "location", new_location)

# Initialize Chat Model
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Session ID (can use UUID or username in production)
SESSION_ID = "default_session"

# Chat function with fact memory
def chat_with_memory(user_input: str) -> str:
    extract_and_store_facts(SESSION_ID, user_input)
    save_message(SESSION_ID, "user", user_input)

    facts = get_user_facts(SESSION_ID)
    facts_str = ", ".join(f"{k.capitalize()} = {v}" for k, v in facts.items())
    system_prompt = f"User facts: {facts_str}" if facts else "No known facts about the user."

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

# Command line interface
if __name__ == "__main__":
    print("🤖 Chatbot with PostgreSQL memory. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower().strip() == "exit":
            break
        reply = chat_with_memory(user_input)
        print(f"Bot: {reply}")
