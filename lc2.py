import os
from dotenv import load_dotenv

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Get API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY not found in .env")

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=openai_api_key
)

# Setup memory
memory = ConversationBufferMemory(return_messages=True)

# Create chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# Chat loop
def run_chat():
    print("✅ LangChain Memory Bot (Ctrl+C or type 'exit' to quit)\n")
    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() == "exit":
                break
            response = conversation.predict(input=user_input)
            print(f"Bot: {response}\n")
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break

if __name__ == "__main__":
    run_chat()
