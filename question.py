#question.py
from rag import setup_rag_chatbot
import os

class Question:
    def __init__(self):
        """Initialize the RAG chatbot."""        
        self.rag_chain = self.rag_init()

    def rag_init(self):
        """Initialize the RAG chatbot and return the chain object."""
        try:
            return setup_rag_chatbot()
        except Exception as e:
            print(f"Error initializing RAG chatbot: {e}")
            return None

    def ask_question(self, question):
        """Ask a question to the RAG chatbot and return the response."""
        if not self.rag_chain:
            print("RAG chatbot is not initialized.")
            return "Error: RAG chatbot initialization failed."

        try:
            response = self.rag_chain.invoke(question)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error during query: {e}"

# ✅ Main function for testing
if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv()
    bot = Question()
    
    while True:
        user_input = input("Ask a question (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = bot.ask_question(user_input)
        print(f"Bot: {response}")
