#rag.py
import os
import time
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from IPython.display import Markdown, clear_output, display
import json
from datetime import datetime
from langchain_core.documents import Document 

from dotenv import load_dotenv

from rag_preprocessing import generate_data

# Custom JSON encoder to handle Timestamp and other non-serializable objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):  
            return obj.isoformat()
        elif hasattr(obj, "to_dict"): 
            return obj.to_dict()
        return super().default(obj)

# Convert dictionary to LangChain Document objects
def dict_to_documents(data):
    documents = []
    for section, content in data.items():
        text = f"{section}: {json.dumps(content, indent=2, cls=CustomJSONEncoder)}"
        documents.append(Document(page_content=text, metadata={"source": "generated_report"}))
    return documents

# Create a vector store from the dictionary (in-memory)
def create_vectorstore_from_dict(data):
    print("Creating vector store from dictionary...\n")
    
    print("Converting dictionary to documents...")
    documents = dict_to_documents(data)
    print(f"Created {len(documents)} documents\n")

    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n")
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} document chunks\n")

    print("Creating embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model loaded\n")

    print("Creating vector store (this may take a while)...")
    total_docs = len(docs)
    
    class EmbeddingProgress:
        def __init__(self, total):
            self.total = total
            self.pbar = tqdm(total=total, desc="Embedding documents")

        def update(self, _):
            self.pbar.update(1)

    progress = EmbeddingProgress(total_docs)

    text_embeddings = []
    for doc in docs:
        embedding = embeddings.embed_query(doc.page_content)
        text_embeddings.append((doc.page_content, embedding))  # Combine text and embedding into a tuple
        progress.update(1)
    progress.pbar.close()

    # Convert embeddings to FAISS format (in-memory)
    vectorstore = FAISS.from_embeddings(text_embeddings, embeddings)  # Pass the list of tuples
    
    print("Vector store created successfully\n")
    
    return vectorstore

# Set up the RAG chatbot with Gemini
def setup_rag_chatbot():

    report_data = generate_data()
    
    vectorstore = create_vectorstore_from_dict(report_data)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("Initializing Gemini model...")
    with tqdm(total=1, desc="Loading Gemini model") as pbar:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")  # or gemini-1.5-pro
        pbar.update(1)
    print("Gemini model initialized\n")
    
    # Create a prompt template
    template = """
    You are a AI chatbot that can answer questions based on Hotel Booking details. 
    Answer the following question based on the retrieved documents.
    Retrieved documents:
    {context}
    
    Question: {question}
    
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    print("Building RAG pipeline...")
    with tqdm(total=1, desc="Creating RAG chain") as pbar:
        rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm)
        pbar.update(1)
    print("RAG pipeline ready\n")
    
    return rag_chain

# Run the chatbot interactively
def run_rag_chatbot():
    rag_chain = setup_rag_chatbot()

    print("\nRAG Chatbot initialized. Type 'exit' to quit.")
    print("Ask questions about the hotel booking report:\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit", "q"]:
            print("Chatbot terminated.")
            break

        try:
            with tqdm(total=3, desc="Processing query") as pbar:
                pbar.set_description("Retrieving context")
                time.sleep(0.5)  # Simulating retrieval delay
                pbar.update(1)
                
                pbar.set_description("Generating response")
                response = rag_chain.invoke(user_input)
                pbar.update(2)
            
            try:
                shell = get_ipython().__class__.__name__
                if shell == 'ZMQInteractiveShell':
                    clear_output(wait=True)
                    display(Markdown(response.content))
                else:
                    print(f"\nChatbot: {response}")
            except NameError:
                print(f"\nChatbot: {response}")

        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    load_dotenv() 
    run_rag_chatbot()