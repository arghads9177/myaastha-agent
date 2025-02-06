import streamlit as st
from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Initialize embeddings and vector store
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Initialize Groq LLM
llm = ChatGroq(model_name="llama-3.1-8b-instant")  # Replace with the correct Groq model name

# Initialize memory for conversation history
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Create RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(),
    memory=memory,
)

# Streamlit UI
st.title("Aastha Co-operative Credit Society - Gen AI Agent")
st.write("Welcome! How can I assist you today?")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_query = st.text_input("Enter your query:")

if user_query:
    # Get response from the QA chain
    response = qa_chain.invoke(user_query)
    # Update chat history
    st.session_state.chat_history.append(f"User: {user_query}")
    st.session_state.chat_history.append(f"AI: {response['result']}")
    
    # Display chat history
    st.write("### Chat History")
    for message in st.session_state.chat_history:
        st.write(message)