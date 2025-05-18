import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from tempfile import NamedTemporaryFile

# --- API KEY Configuration ---
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- Streamlit App Setup ---
st.set_page_config(page_title="📚 Gemini RAG Chatbot", layout="centered")
st.title("🤖 Gemini Chatbot with File-based RAG")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

# --- Initialize session memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Load and process the PDF ---
def process_file(upload):
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(upload.read())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(split_docs, embeddings)

    return vector_store

# --- Chat interface ---
if uploaded_file:
    vectorstore = process_file(uploaded_file)

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
    )

    st.success("PDF processed! Ask your questions below.")
    query = st.chat_input("Ask something about the document...")

    if query:
        response = chain.run(query)
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("Bot", response))

    for role, text in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(text)
else:
    st.info("Please upload a PDF document to begin.")
