import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import tempfile
import os

# ---- Page Config ----
st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")
st.title("💬 Gemini Chatbot with Memory + File RAG")

# ---- API Key Setup ----
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---- Upload File ----
uploaded_file = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

# ---- Load and Split Document ----
def load_and_split(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        st.error("Unsupported file format.")
        return None
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

# ---- Vector Store ----
@st.cache_resource
def create_vectorstore(docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return FAISS.from_documents(docs, embeddings)

# ---- Chat Model + Memory ----
@st.cache_resource
def get_conversational_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )
    return qa_chain

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Initialize RAG and Chain ----
rag_chain = None
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name
    docs = load_and_split(file_path)
    if docs:
        vectorstore = create_vectorstore(docs)
        rag_chain = get_conversational_chain(vectorstore)
        st.success("✅ File processed and RAG model is ready.")

# ---- Chat Input ----
user_input = st.chat_input("Ask me anything...")

if user_input and rag_chain:
    with st.spinner("Thinking..."):
        response = rag_chain({"question": user_input})
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("ai", response["answer"]))

# ---- Display Chat ----
for speaker, message in st.session_state.chat_history:
    if speaker == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)
