
import os
import streamlit as st
from typing import List

# LangChain and Loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document as LangchainDocument
import uuid

# --- 1. Configuration & Mock DB ---
FAISS_INDEX_PATH = "./local_faiss_index"
BASE_DIR = "./university"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Avinash's Requirement: Mock DB for User Access
USER_DB = {
    "student_csc": {"role": "CSC Student", "access": ["CSC"]},
    "student_ece": {"role": "ECE Student", "access": ["ECE"]},
    "admin_staff": {"role": "Administrator", "access": ["administration"]},
    "dean": {"role": "Dean", "access": ["CSC", "ECE", "MECH", "administration"]}
}

# --- 2. Setup Dummy Directories ---
def setup_dummy_directories():
    """Generates the folder structure Avinash asked for so you can test immediately."""
    directories = [
        "university/academic/CSC",
        "university/academic/ECE",
        "university/academic/MECH",
        "university/administration"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        # Create a dummy text file to ensure the folder isn't empty
        dummy_file = os.path.join(directory, f"sample_{os.path.basename(directory)}.txt")
        if not os.path.exists(dummy_file):
            with open(dummy_file, "w") as f:
                f.write(f"This is a highly confidential sample document for the {os.path.basename(directory)} department. It contains secret policies.")

# --- 3. Backend Functions ---
def ingest_university_directory():
    """Scans the university folder and builds the FAISS index automatically."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    docs_to_insert = []
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            department = os.path.basename(root) # e.g., 'CSC', 'administration'
            
            # Support both PDF and TXT for easy testing
            try:
                if file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                else:
                    loader = TextLoader(file_path)
                
                raw_documents = loader.load()
                pages = text_splitter.split_documents(raw_documents) 
                
                for page in pages:
                    # Inject Folder Name as Metadata!
                    page.metadata["read_access"] = [department]
                    page.metadata["name"] = file
                    docs_to_insert.append(page)
            except Exception as e:
                st.error(f"Error loading {file}: {e}")

    if docs_to_insert:
        vectorstore = FAISS.from_documents(docs_to_insert, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)
        return len(docs_to_insert)
    return 0

def query_rag_agent(question: str, allowed_directories: List[str]):
    if not os.path.exists(FAISS_INDEX_PATH):
        return "Error: Database is empty. Please index the directory first.", []

    llm = ChatGroq(model=GROQ_MODEL_NAME, temperature=0.1)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    # PRE-FILTERING: Checks if the document's folder matches the user's allowed folders
    def rbac_filter(metadata: dict) -> bool:
        doc_groups = metadata.get("read_access", [])
        return bool(set(allowed_directories) & set(doc_groups))

    retriever = vectorstore.as_retriever(search_kwargs={"filter": rbac_filter, "k": 4})

    system_prompt = (
        "You are a helpful university AI assistant. Use the following pieces of retrieved context "
        "to answer the user's question. If the answer is not in the context, just say "
        "that you don't know based on the provided documents. Keep the answer clear and concise.\n\n"
        "<context>\n{context}\n</context>"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    response = rag_chain.invoke({"input": question})
    return response["answer"], response.get("context", [])

# --- 4. Streamlit UI ---
st.set_page_config(page_title="University RAG Portal", page_icon="🎓", layout="wide")
st.title("🎓 University Secure RAG Portal")

# Auto-generate dummy folders on startup so it works instantly
setup_dummy_directories()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        
    st.divider()
    
    # Avinash's Requirement 1: DB Authentication
    st.header("👤 Simulate User Login")
    selected_user = st.selectbox("Login As:", list(USER_DB.keys()))
    user_info = USER_DB[selected_user]
    
    st.caption(f"**Role:** {user_info['role']}")
    st.caption(f"**DB Access:** {user_info['access']}")
    
    st.divider()
    
    # Avinash's Requirement 2: Manual Filter Override
    st.header("📁 Manual Directory Filter")
    st.caption("Override the DB rules to test the retriever.")
    all_departments = ["CSC", "ECE", "MECH", "administration"]
    # Default selection to whatever the DB says the user has
    active_filters = st.multiselect("Select Folders to Query:", all_departments, default=user_info['access'])
    
    st.divider()
    
    # Avinash's Requirement 3: Scan folders instead of uploading files
    st.header("🗄️ Database Management")
    if st.button("Scan & Index University Directory", type="primary"):
        with st.spinner("Scanning folders and building FAISS index..."):
            chunks = ingest_university_directory()
            if chunks > 0:
                st.success(f"Success! Indexed {chunks} chunks from the university folders.")
            else:
                st.warning("No files found to index.")

# --- Main Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "context" in message:
            with st.expander("View Retrieved Context"):
                for i, chunk in enumerate(message["context"]):
                    st.markdown(f"**Chunk {i+1} (Folder: `{chunk.metadata.get('read_access', ['Unknown'])[0]}` | File: `{chunk.metadata.get('name', 'Unknown')}`)**")
                    st.write(chunk.page_content)

if prompt := st.chat_input("Ask a question about the university..."):
    if not os.environ.get("GROQ_API_KEY"):
        st.error("Please enter your Groq API Key in the sidebar.")
    elif not os.path.exists(FAISS_INDEX_PATH):
        st.error("No database found! Please click 'Scan & Index' in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Checking permissions and searching..."):
                # Pass the ACTIVE filter (from the UI multiselect) into the RAG agent
                answer, retrieved_docs = query_rag_agent(prompt, active_filters)
                
                st.markdown(answer)
                
                if retrieved_docs:
                    with st.expander(f"View Retrieved Context ({len(retrieved_docs)} chunks used)"):
                        for i, doc in enumerate(retrieved_docs):
                            folder_tag = doc.metadata.get('read_access', ['Unknown'])[0]
                            st.markdown(f"**Chunk {i+1} (Folder: `{folder_tag}` | File: `{doc.metadata.get('name', 'Unknown')}`)**")
                            st.write(doc.page_content)
                
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "context": retrieved_docs
        })
