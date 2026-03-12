
import os
import streamlit as st
from typing import List

# LangChain and Loaders
from langchain_community.document_loaders import PyPDFLoader, TextLoader
rom langchain_community.document_loaders import PyMuPDFLoader, TextLoader
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

# Mock DB for User Access
USER_DB = {
    "student_csc": {"role": "CSC Student", "access": ["CSC"]},
    "student_ece": {"role": "ECE Student", "access": ["ECE"]},
    "admin_staff": {"role": "Administrator", "access": ["administration"]},
    "dean": {"role": "Dean", "access": ["CSC", "ECE", "MECH", "administration"]}
}

# --- Helper: Get available files for the user ---
def get_available_files(allowed_folders: List[str]) -> List[str]:
    """Scans the local directory and returns a list of files the user is allowed to see."""
    available_files = []
    if os.path.exists(BASE_DIR):
        for root, dirs, files in os.walk(BASE_DIR):
            department = os.path.basename(root)
            if department in allowed_folders:
                available_files.extend(files)
    return available_files

# --- 2. Backend Functions ---
def ingest_university_directory():
    """Scans the university folder and builds the FAISS index automatically."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    
    docs_to_insert = []
    
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            department = os.path.basename(root) 
            
            try:
                if file.endswith(".pdf"):
                    loader = PyMuPDFLoader(file_path) # Using PyMuPDFLoader as discussed!
                else:
                    loader = TextLoader(file_path)
                
                raw_documents = loader.load()
                pages = text_splitter.split_documents(raw_documents) 
                
                for page in pages:
                    # Inject Metadata
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

# ---> NEW: target_file parameter added here <---
def query_rag_agent(question: str, allowed_directories: List[str], target_file: Optional[str] = None):
    if not os.path.exists(FAISS_INDEX_PATH):
        return "Error: Database is empty. Please index the directory first.", []

    llm = ChatGroq(model=GROQ_MODEL_NAME, temperature=0.1)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    # ---> NEW: Updated RBAC Filter logic <---
    def rbac_filter(metadata: dict) -> bool:
        # 1. Does the user have access to this folder?
        doc_groups = metadata.get("read_access", [])
        has_folder_access = bool(set(allowed_directories) & set(doc_groups))
        
        # 2. If the user selected a specific file, filter by name too
        if target_file:
            is_target_file = metadata.get("name") == target_file
            return has_folder_access and is_target_file
            
        return has_folder_access

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

# --- 3. Streamlit UI ---
st.set_page_config(page_title="University RAG Portal", page_icon="🎓", layout="wide")
st.title("🎓 University Secure RAG Portal")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        
    st.divider()
    
    st.header("👤 User Login")
    selected_user = st.selectbox("Login As:", list(USER_DB.keys()))
    user_info = USER_DB[selected_user]
    
    st.caption(f"**Role:** {user_info['role']}")
    st.caption(f"**DB Access:** {user_info['access']}")
    
    st.divider()
    
    # ---> NEW: Search Scope UI <---
    st.header("🎯 Search Scope")
    scope_choice = st.radio(
        "Where do you want to search?", 
        ["All My Folders", "Specific Folder", "Specific PDF"]
    )
    
    # Default variables
    active_folders = user_info['access']
    target_file = None
    
    # Update filters based on what the user selects
    if scope_choice == "Specific Folder":
        selected_folder = st.selectbox("Choose Folder:", user_info['access'])
        active_folders = [selected_folder] # Restrict search to just this folder
        
    elif scope_choice == "Specific PDF":
        available_files = get_available_files(user_info['access'])
        if available_files:
            target_file = st.selectbox("Choose File:", available_files)
        else:
            st.warning("No files found in your folders.")
            
    st.divider()
    
    st.header("🗄️ Database Management")
    if st.button("Scan & Index University Directory", type="primary"):
        with st.spinner("Scanning folders and building FAISS index..."):
            chunks = ingest_university_directory()
            if chunks > 0:
                st.success(f"Success! Indexed {chunks} chunks.")
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
            with st.spinner(f"Searching {scope_choice}..."):
                # ---> NEW: Passing target_file to the agent <---
                answer, retrieved_docs = query_rag_agent(prompt, active_folders, target_file)
                
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
