import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# =============================================================================
# Configuration & Constants
# =============================================================================
DATA_PATH = "./data"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "holybot_docs"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def ingest_documents():
    print("🚀 Initializing Holybot 2.0 Document Ingestion...")

    # 1. Clean previous Chroma database if present to ensure a fresh build
    if os.path.exists(CHROMA_PATH):
        print(f"🧹 Removing existing Chroma database at '{CHROMA_PATH}'...")
        shutil.rmtree(CHROMA_PATH)

    # 2. Check and load documents from /data
    documents = []
    
    if os.path.exists(DATA_PATH):
        print(f"📁 Scanning for documents in '{DATA_PATH}'...")
        
        # Load .txt files
        try:
            txt_loader = DirectoryLoader(
                DATA_PATH, 
                glob="**/*.txt", 
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            print(f"  └─ Loaded {len(txt_docs)} .txt document(s).")
        except Exception as e:
            print(f"  └─ Warning loading .txt files: {e}")

        # Load .pdf files
        try:
            pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            print(f"  └─ Loaded {len(pdf_docs)} .pdf document page(s).")
        except Exception as e:
            print(f"  └─ Warning loading .pdf files: {e}")

    else:
        os.makedirs(DATA_PATH, exist_ok=True)
        print(f"⚠️ Created missing '{DATA_PATH}' folder. Place your raw .txt and .pdf files there and re-run.")
        return

    if not documents:
        print("❌ No valid .txt or .pdf files found in /data. Ingestion aborted.")
        return

    # 3. Chunk documents into retrieval-friendly sizes
    print("✂️ Splitting documents into chunks (chunk_size=1000, chunk_overlap=200)...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(documents)
    print(f"  └─ Created {len(chunks)} total text chunks.")

    # 4. Initialize lightweight 384-dimension HuggingFace Embeddings
    print(f"⚙️ Initializing embedding model '{EMBEDDING_MODEL_NAME}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 5. Persist chunks to ChromaDB
    print(f"💾 Ingesting chunks into ChromaDB collection '{COLLECTION_NAME}' at '{CHROMA_PATH}'...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=COLLECTION_NAME
    )
    
    print("✅ Ingestion complete! Holybot 2.0 database is ready.")

if __name__ == "__main__":
    ingest_documents()