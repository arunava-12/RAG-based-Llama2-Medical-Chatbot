from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 

DATA_PATH = 'data/'
DB_FAISS_PATH = 'vectorstore/db_faiss'

# Create vector database
def create_vector_db():
    print("[INFO] Loading documents from:", DATA_PATH)
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} documents.")

    print("[INFO] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"[INFO] Split into {len(texts)} chunks.")

    print("[INFO] Creating embeddings using HuggingFace model...")
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                       model_kwargs={'device': 'cpu'})
    print("[INFO] Embeddings model loaded.")

    print("[INFO] Building FAISS vector database...")
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"[SUCCESS] Vector database saved to: {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()
