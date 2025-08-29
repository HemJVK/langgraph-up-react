import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore

# Initialize a FAISS vector store
embedding_size = 1536  # Dimensions of OpenAI embeddings
index = faiss.IndexFlatL2(embedding_size)
embedding_fn = OpenAIEmbeddings().embed_query
docstore = InMemoryDocstore({})

vector_store = FAISS(embedding_fn, index, docstore, {})

def add_document(document: str):
    """Adds a document to the vector store."""
    vector_store.add_texts([document])

def search_documents(query: str, k: int = 4) -> list[str]:
    """Searches for similar documents in the vector store."""
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]
