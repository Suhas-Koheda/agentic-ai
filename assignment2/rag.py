import os
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

def load_documents():
    """
    Loads all four required documents from the local filesystem.
    """
    docs = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    doc_files = {
        "Company Policy Document": "documents/company_policy.txt",
        "Pricing Guide": "documents/pricing_guide.txt",
        "Technical Manual": "documents/technical_manual.txt",
        "FAQ Document": "documents/faq_document.txt"
    }
    
    for doc_name, relative_path in doc_files.items():
        full_path = os.path.join(base_dir, relative_path)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
                docs.append(Document(
                    page_content=content,
                    metadata={"source": doc_name}
                ))
        else:
            print(f"Warning: File {full_path} not found.")
    return docs

def get_context_for_sources(query: str, sources: List[str]) -> str:
    """
    Filters document splits by source and retrieves matching chunks for a customer query.
    """
    docs = load_documents()
    if not docs:
        raise ValueError("No documents loaded! Check document paths.")
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    splits = text_splitter.split_documents(docs)
    
    # Filter splits by metadata source
    filtered_splits = [s for s in splits if s.metadata.get("source") in sources]
    
    if not filtered_splits:
        return "No relevant documents found for this query."
        
    # Initialize BM25 retriever for this subset of documents
    retriever = BM25Retriever.from_documents(filtered_splits)
    retriever.k = min(3, len(filtered_splits))
    
    matched_docs = retriever.invoke(query)
    
    formatted_chunks = []
    for idx, doc in enumerate(matched_docs):
        source = doc.metadata.get("source", "Unknown Document")
        formatted_chunks.append(
            f"[Source: {source}]\n{doc.page_content.strip()}"
        )
        
    return "\n\n---\n\n".join(formatted_chunks)

def get_retrieved_context(query: str) -> str:
    """
    Retrieves matching document chunks from all documents.
    """
    all_sources = ["Company Policy Document", "Pricing Guide", "Technical Manual", "FAQ Document"]
    return get_context_for_sources(query, all_sources)

if __name__ == "__main__":
    print("Testing Source-filtered RAG Retriever...")
    query = "What is the price of the Standard Plan?"
    # Sales should check Pricing Guide and FAQ Document
    context = get_context_for_sources(query, ["Pricing Guide", "FAQ Document"])
    print(f"Query: {query}")
    print(f"Retrieved Context:\n{context}")
