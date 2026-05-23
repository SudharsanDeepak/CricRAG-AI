import os
import glob
import pandas as pd
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from pypdf import PdfReader

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "ipl_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

class RAGEngine:
    def __init__(self):
        print("Initializing Embedding Model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding Model Loaded Successfully!")
        
        # Initialize Persistent ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
        
        # Get or create collection
        # We define a custom embedding function adapter for Chroma if needed, or pass list of embeddings directly
        self._collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    @property
    def collection(self):
        try:
            self._collection.count()
        except Exception:
            self._collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    @collection.setter
    def collection(self, value):
        self._collection = value

    def extract_text_from_file(self, filepath: str) -> str:
        """Extracts plain text from PDF, MD, or TXT files."""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == ".pdf":
            try:
                reader = PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    text_content = page.extract_text()
                    if text_content:
                        text += text_content + "\n"
                return text
            except Exception as e:
                print(f"Error reading PDF {filepath}: {e}")
                return ""
        elif ext in [".txt", ".md"]:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading text file {filepath}: {e}")
                return ""
        return ""

    def ingest_directory(self, dir_path: str) -> Tuple[int, int]:
        """Ingests all TXT, MD, and PDF files in a directory into ChromaDB."""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            return 0, 0
            
        supported_patterns = ["*.txt", "*.md", "*.pdf"]
        files = []
        for pattern in supported_patterns:
            files.extend(glob.glob(os.path.join(dir_path, pattern)))
            
        total_files = len(files)
        total_chunks = 0
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=120,
            length_function=len
        )
        
        for filepath in files:
            filename = os.path.basename(filepath)
            # Check if this file is already in metadata to avoid duplicate ingestion
            # (In a simple setup we can also just inspect existing IDs or overwrite)
            file_text = self.extract_text_from_file(filepath)
            if not file_text.strip():
                continue
                
            chunks = text_splitter.split_text(file_text)
            if not chunks:
                continue
                
            # Prepare embeddings
            embeddings = self.embedding_model.encode(chunks).tolist()
            
            ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
            
            self.collection.upsert(
                ids=ids,
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas
            )
            total_chunks += len(chunks)
            print(f"Ingested {filename}: Split into {len(chunks)} chunks.")
            
        return total_files, total_chunks

    def ingest_single_file(self, filepath: str) -> int:
        """Ingests a single uploaded file into ChromaDB."""
        if not os.path.exists(filepath):
            return 0
            
        filename = os.path.basename(filepath)
        file_text = self.extract_text_from_file(filepath)
        if not file_text.strip():
            return 0
            
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=120
        )
        chunks = text_splitter.split_text(file_text)
        if not chunks:
            return 0
            
        embeddings = self.embedding_model.encode(chunks).tolist()
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
        
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"Ingested single file {filename}: {len(chunks)} chunks added.")
        return len(chunks)

    def query(self, question: str, n_results: int = 4) -> List[Dict[str, Any]]:
        """Queries the vector database for a given question and returns results with scores."""
        query_embedding = self.embedding_model.encode([question]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        formatted_results = []
        if results and results["documents"]:
            docs = results["documents"][0]
            ids = results["ids"][0]
            metadatas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
            distances = results["distances"][0] if results["distances"] else [0.0] * len(docs)
            
            for doc, doc_id, meta, dist in zip(docs, ids, metadatas, distances):
                # Convert cosine distance to similarity score
                similarity = 1.0 - dist
                formatted_results.append({
                    "id": doc_id,
                    "content": doc,
                    "source": meta.get("source", "Unknown"),
                    "chunk_index": meta.get("chunk_index", 0),
                    "score": round(similarity, 4)
                })
        return formatted_results

    def get_db_stats(self) -> Dict[str, Any]:
        """Returns statistics about the vector collection."""
        try:
            count = self.collection.count()
            # Fetch all metadata to get unique sources
            results = self.collection.get(include=["metadatas"])
            sources = set()
            if results and results["metadatas"]:
                for meta in results["metadatas"]:
                    if meta and "source" in meta:
                        sources.add(meta["source"])
            return {
                "total_chunks": count,
                "unique_sources": list(sources),
                "source_count": len(sources)
            }
        except Exception as e:
            print(f"Error getting DB stats: {e}")
            return {"total_chunks": 0, "unique_sources": [], "source_count": 0}

    def clear_database(self):
        """Clears the collection database."""
        self.chroma_client.delete_collection(COLLECTION_NAME)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print("Database collection cleared!")
