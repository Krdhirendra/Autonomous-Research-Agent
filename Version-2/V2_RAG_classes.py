from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import chromadb
import uuid
from sentence_transformers import SentenceTransformer
from chromadb.config import Settings
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
from collections import defaultdict


class Chunks:
    def __init__(self, documents,chunk_size=1000,chunk_overlap=200):
        self.documents = documents
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self):
        """Split documents into smaller chunks for better RAG performance"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_text(self.documents)
        print(f"Split {len(self.documents)} documents into {len(split_docs)} chunks")
        
        return split_docs
    
# call this function as shown below
"""chunks = split_documents(all_pdf_documents)"""




class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings for {len(texts)} texts...")
        if isinstance(texts[0], str):
            texts = texts
        else:
            texts = [doc for doc in texts]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings


## initialize the embedding manager
"""embedding_manager=EmbeddingManager()
embedding_manager"""



class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""
    
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "data/vector_store"):
        """
        Initialize the vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_text.append(doc.page_content)
            
            # Embedding
            embeddings_list.append(embedding.tolist())
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

#initialise vector store
"""vectorstore=VectorStore()
vectorstore"""



class RAGRetriever:
    """
    Minimal chunk-level retriever.
    Responsibility:
    - Given a query, return relevant chunks from vector store.
    """

    def __init__(self, vector_store, embedding_manager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, score_threshold: float = 0.0, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve chunk-level evidence for a query.

        Returns:
            List of dicts with:
            - content
            - metadata
            - similarity_score
        """

        # 1. Embed query
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # 2. Vector search
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )

        # 3. Build chunk-level results
        retrieved_chunks = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document, metadata, distance in zip(documents, metadatas, distances):
            similarity_score = 1 - distance
            if similarity_score >= score_threshold:
                retrieved_chunks.append({
                    "content": document,
                    "metadata": metadata,
                    "similarity_score": similarity_score
                })

        return retrieved_chunks

    """
    rag_retriever=RAGRetriever(vectorstore,embedding_manager)
    rag_retriever.retrieve("What is attention is all you need")
    """


    def group_chunks_by_paper(self, retrieved_docs:List[Dict[str,Any]],
                            source_key: str="source_file")-> List[Dict[str, Any]]:
        
        """Group retrieved chunks into paper-level objects.
        
        Args: 
            retrieved_docs : Output of the RAG retriever
            source_key: key from metadata identifying the paper
            
        Returns:
            List of paper-level dictionaries """
        
        papers = defaultdict(list)
        #group chunks by papers
        for doc in retrieved_docs:
            metadata = doc.get("metadata", {})
            paper_id = metadata.get(source_key) or metadata.get("source")

            if paper_id is None:
                continue

            papers[paper_id].append(doc)

        paper_objects = []
        for paper_id, chunks in papers.items():
            chunks_sorted = sorted(
                chunks, key=lambda x: x['similarity_score'], reverse=True
            )

            best_chunk = chunks_sorted[0]

            paper_objects.append({
                "source":paper_id,
                "chunks":chunks_sorted,
                "best_chunk":best_chunk['content'],
                "score":best_chunk['similarity_score'],
                "metadata": best_chunk["metadata"]
                })
            
        return paper_objects
    

