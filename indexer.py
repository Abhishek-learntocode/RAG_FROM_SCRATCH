import os
import faiss
import numpy as np
import PyPDF2
from typing import List
from sentence_transformers import SentenceTransformer
import torch
from database import DatabaseManager

class DocumentIndexer:
    def __init__(self, db_manager: DatabaseManager, model_name: str = 'all-MiniLM-L6-v2', index_path: str = 'faiss_index.bin', chunk_size: int = 500, chunk_overlap: int = 50):
        self.db = db_manager
        self.index_path = index_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading embedding model on {self.device}...")
        self.embedding_model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Load or initialize FAISS index
        self.index = self._load_or_create_index()

    def _load_or_create_index(self) -> faiss.IndexIDMap:
        if os.path.exists(self.index_path):
            print(f"Loading FAISS index from {self.index_path}")
            return faiss.read_index(self.index_path)
        else:
            print("Creating new FAISS index")
            base_index = faiss.IndexFlatL2(self.embedding_dim)
            return faiss.IndexIDMap(base_index)

    def save_index(self):
        faiss.write_index(self.index, self.index_path)

    def extract_text(self, file_path: str) -> str:
        if file_path.lower().endswith('.pdf'):
            text = ""
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\\n"
            return text
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()

    def split_text(self, text: str) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            if end >= len(text):
                chunks.append(text[start:].strip())
                break
                
            break_point = end
            last_newline = text.rfind('\\n', start, end)
            last_space = text.rfind(' ', start, end)
            
            if last_newline != -1 and last_newline > start + self.chunk_size // 2:
                break_point = last_newline + 1
            elif last_space != -1 and last_space > start + self.chunk_size // 2:
                break_point = last_space + 1
                
            chunks.append(text[start:break_point].strip())
            
            next_start = break_point - self.chunk_overlap
            if next_start <= start:
                start += 1 # Force forward progress
            else:
                start = next_start
                
        return [c for c in chunks if c]

    def process_document(self, file_path: str):
        print(f"Processing document: {file_path}")
        filename = os.path.basename(file_path)
        
        # 1. Extract text
        text = self.extract_text(file_path)
        if not text.strip():
            print(f"No text extracted from {file_path}")
            return

        # 2. Add to database
        doc_id = self.db.add_document(filename)
        
        # 3. Chunk text
        chunks = self.split_text(text)
        print(f"Created {len(chunks)} chunks.")
        
        # 4. Add chunks to database and get their IDs
        chunk_ids = self.db.add_chunks(doc_id, chunks)
        
        # 5. Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
        
        # 6. Add to FAISS
        ids_array = np.array(chunk_ids, dtype=np.int64)
        self.index.add_with_ids(embeddings, ids_array)
        
        # 7. Save FAISS index
        self.save_index()
        print(f"Successfully indexed {filename}")
