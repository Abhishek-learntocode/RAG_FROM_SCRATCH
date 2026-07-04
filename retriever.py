import torch
from transformers import pipeline
from database import DatabaseManager
from indexer import DocumentIndexer

class RAGPipeline:
    def __init__(self, db_manager: DatabaseManager, indexer: DocumentIndexer, generator_model_name: str = "google/flan-t5-small"):
        self.db = db_manager
        self.indexer = indexer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading generator model ({generator_model_name}) on {self.device}...")
        device_id = 0 if self.device == 'cuda' else -1
        if "t5" in generator_model_name.lower() or "bart" in generator_model_name.lower():
            self.qa_pipeline = pipeline("text2text-generation", model=generator_model_name, device=device_id)
        else:
            self.qa_pipeline = pipeline("text-generation", model=generator_model_name, device=device_id)

    def retrieve(self, query: str, top_k: int = 3) -> list:
        # 1. Embed query
        query_embedding = self.indexer.embedding_model.encode([query], convert_to_numpy=True)
        
        # 2. Search FAISS
        distances, indices = self.indexer.index.search(query_embedding, top_k)
        
        # 3. Fetch from SQLite
        chunk_ids = indices[0].tolist()
        chunk_ids = [cid for cid in chunk_ids if cid != -1]
        
        results = self.db.get_chunks_by_ids(chunk_ids)
        return results

    def generate_answer(self, query: str, contexts: list) -> str:
        if not contexts:
            return "I don't have enough context to answer this question."
            
        # Combine contexts
        context_text = "\\n\\n".join([f"Context (from {filename}):\\n{text}" for _, text, filename in contexts])
        
        # Create prompt
        prompt = f"Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.\\n\\n{context_text}\\n\\nQuestion: {query}\\nAnswer:"
        
        # Generate
        if self.qa_pipeline.task == "text2text-generation":
            response = self.qa_pipeline(prompt, max_length=200, num_return_sequences=1)
            return response[0]['generated_text']
        else:
            response = self.qa_pipeline(prompt, max_new_tokens=150, num_return_sequences=1, truncation=True)
            generated_text = response[0]['generated_text']
            # Extract just the answer part
            answer_start = generated_text.find("Answer:")
            if answer_start != -1:
                return generated_text[answer_start + len("Answer:"):].strip()
            return generated_text.strip()

    def query(self, query_text: str, top_k: int = 3) -> dict:
        print(f"\\nSearching for: '{query_text}'")
        contexts = self.retrieve(query_text, top_k=top_k)
        
        answer = self.generate_answer(query_text, contexts)
        
        return {
            "query": query_text,
            "answer": answer,
            "contexts": contexts
        }
