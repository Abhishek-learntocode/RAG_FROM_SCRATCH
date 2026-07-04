import argparse
import os
from database import DatabaseManager
from indexer import DocumentIndexer
from retriever import RAGPipeline

def init_system(db_path="rag_database.db", index_path="faiss_index.bin"):
    db = DatabaseManager(db_path)
    indexer = DocumentIndexer(db_manager=db, index_path=index_path)
    # Using flan-t5-small as a fast, reasonable default for QA tasks.
    rag = RAGPipeline(db_manager=db, indexer=indexer, generator_model_name="google/flan-t5-small")
    return db, indexer, rag

def main():
    parser = argparse.ArgumentParser(description="Retrieval-Augmented Generation (RAG) QA System")
    parser.add_argument("--ingest", type=str, help="Path to a document (PDF/TXT) or directory to ingest")
    parser.add_argument("--query", type=str, help="Question to ask the system")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive CLI mode")
    
    args = parser.parse_args()
    
    # Initialize components
    db, indexer, rag = init_system()
    
    if args.ingest:
        path = args.ingest
        if os.path.isfile(path):
            indexer.process_document(path)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(('.pdf', '.txt')):
                        full_path = os.path.join(root, file)
                        indexer.process_document(full_path)
        else:
            print(f"Error: {path} is not a valid file or directory.")
            
    if args.query:
        result = rag.query(args.query, top_k=args.top_k)
        print("\\n--- Answer ---")
        print(result['answer'])
        print("\\n--- Sources ---")
        for chunk_id, text, filename in result['contexts']:
            print(f"[{filename}] (Chunk {chunk_id}): {text[:100]}...")
            
    if args.interactive:
        print("\\nWelcome to the RAG QA System! Type 'exit' or 'quit' to stop.")
        while True:
            try:
                user_query = input("\\nAsk a question: ")
                if user_query.lower() in ['exit', 'quit']:
                    break
                if not user_query.strip():
                    continue
                    
                result = rag.query(user_query, top_k=args.top_k)
                print("\\n>> Answer:")
                print(result['answer'])
                print("\\n>> Sources:")
                for chunk_id, text, filename in result['contexts']:
                    print(f" - {filename} (ID: {chunk_id})")
            except KeyboardInterrupt:
                break
                
    if not any([args.ingest, args.query, args.interactive]):
        parser.print_help()

if __name__ == "__main__":
    main()
