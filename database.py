import sqlite3
from typing import List, Tuple

class DatabaseManager:
    def __init__(self, db_path: str = "rag_database.db"):
        self.db_path = db_path
        self._create_tables()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT UNIQUE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER,
                    chunk_text TEXT,
                    chunk_index INTEGER,
                    FOREIGN KEY(document_id) REFERENCES documents(id)
                )
            ''')
            conn.commit()

    def add_document(self, filename: str) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('INSERT INTO documents (filename) VALUES (?)', (filename,))
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Document already exists, return its id
                cursor.execute('SELECT id FROM documents WHERE filename = ?', (filename,))
                return cursor.fetchone()[0]

    def add_chunks(self, document_id: int, chunks: List[str]) -> List[int]:
        chunk_ids = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for i, chunk_text in enumerate(chunks):
                cursor.execute(
                    'INSERT INTO chunks (document_id, chunk_text, chunk_index) VALUES (?, ?, ?)',
                    (document_id, chunk_text, i)
                )
                chunk_ids.append(cursor.lastrowid)
            conn.commit()
        return chunk_ids

    def get_chunks_by_ids(self, chunk_ids: List[int]) -> List[Tuple[int, str, str]]:
        if not chunk_ids:
            return []
        placeholders = ','.join('?' for _ in chunk_ids)
        query = f'''
            SELECT c.id, c.chunk_text, d.filename
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.id IN ({placeholders})
        '''
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, chunk_ids)
            results = cursor.fetchall()
            
            # Sort results to match the order of requested chunk_ids
            id_to_result = {row[0]: row for row in results}
            ordered_results = [id_to_result[cid] for cid in chunk_ids if cid in id_to_result]
            return ordered_results
