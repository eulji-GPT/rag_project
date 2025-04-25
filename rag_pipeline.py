# rag_pipeline.py
from sentence_transformers import SentenceTransformer
import chromadb

# 전역 설정
DB_DIR = "./chroma_db"
COLLECTION_NAME = "notion_docs"  # Notion용 컬렉션으로 수정
EMBED_MODEL = SentenceTransformer("jhgan/ko-sbert-sts")

# 질의 → 임베딩 → 유사 문서 검색

def search_docs(query: str, k: int = 3):
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    query_embedding = EMBED_MODEL.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    return results["documents"][0], results["metadatas"][0]

