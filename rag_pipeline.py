# rag_pipeline.py
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os

# 전역 설정
DB_DIR = "./chroma_db"
COLLECTION_NAME = "docs_ko"
EMBED_MODEL = SentenceTransformer("jhgan/ko-sbert-sts")

# 문서 → Chunk → 임베딩 → ChromaDB 저장
def build_vector_db(data_folder: str = "./data"):
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    for filename in os.listdir(data_folder):
        filepath = os.path.join(data_folder, filename)
        with open(filepath, encoding="utf-8") as f:
            text = f.read()

        chunks = splitter.split_text(text)
        embeddings = EMBED_MODEL.encode(chunks).tolist()
        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in chunks]

        collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)

    print("✅ Vector DB build complete.")

# 질의 → 임베딩 → 유사 문서 검색
def search_docs(query: str, k: int = 3):
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    query_embedding = EMBED_MODEL.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    return results["documents"][0]  # 리스트 반환
