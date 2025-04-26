# rag_pipeline.py
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

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

# 🔧 문서 등록 함수
def build_vector_db(data_folder: str):
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    # 폴더 내 모든 .txt 파일 처리
    files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
    if not files:
        print(f"⚠️ '{data_folder}' 폴더에 .txt 파일이 없습니다.")
        return

    for file_name in files:
        file_path = os.path.join(data_folder, file_name)
        if not os.path.exists(file_path):
            print(f"⚠️ 파일이 존재하지 않음: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_name}_{i}"
            try:
                collection.add(
                    documents=[chunk],
                    embeddings=EMBED_MODEL.encode([chunk]).tolist(),
                    ids=[chunk_id],
                    metadatas=[{"title": file_name, "chunk_index": i}]
                )
                print(f"✅ '{file_name}' chunk {i} 추가됨")
            except chromadb.errors.IDAlreadyExistsError:
                print(f"⚠️ '{file_name}' chunk {i} 이미 존재하여 생략됨")