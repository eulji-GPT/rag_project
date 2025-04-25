import shutil
import os

db_path = "./chroma_db"

if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print("✅ ChromaDB 초기화 완료 (폴더 삭제됨)")
else:
    print("⚠️ ChromaDB 경로가 존재하지 않음:", db_path)
