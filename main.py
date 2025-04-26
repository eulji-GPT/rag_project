# main.py
from rag_pipeline import build_vector_db, search_docs
from ollama_wrapper import get_ollama_answer

# 1. 크로마DB에 문서 등록 (한 번만 실행)
build_vector_db(data_folder="./test_data")

print("💬 RAG 시스템에 오신 걸 환영합니다. 질문을 입력하세요 ('exit' 입력 시 종료):\n")

try:
    while True:
        query = input("🙋 사용자 질문: ")

        if query.lower() in ("exit", "quit"):
            print("👋 종료합니다.")
            break

        # 2. 질문 → 관련 문서 검색
        relevant_chunks, metadatas = search_docs(query)

        print("\n🔎 검색된 문서:")
        for i, chunk in enumerate(relevant_chunks):
            title = metadatas[i].get("title", "제목 없음")
            print(f"[{i+1}] ({title}) {chunk[:100]}...")

        # 3. 검색된 문서 + 질문 → Ollama 응답 (스트리밍 모드)
        print("\n💬 AI 응답:")
        get_ollama_answer(query, relevant_chunks, metadatas, stream=True)
        print("\n" + "-" * 50)

except KeyboardInterrupt:
    print("\n🛑 강제 종료됨.")
