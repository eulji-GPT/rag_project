# main.py
from rag_pipeline import build_vector_db, search_docs
from ollama_wrapper import get_ollama_answer

# 1. 크로마DB에 문서 등록 (최초 1회만 실행하면 됨, 이미 했다면 주석처리해도 됨)
build_vector_db(data_folder="./data")

print("💬 RAG 시스템에 오신 걸 환영합니다. 질문을 입력하세요 ('exit' 입력 시 종료):\n")

try:
    while True:
        query = input("🙋 사용자 질문: ")

        if query.lower() in ("exit", "quit"):
            print("👋 종료합니다.")
            break

        # 2. 질문 → 관련 문서 검색
        relevant_chunks = search_docs(query)

        # 3. 관련 문서 + 질문 → LLM 응답
        response = get_ollama_answer(query, relevant_chunks, stream=True)  # stream=True도 가능
        print("\n🧠 응답:", response)
        print("\n" + "-" * 50)

except KeyboardInterrupt:
    print("\n🛑 강제 종료됨.")
