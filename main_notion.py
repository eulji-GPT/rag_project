# main.py
from rag_pipeline import search_docs  # build_vector_db는 제거
from ollama_wrapper import get_ollama_answer

print("💬 Notion 기반 RAG 시스템입니다. 질문을 입력하세요 ('exit' 입력 시 종료):\n")

try:
    while True:
        query = input("🙋 사용자 질문: ")

        if query.lower() in ("exit", "quit"):
            print("👋 종료합니다.")
            break

        # 1. 질문 → 관련 문서 검색 (ChromaDB에서)
        relevant_chunks = search_docs(query)
        
        print("\n🔎 검색된 문서:")
        for i, chunk in enumerate(relevant_chunks):
            print(f"[{i+1}] {chunk[:100]}...")  # 앞 100자만 미리보기


        # 2. 검색된 문서 + 질문 → Ollama 응답
        response = get_ollama_answer(query, relevant_chunks, stream=True)
        print("\n🧠 응답:", response)
        print("\n" + "-" * 50)

except KeyboardInterrupt:
    print("\n🛑 강제 종료됨.")
