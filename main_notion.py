# main.py
from rag_pipeline import search_docs
from ollama_wrapper import get_ollama_answer

print("💬 Notion 기반 RAG 시스템입니다. 질문을 입력하세요 ('exit' 입력 시 종료):\n")

try:
    while True:
        query = input("🙋 사용자 질문: ")

        if query.lower() in ("exit", "quit"):
            print("👋 종료합니다.")
            break

        # 1. 질문 → 관련 문서 검색 (ChromaDB에서)
        relevant_chunks, metadatas = search_docs(query)

        print("\n🔎 검색된 문서:")
        for i, chunk in enumerate(relevant_chunks):
            title = metadatas[i].get("title", "제목 없음")
            print(f"[{i+1}] ({title}) {chunk[:100]}...")

        # 2. 검색된 문서 + 질문 → Ollama 응답
        response = get_ollama_answer(query, relevant_chunks, metadatas, stream=True)
        print("\n🧠 응답:", response)
        print("\n" + "-" * 50)

except KeyboardInterrupt:
    print("\n🛑 강제 종료됨.")
