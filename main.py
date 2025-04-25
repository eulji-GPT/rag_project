# main.py
from rag_pipeline import build_vector_db, search_docs
from ollama_wrapper import get_ollama_answer

query = "도서관 운영시간이 어떻게 돼?"

# 1. 크로마DB에 문서 등록 (한 번만 실행)
build_vector_db(data_folder="./data")

# 2. 질문 → 검색
relevant_chunks = search_docs(query)

# 3. 검색된 문서 + 질문 → Ollama 응답
response = get_ollama_answer(query, relevant_chunks)
print(response)
