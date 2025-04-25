# ollama_wrapper.py
import ollama

# 🔧 RAG 프롬프트 생성 함수
def build_rag_prompt(query: str, chunks: list[str], metadatas: list[dict]) -> str:
    prompt = "당신은 정보를 요약해서 질문에 정확히 답변하는 AI 비서입니다.\n\n"
    for i, chunk in enumerate(chunks):
        title = metadatas[i].get("title", "알 수 없는 문서")
        prompt += f"[문서 제목: {title}]\n내용:\n{chunk.strip()}\n\n---\n\n"
    prompt += f"질문: {query}\n답변:"
    return prompt

# 🔄 Ollama 호출 함수
def get_ollama_answer(query: str, context_chunks: list[str], metadatas: list[dict], stream: bool = True) -> str:
    prompt = build_rag_prompt(query, context_chunks, metadatas)
    messages = [{"role": "user", "content": prompt}]

    if stream:
        print("\n💬 AI 응답:")
        response = ollama.chat(model="exaone3.5:7.8b", messages=messages, stream=True)
        for chunk in response:
            print(chunk["message"]["content"], end="", flush=True)
        print("\n")
        return ""
    else:
        response = ollama.chat(model="exaone3.5:7.8b", messages=messages)
        return response["message"]["content"]
