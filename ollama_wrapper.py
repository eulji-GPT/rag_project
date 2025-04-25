# ollama_wrapper.py
import ollama

# 문서들과 질문을 묶어 LLM에 전달
def get_ollama_answer(query: str, context_docs: list[str], stream: bool = True) -> str:
    prompt = f"""다음 정보를 참고하여 질문에 답변해 주세요.

[참고 문서]
{chr(10).join(context_docs)}

[질문]
{query}

[답변]
"""

    messages = [{"role": "user", "content": prompt}]

    if stream:
        # 실시간 스트리밍 응답
        response = ollama.chat(model="exaone3.5:7.8b", messages=messages, stream=True)
        print("💬 답변:")
        for chunk in response:
            print(chunk["message"]["content"], end="", flush=True)
        return ""
    else:
        # 한번에 응답
        response = ollama.chat(model="exaone3.5:7.8b", messages=messages)
        return response["message"]["content"]
