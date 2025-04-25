# notion_sync.py
from notion_client import Client
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from notion_config import NOTION_TOKEN, DATABASE_ID

# 🔧 초기 설정
notion = Client(auth=NOTION_TOKEN)
embed_model = SentenceTransformer("jhgan/ko-sbert-sts")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="notion_docs")

# ✅ 제목 추출 함수
def extract_title(page):
    for key in page["properties"]:
        if page["properties"][key]["type"] == "title":
            title_data = page["properties"][key]["title"]
            return title_data[0]["plain_text"] if title_data else "(제목 없음)"
    return "(제목 없음)"

# ✅ 본문 내용 추출 함수
def fetch_page_content(page_id):
    blocks = notion.blocks.children.list(block_id=page_id)["results"]
    content = ""
    for block in blocks:
        if block["type"] == "paragraph":
            texts = block["paragraph"]["rich_text"]
            for t in texts:
                content += t.get("plain_text", "") + " "
        content += "\n"
    return content.strip()

# ✅ Notion DB → ChromaDB 동기화 함수
def sync_notion_to_chroma():
    pages = notion.databases.query(database_id=DATABASE_ID)["results"]
    print(f"🔄 총 {len(pages)}개의 페이지를 동기화합니다.")

    for idx, page in enumerate(pages):
        title = extract_title(page)
        content = fetch_page_content(page["id"])
        full_text = f"제목: {title}\n내용:\n{content}"

        embedding = embed_model.encode([full_text]).tolist()
        doc_id = f"notion_{page['id'].replace('-', '')}"

        # 중복 방지: 이미 존재하는 ID는 생략
        try:
            collection.add(
                documents=[full_text],
                embeddings=embedding,
                ids=[doc_id],
                metadatas=[{"title": title}]
            )
            print(f"✅ [{idx+1}] '{title}' 추가됨")
        except chromadb.errors.IDAlreadyExistsError:
            print(f"⚠️  [{idx+1}] '{title}' 이미 존재하여 생략됨")

if __name__ == "__main__":
    sync_notion_to_chroma()