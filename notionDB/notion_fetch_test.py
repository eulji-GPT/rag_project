# notion_fetch_test.py
from notion_client import Client
from notion_config import NOTION_TOKEN, DATABASE_ID

# Notion 클라이언트 생성
notion = Client(auth=NOTION_TOKEN)

# 데이터베이스에서 페이지 가져오기
def fetch_notion_data():
    results = notion.databases.query(database_id=DATABASE_ID)["results"]

    print("🔍 Notion DB에서 불러온 항목:")
    for i, page in enumerate(results):
        # 기본적으로 제목 속성 추출
        title_prop = page["properties"]["Name"]
        title = title_prop["title"][0]["plain_text"] if title_prop["title"] else "(제목 없음)"

        print(f"{i+1}. 제목: {title} / ID: {page['id']}")

if __name__ == "__main__":
    fetch_notion_data()
