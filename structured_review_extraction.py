"""
PydanticAI 結構化資料提取範例
將非結構化的影評文字轉換為結構化的 Python 物件

文檔可參考：https://ai.pydantic.dev/models/google/
"""

from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel

# 載入 .env 檔案中的環境變數
load_dotenv()

# 1. 定義結構化資料模型 (使用 Pydantic BaseModel)
# 這是 PydanticAI 的核心特色：型別安全的資料結構
class FilmCriticReview(BaseModel):
    film_name: str = Field(description="被評論的電影名稱")
    rating: int = Field(description="從 1 到 10 的評分", ge=1, le=10)
    critique: str = Field(description="一句話的簡短評論")

# 2. 準備非結構化的原始文字（真實場景中可能來自用戶輸入、檔案等）
unstructured_review = """
我昨天看了駭客任務, 整體超酷的, 我給七點二分
"""

# 3. 建立 AI 模型（支援多種 LLM：OpenAI, Anthropic, Google 等）
model = GoogleModel('gemini-2.0-flash-exp')

# 4. 建立 PydanticAI Agent，指定輸出格式
# 這是 PydanticAI 的魔法：自動將 LLM 輸出轉換為 Pydantic 模型
agent = Agent(model, output_type=FilmCriticReview)

# 5. 執行結構化提取,
# 同步執行：阻塞式調用，直到獲得完整結果才返回
result = agent.run_sync(f"從以下影評中提取結構化資訊: '{unstructured_review}'")

# 6. 獲得型別安全的結構化資料
# PydanticAI 確保輸出符合定義的模型格式
review_object = result.output
print("原始輸出:")
print(review_object)

print("\n=== 結構化資料提取結果 ===")
print(f"電影名稱: {review_object.film_name}")
print(f"評分: {review_object.rating}/10")
print(f"評論: {review_object.critique}")

# 7. 型別安全的好處：IDE 會提供自動完成和型別檢查
# review_object.film_name 會被識別為 str 型別
# review_object.rating 會被識別為 int 型別且限制在 1-10 之間
