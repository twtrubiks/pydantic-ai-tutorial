"""
Gemini Embedding 範例（聚焦：用向量相似度做語意搜尋）

主題：Embeddings（文字向量 + cosine 相似度）
    用 Pydantic AI 的 Embedder 把句子轉成向量，再用 cosine 相似度比較語意。
    示範一個「語意搜尋」小場景：給一句查詢，從候選句子裡找出語意最接近的那幾句。

重點：
    - Embedder 是 Pydantic AI 的統一 embeddings 介面，跨 provider 寫法一致
    - embed_query() 給「查詢」、embed_documents() 給「被檢索的內容」——
      某些模型會針對這兩者做不同最佳化，分開呼叫語意更準
    - EmbeddingSettings(dimensions=768) 可調向量維度（越大越精細，索引也越大）

文檔參考：
    - https://pydantic.dev/docs/ai/guides/embeddings/
    - Gemini embeddings：https://ai.google.dev/gemini-api/docs/embeddings

執行前：
    1. 前往 https://aistudio.google.com/apikey 取得 API Key（格式 AIza...）
    2. 在 .env 檔案中設置 GOOGLE_API_KEY=AIza...
    3. python gemini_embedding_demo.py
"""

import asyncio
import math
import os

from dotenv import load_dotenv
from pydantic_ai import Embedder
from pydantic_ai.embeddings import EmbeddingSettings
from pydantic_ai.embeddings.google import GoogleEmbeddingModel
from pydantic_ai.providers.google import GoogleProvider

# 載入 .env 檔案中的環境變數
load_dotenv()

# ========== 模型設定 ==========

# 設置 Google API Key（前往 https://aistudio.google.com/apikey 取得，格式 AIza...）
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

MODEL_NAME = "gemini-embedding-2"

# 向量維度：可選 128~3072，常用 768 / 1536 / 3072；越大越精細、索引也越大。
DIM = 768


def build_embedder() -> Embedder:
    """建立 Embedder（延後到確認有 API Key 後才建，避免 import 即報錯）"""
    model = GoogleEmbeddingModel(
        MODEL_NAME,
        provider=GoogleProvider(api_key=GOOGLE_API_KEY),
    )
    return Embedder(model, settings=EmbeddingSettings(dimensions=DIM))


def cosine(a, b) -> float:
    """cosine 相似度：1 = 語意幾乎相同、0 = 不相關。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


# ========== 主題：語意搜尋 ==========


async def demo_semantic_search(embedder: Embedder):
    """語意搜尋：給一句查詢，從候選句子裡找語意最接近的。"""
    print("\n🔎 主題：用 embedding + cosine 做語意搜尋")
    print("=" * 60)

    query = "如何學習 Python 程式設計？"
    candidates = [
        "Python 入門教學與練習題",
        "新手該怎麼開始寫程式？",
        "今天台北的天氣如何？",
        "紅燒牛肉麵食譜",
    ]

    # 查詢用 embed_query、被檢索內容用 embed_documents（某些模型會分別最佳化）
    q_result = await embedder.embed_query(query)
    q_vec = q_result.embeddings[0]

    doc_result = await embedder.embed_documents(candidates)

    print(f"  模型：{MODEL_NAME}　維度：{len(q_vec)}")
    print(f"  查詢：{query}\n")

    # 逐一算查詢與每個候選的 cosine 相似度，再由高到低排序
    scored = [
        (cosine(q_vec, doc_result.embeddings[i]), c) for i, c in enumerate(candidates)
    ]
    scored.sort(reverse=True)

    print("  依語意相似度排序（高 → 低）：")
    for sim, c in scored:
        print(f"    {sim:.4f}  {c}")


# ========== 說明 ==========


def explain():
    """整理重點"""
    print("\n\n" + "=" * 60)
    print("💡 重點整理")
    print("=" * 60)

    print("\n1. Embeddings（語意向量）：")
    print("   • Embedder 是 Pydantic AI 的統一介面，換 provider 不用改呼叫方式")
    print("   • embed_query() 給查詢、embed_documents() 給被檢索的內容（分開最佳化）")
    print("   • result.embeddings[i] 就是第 i 句的向量（一串 float）")
    print("   • cosine 相似度：1 ≈ 語意相同、0 ≈ 不相關，常用於語意搜尋 / 比對 / 分群")
    print(
        "   • EmbeddingSettings(dimensions=768) → 調維度：越大越精細，索引也越大、越慢"
    )


# ========== 主程式 ==========


async def main():
    """主程式入口"""
    if not GOOGLE_API_KEY:
        print("⚠️  警告：請在 .env 檔案中設置 GOOGLE_API_KEY")
        print("💡 前往 https://aistudio.google.com/apikey 取得 API Key（格式 AIza...）")
        return

    try:
        # 確認有 Key 後才建立 Embedder
        embedder = build_embedder()

        # 語意搜尋
        await demo_semantic_search(embedder)

        # 重點整理
        explain()

    except Exception as e:
        print(f"\n❌ 執行出錯：{e}")
        print(
            "💡 請檢查 GOOGLE_API_KEY 是否正確設置在 .env 檔案中，以及模型名稱是否可用"
        )


if __name__ == "__main__":
    # 執行異步主程式
    asyncio.run(main())
