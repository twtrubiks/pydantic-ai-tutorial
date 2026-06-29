"""
PydanticAI 多輪對話與訊息歷史示範 (使用 GitHub Models)
展示 Message History 如何讓 Agent「記住」先前的對話

這個範例展示了：
1. 無記憶 vs 有記憶 - 為什麼一次性問答不夠用
2. message_history - 把上一輪對話餵回去，維持上下文
3. all_messages() / new_messages() - 取出對話訊息
4. 序列化存檔 / 讀回 - 用 ModelMessagesTypeAdapter 把對話存成 JSON，
   下次開啟程式再接著聊（跨 session 持久化）
5. conversation_id - 同一段對話的多次 run 會共享同一個 ID
6. History Processor (V2 capabilities) - 只保留最近 N 則訊息以節省 token

GitHub Models 文檔：https://docs.github.com/en/github-models
Pydantic AI 文檔：https://ai.pydantic.dev/message-history/
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from pydantic_ai import Agent, ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.github import GitHubProvider

# 載入 .env 檔案中的環境變數
load_dotenv()

# 設置 GitHub API Key
GITHUB_API_KEY = os.getenv('GITHUB_API_KEY')
model_name = "gpt-4o-mini"

# 檢查 API Key
if not GITHUB_API_KEY:
    print("⚠️ 警告：請在 .env 檔案中設置 GITHUB_API_KEY")
    print("💡 執行步驟：")
    print("   1. 前往 GitHub 設置 > Developer settings > Personal access tokens")
    print("   2. 創建新的 token 並添加 'models:read' 權限")
    print("   3. 在 .env 檔案中添加：GITHUB_API_KEY=github_pat_...")
    exit(1)

# 建立 AI 模型（使用 GitHub Models）
model = OpenAIChatModel(
    model_name,
    provider=GitHubProvider(api_key=GITHUB_API_KEY),
)

# 對話歷史的存檔路徑（demo 2 會用到）
HISTORY_FILE = Path(__file__).parent / "conversation_history.json"


# ========== Demo 1: 無記憶 vs 有記憶 ==========

async def demo_1_memory_vs_no_memory():
    """對比：不傳 message_history（失憶） vs 傳 message_history（有記憶）"""
    print("\n" + "=" * 60)
    print("📌 Demo 1: 無記憶 vs 有記憶")
    print("=" * 60)

    agent = Agent(
        model,
        system_prompt="你是一個友善的助理，請用繁體中文回答，回答盡量簡短。",
    )

    # 第一輪：告訴 AI 一個資訊
    print("\n👤 使用者: 我叫小明，我最喜歡的數字是 7。")
    result1 = await agent.run("我叫小明，我最喜歡的數字是 7。")
    print(f"🤖 AI: {result1.output}")

    # 情境 A：不傳歷史 → AI 失憶，不知道前面說過什麼
    print("\n--- ❌ 情境 A：不傳 message_history（每次都是全新對話）---")
    print("👤 使用者: 我最喜歡的數字是多少？")
    result_no_mem = await agent.run("我最喜歡的數字是多少？")
    print(f"🤖 AI: {result_no_mem.output}")
    print("   👉 AI 不知道，因為它沒有前一輪的記憶")

    # 情境 B：傳入第一輪的歷史 → AI 記得
    print("\n--- ✅ 情境 B：傳入 message_history（延續同一段對話）---")
    print("👤 使用者: 我最喜歡的數字是多少？")
    result_with_mem = await agent.run(
        "我最喜歡的數字是多少？",
        message_history=result1.all_messages(),  # 把上一輪的完整對話餵回去
    )
    print(f"🤖 AI: {result_with_mem.output}")
    print("   👉 AI 答得出來，因為歷史裡有「最喜歡 7」這個資訊")


# ========== Demo 2: 連續多輪 + 存檔讀回 ==========

async def demo_2_persist_and_resume():
    """連續多輪對話，存成 JSON，再讀回來接著聊（模擬關掉程式後重開）"""
    print("\n" + "=" * 60)
    print("📌 Demo 2: 連續多輪對話 + 存檔 / 讀回（跨 session 持久化）")
    print("=" * 60)

    agent = Agent(
        model,
        system_prompt="你是一個記憶力很好的助理，請用繁體中文回答，回答盡量簡短。",
    )

    # --- 第一段 session：聊兩輪後存檔 ---
    print("\n--- 🟢 第一段 session ---")
    history: list[ModelMessage] = []  # 累積整段對話
    first_conversation_id = None

    for user_input in ["我下週要去日本京都旅遊。", "幫我推薦一個當地必吃的食物。"]:
        print(f"👤 使用者: {user_input}")
        result = await agent.run(user_input, message_history=history)
        print(f"🤖 AI: {result.output}")
        history = result.all_messages()  # 滾動更新：包含到目前為止的完整對話
        first_conversation_id = result.conversation_id

    # 把對話歷史序列化成 JSON 存檔
    # ModelMessagesTypeAdapter 是官方提供的轉換器，能在 Python 物件 <-> JSON 間轉換
    HISTORY_FILE.write_bytes(ModelMessagesTypeAdapter.dump_json(history))
    print(f"\n💾 已將 {len(history)} 則訊息存到 {HISTORY_FILE.name}")

    # --- 模擬：關掉程式、重新開啟，從檔案讀回對話 ---
    print("\n--- 🔵 第二段 session（假裝是重開程式後）---")
    loaded_bytes = HISTORY_FILE.read_bytes()
    loaded_history = ModelMessagesTypeAdapter.validate_json(loaded_bytes)
    print(f"📂 從檔案讀回 {len(loaded_history)} 則訊息")

    # 用「它」這種代名詞測試 AI 是否真的記得前面聊的京都
    print("👤 使用者: 我剛剛說要去哪個城市？那裡還有什麼景點推薦？")
    result2 = await agent.run(
        "我剛剛說要去哪個城市？那裡還有什麼景點推薦？",
        message_history=loaded_history,
    )
    print(f"🤖 AI: {result2.output}")

    # conversation_id 會跨 run 延續：同一段對話即使存檔讀回，ID 仍相同
    print("\n🔗 conversation_id 驗證:")
    print(f"   第一段 session: {first_conversation_id}")
    print(f"   第二段 session: {result2.conversation_id}")
    print(f"   是否為同一段對話: {first_conversation_id == result2.conversation_id}")


# ========== Demo 3: History Processor 省 token (V2 capabilities) ==========

def keep_recent_messages(messages: list[ModelMessage]) -> list[ModelMessage]:
    """只保留最近 6 則訊息，避免對話越長、每次送出的 token 越多

    注意：切片時要小心 tool call 與 return 必須成對，否則模型可能報錯。
    本範例是純文字對話，沒有工具調用，所以可以安全切片。
    """
    max_messages = 6
    if len(messages) > max_messages:
        kept = messages[-max_messages:]
        print(f"   ✂️  History Processor: {len(messages)} 則 → 只送出最近 {len(kept)} 則")
        return kept
    return messages


async def demo_3_history_processor():
    """用 ProcessHistory capability 控制送進模型的訊息量（省 token）"""
    print("\n" + "=" * 60)
    print("📌 Demo 3: History Processor 省 token (V2 capabilities)")
    print("=" * 60)

    # V2 把歷史處理器掛在 capabilities 上（V1 的 history_processors= 參數已移除）
    agent = Agent(
        model,
        system_prompt="你是一個友善的助理，請用繁體中文回答，回答盡量簡短。",
        capabilities=[ProcessHistory(keep_recent_messages)],
    )

    # 連續聊多輪，讓對話歷史不斷變長
    history: list[ModelMessage] = []
    questions = [
        "1 加 1 等於多少？",
        "再乘以 10 呢？",
        "再減掉 5 呢？",
        "結果是質數嗎？",
    ]
    for q in questions:
        print(f"\n👤 使用者: {q}")
        result = await agent.run(q, message_history=history)
        print(f"🤖 AI: {result.output}")
        history = result.all_messages()
        print(f"   📊 累積歷史 {len(history)} 則 | 本輪送出 token: {result.usage.input_tokens}")

    print("\n💡 觀察：對話越長，累積歷史越多，但 Processor 讓「實際送出」維持在固定上限，")
    print("   避免 input token 隨對話無限膨脹（這就是省錢的關鍵）。")


# ========== 主程式 ==========

async def main():
    """主程式入口"""
    print("🗂️  PydanticAI 多輪對話與訊息歷史展示 (GitHub Models)")
    print(f"🚀 使用模型：{model_name}")

    try:
        await demo_1_memory_vs_no_memory()
        await demo_2_persist_and_resume()
        await demo_3_history_processor()

        print("\n" + "=" * 60)
        print("✨ 範例完成！")
        print("💡 重點回顧:")
        print("   - message_history= 是維持多輪對話上下文的關鍵")
        print("   - all_messages() 取完整歷史；new_messages() 只取本輪新增")
        print("   - ModelMessagesTypeAdapter 讓對話能存成 JSON、跨 session 還原")
        print("   - conversation_id 自動串起同一段對話的多次 run")
        print("   - ProcessHistory capability 可控制送出的訊息量，避免 token 暴增")

    except Exception as e:
        print(f"\n❌ 執行出錯: {e}")
        print("💡 可能的原因：")
        print("   - GitHub API Key 未設置或無效")
        print("   - GitHub token 缺少 'models:read' 權限")
        print("   - 網路連接問題或模型暫時不可用")


if __name__ == "__main__":
    asyncio.run(main())
