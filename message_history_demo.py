"""
PydanticAI 多輪對話與訊息歷史示範 (使用 NVIDIA NIM)
展示 Message History 如何讓 Agent「記住」先前的對話

這個範例展示了：
1. 無記憶 vs 有記憶 - 為什麼一次性問答不夠用
2. message_history - 把上一輪對話餵回去，維持上下文
3. all_messages() / new_messages() - 取出對話訊息
4. Conversation (V2.55) - 用 conversation= 接續對話，再用 ConversationTypeAdapter
   存成 JSON，下次開啟程式再接著聊（跨 session 持久化）
5. conversation_id / usage - 同一段對話共享同一個 ID，usage 也會跨 run 累積
6. History Processor (V2 capabilities) - 只保留最近 N 則訊息以節省 token

NVIDIA NIM：https://build.nvidia.com/
Pydantic AI 文檔：https://pydantic.dev/docs/ai/core-concepts/message-history/
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from pydantic_ai import Agent, Conversation, ConversationTypeAdapter, ModelMessage
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# 載入 .env 檔案中的環境變數
load_dotenv()

# 設置 NVIDIA API Key
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
model_name = "nvidia/nemotron-3-super-120b-a12b"

# 檢查 API Key
if not NVIDIA_API_KEY:
    print("⚠️ 警告：請在 .env 檔案中設置 NVIDIA_API_KEY")
    print("💡 執行步驟：")
    print("   1. 前往 https://build.nvidia.com/ 註冊並獲取 API Key")
    print("   2. 在 .env 檔案中添加：NVIDIA_API_KEY=nvapi-...")
    exit(1)

# 建立 AI 模型（透過 OpenAIProvider 指向 NVIDIA NIM 的 OpenAI 相容端點）
model = OpenAIChatModel(
    model_name,
    provider=OpenAIProvider(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    ),
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
    # Conversation 把 messages、usage、conversation_id（還有待核准的工具呼叫）打包在一起，
    # 用 conversation= 傳回去，比只傳 message_history= 完整（V2.55 官方建議的寫法）
    conversation: Conversation | None = None  # None = 開新對話

    for user_input in ["我下週要去日本京都旅遊。", "幫我推薦一個當地必吃的食物。"]:
        print(f"👤 使用者: {user_input}")
        result = await agent.run(user_input, conversation=conversation)
        print(f"🤖 AI: {result.output}")
        conversation = result.conversation  # 滾動更新：包含到目前為止的完整對話
    assert conversation is not None  # 跑過兩輪一定有值（讓型別檢查器知道不是 None）

    # 把整段對話序列化成 JSON 存檔
    # ConversationTypeAdapter 是官方提供的轉換器，能在 Python 物件 <-> JSON 間轉換
    HISTORY_FILE.write_bytes(ConversationTypeAdapter.dump_json(conversation))
    print(f"\n💾 已將 {len(conversation.messages)} 則訊息存到 {HISTORY_FILE.name}")

    # --- 模擬：關掉程式、重新開啟，從檔案讀回對話 ---
    print("\n--- 🔵 第二段 session（假裝是重開程式後）---")
    loaded = ConversationTypeAdapter.validate_json(HISTORY_FILE.read_bytes())
    print(f"📂 從檔案讀回 {len(loaded.messages)} 則訊息，之前已累積 {loaded.usage.requests} 次請求")

    # 用「它」這種代名詞測試 AI 是否真的記得前面聊的京都
    print("👤 使用者: 我剛剛說要去哪個城市？那裡還有什麼景點推薦？")
    result2 = await agent.run(
        "我剛剛說要去哪個城市？那裡還有什麼景點推薦？",
        conversation=loaded,
    )
    print(f"🤖 AI: {result2.output}")

    # conversation_id 會跨 run 延續：同一段對話即使存檔讀回，ID 仍相同
    print("\n🔗 conversation_id 驗證:")
    print(f"   第一段 session: {conversation.conversation_id}")
    print(f"   第二段 session: {result2.conversation_id}")
    print(f"   是否為同一段對話: {conversation.conversation_id == result2.conversation_id}")

    # usage 也跟著對話累積；只傳 message_history= 的話，每次 run 都從 0 重算，
    # UsageLimits(request_limit=...) 就只能限制單次 run，管不到整段對話
    print("\n📊 usage 累積驗證:")
    print(f"   conversation= 這次 run 的 usage.requests: {result2.usage.requests}（含先前兩輪）")
    print("   若改用 message_history=，這裡只會是 1")


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
    print("🗂️  PydanticAI 多輪對話與訊息歷史展示 (NVIDIA NIM)")
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
        print("   - conversation= 一次帶上訊息、usage、conversation_id，是 V2.55 建議的接續方式")
        print("   - ConversationTypeAdapter 讓整段對話能存成 JSON、跨 session 還原")
        print("   - conversation_id 自動串起同一段對話的多次 run")
        print("   - ProcessHistory capability 可控制送出的訊息量，避免 token 暴增")

    except Exception as e:
        print(f"\n❌ 執行出錯: {e}")
        print("💡 可能的原因：")
        print("   - NVIDIA API Key 未設置或無效")
        print("   - 網路連接問題或模型暫時不可用")


if __name__ == "__main__":
    asyncio.run(main())
