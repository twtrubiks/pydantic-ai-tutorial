"""
Gemini 3 Thinking 控制範例（聚焦：降低 thinking 省錢）

主題：Thinking 控制（thinking_level 高 vs 最省）
    Gemini 3 系列預設會「思考」（thinking），思考會額外消耗 token、增加延遲。
    透過 GoogleModelSettings 的 google_thinking_config 把 thinking_level 設成 MINIMAL，
    即可把思考降到最低，省 token 也更快。本範例直接「HIGH vs MINIMAL」對比 thinking token 與耗時。

模型與設定說明：
    Gemini 3 系列使用 thinking_level（MINIMAL/LOW/MEDIUM/HIGH）控制思考強度，
    無法像 Gemini 2.5 那樣用 thinking_budget=0「完全關閉」；MINIMAL 是最省的等級。
    （若需要「完全關閉」思考，請改用 gemini-2.5-flash + thinking_budget=0。）

文檔參考：
    - thinking：https://ai.pydantic.dev/thinking/
    - Gemini thinking：https://ai.google.dev/gemini-api/docs/thinking

執行前：
    1. 前往 https://aistudio.google.com/apikey 取得 API Key（格式 AIza...）
    2. 在 .env 檔案中設置 GOOGLE_API_KEY=AIza...
    3. python gemini_thinking_demo.py
"""

import asyncio
import os
import time

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.messages import ThinkingPart
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
from pydantic_ai.providers.google import GoogleProvider

# 載入 .env 檔案中的環境變數
load_dotenv()

# ========== 模型設定 ==========

# 設置 Google API Key（前往 https://aistudio.google.com/apikey 取得，格式 AIza...）
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini 3 用 thinking_level（MINIMAL/LOW/MEDIUM/HIGH）控制思考強度，
# 無法用 thinking_budget=0 完全關閉；MINIMAL 是最省的等級。
MODEL_NAME = "gemini-3-flash-preview"


def build_model() -> GoogleModel:
    """建立 Gemini 模型（延後到確認有 API Key 後才建，避免 import 即報錯）"""
    return GoogleModel(
        MODEL_NAME,
        provider=GoogleProvider(api_key=GOOGLE_API_KEY),
    )


# 兩組 thinking 設定，等下直接拿來對比 ----------------------------------------

# 高思考：thinking_level=HIGH 讓模型盡量多思考，
# include_thoughts=True 會把思考摘要一併回傳，方便我們在軌跡裡看到它。
THINKING_HIGH = GoogleModelSettings(
    google_thinking_config={"thinking_level": "HIGH", "include_thoughts": True},
)

# 最省思考：thinking_level=MINIMAL → 思考最少 → 省 token、更快（這就是 Gemini 3 的「省錢開關」）
THINKING_MINIMAL = GoogleModelSettings(
    google_thinking_config={"thinking_level": "MINIMAL"},
)


# ========== 主題 A：Thinking 控制（降思考省錢）==========


def print_usage(label: str, result, elapsed: float):
    """印出單次執行的 token 使用與耗時，重點看 thinking token"""
    usage = result.usage
    thoughts = usage.details.get("thoughts_tokens", 0) if usage.details else 0

    print(f"\n  📊 {label}")
    print(f"     輸入 token：{usage.input_tokens}")
    print(f"     輸出 token：{usage.output_tokens}")
    print(
        f"     其中 thinking token：{thoughts}  {'🧠 有思考' if thoughts else '🚫 沒思考'}"
    )
    print(f"     耗時：{elapsed:.2f} 秒")


def has_thinking(result) -> bool:
    """掃描訊息，看是否真的出現 ThinkingPart（思考內容）"""
    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, ThinkingPart):
                return True
    return False


async def demo_thinking_control(model: GoogleModel):
    """對比 thinking 高 / 最省：證明降到 MINIMAL 後省 token 又更快"""
    print("\n🧠 主題 A：Thinking 控制（降思考省錢）")
    print("=" * 60)

    # 用一題需要一點推理的問題，思考的差異才看得出來
    question = "一個水池有兩個進水管，A 管單獨注滿要 6 小時，B 管單獨注滿要 4 小時，兩管齊開要幾小時注滿？請給最後答案。"

    # 注意：這裡用同一個 agent，只在 run() 時用 model_settings 切換 thinking 等級
    agent = Agent(model, system_prompt="你是數學小老師，請用繁體中文簡短作答。")

    # --- 高思考 ---
    print("\n📌 情況 1：高思考（thinking_level=HIGH）")
    print("-" * 40)
    start = time.perf_counter()
    result_high = await agent.run(question, model_settings=THINKING_HIGH)
    elapsed_high = time.perf_counter() - start
    print(f"  🤖 回答：{result_high.output.strip()}")
    print(
        f"  🔍 軌跡是否含思考內容（ThinkingPart）：{'是' if has_thinking(result_high) else '否'}"
    )
    print_usage("高思考", result_high, elapsed_high)

    # --- 最省思考 ---
    print("\n📌 情況 2：最省思考（thinking_level=MINIMAL，省錢模式）")
    print("-" * 40)
    start = time.perf_counter()
    result_minimal = await agent.run(question, model_settings=THINKING_MINIMAL)
    elapsed_minimal = time.perf_counter() - start
    print(f"  🤖 回答：{result_minimal.output.strip()}")
    print(
        f"  🔍 軌跡是否含思考內容（ThinkingPart）：{'是' if has_thinking(result_minimal) else '否'}"
    )
    print_usage("最省思考", result_minimal, elapsed_minimal)

    # --- 對比結論 ---
    thoughts_high = (
        result_high.usage.details.get("thoughts_tokens", 0)
        if result_high.usage.details
        else 0
    )
    thoughts_minimal = (
        result_minimal.usage.details.get("thoughts_tokens", 0)
        if result_minimal.usage.details
        else 0
    )
    out_high = result_high.usage.output_tokens
    out_minimal = result_minimal.usage.output_tokens
    print("\n  💰 對比結論：")
    print(
        f"     thinking token：{thoughts_high} → {thoughts_minimal}"
        f"（省下約 {max(thoughts_high - thoughts_minimal, 0)} 個思考 token）"
    )
    print(f"     輸出 token：{out_high} → {out_minimal}")
    print(f"     耗時：{elapsed_high:.2f} 秒 → {elapsed_minimal:.2f} 秒")
    print("     ⇒ 降到 MINIMAL 通常更省、更快；代價是複雜推理題的正確率可能下降。")


# ========== 說明 ==========


def explain():
    """整理重點"""
    print("\n\n" + "=" * 60)
    print("💡 重點整理")
    print("=" * 60)

    print("\n1. Thinking 控制（省錢開關）：")
    print(
        "   • GoogleModelSettings(google_thinking_config={'thinking_level': 'MINIMAL'}) → 思考最少、最省"
    )
    print(
        "   • thinking_level 由低到高：MINIMAL < LOW < MEDIUM < HIGH，等級越高思考越多"
    )
    print("   • include_thoughts=True → 回傳思考摘要（方便除錯，但會多花輸出 token）")
    print("   • 省錢證據：usage.details['thoughts_tokens'] 在 MINIMAL 時大幅下降")
    print(
        "   • 注意：Gemini 3 無法完全關閉思考（沒有 budget=0）；要完全關閉需改用 gemini-2.5-flash + thinking_budget=0"
    )


# ========== 主程式 ==========


async def main():
    """主程式入口"""
    if not GOOGLE_API_KEY:
        print("⚠️  警告：請在 .env 檔案中設置 GOOGLE_API_KEY")
        print("💡 前往 https://aistudio.google.com/apikey 取得 API Key（格式 AIza...）")
        return

    try:
        # 確認有 Key 後才建立模型
        model = build_model()

        # Thinking 省錢
        await demo_thinking_control(model)

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
