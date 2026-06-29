"""
Output Mode 範例（聚焦：NativeOutput / PromptedOutput 兩種結構化輸出模式）

Pydantic AI 取得結構化輸出共有三種模式：
- ToolOutput（預設）：用 tool calling 回傳結構化資料，相容性最廣。
- NativeOutput：schema 走 API 的 response_format / JSON schema（不進 prompt），
  端點有支援才會被「強制」，只有支援的模型 / 端點才能用。
- PromptedOutput：把 JSON schema 塞進 prompt，請模型「自己」回 JSON，
  任何模型都能用，但純靠提示約束、可靠度略低。

注意：兩種模式模型「都有可能不遵守」——
  Native 端點若不支援會被忽略；Prompted 純靠指令、模型可能回非預期格式。
  但不論哪種，pydantic_ai 最後都會用 Pydantic 驗證回傳值，不合就重試，
  所以不會默默把壞資料丟給你（重試用完才拋 UnexpectedModelBehavior）。

本範例用同一個 Recipe 資料結構，分別跑 NativeOutput 與 PromptedOutput，
證明兩者都能得到同樣型別的結構化結果，並說明差異與選用時機。

模型沿用本專案其他 NVIDIA NIM 範例（output_validator_demo.py 等）的
Nemotron（OpenAI 相容端點）。

文檔參考：https://ai.pydantic.dev/output/#structured-output
"""

import asyncio
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, NativeOutput, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# 載入 .env 檔案中的環境變數
load_dotenv()

# ========== 模型設定 ==========

# 設置 NVIDIA API Key（前往 https://build.nvidia.com/ 取得，格式 nvapi-...）
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# 透過 OpenAIProvider 指向 NVIDIA 的 OpenAI 相容端點
MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"
model = OpenAIChatModel(
    MODEL_NAME,
    provider=OpenAIProvider(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    ),
)

# ========== 資料結構定義 ==========


class Recipe(BaseModel):
    """一道料理的結構化資料"""

    dish: str = Field(description="料理名稱")
    servings: int = Field(description="幾人份", ge=1)
    ingredients: list[str] = Field(description="所需食材清單")
    steps: list[str] = Field(description="製作步驟（精簡條列）")


# ========== Output Mode 測試 ==========


async def run_one_mode(mode_label: str, output_type) -> Recipe:
    """用指定的 output mode 跑同一個任務，回傳解析後的 Recipe"""
    agent = Agent(
        model,
        output_type=output_type,
        system_prompt="你是料理助手，請用繁體中文回答。",
    )
    result = await agent.run(
        "幫我規劃一道 2 人份的『蕃茄炒蛋』，給出食材與精簡步驟。",
    )
    recipe = result.output
    print(f"\n  ✅ [{mode_label}] 解析成功，型別：{type(recipe).__name__}")
    print(f"     料理：{recipe.dish}（{recipe.servings} 人份）")
    print(f"     食材：{', '.join(recipe.ingredients)}")
    print(f"     步驟：{len(recipe.steps)} 步")
    return recipe


async def demo_output_modes():
    """示範 NativeOutput 與 PromptedOutput 兩種 output mode"""
    print("\n📦 NativeOutput / PromptedOutput（兩種 output mode）")
    print("=" * 60)

    # 模式 1：NativeOutput —— 用原生 JSON schema（供應商端強制格式）
    print("\n📌 模式 1：NativeOutput（模型原生結構化輸出）")
    print("-" * 40)
    print(
        "  說明：schema 走頂層 response_format（不進 prompt）；端點支援才會被強制，回傳後仍由 pydantic_ai 驗證。"
    )
    await run_one_mode("NativeOutput", NativeOutput(Recipe))

    # 模式 2：PromptedOutput —— 把 schema 塞進 prompt，請模型自己回 JSON
    print("\n📌 模式 2：PromptedOutput（提示式結構化輸出）")
    print("-" * 40)
    print(
        "  說明：把 schema 寫進 prompt 請模型『自己』回 JSON，任何模型都能用、但靠提示約束。"
    )
    await run_one_mode("PromptedOutput", PromptedOutput(Recipe))


# ========== 說明 ==========


def explain():
    """整理重點"""
    print("\n\n" + "=" * 60)
    print("💡 重點整理")
    print("=" * 60)

    print("\n1. 三種 output mode：")
    print("   • ToolOutput（預設）：tool calling 回傳，相容性最廣")
    print(
        "   • NativeOutput：schema 走 API response_format（不進 prompt），端點支援才強制"
    )
    print("   • PromptedOutput：schema 塞進 prompt，任何模型可用、純靠提示約束")

    print("\n2. 兩種都可能不遵守，但有安全網：")
    print("   • Native 端點不支援會被忽略；Prompted 純靠指令、模型可能回非預期格式")
    print("   • 不論哪種，pydantic_ai 都會用 Pydantic 驗證回傳值，不合就重試")
    print("   • 重試用完仍不合 → 拋 UnexpectedModelBehavior，不會默默給壞資料")

    print("\n3. 怎麼選？")
    print("   • 端點支援原生結構化輸出（response_format json_schema）→ 優先 NativeOutput")
    print("   • 端點不支援 tool / 原生 JSON、或想最大相容 → PromptedOutput")
    print("   • 需要多工具混合呼叫的流程 → 維持預設 ToolOutput")


# ========== 主程式 ==========


async def main():
    """主程式入口"""
    if not NVIDIA_API_KEY:
        print("⚠️  警告：請在 .env 檔案中設置 NVIDIA_API_KEY")
        print("💡 前往 https://build.nvidia.com/ 取得 API Key（格式 nvapi-...）")
        return

    try:
        # 示範兩種 output mode
        await demo_output_modes()

        # 重點整理
        explain()

    except Exception as e:
        print(f"\n❌ 執行出錯：{e}")
        print("💡 請檢查 NVIDIA_API_KEY 是否正確設置在 .env 檔案中，以及模型名稱是否可用")


if __name__ == "__main__":
    # 執行異步主程式
    asyncio.run(main())
