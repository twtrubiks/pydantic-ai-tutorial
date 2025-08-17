"""
PydanticAI 工具系統示範 - 骰子遊戲範例
展示 Agent Tools 和 Function Calling 功能

這個範例展示了：
1. @agent.tool_plain - 無上下文工具（純函數）
2. @agent.tool - 有上下文工具（可訪問 RunContext）
3. Function Calling 的自動選擇機制
4. AI 如何根據對話內容決定使用哪些工具

文檔參考：https://ai.pydantic.dev/tools/
"""

import random
import asyncio
from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel

# 載入 .env 檔案中的環境變數
load_dotenv()

# 1. 建立 AI 模型（支援多種 LLM：OpenAI, Anthropic, Google 等）
model = GoogleModel('gemini-2.0-flash-exp')

# 2. 建立 PydanticAI Agent 並定義依賴類型
# deps_type=str 表示依賴類型是字串（玩家姓名）
dice_agent = Agent(
    model,
    deps_type=str,  # 玩家姓名作為依賴
    system_prompt=(
        "你是一個友善的骰子遊戲主持人。"
        "玩家會告訴你他們的猜測，你需要：\n"
        "1. 擲骰子看結果\n"
        "2. 取得玩家姓名\n"
        "3. 比較猜測和結果\n"
        "4. 用玩家的姓名宣布結果，並給予鼓勵或安慰\n"
        "請保持遊戲的趣味性和互動性！"
    ),
)

# ========== 3. 工具定義 ==========

@dice_agent.tool_plain  # 無上下文工具：純函數，不需要訪問 RunContext
def roll_dice() -> str:
    """擲一個六面骰子並返回結果"""
    result = random.randint(1, 6)
    print(f"🎲 骰子結果: {result}")  # 用於觀察工具執行
    return str(result)

@dice_agent.tool  # 有上下文工具：可以訪問 RunContext 獲取依賴和其他資訊
def get_player_name(ctx: RunContext[str]) -> str:
    """獲取當前玩家的姓名"""
    player_name = ctx.deps
    print(f"👤 玩家姓名: {player_name}")  # 用於觀察工具執行
    return player_name

@dice_agent.tool  # 帶參數的上下文工具
def check_guess(ctx: RunContext[str], guess: int, actual: int) -> str:
    """檢查玩家的猜測是否正確"""
    print(f"🔍 比較: 猜測={guess}, 實際={actual}")  # 用於觀察工具執行

    if guess == actual:
        return "完全正確！"
    elif abs(guess - actual) == 1:
        return "很接近了！"
    else:
        return "差得有點遠！"

@dice_agent.tool_plain
def get_game_rules() -> str:
    """獲取遊戲規則"""
    return (
        "骰子遊戲規則：\n"
        "1. 猜測一個 1-6 的數字\n"
        "2. 擲骰子看結果\n"
        "3. 猜對了就贏，猜錯了也不要氣餒！"
    )

# ========== 4. 遊戲執行函數 ==========

async def play_dice_game(player_name: str, guess_text: str):
    """執行一局骰子遊戲"""
    print(f"\n{'='*50}")
    print(f"🎮 玩家: {player_name}")
    print(f"💭 猜測: {guess_text}")
    print(f"{'='*50}")

    # 5. 執行 Agent：AI 會自動決定使用哪些工具
    result = await dice_agent.run(guess_text, deps=player_name)

    print(f"🤖 AI 回應: {result.output}")

    # 查看 AI 與模型的完整對話過程
    print("\n📋 對話記錄:")
    for i, message in enumerate(result.all_messages(), 1):
        print(f"  {i}. {type(message).__name__}: {str(message)[:100]}...")

    return result

async def demonstrate_tools():
    """展示不同的工具使用情境"""

    print("🎯 PydanticAI 工具系統展示 - 骰子遊戲範例")
    print("=" * 60)

    # 測試案例 1: 基本遊戲
    # await play_dice_game("小明", "我猜是 2")

    # 測試案例 2: 詢問規則
    await play_dice_game("小華", "這個遊戲怎麼玩？")

    # 測試案例 3: 多次猜測
    # await play_dice_game("小美", "我想玩兩次，第一次猜 1，第二次猜 6")

    # 測試案例 4: 非數字猜測
    # await play_dice_game("小李", "我猜是最大的數字")

def analyze_tools():
    """分析工具的設計模式"""

    print("\n" + "="*60)
    print("🔍 工具分析")
    print("="*60)

    print("\n1. @agent.tool_plain - 無上下文工具:")
    print("   ✅ roll_dice() - 擲骰子，不需要玩家資訊")
    print("   ✅ get_game_rules() - 獲取規則，與玩家無關")
    print("   📌 特點: 純函數，輸出只依賴參數")

    print("\n2. @agent.tool - 有上下文工具:")
    print("   ✅ get_player_name() - 需要從 RunContext 獲取玩家姓名")
    print("   ✅ check_guess() - 雖然有參數，但可能需要玩家資訊做個性化回應")
    print("   📌 特點: 可以訪問 ctx.deps, ctx.usage 等上下文資訊")

    print("\n3. AI 如何決定使用工具:")
    print("   🧠 根據用戶輸入的意圖")
    print("   🧠 根據工具的文檔字符串（docstring）")
    print("   🧠 根據對話的上下文需求")

    print("\n4. 執行順序:")
    print("   📝 AI 讀取系統提示")
    print("   📝 分析用戶輸入")
    print("   🔧 決定需要哪些工具")
    print("   🔄 依序或並行調用工具")
    print("   💬 整合結果生成回應")

# ========== 6. 主程式 ==========

async def main():
    """主程式入口"""
    try:
        # 展示工具使用
        await demonstrate_tools()

        # 分析工具設計
        analyze_tools()

        print("\n" + "="*60)
        print("✨ 範例完成！")
        print("💡 觀察重點:")
        print("   - AI 如何自主選擇使用哪些工具")
        print("   - 工具的文檔字符串如何指導 AI")
        print("   - @tool_plain vs @tool 的使用場景")
        print("   - RunContext 如何傳遞上下文資訊")

    except Exception as e:
        print(f"❌ 執行出錯: {e}")
        print("💡 請檢查 GOOGLE_API_KEY 是否正確設置在 .env 檔案中")

if __name__ == "__main__":
    # 執行異步主程式
    asyncio.run(main())