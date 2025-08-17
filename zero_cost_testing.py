"""
零成本測試範例 - 展示 PydanticAI 的測試優勢
使用 TestModel 和 FunctionModel 進行單元測試
完全不需要真實 API 調用，確保測試快速且可預測

三個核心測試場景：
1. TestModel - 自動模擬工具調用（含自定義輸出）
2. FunctionModel - 完全控制 AI 回應
3. 批次測試 - 展示並行測試效能
"""

import asyncio
import random
from typing import List
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart

# === 創建骰子遊戲代理 ===
dice_agent = Agent(
    TestModel(),  # 使用 TestModel 避免需要 API key
    deps_type=str,  # 玩家姓名作為依賴
    system_prompt=(
        "你是一個友善的骰子遊戲主持人。"
        "玩家會告訴你他們的猜測，你需要："
        "1. 擲骰子看結果\n"
        "2. 取得玩家姓名\n"
        "3. 比較猜測和結果\n"
        "4. 用玩家的姓名宣布結果"
    )
)

# === 定義工具 ===
@dice_agent.tool_plain
def roll_dice() -> str:
    """擲一個六面骰子並返回結果"""
    result = random.randint(1, 6)
    print(f"  🎲 工具調用: roll_dice() = {result}")
    return str(result)

@dice_agent.tool
def get_player_name(ctx: RunContext[str]) -> str:
    """獲取當前玩家的姓名"""
    player_name = ctx.deps
    print(f"  👤 工具調用: get_player_name() = {player_name}")
    return player_name

@dice_agent.tool
def check_guess(ctx: RunContext[str], guess: int, actual: int) -> str:
    """檢查玩家的猜測是否正確"""
    print(f"  🔍 工具調用: check_guess({guess}, {actual})")

    if guess == actual:
        return "完全正確！"
    elif abs(guess - actual) == 1:
        return "很接近了！"
    else:
        return "差得有點遠！"


# === 測試 1: TestModel 完整功能展示 ===
async def test_test_model():
    """
    TestModel 的核心優勢：
    1. 自動處理工具調用，返回 JSON 格式結果
    2. 驗證工具註冊正確性
    3. 支援自定義輸出文字
    """
    print("\n" + "="*50)
    print("測試 1: TestModel - 自動模擬與自定義輸出")
    print("="*50)

    # Part A: 基本功能 - 自動工具調用
    print("\n📌 基本功能：自動工具調用")
    test_model = TestModel()

    with dice_agent.override(model=test_model):
        result = await dice_agent.run("我猜是 3", deps="測試玩家")

        print(f"✅ 自動生成輸出: {result.output}")

        # 驗證工具註冊
        if test_model.last_model_request_parameters:
            tools = test_model.last_model_request_parameters.function_tools
            print(f"✅ 已註冊 {len(tools)} 個工具: {[t.name for t in tools]}")

        print(f"✅ 訊息交換: {len(result.all_messages())} 條")

    # Part B: 進階功能 - 自定義輸出
    print("\n📌 進階功能：自定義輸出")
    custom_model = TestModel(custom_output_text="恭喜！你贏了！🎉")

    with dice_agent.override(model=custom_model):
        result = await dice_agent.run("我猜是 6", deps="幸運玩家")
        print(f"✅ 自定義輸出: {result.output}")

    return result


# === 測試 2: FunctionModel 精確場景控制 ===
async def test_function_model():
    """
    FunctionModel 的核心優勢：
    1. 完全控制回應內容
    2. 可模擬複雜的工具調用流程
    3. 適合測試特定場景和邊界條件
    """
    print("\n" + "="*50)
    print("測試 2: FunctionModel - 完全控制回應")
    print("="*50)

    # 定義模擬函數：展示工具調用流程
    async def simulate_game_flow(messages: List[ModelMessage], info: AgentInfo) -> ModelResponse:
        """模擬完整的遊戲流程"""

        # 第一次調用：返回工具調用
        if len(messages) == 1:
            print("\n📋 模擬 AI 決策：調用工具")
            return ModelResponse(parts=[
                ToolCallPart('get_player_name', {}, 'call_1'),
                ToolCallPart('roll_dice', {}, 'call_2'),
                ToolCallPart('check_guess', {'guess': 3, 'actual': 4}, 'call_3')
            ])

        # 第二次調用：基於工具結果生成最終回應
        else:
            print("📋 模擬 AI 決策：生成回應")
            return ModelResponse(parts=[
                TextPart("測試玩家，你猜的是 3，骰子結果是 4！差一點就猜中了！再接再厲！")
            ])

    function_model = FunctionModel(simulate_game_flow)

    with dice_agent.override(model=function_model):
        result = await dice_agent.run("我猜是 3", deps="測試玩家")
        print(f"\n✅ 完全控制的回應: {result.output}")
        print(f"✅ 訊息往返: {len(result.all_messages())} 條")

    return result


# === 測試 3: 批次測試效能展示 ===
async def test_batch_performance():
    """
    批次測試的核心優勢：
    1. 並行執行，效率極高
    2. 無網路延遲，毫秒級完成
    3. 零 API 成本
    """
    print("\n" + "="*50)
    print("測試 3: 批次測試 - 展示零成本效能")
    print("="*50)

    # 準備測試案例
    test_model = TestModel()
    test_cases = [
        ("玩家1", "我猜 1"),
        ("玩家2", "我猜 2"),
        ("玩家3", "我猜 3"),
        ("玩家4", "我猜 4"),
        ("玩家5", "我猜 5"),
        ("玩家6", "我猜 6"),
        ("玩家7", "最大的數字"),
        ("玩家8", "中間的數字"),
        ("玩家9", "隨機猜一個"),
        ("玩家10", "我選擇 3"),
    ]

    print(f"\n🚀 並行測試 {len(test_cases)} 個案例...")

    with dice_agent.override(model=test_model):
        start_time = asyncio.get_event_loop().time()

        # 並行執行所有測試
        tasks = [
            dice_agent.run(guess, deps=player)
            for player, guess in test_cases
        ]
        results = await asyncio.gather(*tasks)

        elapsed = asyncio.get_event_loop().time() - start_time

        # 展示效能統計
        print("\n📊 效能統計:")
        print(f"  ✅ 完成測試: {len(results)} 個")
        print(f"  ✅ 總耗時: {elapsed:.3f} 秒")
        print(f"  ✅ 平均耗時: {elapsed/len(results)*1000:.1f} 毫秒/測試")
        print("  ✅ API 成本: $0.00 (完全免費！)")

    return results


# === 主程式 ===
async def main():
    """執行三個核心測試場景"""
    print("\n🎯 PydanticAI 零成本測試示範")
    print("="*60)
    print("展示三個核心測試場景：")
    print("1️⃣  TestModel - 自動模擬與自定義輸出")
    print("2️⃣  FunctionModel - 完全控制回應流程")
    print("3️⃣  批次測試 - 並行執行展示效能")

    # 執行三個核心測試
    await test_test_model()
    await test_function_model()
    await test_batch_performance()

    print("\n" + "="*60)
    print("✨ 測試完成！")
    print("\n💡 PydanticAI 測試優勢總結：")
    print("┌─────────────────────────────────────────┐")
    print("│ • TestModel：自動模擬 + 自定義輸出     │")
    print("│ • FunctionModel：完全控制工具調用      │")
    print("│ • 並行測試：10個測試 < 0.02秒          │")
    print("│ • 零成本：不消耗任何 API 額度          │")
    print("│ • 可預測：結果一致，適合 CI/CD         │")
    print("└─────────────────────────────────────────┘")


if __name__ == "__main__":
    asyncio.run(main())