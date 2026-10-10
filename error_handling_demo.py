"""
ModelRetry 重試機制範例

這個範例展示了 Pydantic AI 的 ModelRetry 功能：
1. 工具層級的重試機制
2. ModelRetry 引導 AI 自我修正
3. 不完整輸入的智能處理（如：兩個字找三個字）
4. 使用限制防止無限重試
5. 透過 result.all_messages() 觀察自我修正軌跡（print_retry_trace）

三個場景：
- 場景 1：查不存在用戶 → 工具回傳 None，不觸發重試（對照組）
- 場景 2：兩個字找三個字 → ModelRetry 引導 AI 修正輸入後重試
- 場景 3：不穩定 API → ModelRetry 處理暫時性失敗，自動重試直到成功

文檔參考：https://pydantic.dev/docs/ai/core-concepts/agent/#reflection-and-self-correction
"""

import asyncio
import os
from dataclasses import dataclass
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.messages import RetryPromptPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.usage import UsageLimits

# 載入 .env 檔案中的環境變數
load_dotenv()

# ========== 簡單的模擬資料庫 ==========

class UserDatabase:
    """模擬用戶資料庫"""

    def __init__(self):
        # 用戶資料（包含三個字的姓名）
        self.users = {
            "張三": {"id": 1, "email": "zhang@example.com", "age": 25},
            "李四": {"id": 2, "email": "li@example.com", "age": 30},
            "王五": {"id": 3, "email": "wang@example.com", "age": 35},
            "張小華": {"id": 4, "email": "zhangxh@example.com", "age": 28},  # 新增：用於測試兩個字查詢
        }

        # API 調用計數器（模擬失敗情況）
        self.api_call_count = 0

    def get_user_by_name(self, name: str) -> dict | None:
        """根據姓名查找用戶"""
        return self.users.get(name)

    def validate_api_service(self) -> bool:
        """模擬不穩定的 API 服務"""
        self.api_call_count += 1

        # 前兩次失敗，第三次成功（演示重試）
        if self.api_call_count <= 2:
            print(f"  ❌ API 調用失敗（第 {self.api_call_count} 次）")
            return False
        else:
            print(f"  ✅ API 調用成功（第 {self.api_call_count} 次）")
            return True

# ========== 資料結構定義 ==========

@dataclass
class UserQueryDependencies:
    """用戶查詢的依賴注入"""
    database: UserDatabase

class UserProfile(BaseModel):
    """用戶資料輸出格式"""
    name: str = Field(default="N/A", description="用戶的完整姓名，如果找不到則為 N/A")
    email: str = Field(default="N/A", description="用戶的電子郵件地址，如果找不到則為 N/A")
    age: int = Field(default=0, description="用戶年齡，如果找不到則為 0", ge=0, le=120)
    message: str = Field(description="查詢結果訊息")

# ========== 模型設定 ==========

# 設置 Anthropic API Key
# 你需要前往 https://platform.claude.com/settings/keys 取得 API Key（格式為 sk-ant-...）
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# 這個範例需要模型「聽得懂 ModelRetry 的提示並修正參數」，也會用到無參數工具。
# 實測 NVIDIA Nemotron 呼叫無參數工具時會自己編參數（如 {"__noop": ""}），
# 收到重試提示也不會改用完整姓名，因此改用便宜的 Claude Haiku。
model_name = "claude-haiku-5-5"
model = AnthropicModel(
    model_name,
    provider=AnthropicProvider(api_key=ANTHROPIC_API_KEY),
)

# ========== Agent 建立 ==========

user_agent = Agent(
    model,
    deps_type=UserQueryDependencies,
    output_type=UserProfile,
    retries=2,  # 全局重試次數
    system_prompt=(
        "你是一個用戶資料查詢助手。\n"
        "請根據用戶的請求查詢並返回用戶資料。\n"
        "如果用戶不存在，請在 message 中說明用戶不存在。"
    ),
)

# ========== 工具定義（含 ModelRetry 重試機制）==========

@user_agent.tool(retries=2)
def get_user_by_name(ctx: RunContext[UserQueryDependencies], name: str) -> dict | None:
    """
    根據姓名查找用戶資料

    Args:
        name: 用戶姓名

    Returns:
        用戶資料字典，如果找不到則返回 None
    """
    print(f"\n🔍 嘗試查詢用戶：{name}")

    user_data = ctx.deps.database.get_user_by_name(name)

    if user_data is None:
        # 檢查是否為不完整的姓名（例如只有兩個字）
        if len(name) == 2:
            # 嘗試在資料庫中尋找匹配的三個字姓名
            possible_matches = [
                full_name for full_name in ctx.deps.database.users.keys()
                if full_name.startswith(name)
            ]

            if possible_matches:
                print(f"  💡 找到可能的匹配：{possible_matches}")
                # 使用 ModelRetry 引導 AI 學習並重試
                raise ModelRetry(
                    f'找不到完全匹配 "{name}" 的用戶。\n'
                    f'但找到以 "{name}" 開頭的用戶：{", ".join(possible_matches)}。\n'
                    f'請直接用完整的三個字姓名再呼叫一次 get_user_by_name 查詢。'
                )

        print(f"  ❌ 找不到用戶：{name}")
        # 返回 None 表示用戶不存在
        return None

    print(f"  ✅ 找到用戶：{user_data}")
    return user_data

@user_agent.tool(retries=3)
def validate_external_api(ctx: RunContext[UserQueryDependencies]) -> bool:
    """
    調用外部 API 服務（模擬不穩定的服務）
    """
    print("\n📡 調用外部 API...")

    success = ctx.deps.database.validate_api_service()

    if not success:
        # API 失敗時，使用 ModelRetry 觸發重試
        raise ModelRetry(
            'API 服務暫時不可用，這是暫時性的網路問題。'
            '請直接再呼叫一次 validate_external_api，通常重試幾次就會成功。'
        )

    return True

# ========== 觀察工具 ==========

def print_result(result):
    """印出查詢結果"""
    print(f"   姓名：{result.output.name}")
    print(f"   Email：{result.output.email}")
    print(f"   年齡：{result.output.age}")
    print(f"   訊息：{result.output.message}")

def print_retry_trace(result):
    """印出完整對話軌跡，觀察 ModelRetry 的自我修正過程"""
    print("\n🔬 ModelRetry 自我修正軌跡：")
    tool_calls = 0
    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                tool_calls += 1
                print(f"  [{tool_calls}] 🛠️  AI 呼叫工具 {part.tool_name}(args={part.args})")
            elif isinstance(part, RetryPromptPart):
                print(f"      ↩️  收到重試提示：{part.content}")
            elif isinstance(part, ToolReturnPart):
                print(f"      ✅ 工具回傳：{part.content}")

    # 證據：同一個工具被呼叫超過一次，代表 AI 收到回饋後改了參數重試
    print(f"\n  📊 工具總共被呼叫 {tool_calls} 次", end="")
    print("（> 1 次 ⇒ 發生了自我修正/重試）" if tool_calls > 1 else "（沒有觸發重試）")

# ========== 測試函數 ==========

async def test_retry_scenarios():
    """測試重試機制（三個場景）"""
    print("\n🛡️ Pydantic AI ModelRetry 重試機制展示")
    print("=" * 60)

    # 建立依賴
    deps = UserQueryDependencies(database=UserDatabase())

    # 場景 1：查詢不存在的用戶（工具直接返回 None，不觸發重試）
    print("\n\n📌 場景 1：查詢完全不存在的用戶")
    print("-" * 40)

    try:
        result = await user_agent.run(
            "查詢用戶 '小明' 的資料",
            deps=deps,
            usage_limits=UsageLimits(request_limit=10)
        )

        print("\n✅ 查詢完成！")
        print_result(result)

        # 對照組：沒有觸發重試，工具只會被呼叫一次
        print_retry_trace(result)

    except Exception as e:
        print(f"❌ 查詢失敗：{e}")

    # 場景 2：不完整姓名查詢（兩個字找三個字，觸發 ModelRetry 修正輸入）
    print("\n\n📌 場景 2：不完整姓名查詢（兩個字 → 三個字）")
    print("-" * 40)

    try:
        result = await user_agent.run(
            "查詢用戶 '張小' 的資料",  # 只給兩個字
            deps=deps,
            usage_limits=UsageLimits(request_limit=10)
        )

        print("\n✅ 查詢成功！")
        print_result(result)

        # 印出對話軌跡，觀察「兩個字 → 三個字」的自我修正
        print_retry_trace(result)

    except Exception as e:
        print(f"❌ 查詢失敗：{e}")

    # 場景 3：API 服務不穩定（暫時性失敗，自動重試直到成功）
    print("\n\n📌 場景 3：調用不穩定的外部 API")
    print("-" * 40)

    # 重置 API 計數器
    deps.database.api_call_count = 0

    try:
        result = await user_agent.run(
            "查詢張三的資料並驗證外部服務",
            deps=deps,
            usage_limits=UsageLimits(request_limit=10)
        )

        print("\n✅ 查詢成功！")
        print_result(result)

        # 軌跡會看到 validate_external_api 被呼叫多次（前兩次失敗、第三次成功）
        print_retry_trace(result)

    except Exception as e:
        print(f"❌ 查詢失敗：{e}")

def explain_retry_mechanism():
    """說明重試機制"""
    print("\n\n" + "="*60)
    print("💡 ModelRetry 重試機制說明")
    print("="*60)

    print("\n1. 什麼是 ModelRetry？")
    print("   • 一種特殊的異常，用於引導 AI 重新嘗試")
    print("   • 不是簡單的重試，而是提供錯誤訊息讓 AI 學習改進")

    print("\n2. 重試設定層級：")
    print("   • 工具層級：@tool(retries=2) - 單一工具的重試次數")
    print("   • Agent 層級：Agent(retries=2) - 全局重試次數")

    print("\n3. 本範例展示的場景：")
    print("   • 場景 1：查詢不存在的用戶 → 直接返回不存在（對照組，不重試）")
    print("   • 場景 2：兩個字找三個字 → ModelRetry 引導 AI 學習完整姓名")
    print("   • 場景 3：API 失敗 → 自動重試直到成功")

    print("\n4. 最佳實踐：")
    print("   • 提供明確的錯誤訊息和改進建議")
    print("   • 設定合理的重試次數（通常 2-3 次）")
    print("   • 使用 UsageLimits 防止無限重試")
    print("   • 智能處理不完整輸入（如兩個字補全為三個字）")

    print("\n5. 觀察自我修正：")
    print("   • print_retry_trace() 走訪 result.all_messages()")
    print("   • 看 ToolCallPart / RetryPromptPart / ToolReturnPart")
    print("   • 工具被呼叫 > 1 次 ⇒ 證明真的發生了重試/自我修正")

# ========== 主程式 ==========

async def main():
    """主程式入口"""
    try:
        # 測試重試場景
        await test_retry_scenarios()

        # 說明重試機制
        explain_retry_mechanism()

        print("\n" + "="*60)
        print("✨ ModelRetry 範例完成！")
        print("\n💡 重點觀察：")
        print("   • ModelRetry 如何引導 AI 改進參數")
        print("   • 工具層級的重試設定")
        print("   • 自動重試處理暫時性錯誤")

    except Exception as e:
        print(f"❌ 執行出錯：{e}")
        print("💡 請檢查 ANTHROPIC_API_KEY 是否正確設置在 .env 檔案中")

if __name__ == "__main__":
    # 執行異步主程式
    asyncio.run(main())
