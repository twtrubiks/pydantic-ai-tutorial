"""
ModelRetry 重試機制範例

這個範例展示了 Pydantic AI 的 ModelRetry 功能：
1. 工具層級的重試機制
2. ModelRetry 引導 AI 自我修正
3. 不完整輸入的智能處理（如：兩個字找三個字）
4. 使用限制防止無限重試

文檔參考：https://ai.pydantic.dev/agents/#retries
"""

import asyncio
import os
from dataclasses import dataclass
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.github import GitHubProvider
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

# 設置 GitHub API Key
# 你需要在 GitHub 設置中創建 Personal Access Token，並添加 'models:read' 權限
GITHUB_API_KEY = os.getenv('GITHUB_API_KEY')

# GitHub Models 使用前綴格式的模型名稱
model_name = "gpt-4o-mini"
model = OpenAIModel(
    model_name,
    provider=GitHubProvider(api_key=GITHUB_API_KEY),
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
                    f'請使用完整的三個字姓名重試。'
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
            'API 服務暫時不可用，請重試。'
            '這是暫時性的網路問題。'
        )

    return True

# ========== 測試函數 ==========

async def test_retry_scenarios():
    """測試重試機制"""
    print("\n🛡️ Pydantic AI ModelRetry 重試機制展示")
    print("=" * 60)

    # 建立依賴
    deps = UserQueryDependencies(database=UserDatabase())

    # 場景 1：查詢不存在的用戶（直接返回不存在）
    # print("\n📌 場景 1：查詢完全不存在的用戶")
    # print("-" * 40)

    # try:
    #     result = await user_agent.run(
    #         "查詢用戶 '小明' 的資料",
    #         deps=deps,
    #         usage_limits=UsageLimits(request_limit=10)
    #     )

    #     print("\n✅ 查詢完成！")
    #     print(f"   姓名：{result.output.name}")
    #     print(f"   Email：{result.output.email}")
    #     print(f"   年齡：{result.output.age}")
    #     print(f"   訊息：{result.output.message}")

    # except Exception as e:
    #     print(f"❌ 查詢失敗：{e}")

    # 場景 2：不完整姓名查詢（兩個字找三個字，觸發 ModelRetry 學習）
    print("\n\n📌 場景 2：不完整姓名查詢（兩個字 → 三個字）")
    print("-" * 40)

    try:
        result = await user_agent.run(
            "查詢用戶 '張小' 的資料",  # 只給兩個字
            deps=deps,
            usage_limits=UsageLimits(request_limit=10)
        )

        print("\n✅ 查詢成功！")
        print(f"   姓名：{result.output.name}")
        print(f"   Email：{result.output.email}")
        print(f"   年齡：{result.output.age}")
        print(f"   訊息：{result.output.message}")

    except Exception as e:
        print(f"❌ 查詢失敗：{e}")

    # 場景 3：API 服務不穩定（需要多次重試）
    # print("\n\n📌 場景 3：調用不穩定的外部 API")
    # print("-" * 40)

    # # 重置 API 計數器
    # deps.database.api_call_count = 0

    # try:
    #     result = await user_agent.run(
    #         "查詢張三的資料並驗證外部服務",
    #         deps=deps,
    #         usage_limits=UsageLimits(request_limit=10)
    #     )

    #     print("\n✅ 查詢成功！")
    #     print(f"   姓名：{result.output.name}")
    #     print(f"   Email：{result.output.email}")
    #     print(f"   年齡：{result.output.age}")
    #     print(f"   訊息：{result.output.message}")

    # except Exception as e:
    #     print(f"❌ 查詢失敗：{e}")

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
    print("   • 場景 1：查詢不存在的用戶 → 直接返回不存在")
    print("   • 場景 2：兩個字找三個字 → ModelRetry 引導 AI 學習完整姓名")
    print("   • 場景 3：API 失敗 → 自動重試直到成功")

    print("\n4. 最佳實踐：")
    print("   • 提供明確的錯誤訊息和改進建議")
    print("   • 設定合理的重試次數（通常 2-3 次）")
    print("   • 使用 UsageLimits 防止無限重試")
    print("   • 智能處理不完整輸入（如兩個字補全為三個字）")

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
        print("💡 請檢查 GITHUB_API_KEY 是否正確設置在 .env 檔案中")

if __name__ == "__main__":
    # 執行異步主程式
    asyncio.run(main())