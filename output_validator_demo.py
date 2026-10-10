"""
PydanticAI 輸出驗證層示範 - 用戶資料查詢範例 (使用 NVIDIA NIM)
展示 @agent.output_validator 與 ModelRetry 的跨欄位語意驗證

這個範例展示了：
1. @agent.output_validator - 在 AI 產生「最終結構化輸出」後，對整包輸出再做一次語意驗證
2. 驗證不通過時 raise ModelRetry，引導 AI 重新生成（跨欄位一致性檢查）
3. 透過對話軌跡觀察輸出層 ModelRetry 的自我修正過程
4. 使用 NVIDIA NIM（OpenAI 相容端點）訪問 Nemotron 等模型

與「工具層 ModelRetry」的差異：
- 工具層：驗證單一工具的輸入／執行
- 輸出層：驗證 AI 組裝出來的最終結果（跨欄位一致性）

NVIDIA NIM 文檔：https://build.nvidia.com/
Pydantic AI 文檔：https://pydantic.dev/docs/ai/core-concepts/output/#output-validator-functions
"""

import asyncio
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import RetryPromptPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

# 載入 .env 檔案中的環境變數
load_dotenv()

# ========== 1. 簡單的模擬資料庫 ==========


class UserDatabase:
    """模擬用戶資料庫"""

    def __init__(self):
        self.users = {
            "張三": {"id": 1, "email": "zhang@example.com", "age": 25},
            "李四": {"id": 2, "email": "li@example.com", "age": 30},
            "王五": {"id": 3, "email": "wang@example.com", "age": 35},
        }

    def get_user_by_name(self, name: str) -> dict | None:
        """根據姓名查找用戶"""
        return self.users.get(name)


# ========== 2. 資料結構定義 ==========


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


# ========== 3. 模型設定（NVIDIA NIM）==========

# 設置 NVIDIA API Key
# 你需要前往 https://build.nvidia.com/ 取得 API Key（格式為 nvapi-...）
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# NVIDIA NIM 模型名稱
MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"

# 檢查 API Key
if not NVIDIA_API_KEY:
    print("⚠️ 警告：請在 .env 檔案中設置 NVIDIA_API_KEY")
    print("💡 執行步驟：")
    print("   1. 前往 https://build.nvidia.com/ 註冊並獲取 API Key")
    print("   2. 在 .env 檔案中添加：NVIDIA_API_KEY=nvapi-...")
    exit(1)

# 透過 OpenAIProvider 指向 NVIDIA 的 OpenAI 相容端點
model = OpenAIChatModel(
    MODEL_NAME,
    provider=OpenAIProvider(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    ),
)

print(f"🚀 使用 NVIDIA NIM 模型：{MODEL_NAME}")
print("📡 API 端點：https://integrate.api.nvidia.com/v1")
print("✅ 透過 OpenAI 相容介面接入 NVIDIA NIM")

# ========== 4. Agent 建立 ==========

user_agent = Agent(
    model,
    deps_type=UserQueryDependencies,
    output_type=UserProfile,
    retries=2,  # 輸出驗證失敗時，最多重新生成的次數
    system_prompt=(
        "你是一個用戶資料查詢助手。\n"
        "請根據用戶的請求查詢並返回用戶資料。\n"
        "如果用戶不存在，請在 message 中說明用戶不存在。"
    ),
)

# ========== 5. 工具定義（單純查資料，不含重試）==========


@user_agent.tool
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
        print(f"  ❌ 找不到用戶：{name}")
        return None

    print(f"  ✅ 找到用戶：{user_data}")
    return user_data


# ========== 6. 輸出驗證層（output_validator）==========


@user_agent.output_validator
def validate_user_profile(
    ctx: RunContext[UserQueryDependencies], output: UserProfile
) -> UserProfile:
    """
    輸出驗證層：在 AI 產生「最終結構化輸出」後，對整包輸出做語意驗證。

    這裡做的是 pydantic Field 驗證做不到的「跨欄位邏輯」檢查，
    不通過時 raise ModelRetry，引導 AI 重新生成整包輸出。
    """
    print(f"\n🧪 輸出驗證層：檢查最終輸出 {output!r}")

    # 規則 1：有 email 卻沒有 name → 自相矛盾，要求 AI 補上姓名
    if output.email != "N/A" and output.name == "N/A":
        raise ModelRetry("輸出不一致：有 email 卻沒有 name，請補上完整姓名後重新輸出。")

    # 規則 2：查到用戶時，message 必須包含 email，方便使用者直接取得聯絡方式
    if output.name != "N/A" and output.email not in output.message:
        raise ModelRetry(
            f"查到用戶 {output.name}，但 message 沒有包含其 email（{output.email}）。"
            f"請把 email 一併寫進 message 後重新輸出。"
        )

    print("  ✅ 輸出驗證通過")
    return output


# ========== 7. 觀察工具 ==========


def print_retry_trace(result):
    """印出完整對話軌跡，觀察輸出層 ModelRetry 的自我修正過程"""
    print("\n🔬 ModelRetry 自我修正軌跡：")
    retries = 0
    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                print(f"  🛠️  AI 呼叫工具 {part.tool_name}(args={part.args})")
            elif isinstance(part, ToolReturnPart):
                print(f"      ✅ 工具回傳：{part.content}")
            elif isinstance(part, RetryPromptPart):
                retries += 1
                print(f"      ↩️  收到重試提示：{part.content}")

    # 本範例工具不再 raise ModelRetry，所以這裡的重試都來自「輸出驗證層」
    print(f"\n  📊 共觸發 {retries} 次重試", end="")
    print(
        "（輸出層驗證有引導 AI 自我修正）"
        if retries > 0
        else "（一次就通過，未觸發重試）"
    )


# ========== 8. 測試 ==========


async def test_output_validation():
    """測試輸出驗證層"""
    print("\n🛡️ Pydantic AI 輸出驗證層（output_validator）展示")
    print("=" * 60)

    deps = UserQueryDependencies(database=UserDatabase())

    print("\n📌 查詢用戶 '張三'（觀察輸出層驗證）")
    print("-" * 40)

    result = await user_agent.run(
        "查詢用戶 '張三' 的資料",
        deps=deps,
        usage_limits=UsageLimits(request_limit=10),
    )

    print("\n✅ 查詢成功！")
    print(f"   姓名：{result.output.name}")
    print(f"   Email：{result.output.email}")
    print(f"   年齡：{result.output.age}")
    print(f"   訊息：{result.output.message}")

    # 若輸出層觸發 ModelRetry，這裡會看到 RetryPromptPart
    print_retry_trace(result)

    # 顯示使用統計（注意：v2 起 result.usage 是 property，不是 method）
    usage = result.usage
    print("\n📈 使用統計:")
    print(f"  • 輸入 token: {usage.input_tokens}")
    print(f"  • 輸出 token: {usage.output_tokens}")
    print(f"  • 總計 token: {usage.total_tokens}")


def explain_output_validator():
    """說明輸出驗證層"""
    print("\n\n" + "=" * 60)
    print("💡 輸出驗證層（output_validator）說明")
    print("=" * 60)

    print("\n1. 它是什麼？")
    print("   • 用 @agent.output_validator 註冊的函式")
    print("   • 在 AI 產生最終結構化輸出後執行，對「整包輸出」做語意驗證")
    print("   • 驗證不通過時 raise ModelRetry，引導 AI 重新生成")

    print("\n2. 跟工具層 ModelRetry 的差異：")
    print("   • 工具層：驗證單一工具的輸入／執行")
    print("   • 輸出層：驗證 AI 組裝出來的最終結果（跨欄位一致性）")

    print("\n3. 本範例的兩條規則：")
    print("   • 規則 1：有 email 卻沒有 name → 自相矛盾")
    print("   • 規則 2：查到用戶時，message 必須包含其 email")

    print("\n4. 觸發時機提醒：")
    print("   • 規則 2 是否觸發，取決於模型第一次輸出有沒有把 email 寫進 message")
    print("   • 若一次就符合，validator 直接放行（這是正確的安全閘行為）")


# ========== 9. 主程式 ==========


async def main():
    """主程式入口"""
    try:
        await test_output_validation()
        explain_output_validator()
    except Exception as e:
        print(f"❌ 執行出錯：{e}")
        print("💡 可能的原因：")
        print("   - NVIDIA API Key 未設置或無效")
        print("   - 網路連接問題")
        print("   - 模型暫時不可用或名稱錯誤")
        print("💡 請檢查 NVIDIA_API_KEY 是否正確設置在 .env 檔案中")


if __name__ == "__main__":
    asyncio.run(main())
