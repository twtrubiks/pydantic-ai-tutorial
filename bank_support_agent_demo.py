"""
PydanticAI 銀行客服代理示範
展示依賴注入、動態系統提示和工具系統

這個範例展示了：
1. 依賴注入（Dependency Injection）
2. 動態系統提示（Dynamic System Prompts）
3. 工具系統與資料庫整合
4. 結構化輸出與型別安全

文檔參考：https://ai.pydantic.dev/agents/
"""

from dataclasses import dataclass
import asyncio
import os
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.github import GitHubProvider

# 載入 .env 檔案中的環境變數
load_dotenv()

# 1. 模擬資料庫連接類別
class DatabaseConn:
    async def customer_name(self, id: int) -> str:
        # 模擬資料庫查詢，根據 ID 返回客戶姓名
        customer_names = {
            123: "小明",
            456: "老王",
            789: "張三"
        }
        return customer_names.get(id, "Unknown Customer")

    async def customer_balance(self, id: int, include_pending: bool) -> float:
        # 模擬資料庫查詢，返回客戶餘額
        base_balance = 1234.56
        pending_amount = 100.00 if include_pending else 0.00
        return base_balance + pending_amount

# 2. 依賴注入類別
# SupportDependencies 用於將數據、連接和邏輯傳遞到 Agent 中
# 依賴注入提供了一種型別安全的方式來自訂代理的行為
@dataclass
class SupportDependencies:
    customer_id: int  # 客戶 ID
    db: DatabaseConn  # 資料庫連接


# 3. 輸出結構定義
# 這個 Pydantic 模型定義了 Agent 返回的結構化輸出
class SupportOutput(BaseModel):
    support_advice: str = Field(description="給客戶的建議")
    block_card: bool = Field(description="是否封鎖客戶的卡片")
    risk: int = Field(description='查詢的風險等級', ge=0, le=10)

GITHUB_API_KEY = os.getenv('GITHUB_API_KEY')
model_name = "gpt-4o-mini"

# 4. 建立 AI 模型（支援多種 LLM：OpenAI, Anthropic, Google 等）
model = OpenAIChatModel(
    model_name,
    provider=GitHubProvider(api_key=GITHUB_API_KEY),
)


# 5. 建立銀行客服 Agent
# 這個 Agent 將作為銀行的一線客服支援
# Agent 在接受的依賴類型和返回的輸出類型方面是通用的
# 在這種情況下，支援代理的類型是 `Agent[SupportDependencies, SupportOutput]`
support_agent = Agent(
    model,
    deps_type=SupportDependencies,  # 依賴類型
    # Agent 的回應將保證是 SupportOutput 類型，如果驗證失敗會自動重試
    output_type=SupportOutput,  # 輸出類型
    system_prompt=(
        '你是我們銀行的客服代理，為客戶提供支援並判斷他們查詢的風險等級。'
        '請務必用中文回應客戶，所有輸出都必須是中文。'
    ),
)


# 6. 動態系統提示
# 動態系統提示可以使用依賴注入，根據客戶資料調整提示內容
# 依賴通過 `RunContext` 參數傳遞，該參數使用上面的 `deps_type` 進行參數化
# 如果類型註解錯誤，靜態類型檢查器會捕獲它
@support_agent.system_prompt
async def add_customer_name(ctx: RunContext[SupportDependencies]) -> str:
    # 從資料庫獲取客戶姓名
    customer_name = await ctx.deps.db.customer_name(id=ctx.deps.customer_id)
    result = f"客戶姓名是 {customer_name!r}"
    return result


@support_agent.system_prompt
async def add_customer_status(ctx: RunContext[SupportDependencies]) -> str:
    if ctx.deps.customer_id in [123, 456]:
        result = "這是VIP客戶，請提供優質服務"
        return result
    return "這是普通客戶"


# 7. 工具函數定義
# `tool` 讓你註冊 LLM 在回應用戶時可能調用的函數
# 依賴通過 `RunContext` 傳遞，其他參數成為傳遞給 LLM 的工具模式
# Pydantic 用於驗證這些參數，錯誤會傳回給 LLM 以便重試
@support_agent.tool
async def customer_balance(
        ctx: RunContext[SupportDependencies], include_pending: bool
) -> float:
    """返回客戶的當前帳戶餘額。"""
    # 工具的文檔字符串會傳遞給 LLM 作為工具的描述
    # 參數描述從文檔字符串中提取並添加到發送給 LLM 的參數模式中
    balance = await ctx.deps.db.customer_balance(
        id=ctx.deps.customer_id,
        include_pending=include_pending,  # 是否包含待處理交易
    )
    return balance


# 8. 主程式演示
async def main():
    # 創建依賴對象，包含客戶 ID 和資料庫連接
    deps = SupportDependencies(customer_id=456, db=DatabaseConn())

    print("🏦 銀行客服 AI 代理示範")
    print("=" * 50)

    # 異步運行 Agent，與 LLM 進行對話直到達到最終回應
    # 即使在這個簡單的情況下，Agent 也會與 LLM 交換多條消息，因為會調用工具來檢索輸出
    result = await support_agent.run('我的餘額是多少？', deps=deps)

    # `result.output` 將通過 Pydantic 驗證以保證它是 `SupportOutput`
    # 由於 Agent 是通用的，它也會被類型化為 `SupportOutput` 以幫助靜態類型檢查
    print("🤖 AI 回應結果:")
    print(result.output)

    # 查看 AI 與模型的完整對話過程
    print("\n📋 對話記錄:")
    for i, message in enumerate(result.all_messages(), 1):
        print(f"  {i}. {type(message).__name__}: {str(message)[:500]}...")

    # 顯示使用統計
    usage = result.usage
    print("\n📊 使用統計:")
    print(f"   請求次數: {usage.requests}")
    print(f"   輸入 tokens: {usage.input_tokens}")
    print(f"   輸出 tokens: {usage.output_tokens}")
    print(f"   總計 tokens: {usage.total_tokens}")

    print("\n" + "=" * 50)
    print("💡 PydanticAI 特色展示：")
    print("   ✅ 依賴注入：將客戶資料安全傳遞給 Agent")
    print("   ✅ 動態系統提示：根據客戶狀態調整提示")
    print("   ✅ 工具整合：自動調用資料庫查詢函數")
    print("   ✅ 結構化輸出：保證返回正確的資料格式")
    print("   ✅ 型別安全：編譯時檢查和運行時驗證")
    print("   ✅ 透明對話：完整記錄 AI 與模型的互動過程")

    # 額外測試案例（取消註解可測試）
    # print("\n--- 測試遺失卡片情況 ---")
    # result = await support_agent.run('我的卡片遺失了！', deps=deps)
    # print(result.output)


# 9. 主程式執行
if __name__ == '__main__':
    asyncio.run(main())