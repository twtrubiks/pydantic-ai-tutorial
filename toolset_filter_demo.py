"""
Toolset 過濾範例（聚焦：用 filtered() 做唯讀安全白名單）

把工具過濾當成「安全護欄」：用 filtered() 從完整工具集衍生出「唯讀版」，
危險的寫入工具直接不會出現在送給模型的工具清單裡——模型不是被勸退，而是「無從呼叫」。
這比在 system prompt 叮嚀「不要刪資料」可靠得多。

重點：
    - 用 metadata={"access": "read"/"write"} 標記工具屬性，過濾就不必硬編工具名稱
    - toolset.filtered(fn)：fn(ctx, tool_def) 回傳 True 才保留該工具
    - 想驗證護欄：await toolset.get_tools(ctx) 印出模型「真正看得到」的工具

文檔參考：https://ai.pydantic.dev/toolsets/#filtering-tools

執行前：
    1. 前往 https://build.nvidia.com/ 取得 API Key（格式 nvapi-...）
    2. 在 .env 檔案中設置 NVIDIA_API_KEY=nvapi-...
    3. python toolset_filter_demo.py
"""

import asyncio
import os

from dotenv import load_dotenv
from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RunUsage, UsageLimits

# 載入 .env 檔案中的環境變數
load_dotenv()

# ========== 模擬資料庫 ==========


class EmployeeDatabase:
    """模擬員工資料庫（唯讀查詢 + 危險的刪除操作）"""

    def __init__(self):
        self.employees = {
            "E001": "Alice（工程師）",
            "E002": "Bob（設計師）",
        }

    def list_all(self) -> list[str]:
        return [f"{eid}: {name}" for eid, name in self.employees.items()]

    def delete(self, emp_id: str) -> bool:
        return self.employees.pop(emp_id, None) is not None


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

# ========== 工具集（用 metadata 標記讀 / 寫）==========

# 把工具放在同一個 FunctionToolset，之後用 filtered() 衍生出受限版本。
employee_toolset = FunctionToolset[EmployeeDatabase]()


@employee_toolset.tool(metadata={"access": "read"})
def list_employees(ctx: RunContext[EmployeeDatabase]) -> str:
    """列出所有員工。"""
    print("  🔧 [唯讀] list_employees()")
    return "；".join(ctx.deps.list_all())


@employee_toolset.tool(metadata={"access": "write"})
def delete_employee(ctx: RunContext[EmployeeDatabase], emp_id: str) -> str:
    """刪除一名員工（危險的破壞性操作）。"""
    print(f"  🔴 [寫入] delete_employee({emp_id}) — 真的刪除了資料！")
    ok = ctx.deps.delete(emp_id)
    return f"已刪除員工 {emp_id}" if ok else f"查無員工 {emp_id}"


# ========== 護欄：filtered — 唯讀白名單 ==========

# filter_func(ctx, tool_def) 回傳 True 才保留；只看 metadata 的 access，不必硬編工具名稱。
readonly_toolset = employee_toolset.filtered(
    lambda ctx, tool_def: (tool_def.metadata or {}).get("access") == "read"
)


# ========== 輔助函數 ==========


def make_agent(toolset) -> Agent[EmployeeDatabase, str]:
    """用指定的（可能已過濾的）工具集建立 Agent。"""
    return Agent(
        model,
        deps_type=EmployeeDatabase,
        toolsets=[toolset],
        system_prompt=(
            "你是員工資料管理助手。只能使用系統實際提供給你的工具。\n"
            "如果沒有任何工具能完成請求，請直接說明你沒有權限，不要假裝完成。"
        ),
    )


async def show_visible_tools(toolset, deps: EmployeeDatabase, label: str):
    """印出模型『實際看得到』的工具清單——護欄是否生效的鐵證。"""
    ctx = RunContext(deps=deps, model=model, usage=RunUsage())
    tools = await toolset.get_tools(ctx)
    visible = [
        f"{name}[{(tool.tool_def.metadata or {}).get('access', '?')}]"
        for name, tool in tools.items()
    ]
    print(f"  👀 {label} 看得到的工具：{visible}")


def print_tool_trace(result):
    """掃描對話軌跡，列出模型『實際呼叫了哪些工具』。"""
    calls = [
        part.tool_name
        for msg in result.all_messages()
        for part in msg.parts
        if isinstance(part, ToolCallPart)
    ]
    print(f"  🔬 模型呼叫的工具：{calls or '（無）'}")


# ========== 測試場景 ==========


async def run_delete_request(toolset, label: str):
    """對指定工具集發出同一個危險請求『刪除 E002』，看護欄是否擋得住。"""
    print(f"\n📌 {label}")
    print("-" * 50)

    deps = EmployeeDatabase()
    agent = make_agent(toolset)

    await show_visible_tools(toolset, deps, label)
    print("  💬 使用者請求：請刪除員工 E002")

    result = await agent.run(
        "請刪除員工 E002", deps=deps, usage_limits=UsageLimits(request_limit=5)
    )
    print_tool_trace(result)
    print(f"  🗄️  資料庫剩餘員工：{list(deps.employees.keys())}")
    print(f"  🤖 模型回覆：{result.output}")


# ========== 說明 ==========


def explain():
    """說明 Toolset 過濾護欄。"""
    print("\n\n" + "=" * 60)
    print("💡 Toolset 過濾（filtered）說明")
    print("=" * 60)

    print("\n1. 為什麼在「工具集層」做護欄？")
    print("   • 被擋掉的工具不會出現在送給模型的清單裡 → 模型無從呼叫")
    print("   • 比在 system prompt 叮嚀「不要刪資料」可靠得多")

    print("\n2. 怎麼做：")
    print("   • toolset.filtered(fn)：fn(ctx, tool_def)->bool，回傳 True 才保留")
    print("   • 從既有工具集衍生新工具集，不會動到原本的工具")

    print("\n3. 小技巧：")
    print("   • 用 metadata={'access':'read'/'write'} 標記工具，過濾就不必硬編名稱")
    print("   • 想驗證護欄：await toolset.get_tools(ctx) 印出模型真正看得到的工具")


# ========== 主程式 ==========


async def main():
    """主程式入口"""
    print("\n🛡️ Pydantic AI Toolset 過濾（filtered）展示")
    print("=" * 60)

    if not NVIDIA_API_KEY:
        print("⚠️  警告：請在 .env 檔案中設置 NVIDIA_API_KEY")
        print("💡 前往 https://build.nvidia.com/ 取得 API Key（格式 nvapi-...）")
        return

    try:
        # 場景 A：完整工具集 → 模型看得到 delete_employee，危險請求真的被執行
        await run_delete_request(employee_toolset, "場景 A：完整工具集（沒有護欄）")

        # 場景 B：唯讀白名單 → delete_employee 被擋在工具集外，資料安然無恙
        await run_delete_request(readonly_toolset, "場景 B：唯讀白名單（filtered）")

        explain()

    except Exception as e:
        print(f"\n❌ 執行出錯：{e}")
        print("💡 請檢查 NVIDIA_API_KEY 是否正確設置在 .env 檔案中")


if __name__ == "__main__":
    asyncio.run(main())
