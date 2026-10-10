"""
MCP Client 範例（聚焦：用 MCPToolset 把 Odoo MCP server 掛上 Agent）

這個範例展示 Pydantic AI 作為「MCP Client」的用法：
1. 用 MCPToolset 連到一個「已經獨立跑起來」的 MCP server
   （傳 http(s):// 開頭的 URL，預設就走 Streamable HTTP）
2. 把這個 MCP server 當成 toolset 掛上 Agent —— server 暴露的工具會自動變成
   LLM 可呼叫的工具，Agent 本身完全不需要知道 Odoo / XML-RPC 的存在
3. 用自然語言下指令，觀察 LLM 自動挑選並呼叫 MCP 工具

整條資料流：
    使用者 → Pydantic AI Agent → LLM（決定呼叫哪個工具）
           → MCP Client（MCPToolset）
           → (HTTP /mcp) → odoo_mcp_server.py
           → (XML-RPC) → Odoo

注意：
- 本檔「不會」幫你架 / 啟動 MCP server，只示範 client 端程式長怎樣。
- 跑之前，Odoo MCP server 要先用 HTTP transport 跑起來，例如：
      python odoo_mcp_server.py --transport http --host 0.0.0.0 --port 8000
  （ODOO_URL / ODOO_DATABASE / ODOO_API_KEY 是「server 那端」的環境變數，client 不用帶）
- MCP server 來源：https://github.com/twtrubiks/odoo19-mcp-server
- 為了安全，本範例只下「唯讀」類的查詢；建議 server 端也開 READONLY_MODE=true。

文檔參考：
- https://pydantic.dev/docs/ai/mcp/client/
- https://pydantic.dev/docs/ai/mcp/client/#streamable-http
"""

import asyncio
import os

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import UsageLimits

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

# ========== MCP Client 設定（重點）==========

# Odoo MCP server 的 Streamable HTTP 端點。
# server 用 `--transport http --host 0.0.0.0 --port 8000` 跑起來後，
# 對外的 endpoint 路徑是 /mcp。
ODOO_MCP_URL = os.getenv("ODOO_MCP_URL", "http://localhost:8000/mcp")

# 這一行就是「MCP Client」：它只負責連到 HTTP endpoint、用 MCP 協定拿工具，
# 不知道（也不需要知道）背後是 Odoo 還是 XML-RPC。
# MCPToolset 會依參數自動選傳輸層：給 http(s) URL → Streamable HTTP；給指令 → stdio。
odoo_mcp = MCPToolset(ODOO_MCP_URL)

# ========== Agent 建立 ==========

# 把 odoo_mcp 放進 toolsets，server 暴露的工具（list_models / search_records /
# count_records / get_fields ...）就會自動成為這個 Agent 可用的工具。
#
# 補充：可以同時混用很多工具來源（把 MCP 當成「另一組 tools」看就好）——
# 本地函式工具用 tools=[...] 或 @agent.tool，多個 MCP 用 toolsets=[a_mcp, b_mcp, ...]，
# 全部會合併成同一份工具清單給 LLM。例如：
#     Agent(model, tools=[my_func], toolsets=[odoo_mcp, github_mcp])
# （撞名請用 odoo_mcp.prefixed('odoo')；工具太多可用 filtered()/prepared() 收斂，詳見下方說明。）
odoo_agent = Agent(
    model,
    toolsets=[odoo_mcp],
    system_prompt=(
        "你是一個 Odoo ERP 查詢助手。\n"
        "你可以使用系統提供的工具（來自 Odoo MCP server）來查詢資料。\n"
        "請只做唯讀查詢（列出 model、查欄位、搜尋、計數、讀取），不要做新增 / 修改 / 刪除。\n"
        "回答時請用繁體中文，並簡潔說明你查到的結果。"
    ),
)


# ========== 輔助函數 ==========


async def show_mcp_tools():
    """印出「Agent 透過 MCP 實際拿到的工具清單」——掛上 toolset 是否生效的鐵證。"""
    # 注意：要在 `async with odoo_agent:` 開啟連線後才拿得到工具。
    # list_tools() 直接向 MCP server 問「你有哪些工具」，回傳原始的 MCP Tool。
    tools = await odoo_mcp.list_tools()
    print(f"  👀 從 MCP server 拿到的工具（共 {len(tools)} 個）：")
    for tool in tools:
        desc = (tool.description or "").splitlines()[0] if tool.description else ""
        print(f"      • {tool.name} — {desc}")


def print_tool_trace(result):
    """掃描對話軌跡，列出 LLM 實際呼叫了哪些 MCP 工具、帶了什麼參數。"""
    print("\n  🔬 MCP 工具呼叫軌跡：")
    called = 0
    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                called += 1
                print(f"      [{called}] 🛠️  呼叫 {part.tool_name}(args={part.args})")
            elif isinstance(part, ToolReturnPart):
                # MCP 回傳內容可能較長，截斷顯示
                content = str(part.content)
                preview = content if len(content) <= 200 else content[:200] + " ..."
                print(f"          ↩️  回傳：{preview}")
    if called == 0:
        print("      （LLM 沒有呼叫任何 MCP 工具）")


def print_usage(result):
    """印出這次 run 的用量統計——Pydantic AI 內建，不用自己累加。

    result.usage 回傳 RunUsage（注意：新版是「屬性」，不要加括號）：
    - requests：總共呼叫了幾次 AI（就是 request_limit 在管的那個數字）
    - tool_calls：工具被呼叫幾次
    - input_tokens / output_tokens：token 用量
    """
    u = result.usage  # 新版是 property；舊寫法 result.usage() 會跳 deprecation 警告
    print("\n  📊 本次 run 用量（result.usage）：")
    print(f"      • 呼叫 AI 次數 requests：{u.requests}（上限 request_limit=8）")
    print(f"      • 工具呼叫次數 tool_calls：{u.tool_calls}")
    print(f"      • token：輸入 {u.input_tokens} / 輸出 {u.output_tokens}")


# ========== 測試場景 ==========


async def scenario_a_list_models():
    """場景 A：用自然語言問「有哪些 model」→ LLM 自動呼叫 list_models。"""
    print("\n📌 場景 A：列出與『聯絡人 / 夥伴』相關的 model")
    print("-" * 50)

    question = "系統裡有哪些跟『聯絡人 / 合作夥伴 (partner)』有關的 model？列幾個就好。"
    print(f"  💬 使用者：{question}")

    result = await odoo_agent.run(question, usage_limits=UsageLimits(request_limit=8))
    print(f"\n  🤖 回答：{result.output}")
    print_tool_trace(result)
    print_usage(result)


async def scenario_b_count_records():
    """場景 B：問「res.partner 有幾筆」→ LLM 自動呼叫 count_records。"""
    print("\n📌 場景 B：計算聯絡人總數")
    print("-" * 50)

    question = "幫我數一下 res.partner（聯絡人）總共有幾筆資料。"
    print(f"  💬 使用者：{question}")

    result = await odoo_agent.run(question, usage_limits=UsageLimits(request_limit=8))
    print(f"\n  🤖 回答：{result.output}")
    print_tool_trace(result)
    print_usage(result)


# ========== 說明 ==========


def explain_mcp_client():
    """說明 MCP Client 的運作。"""
    print("\n\n" + "=" * 60)
    print("💡 Pydantic AI 作為 MCP Client 說明")
    print("=" * 60)

    print("\n1. 角色分工：")
    print("   • Pydantic AI = MCP Client（連線方），不是 server")
    print("   • odoo_mcp_server.py = MCP Server（懂 Odoo XML-RPC 的那一端）")
    print("   • Agent 只跟 MCP 協定對話，不知道背後是 Odoo")

    print("\n2. 完整資料流：")
    print("   使用者 → Agent → LLM（選工具）")
    print("          → MCPToolset →(HTTP /mcp)→ odoo_mcp_server")
    print("          →(XML-RPC)→ Odoo")

    print("\n3. 為什麼用 Streamable HTTP（而非 stdio）？")
    print("   • stdio：Agent 把 server 當子行程啟動，適合同機")
    print("   • streamable HTTP：連到『已獨立跑起來』的 server，適合遠端 / 容器部署")
    print("   • 兩者只差在傳輸層，再往後的 →XML-RPC→Odoo 完全一樣")

    print("\n4. 使用重點：")
    print("   • toolsets=[odoo_mcp] → server 的工具自動變成 LLM 可呼叫的工具")
    print("   • 必須在 `async with odoo_agent:` 內執行，才會開啟 MCP 連線")
    print("   • Odoo 連線資訊在『server 端』env，client 端只需要 endpoint URL")

    print("\n5. 可以同時混用很多工具來源（把 MCP 當成 tools 看就好）：")
    print(
        "   • 所有來源最後會合併成『同一份工具清單』送給 LLM，LLM 不在乎工具是誰提供的"
    )
    print("   • 同一個 Agent 可同時掛：")
    print("       - 本地函式工具：@agent.tool / tools=[...]")
    print("       - 多個 MCP server：toolsets=[odoo_mcp, github_mcp, ...]")
    print("     例：Agent(model, tools=[my_func], toolsets=[odoo_mcp, github_mcp])")
    print("   • 注意撞名：多個來源若有同名工具會報錯 → 用 .prefixed('odoo') 加命名空間")
    print("       例：MCPToolset(url).prefixed('odoo') → odoo_list_models")
    print(
        "   • 工具太多會吃 token、也容易選錯 → 用 filtered()/prepared() 每次只放行需要的子集"
    )


# ========== 主程式 ==========


async def main():
    """主程式入口"""
    print("\n🔌 Pydantic AI MCP Client（Odoo / Streamable HTTP）展示")
    print("=" * 60)
    print(f"  連線目標：{ODOO_MCP_URL}")

    try:
        # `async with odoo_agent:` 會開啟所有 toolset（含 MCP server）的連線，
        # 結束時自動關閉。所有 agent.run(...) 都要在這個區塊內。
        async with odoo_agent:
            await show_mcp_tools()
            await scenario_a_list_models()
            # await scenario_b_count_records()

        explain_mcp_client()

    except Exception as e:
        print(f"\n❌ 執行出錯：{e}")
        print("💡 請確認：")
        print("   1. Odoo MCP server 已用 HTTP transport 跑起來：")
        print(
            "      python odoo_mcp_server.py --transport http --host 0.0.0.0 --port 8000"
        )
        print(f"   2. ODOO_MCP_URL 是否正確（目前：{ODOO_MCP_URL}）")
        print("   3. NVIDIA_API_KEY 是否正確設置在 .env 檔案中")


if __name__ == "__main__":
    asyncio.run(main())
