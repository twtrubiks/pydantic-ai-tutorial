"""
工具核准範例（Human-in-the-Loop）：危險操作先問過人再執行

情境：銀行轉帳助理。查餘額直接執行；轉帳超過 1,000 元、關閉帳戶都要客戶本人核准。

主題：
    A. 暫停式核准：遇到要核准的工具，agent 先結束這次 run，回傳 DeferredToolRequests。
       把整段對話（Conversation）存成 JSON，模擬「等主管隔天審核」，讀回後帶著核准結果接著跑。
    B. 中途核准（HandleDeferredToolCalls capability）：在同一次 run 裡直接問人，不必結束再重開，
       適合核准的人就在同一個程式旁邊（終端機、桌面 App）的情況。

兩種要求核准的寫法：
    1. Tool(fn, requires_approval=True)：每次呼叫都要核准（關閉帳戶）
    2. 在工具裡 raise ApprovalRequired：依參數決定要不要核准（轉帳超過 1,000 元才要）
       核准後 Pydantic AI 會用同樣的參數再呼叫一次工具，這時 ctx.tool_call_approved 為 True

為什麼用 Claude Haiku：
    查餘額、關閉帳戶都是沒有參數的工具，NVIDIA Nemotron 遇到這種工具會自己編參數而失敗。

文檔參考：
    - Deferred Tools：https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/
    - HandleDeferredToolCalls：https://pydantic.dev/docs/ai/capabilities/handle-deferred-tool-calls/

執行前：
    1. 在 .env 檔案中設置 ANTHROPIC_API_KEY=sk-ant-...
    2. python tool_approval_demo.py（過程中會在終端機問你 y/N）
"""

import asyncio
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic_ai import (
    Agent,
    ApprovalRequired,
    ConversationTypeAdapter,
    DeferredToolRequests,
    DeferredToolResults,
    RunContext,
    Tool,
    ToolDenied,
)
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

# 載入 .env 檔案中的環境變數
load_dotenv()

# ========== 模型設定 ==========

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL_NAME = "claude-haiku-5-5"

# 關掉 adaptive thinking 省 token；回答只要幾句話，max_tokens 設小一點
SETTINGS = AnthropicModelSettings(thinking=False, max_tokens=1024)


def build_model() -> AnthropicModel:
    """建立 Claude 模型（延後到確認有 API Key 後才建，避免 import 即報錯）"""
    return AnthropicModel(MODEL_NAME, provider=AnthropicProvider(api_key=ANTHROPIC_API_KEY))


# ========== 模擬的銀行帳戶（當作 deps 傳給工具）==========

TRANSFER_LIMIT = 1_000  # 單筆超過這個金額要人工核准


@dataclass
class Bank:
    """一個客戶的帳戶；每個主題各用一份新的，最後用它驗證「實際上」執行了什麼"""

    balance: int = 50_000
    payees: dict[str, str] = field(default_factory=lambda: {"A-001": "媽媽", "B-002": "房東"})
    executed: list[str] = field(default_factory=list)  # 真正執行過的操作
    closed: bool = False


# ========== 工具 ==========


def get_balance(ctx: RunContext[Bank]) -> str:
    """查詢目前帳戶餘額"""
    return f"目前餘額 {ctx.deps.balance:,} 元"


def transfer(ctx: RunContext[Bank], to_account: str, amount: int) -> str:
    """轉帳到約定帳戶

    Args:
        to_account: 約定帳號，A-001 是媽媽、B-002 是房東
        amount: 金額（新台幣，整數）
    """
    bank = ctx.deps
    if to_account not in bank.payees:
        return f"{to_account} 不是約定帳號，無法轉帳"
    if amount > TRANSFER_LIMIT and not ctx.tool_call_approved:
        # 第一次呼叫先不執行，交給人核准；metadata 會跟著 DeferredToolRequests 一起回傳
        raise ApprovalRequired(metadata={"reason": f"單筆超過 {TRANSFER_LIMIT:,} 元"})
    if amount > bank.balance:
        return f"餘額不足（目前 {bank.balance:,} 元）"
    bank.balance -= amount
    bank.executed.append(f"轉帳 {amount:,} 元給{bank.payees[to_account]}（{to_account}）")
    return f"已轉帳 {amount:,} 元給{bank.payees[to_account]}，餘額 {bank.balance:,} 元"


def close_account(ctx: RunContext[Bank]) -> str:
    """結清並關閉帳戶（不可復原）"""
    ctx.deps.closed = True
    ctx.deps.executed.append("關閉帳戶")
    return "帳戶已關閉"


INSTRUCTIONS = (
    "你是銀行的轉帳助理，請用繁體中文、簡短回答。"
    "客戶要求的操作請直接呼叫工具，需要核准的操作系統會自動向客戶確認，你不用再用文字詢問。"
    "操作被拒絕時，請如實告知客戶沒有執行。"
)


def build_agent(model: AnthropicModel, **kwargs) -> Agent:
    """兩個主題共用同一組工具，差別只在 output_type 或 capabilities"""
    return Agent(
        model,
        deps_type=Bank,
        instructions=INSTRUCTIONS,
        model_settings=SETTINGS,
        tools=[
            get_balance,
            transfer,  # 依金額決定，在工具裡 raise ApprovalRequired
            Tool(close_account, requires_approval=True),  # 每次都要核准
        ],
        **kwargs,
    )


# ========== 共用：問人、印出結果 ==========


def ask_human(call: ToolCallPart, reason: str | None) -> bool | ToolDenied:
    """在終端機問客戶要不要核准；拒絕時用 ToolDenied 自訂要告訴模型的訊息"""
    args = ", ".join(f"{k}={v!r}" for k, v in call.args_as_dict().items())
    print(f"\n  🙋 需要核准：{call.tool_name}({args})" + (f"，原因：{reason}" if reason else ""))
    try:
        answer = input("     核准嗎？(y/N) ").strip().lower()
    except EOFError:  # 沒有可讀的輸入（例如 stdin 被關掉）就當作拒絕
        answer = ""
    if answer == "y":
        print("     ✅ 已核准")
        return True
    print("     ❌ 已拒絕")
    return ToolDenied("客戶拒絕了這個操作，沒有執行。")


def collect_approvals(requests: DeferredToolRequests) -> DeferredToolResults:
    """逐一詢問等待核准的呼叫，組成 DeferredToolResults"""
    approvals = {
        call.tool_call_id: ask_human(call, requests.metadata.get(call.tool_call_id, {}).get("reason"))
        for call in requests.approvals
    }
    # build_results 會檢查每個 tool_call_id 都對得上，打錯 ID 會在這裡就報錯
    return requests.build_results(approvals=approvals)


def print_tool_trace(messages: list[ModelMessage]):
    """列出模型呼叫了哪些工具、每個工具回了什麼（被拒絕的會標 outcome=denied）"""
    for message in messages:
        for part in message.parts:
            if isinstance(message, ModelResponse) and isinstance(part, ToolCallPart):
                print(f"     🔧 {part.tool_name}({part.args_as_dict()})")
            elif isinstance(part, ToolReturnPart):
                mark = "🚫" if part.outcome == "denied" else "↩️ "
                print(f"     {mark} {part.tool_name} → {part.content}")


def print_ledger(bank: Bank):
    """用帳戶資料獨立驗證，不只相信模型的說法"""
    print("\n  📒 帳戶實際狀態（程式驗證）：")
    for item in bank.executed or ["（沒有執行任何操作）"]:
        print(f"     • {item}")
    print(f"     餘額 {bank.balance:,} 元，帳戶{'已關閉' if bank.closed else '仍正常'}")


# ========== 主題 A：暫停式核准（run 結束 → 存檔 → 讀回 → 接著跑）==========


async def demo_pause_and_resume(model: AnthropicModel):
    print("\n⏸️  主題 A：暫停式核准（DeferredToolRequests + Conversation 存檔）")
    print("=" * 60)

    # output_type 要包含 DeferredToolRequests，run 才能在等待核准時「正常結束」
    agent = build_agent(model, output_type=[str, DeferredToolRequests])
    bank = Bank()

    prompt = "幫我轉 500 元給媽媽，再轉 30,000 元給房東繳房租。"
    print(f"\n👤 客戶：{prompt}")
    result = await agent.run(prompt, deps=bank)
    print_tool_trace(result.new_messages())

    if not isinstance(result.output, DeferredToolRequests):
        print(f"\n🤖 {result.output}（這次沒有需要核准的操作）")
        return

    print(f"\n  ⏸️  run 結束，output 是 DeferredToolRequests：{len(result.output.approvals)} 個呼叫等待核准")
    print(f"     不用核准的已經先執行：{bank.executed}")

    # Conversation 會連同 deferred_tool_requests（含 metadata）一起存；只存 messages 的話，
    # 讀回來分不出哪個呼叫在等核准、為什麼要核准
    stored = ConversationTypeAdapter.dump_json(result.conversation)
    print(f"  💾 整段對話存成 JSON（{len(stored):,} bytes），模擬等主管隔天審核……")

    conversation = ConversationTypeAdapter.validate_json(stored)
    requests = conversation.deferred_tool_requests
    assert requests is not None  # run 是因為等核准才結束的，一定有值
    print(f"  📂 讀回對話，待核准：{[call.tool_name for call in requests.approvals]}")
    results = collect_approvals(requests)

    # 不需要新的 prompt：帶著核准結果接續同一段對話
    result = await agent.run(conversation=conversation, deferred_tool_results=results, deps=bank)
    print("\n  ▶️  接續執行：")
    print_tool_trace(result.new_messages())
    print(f"\n🤖 {result.output}")
    print_ledger(bank)


# ========== 主題 B：中途核准（HandleDeferredToolCalls，run 不中斷）==========


async def demo_inline_handler(model: AnthropicModel):
    print("\n\n▶️  主題 B：中途核准（HandleDeferredToolCalls capability）")
    print("=" * 60)

    def handle(ctx: RunContext[Bank], requests: DeferredToolRequests) -> DeferredToolResults:
        """遇到要核准的呼叫時，Pydantic AI 會在 run 進行中呼叫這個 handler"""
        print(f"\n  ⏯️  handler 被呼叫（第 {ctx.run_step} 步），run 還在進行中")
        return collect_approvals(requests)

    # 有 handler 就不用在 output_type 加 DeferredToolRequests，output 一定是 str
    agent = build_agent(model, capabilities=[HandleDeferredToolCalls(handler=handle)])
    bank = Bank()

    prompt = "先幫我查餘額，然後轉 2,000 元給房東，最後把我的帳戶關掉。"
    print(f"\n👤 客戶：{prompt}")
    result = await agent.run(prompt, deps=bank)

    print("\n  🧾 整個 run 的工具軌跡：")
    print_tool_trace(result.all_messages())
    print(f"\n🤖 {result.output}")
    print(f"  📊 一次 run 共 {result.usage.requests} 次模型請求")
    print_ledger(bank)


# ========== 說明 ==========


def explain():
    print("\n\n" + "=" * 60)
    print("💡 重點整理")
    print("=" * 60)

    print("\n1. 什麼時候要核准：")
    print("   • Tool(fn, requires_approval=True)：每次都要（刪除、關帳戶這類不可逆操作）")
    print("   • 工具裡 raise ApprovalRequired(metadata=...)：依參數決定（金額門檻、受保護的檔案）")
    print("   • 核准後工具會用同樣參數再跑一次，ctx.tool_call_approved 為 True")

    print("\n2. 兩種處理方式：")
    print("   • 暫停式：output_type=[str, DeferredToolRequests]，run 結束後再用")
    print("     deferred_tool_results= 接續；核准的人不在線上（主管審核、前端按鈕）時用")
    print("   • 中途核准：capabilities=[HandleDeferredToolCalls(handler)]，一次 run 跑完；")
    print("     核准的人就在同一個程式旁邊時用。兩者可以並用，handler 回 None 就交給暫停式")

    print("\n3. 回覆核准結果：")
    print("   • True / ToolApproved(override_args=...)：核准（可以順便改參數）")
    print("   • ToolDenied('訊息')：拒絕，訊息會交給模型，讓它跟使用者解釋")
    print("   • requests.build_results(approvals=...)：會檢查 tool_call_id 都對得上")

    print("\n4. 暫停期間要存 Conversation，不要只存 messages：")
    print("   • result.conversation 包含 deferred_tool_requests 和 metadata，讀回後才知道在等什麼")

    print("\n5. 核准不是權限控管：")
    print("   • 它防的是「模型自作主張」，不是惡意的使用者；敏感操作在工具裡仍要自己檢查權限")


# ========== 主程式 ==========


async def main():
    if not ANTHROPIC_API_KEY:
        print("⚠️  警告：請在 .env 檔案中設置 ANTHROPIC_API_KEY")
        print("💡 前往 https://platform.claude.com/settings/keys 取得 API Key（格式 sk-ant-...）")
        return

    try:
        model = build_model()
        print(f"🚀 使用 Claude 模型：{MODEL_NAME}")
        print(f"🏦 初始餘額 50,000 元；單筆轉帳超過 {TRANSFER_LIMIT:,} 元、關閉帳戶都要核准")

        await demo_pause_and_resume(model)
        await demo_inline_handler(model)
        explain()

    except Exception as e:
        print(f"\n❌ 執行出錯：{e}")
        print("💡 請檢查 ANTHROPIC_API_KEY 是否正確設置在 .env 檔案中，以及模型名稱是否可用")


if __name__ == "__main__":
    asyncio.run(main())
