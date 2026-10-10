"""
Claude Advisor 範例：便宜的模型做事，遇到關鍵判斷再請教更強的模型

情境：星河科技的出貨排程助理。合約寫「下單隔天起算，第 20 個工作天出貨」，公司休假日、補班日要用工具查。
陷阱：從 12 月中往後數 20 個工作天會跨年，只查 2026 年的行事曆，就會漏掉 2027 年的休假日。

主題：
    A. 原生 Advisor（AdvisorTool）：Haiku 5.5 當執行者，Opus 5.5 當顧問，對照「Haiku 單獨」「Opus 單獨」。
       顧問在 Anthropic 的伺服器上執行，直接讀整段對話（instructions、工具定義、工具結果），
       執行者呼叫時不用寫問題，也不需要多一次來回。
    B. 本機 Advisor（pydantic-ai-harness）：NVIDIA Nemotron 當執行者，Claude Opus 5.5 當顧問（跨供應商）。
       顧問變成一般的函式工具，執行者要自己把問題寫進 prompt；建議的內容看得到，費用也會算進 usage.cost。

費用重點：
    - 原生 Advisor 的 token 不會算進 usage 的 input / output token，usage.cost 也不含顧問費
      （顧問用 Opus 的價格計費，跟執行者不同）。要看真正的帳單，得自己從 usage.details 的 advisor_* 換算。
    - 顧問每次都會讀完整段對話再思考，短任務可能比直接用 Opus 還貴；
      適合「大部分步驟很機械、少數關鍵判斷要很準」的長任務（寫程式、多步驟研究）。

文檔參考：
    - Anthropic advisor tool：https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool
    - Native Tools：https://pydantic.dev/docs/ai/tools-toolsets/native-tools/

執行前：
    1. 在 .env 檔案中設置 ANTHROPIC_API_KEY=sk-ant-...（主題 B 另外需要 NVIDIA_API_KEY，沒有就跳過）
    2. 需要 pydantic-ai >= 2.55、pydantic-ai-harness（pip install pydantic-ai-harness）
    3. python claude_advisor_demo.py（跑一次大約 0.15 美元，大部分是顧問費）
"""

import asyncio
import os
import re
import unicodedata
from datetime import date, timedelta
from decimal import Decimal

from dotenv import load_dotenv
from genai_prices import Usage, calc_price
from pydantic_ai import Agent, RunUsage
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.native_tools import AdvisorTool
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_harness.advisor import Advisor

# 載入 .env 檔案中的環境變數
load_dotenv()

# ========== 模型設定 ==========

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

EXECUTOR_MODEL = "claude-haiku-5-5"  # 執行者：便宜、快，負責查資料、算日期、回覆
ADVISOR_MODEL = "claude-opus-5-5"  # 顧問：貴、強，只在關鍵時刻出意見
NVIDIA_MODEL = "nvidia/nemotron-3-super-120b-a12b"  # 主題 B 的執行者

# 執行者關掉 adaptive thinking，讓差異來自「有沒有顧問」；顧問在伺服器上照常思考
SETTINGS = AnthropicModelSettings(thinking=False, max_tokens=2048)

# 顧問每次最多輸出多少 token（含思考，最小 1024；官方建議從 2048 開始）
ADVISOR_MAX_TOKENS = 2048


def build_claude(model_name: str) -> AnthropicModel:
    """建立 Claude 模型（延後到確認有 API Key 後才建，避免 import 即報錯）"""
    return AnthropicModel(model_name, provider=AnthropicProvider(api_key=ANTHROPIC_API_KEY))


def build_nvidia() -> OpenAIChatModel:
    """NVIDIA NIM 走 OpenAI 相容端點"""
    return OpenAIChatModel(
        NVIDIA_MODEL,
        provider=OpenAIProvider(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY),
    )


# ========== 公司行事曆（工具）與標準答案 ==========

CALENDAR = {
    2026: {
        "holidays": {"2026-12-25": "聖誕節（公司福利假）", "2026-12-31": "年終盤點"},
        "makeup_workdays": {"2026-12-19": "週六補班"},
    },
    2027: {
        "holidays": {"2027-01-01": "元旦", "2027-01-08": "尾牙隔天公休"},
        "makeup_workdays": {},
    },
}

ORDER_DATE = date(2026, 12, 15)  # 週二
WORKDAYS = 20


def get_company_calendar(year: int) -> dict:
    """查詢公司某一年的行事曆。

    Args:
        year: 西元年，例如 2026

    Returns:
        holidays 是休假日（不算工作天），makeup_workdays 是補班日（週六但要上班，算工作天）
    """
    return CALENDAR.get(year, {"holidays": {}, "makeup_workdays": {}})


def expected_ship_date() -> date:
    """用程式算標準答案，拿來驗證模型的回答"""
    holidays = {date.fromisoformat(d) for cal in CALENDAR.values() for d in cal["holidays"]}
    makeups = {date.fromisoformat(d) for cal in CALENDAR.values() for d in cal["makeup_workdays"]}
    day, count = ORDER_DATE, 0
    while count < WORKDAYS:
        day += timedelta(days=1)
        if (day.weekday() < 5 and day not in holidays) or day in makeups:
            count += 1
    return day


INSTRUCTIONS = """\
你是星河科技的出貨排程助理，用繁體中文回答。
公司的休假日、補班日一定要用 get_company_calendar 查詢，不要自己猜。"""

# 只加在有顧問的 agent 上：沒有顧問卻叫它請教顧問，模型會到處找這個工具
# Haiku 當執行者容易忘記問顧問，官方建議在 instructions 寫清楚「什麼時候要問」
ADVISOR_HINT = """
你有一個 advisor 工具，可以請教更強的模型。
開始查資料、計算之前，先呼叫 advisor 確認做法；回覆客戶之前，再呼叫一次確認答案。"""

QUESTION = (
    f"客戶在 {ORDER_DATE}（週二）下單，合約寫「下單隔天起算，第 {WORKDAYS} 個工作天出貨」。"
    "工作天是週一到週五，扣掉公司休假日；補班日也算工作天。"
    "請問出貨日是哪一天？最後一行只寫日期 YYYY-MM-DD。"
)


# ========== 共用輔助函式 ==========


def extract_date(output: str) -> date | None:
    """取回答裡最後一個 YYYY-MM-DD"""
    found = re.findall(r"\d{4}-\d{2}-\d{2}", output)
    return date.fromisoformat(found[-1]) if found else None


def format_usd(cost: Decimal | None) -> str:
    return f"${cost:.4f}" if cost is not None else "（查無價格）"


def pad(text: str, width: int) -> str:
    """補空白到指定顯示寬度（中文字占 2 格），讓對照表對齊"""
    shown = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)
    return text + " " * max(width - shown, 1)


def native_advisor_cost(usage: RunUsage) -> Decimal:
    """原生 Advisor 的費用：Pydantic AI 把顧問 token 放在 usage.details 的 advisor_*，用顧問模型的價格自己換算"""
    details = usage.details
    if not details.get("advisor_iterations"):
        return Decimal(0)
    cache_read = details.get("advisor_cache_read_input_tokens", 0)
    cache_write = details.get("advisor_cache_creation_input_tokens", 0)
    price = calc_price(
        Usage(
            # genai-prices 的 input_tokens 要包含快取的部分；Anthropic 回傳的 advisor_input_tokens 不含
            input_tokens=details.get("advisor_input_tokens", 0) + cache_read + cache_write,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            output_tokens=details.get("advisor_output_tokens", 0),
        ),
        model_ref=ADVISOR_MODEL,
        provider_id="anthropic",
    )
    return price.total_price


def describe_native_advice(content: object) -> str:
    """原生顧問的回傳有三種：明文、加密、錯誤"""
    if not isinstance(content, dict):
        return str(content)
    kind = content.get("type")
    if kind == "advisor_redacted_result":
        return "🔒 建議已加密（advisor_redacted_result）：只有執行者在伺服器上讀得到，程式看不到內容"
    if kind == "advisor_result":
        return f"📝 顧問建議：{content.get('text', '')[:200]}"
    return f"⚠️ 顧問沒回答（{content.get('error_code')}），執行者會自己繼續做"


def print_trace(messages: list[ModelMessage]) -> None:
    """依序印出工具呼叫、顧問諮詢，以及執行者中途說的話（最後的回答另外印）"""
    responses = [m for m in messages if isinstance(m, ModelResponse)]
    # 原生顧問的結果有時會落在下一個回應，先用 tool_call_id 配對好，跟呼叫印在一起
    native_returns = {
        part.tool_call_id: part.content
        for message in responses
        for part in message.parts
        if isinstance(part, NativeToolReturnPart)
    }
    for message in messages:
        for part in message.parts:
            if isinstance(part, NativeToolCallPart) and part.tool_name == "advisor":
                print("   🧭 請教顧問（原生：不用寫問題，顧問直接讀整段對話）")
                print(f"      {describe_native_advice(native_returns.get(part.tool_call_id))}")
            elif isinstance(part, ToolCallPart) and part.tool_name == "advisor":
                prompt = str(part.args_as_dict().get("prompt", "")).replace("\n", " ")
                print(f"   🧭 請教顧問（本機：執行者自己寫問題）：{prompt[:120]}…")
            elif isinstance(part, ToolReturnPart) and part.tool_name == "advisor":
                advice = str(part.content).replace("\n", " ")
                print(f"      📝 顧問建議：{advice[:200]}…")
            elif isinstance(part, ToolCallPart):
                print(f"   🔧 {part.tool_name}({part.args_as_dict()})")
            elif isinstance(part, TextPart) and message is not responses[-1] and part.content.strip():
                print(f"   💬 {part.content.strip()[:120]}")


def grade(output: str) -> tuple[date | None, bool]:
    answer = extract_date(output)
    return answer, answer == expected_ship_date()


# ========== 主題 A：原生 Advisor（Haiku 做事、Opus 指點）==========


async def demo_native_advisor():
    print("\n" + "=" * 70)
    print("📘 主題 A：原生 Advisor（AdvisorTool）— Haiku 做事、Opus 指點")
    print("=" * 70)
    print(f"❓ {QUESTION}")
    print(f"🎯 標準答案（程式算的）：{expected_ship_date()}")

    haiku = build_claude(EXECUTOR_MODEL)
    advisor = AdvisorTool(model=ADVISOR_MODEL, max_tokens=ADVISOR_MAX_TOKENS)

    setups = [
        ("Haiku 單獨", Agent(haiku, instructions=INSTRUCTIONS, tools=[get_company_calendar], model_settings=SETTINGS)),
        (
            "Haiku + Opus 顧問",
            Agent(
                haiku,
                instructions=INSTRUCTIONS + ADVISOR_HINT,
                tools=[get_company_calendar],
                model_settings=SETTINGS,
                # 原生工具用 NativeTool capability 掛上去；beta header（advisor-tool-2026-03-01）Pydantic AI 會自動加
                capabilities=[NativeTool(advisor)],
            ),
        ),
        (
            "Opus 單獨",
            Agent(
                build_claude(ADVISOR_MODEL),
                instructions=INSTRUCTIONS,
                tools=[get_company_calendar],
                model_settings=SETTINGS,
            ),
        ),
    ]

    rows = []
    for label, agent in setups:
        print(f"\n▶️  {label}")
        result = await agent.run(QUESTION)
        print_trace(result.all_messages())
        answer, correct = grade(result.output)
        usage = result.usage
        consults = usage.details.get("advisor_iterations", 0)
        executor_cost = usage.cost
        advisor_cost = native_advisor_cost(usage)
        print(f"   {'✅' if correct else '❌'} 回答：{answer}")
        if consults:
            print(
                f"   🧾 執行者 token：輸入 {usage.input_tokens:,}、輸出 {usage.output_tokens:,}"
                f"（usage.cost {format_usd(executor_cost)}）"
            )
            print(
                f"   🧾 顧問 token（usage.details，不在上面的總數裡）：請教 {consults} 次，"
                f"輸入 {usage.details.get('advisor_input_tokens', 0):,}、"
                f"輸出 {usage.details.get('advisor_output_tokens', 0):,}（含思考）→ 自己換算 {format_usd(advisor_cost)}"
            )
        rows.append((label, answer, correct, executor_cost, advisor_cost))

    print("\n📊 對照表")
    print(f"   {pad('組合', 20)}{pad('回答', 15)}{pad('usage.cost', 12)}{pad('顧問費', 10)}實際合計")
    for label, answer, correct, executor_cost, advisor_cost in rows:
        total = (executor_cost or Decimal(0)) + advisor_cost
        mark = "✅" if correct else "❌"
        print(
            f"   {pad(label, 20)}{mark} {pad(str(answer), 12)}{pad(format_usd(executor_cost), 12)}"
            f"{pad(format_usd(advisor_cost), 10)}{format_usd(total)}"
        )
    print("   💡 usage.cost 不含原生顧問的費用；只看 usage.cost 會以為「加了顧問幾乎不花錢」")
    print("   💡 Haiku 單獨偶爾也會答對（多跑幾次看看）；常見錯誤是只查 2026 年，或日期數錯")


# ========== 主題 B：本機 Advisor（NVIDIA 做事、Claude 指點）==========


async def demo_local_advisor():
    print("\n" + "=" * 70)
    print("📗 主題 B：本機 Advisor（pydantic-ai-harness）— NVIDIA 做事、Claude 指點")
    print("=" * 70)

    if not NVIDIA_API_KEY:
        print("⏭️  沒有設定 NVIDIA_API_KEY，跳過主題 B")
        return

    agent = Agent(
        build_nvidia(),
        instructions=INSTRUCTIONS + ADVISOR_HINT,
        tools=[get_company_calendar],
        # 執行者不是 Anthropic，Advisor 會自動改用本機模式：
        # 掛一個叫 advisor 的函式工具，裡面再用 Claude Opus 跑一個獨立的 agent
        capabilities=[Advisor(f"anthropic:{ADVISOR_MODEL}", max_tokens=ADVISOR_MAX_TOKENS)],
    )

    print(f"▶️  {NVIDIA_MODEL} + Opus 顧問（NVIDIA 比較慢，大約要 1 分鐘）")
    result = await agent.run(QUESTION)
    print_trace(result.all_messages())
    answer, correct = grade(result.output)
    print(f"   {'✅' if correct else '❌'} 回答：{answer}（標準答案 {expected_ship_date()}）")
    print(f"   🧾 usage.requests：{result.usage.requests}（含顧問的請求）")
    print(f"   🧾 usage.cost：{format_usd(result.usage.cost)}（只有顧問的 Opus 費用；NVIDIA 查無價格，不計）")
    print("   💡 本機顧問是一般的模型請求，費用會正常算進 usage.cost；建議內容也是明文，看得到")


# ========== 重點說明 ==========


def explain():
    print("\n" + "=" * 70)
    print("📚 重點整理")
    print("=" * 70)
    print("""
1. 原生 Advisor 怎麼開
   capabilities=[NativeTool(AdvisorTool(model="claude-opus-5-5"))]
   beta header 由 Pydantic AI 自動加上。Haiku 5.5 當執行者時，顧問可以是 Opus、Sonnet 5.5 等更強的模型。

2. 顧問看得到什麼
   原生：整段對話（instructions、工具定義、工具結果、執行者寫到一半的內容），執行者呼叫時不帶參數。
   Opus / Sonnet 5.x 當顧問時，建議會加密回傳，程式讀不到；Opus 4.8 這類舊款顧問才是明文。

3. Haiku 容易忘記問顧問
   在 instructions 寫清楚什麼時候要呼叫（官方建議：動手之前、交付之前、卡關時）。

4. 帳單要自己算
   原生顧問的 token 在 usage.details 的 advisor_input_tokens / advisor_output_tokens，
   不在 input_tokens、output_tokens 裡，usage.cost 也不含。用 genai-prices 以顧問模型的價格換算。
   max_tokens 限制每次顧問的輸出（含思考），max_uses 限制「單一請求」內的次數（不是整個 run）。

5. 本機 Advisor（pydantic-ai-harness）
   Advisor("anthropic:claude-opus-5-5")：執行者也是 Anthropic 就用原生，其他供應商自動改成本機工具。
   本機模式：可以跨供應商、建議看得到、費用算進 usage.cost；
   但顧問只看到執行者寫的 prompt（forward_history=True 才會帶上完整對話），而且多一次來回。

6. 什麼時候值得用
   長任務、大部分步驟很機械、少數關鍵判斷要很準（例如寫程式的 agent）。
   像本範例這種短任務，直接用 Opus 可能更便宜；顧問的價值在「整段任務只需要強模型出手幾次」。
""")


async def main():
    """
    主程式：執行所有示範

    主題 A：原生 Advisor，Haiku / Haiku + Opus 顧問 / Opus 三種組合對照答案與費用
    主題 B：本機 Advisor，NVIDIA 執行者 + Claude 顧問（需要 NVIDIA_API_KEY）
    """
    print("\n" + "=" * 70)
    print("🧭 Claude Advisor 示範：便宜的模型做事，關鍵判斷請教強模型")
    print(f"   執行者：{EXECUTOR_MODEL}　顧問：{ADVISOR_MODEL}")
    print("=" * 70)

    if not ANTHROPIC_API_KEY:
        print("⚠️  警告：請在 .env 檔案中設置 ANTHROPIC_API_KEY")
        print("💡 前往 https://platform.claude.com/settings/keys 取得 API Key（格式 sk-ant-...）")
        return

    await demo_native_advisor()
    await demo_local_advisor()
    explain()


if __name__ == "__main__":
    asyncio.run(main())
