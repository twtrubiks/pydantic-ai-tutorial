"""
Pydantic AI Harness 範例（聚焦：用 pydantic-ai 組出一個簡化版 Claude Code）

pydantic-ai-harness 是 Pydantic 官方另一個套件，把 Claude Code 這類 coding agent 的零件
做成一個個 capability，掛到 Agent 上就能用：
    - Coder：讀檔、寫檔、改檔、grep、shell、子代理，十行內做出簡化版 Claude Code
    - Skills：讀 SKILL.md（跟 Claude Code skills 同一個 agentskills.io 標準），
      模型平常只看到每個 skill 的名稱和描述，需要時才用 load_capability 載入內容
    - RepoContext：自動讀 CLAUDE.md / AGENTS.md，也能列出 .claude/skills、.claude/agents 裡有什麼
    - SystemReminders：照 Claude Code 的做法，用 <system-reminder> 標籤在長任務中途重申規則

主題：
    A. RepoContext + Skills（不給 shell，唯讀）：問 agent 這個專案有哪些 coding-assistant 設定
    B. Coder + Skills + SystemReminders：讓 agent 自己跑測試、找 bug、修好，再依 skill 寫修正說明

安全性：
    Coder 的 shell 沒有指令白名單，模型可以在你的電腦上執行任意指令。
    本範例每次都在暫存目錄建一個迷你專案給 agent 改，跑完自動刪除；
    LocalWorkspace 只把 PATH、HOME、LANG 傳給 shell，不會把 API Key 交給模型執行的指令。

文檔參考：
    - Harness 總覽：https://pydantic.dev/docs/ai/harness/
    - Coder：https://pydantic.dev/docs/ai/harness/coder/
    - Skills：https://pydantic.dev/docs/ai/harness/skills/
    - RepoContext：https://pydantic.dev/docs/ai/harness/repo-context/
    - SystemReminders：https://pydantic.dev/docs/ai/harness/system-reminders/

執行前：
    1. pip install "pydantic-ai-harness[coder,skills]"
    2. 在 .env 檔案中設置 ANTHROPIC_API_KEY=sk-ant-...
    3. python claude_harness_demo.py
"""

import asyncio
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness import RepoContext, Skills, SystemReminders
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.system_reminders import Reminder, ReminderFiredEvent

# 載入 .env 檔案中的環境變數
load_dotenv()

# ========== 模型設定 ==========

# 設置 Anthropic API Key（前往 https://platform.claude.com/settings/keys 取得，格式 sk-ant-...）
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Haiku 5.5：最便宜，coding agent 會來回呼叫很多次工具，用便宜的模型練習比較安心
MODEL_NAME = "claude-haiku-5-5"


def build_model() -> AnthropicModel:
    """建立 Claude 模型（延後到確認有 API Key 後才建，避免 import 即報錯）"""
    return AnthropicModel(
        MODEL_NAME,
        provider=AnthropicProvider(api_key=ANTHROPIC_API_KEY),
    )


# ========== 迷你專案：給 agent 操作的範例 repo ==========

# RepoContext 會自動把這份 CLAUDE.md 放進 instructions
CLAUDE_MD = """\
# 迷你計算機專案

- 一律使用繁體中文回覆。
- 只能修改 calc.py，不要修改測試檔 test_calc.py。
- 修改程式後，必須執行 `python test_calc.py` 確認測試通過。
- 回覆的最後一行固定寫：「—— 已依 CLAUDE.md 規範完成」
"""

# Skills 只會先把 name + description 給模型看，模型判斷需要時才載入內文
RELEASE_NOTES_SKILL = """\
---
name: release-notes
description: 撰寫修正說明、版本更新說明（release notes / changelog）前，一定要先載入這個 skill，照裡面的格式寫。
---

請嚴格使用以下格式撰寫：

## 🐛 修正
- 每個修正一行，格式：`檔案名稱` 函式名稱：問題 → 修正方式

## ✅ 驗證
- 列出執行過的測試指令與結果
"""

CODE_REVIEW_SKILL = """\
---
name: code-review
description: 進行程式碼審查（code review）時使用。
---

依嚴重程度（高 / 中 / 低）列出問題，每個問題附上檔案與行號。
"""

# 故意埋兩個 bug，讓 agent 自己跑測試找出來
CALC_PY = '''\
def average(numbers: list[float]) -> float:
    """計算平均值"""
    return sum(numbers) / (len(numbers) - 1)


def discount(price: float, percent: float) -> float:
    """依百分比打折，percent=20 代表打八折"""
    return price * percent / 100
'''

TEST_CALC_PY = """\
from calc import average, discount


def test_average():
    assert average([1, 2, 3]) == 2


def test_discount():
    assert discount(100, 20) == 80


if __name__ == "__main__":
    test_average()
    test_discount()
    print("全部測試通過")
"""


def create_demo_repo(root: Path) -> None:
    """在暫存目錄建立迷你專案：CLAUDE.md、兩個 skill、有 bug 的程式和測試"""
    (root / "CLAUDE.md").write_text(CLAUDE_MD, encoding="utf-8")
    for name, content in [("release-notes", RELEASE_NOTES_SKILL), ("code-review", CODE_REVIEW_SKILL)]:
        skill_dir = root / ".claude" / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    (root / "calc.py").write_text(CALC_PY, encoding="utf-8")
    (root / "test_calc.py").write_text(TEST_CALC_PY, encoding="utf-8")


# ========== 觀察工具 ==========


def print_tool_trace(result) -> None:
    """走訪 result.all_messages()，印出 agent 依序呼叫了哪些工具"""
    print("\n🔬 工具呼叫軌跡：")
    count = 0
    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, ToolCallPart):
                count += 1
                args = part.args_as_dict()
                # edit_file 一次改多處時 replacements 是清單，只顯示筆數
                brief = ", ".join(
                    f"{k}=[{len(v)} 處]" if isinstance(v, list) else f"{k}={str(v)[:60]!r}"
                    for k, v in args.items()
                )
                print(f"  [{count:02d}] 🛠️  {part.tool_name}({brief})")
            elif isinstance(part, ToolReturnPart) and part.tool_name == "load_capability":
                # 載入 skill 的回傳是 {'instructions': '# Skill: <name>\n...'}
                content = part.content
                text = content.get("instructions", "") if isinstance(content, dict) else str(content)
                print(f"       📖 載入：{text.strip().splitlines()[0]}")


def print_usage(result) -> None:
    """印出這次 run 的請求數與 token 用量"""
    usage = result.usage
    print(
        f"\n📈 用量：{usage.requests} 次請求，輸入 {usage.input_tokens} token"
        f"（快取讀取 {usage.cache_read_tokens}），輸出 {usage.output_tokens} token"
    )


# ========== 主題 A：RepoContext + Skills（唯讀）==========


async def demo_repo_context_and_skills(model: AnthropicModel, repo: Path) -> None:
    """不給 shell 和寫檔工具，只看 agent 能不能認出這個 repo 的 Claude Code 設定"""
    print("\n\n📌 主題 A：RepoContext + Skills（唯讀，不給 shell）")
    print("-" * 60)

    agent = Agent(
        model,
        capabilities=[
            LocalWorkspace(repo),
            # expose_inventory_tool=True：多給一個 inventory_agent_context 工具，列出 .claude 底下的資產
            RepoContext(expose_inventory_tool=True),
            # Skills 不會自動找 .claude/skills，要明確指定目錄
            Skills(".claude/skills"),
        ],
    )

    result = await agent.run(
        "這個專案有哪些 coding-assistant 設定？請說明 CLAUDE.md 的規則，以及有哪些 skill 可以用。"
        "不用載入 skill 的內容。",
        usage_limits=UsageLimits(request_limit=10),
    )

    # CLAUDE.md 是放在 instructions，不是工具呼叫，從第一個請求的 instructions 確認
    first_request = result.all_messages()[0]
    loaded = "迷你計算機專案" in (first_request.instructions or "")
    print(f"\n📄 CLAUDE.md 是否已自動放進 instructions：{'✅ 是' if loaded else '❌ 否'}")

    print_tool_trace(result)
    print(f"\n🤖 回答：\n{result.output}")
    print_usage(result)


# ========== 主題 B：Coder + Skills + SystemReminders ==========


async def demo_coder(model: AnthropicModel, repo: Path) -> None:
    """讓 agent 自己跑測試、找 bug、修好，再依 release-notes skill 寫修正說明"""
    print("\n\n📌 主題 B：Coder + Skills + SystemReminders（簡化版 Claude Code）")
    print("-" * 60)

    agent = Agent(
        model,
        capabilities=[
            # 工作目錄 = 暫存的迷你專案；shell 指令也從這裡開始執行
            LocalWorkspace(repo),
            # Coder 已內含 RepoContext，CLAUDE.md 會自動載入
            Coder(),
            Skills(".claude/skills"),
            # 每 3 次模型請求重申一次規則，包在 <system-reminder> 標籤裡，不寫進對話歷史
            SystemReminders(
                reminders=[Reminder("只能修改 calc.py，不要修改 test_calc.py。", interval=3)],
            ),
        ],
    )

    # SystemReminders 不會寫進 message history，要訂閱事件才看得到它何時觸發
    @agent.on_event(ReminderFiredEvent)
    async def show_reminder(ctx, event: ReminderFiredEvent) -> None:
        print(f"  🔔 第 {ctx.run_step} 步觸發 system reminder：{event.text.splitlines()[1]}")

    test_before = (repo / "test_calc.py").read_text(encoding="utf-8")

    print("🧑 任務：test_calc.py 跑不過，請找出原因並修好，最後寫一份修正說明。\n")
    result = await agent.run(
        "test_calc.py 跑不過，請找出原因並修好，最後寫一份修正說明。",
        usage_limits=UsageLimits(request_limit=30),
    )

    print_tool_trace(result)
    print(f"\n🤖 回答：\n{result.output}")
    print_usage(result)

    # 不相信 agent 的自我宣稱：自己再跑一次測試，並確認測試檔沒被改
    print("\n🔍 獨立驗證：")
    check = subprocess.run(
        [sys.executable, "test_calc.py"], cwd=repo, capture_output=True, text=True
    )
    print(f"  • 測試結果：{'✅ 通過' if check.returncode == 0 else '❌ 失敗'}")
    unchanged = (repo / "test_calc.py").read_text(encoding="utf-8") == test_before
    print(f"  • test_calc.py 未被修改：{'✅ 是' if unchanged else '❌ 否'}")
    print("  • 修正後的 calc.py：")
    for line in (repo / "calc.py").read_text(encoding="utf-8").splitlines():
        print(f"      {line}")


# ========== 說明 ==========


def explain() -> None:
    """整理重點"""
    print("\n\n" + "=" * 60)
    print("💡 重點整理")
    print("=" * 60)

    print("\n1. 每個零件都是 capability，掛到 Agent(capabilities=[...]) 就能用：")
    print("   • LocalWorkspace(path)：指定 agent 工作的目錄，Coder / Skills / RepoContext 都從這裡讀")
    print("   • Coder()：read_file / write_file / edit_file / list_files / grep / shell / delegate_task")
    print("   • Coder 本身就是 FileSystem + Shell + RepoContext + SubAgents + 壓縮 / 快取等零件組成")

    print("\n2. Skills：")
    print("   • 一開始只把每個 skill 的 name + description 給模型看，省 token")
    print("   • 模型判斷需要時才呼叫 load_capability 載入內文（看軌跡裡的 📖）")
    print("   • 不會自動搜尋 .claude/skills，要明確傳入目錄")

    print("\n3. RepoContext：")
    print("   • CLAUDE.md / AGENTS.md 在 run 開始時讀一次，放進 instructions（快取前綴不會變）")
    print("   • expose_inventory_tool=True 可以列出 .claude 底下的 skills、agents、hooks")

    print("\n4. SystemReminders：")
    print("   • 長任務做久了模型會忘記規則，定期用 <system-reminder> 重申")
    print("   • 只放在每次請求的尾端，不寫進對話歷史，也不會打破 prompt 快取")

    print("\n5. 安全性：")
    print("   • Coder 的 shell 沒有白名單，務必讓它在暫存目錄或 sandbox（Modal / E2B）裡跑")
    print("   • 想讓另一個 agent 只能看不能改，可用 ReadOnlyWorkspace")


# ========== 主程式 ==========


async def main():
    """主程式入口"""
    if not ANTHROPIC_API_KEY:
        print("⚠️  警告：請在 .env 檔案中設置 ANTHROPIC_API_KEY")
        print("💡 前往 https://platform.claude.com/settings/keys 取得 API Key（格式 sk-ant-...）")
        return

    try:
        model = build_model()
        print(f"🚀 使用 Claude 模型：{MODEL_NAME}")

        # 每次都在暫存目錄建立全新的迷你專案，跑完自動刪除
        with tempfile.TemporaryDirectory(prefix="harness_demo_") as tmp:
            repo = Path(tmp)
            create_demo_repo(repo)
            print(f"📁 迷你專案位置：{repo}")

            # 主題 A：認出 repo 的 Claude Code 設定
            await demo_repo_context_and_skills(model, repo)

            # 主題 B：自己修 bug 的 coding agent
            await demo_coder(model, repo)

        # 重點整理
        explain()

    except Exception as e:
        print(f"\n❌ 執行出錯：{e}")
        print("💡 請檢查 ANTHROPIC_API_KEY 是否正確設置，以及是否已安裝 pydantic-ai-harness[coder,skills]")


if __name__ == "__main__":
    # 執行異步主程式
    asyncio.run(main())
