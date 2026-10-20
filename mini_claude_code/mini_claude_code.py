"""
迷你 Claude Code（聚焦：只用 anthropic SDK，手寫 coding agent 的核心迴圈）

Claude Code 去掉 UI、telemetry、feature flag 之後，剩下的骨架其實很小：
    1. 一個 while True 迴圈：呼叫模型 → 看 stop_reason → 是 tool_use 就執行工具、把結果接回去再呼叫
    2. 六個工具：read_file / write_file / edit_file / bash / glob / grep
       （--web 再加上 Anthropic 伺服器端的 web_search / web_fetch：搜尋在 API 那邊跑，結果直接回在同一個回應裡）
    3. 一份 system prompt：身分 + 環境資訊 + 專案的 CLAUDE.md
    4. 三件維持長對話不爆炸的事：prompt caching、工具輸出截斷、context 太大時 compaction
    5. 一層安全網：會改東西的工具（寫檔、改檔、shell）執行前先問人
    6. 兩個擴充點：Skills（.claude/skills/<名稱>/SKILL.md，system prompt 只列名稱與描述，要用時才用 skill 工具載入全文）
       與子代理（agent 工具：用同一個 class 再開一個獨立 context 的迴圈，只把最後的報告回傳給主代理）

這個檔案就是把這幾件事各寫一遍，不依賴任何 agent 框架，每一行都看得到。
對照組：claude_harness_demo.py 用 pydantic-ai-harness 的 Coder 一行做到同樣的事，但看不到迴圈。

從 Claude Code 原始碼借來的小細節：
    - 唯讀工具可以並行執行，會寫入的工具一律依序執行
    - edit_file 要求「先 read 過、而且讀完之後檔案沒被別人改過」才能改，old_string 必須唯一
    - 每個 tool_use 一定要有對應的 tool_result（即使被 Ctrl+C 中斷也要補一個），否則下一輪 API 會拒絕
    - stop_reason 是 max_tokens 時，不是接著講，而是丟掉被截斷的回應、把 max_tokens 加倍重送
    - 會隨時間變動的資訊（日期、git status）放在第一則 user 訊息，不放 system prompt，避免每輪都打破快取
    - Skills 採漸進式載入：system prompt 只有一行描述，模型判斷用得上才呼叫 skill 工具拿全文；/名稱 可手動觸發
    - 子代理有自己的空 context，中間讀了幾十個檔案、跑了多長的指令都不會污染主代理，只有最後的報告回去

Claude API 用到的功能：
    - Streaming：client.messages.stream，邊收邊印（回答用 rich 的 Markdown 邊串流邊渲染）
    - Prompt caching：system prompt 放一個顯式斷點；對話歷史交給頂層 cache_control 的「自動快取」
    - Adaptive thinking：5.x 模型預設開啟，thinking block 必須原樣送回，所以 assistant 內容整塊存
    - Thinking block 綁定前綴：5.5 系列的 thinking block 記錄了產生它時的 system、tools 與之前每一則訊息，
      之後只要改過其中任何一處，送回時就對不上（2026-08-31 以後建立的帳號直接 400）。所以歷史一律只 append 不改，
      --resume 原樣還原存檔時的 system prompt 與 tools，日期這類會變的資訊放在 user 訊息裡
    - effort：output_config.effort 控制思考深度（low / medium / high / xhigh / max）
    - 對話中途改 effort（/effort）：塞一則只有 output_config 的 system 訊息到 messages，前綴不變、快取不失效（beta）
    - 對話中途換模型（/model）：快取綁模型，一定重建一次；舊模型的 thinking block 新模型讀不懂會被靜默丟掉（不計費）
    - stop_reason 新值 model_context_window_exceeded：直接拿來觸發 compaction
    - refusal：被安全分類器拒絕時，已串流出的半截輸出整個丟掉；Opus 5.5 / Sonnet 5.5 等模型加上
      伺服器端 fallbacks（beta），被拒時 API 在同一次請求裡自動改用備援模型重跑
    - 伺服器端工具（--web）：tools 裡多放 web_search / web_fetch 的定義就好，沒有本機要執行的東西；
      回應裡會多出 server_tool_use 與 web_search_tool_result 等 block，跟 thinking 一樣整塊原樣存回；
      伺服器端迴圈跑滿會回 stop_reason pause_turn，直接再呼叫一次；主代理和子代理都能用（子代理也能上網查資料）

安全性：
    bash 工具沒有白名單，模型可以在你的電腦上執行任意指令。預設每個會改動的工具都會先問你 y/n/a，
    --yolo 會跳過詢問，請只在你信得過的目錄或 sandbox 裡用。
    不過有三類操作不管 [a] 或 --yolo 都一律詢問：危險指令（rm -rf、git push -f、curl | sh、sudo…）、
    碰到金鑰憑證的路徑（.env、~/.ssh…）、寫檔或改檔到工作目錄之外；regex 只是多一層保險，不是 sandbox。
    唯讀工具讀工作目錄外的檔案只是一般詢問，[a] 或 --yolo 可放行（bash 本來就能 cat 任何檔案，擋 read_file 沒意義）。
    --dangerously-skip-permissions 連上面三類都不問，跟 Claude Code 的同名選項一樣，只適合丟在容器裡跑。
    執行 shell 時會把 .env 載入的所有變數和名字含 API_KEY / SECRET 的環境變數拿掉，避免模型執行的指令拿到金鑰。

文檔參考：
    - Tool use：https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
    - Streaming：https://platform.claude.com/docs/en/build-with-claude/streaming
    - Prompt caching（含自動快取）：https://platform.claude.com/docs/en/build-with-claude/prompt-caching
    - Thinking 與工具迴圈：https://platform.claude.com/docs/en/build-with-claude/thinking-tool-workflows
    - Effort：https://platform.claude.com/docs/en/build-with-claude/effort
    - 快取失效規則與逃生口（含中途改 effort）：https://platform.claude.com/docs/en/build-with-claude/prompt-caching
    - Web search / web fetch 伺服器端工具：https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool

執行前：
    1. 在 .env 檔案中設置 ANTHROPIC_API_KEY=sk-ant-...
    2. 需要 anthropic >= 1.13（開發時實測的版本）與 prompt_toolkit（互動模式的 / 選單）；grep / glob 工具優先使用 ripgrep（rg），沒有也能跑
    3. 目前所在目錄就是 agent 的工作目錄，先 cd 到要讓它工作的專案，再用這支檔案的路徑執行，例如：
       python path/to/mini_claude_code.py                      # 互動模式
       python path/to/mini_claude_code.py "修好 calc.py 的 bug"   # 單次模式，跑完就退出
       python path/to/mini_claude_code.py --resume              # 接續上一次對話
       python path/to/mini_claude_code.py --web "查 anthropic SDK 最新版怎麼用 web_fetch"   # 開放模型上網搜尋（搜尋另外計費）
       python path/to/mini_claude_code.py --help                # 其他選項（--model、--effort、--yolo、--show-thinking…）
    4. Skills（選用）：在 .claude/skills/<名稱>/SKILL.md（專案）或 ~/.claude/skills/<名稱>/SKILL.md（全域）
       放 frontmatter 有 name / description 的 markdown，啟動時會列進 system prompt，互動模式打 /名稱 可直接執行
       這支檔案旁邊 skills/ 底下的是內建範例（brainstorming），不管在哪個目錄執行都會載入，可以照自己的需求增加
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic
from dotenv import dotenv_values, load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, ThreadedCompleter
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table

# 載入 .env 檔案中的環境變數
load_dotenv()

# 不交給 bash 工具的環境變數：.env 載入的全部（裡面通常全是金鑰），加上名字含 API_KEY / SECRET 的
SECRET_ENV_KEYS = set(dotenv_values()) | {"ANTHROPIC_API_KEY"}


def is_secret_env(name: str) -> bool:
    return name in SECRET_ENV_KEYS or "API_KEY" in name or "SECRET" in name


console = Console(highlight=False)

# ========== 設定 ==========

# Haiku 5.5：最便宜也最快，coding agent 一個任務會來回呼叫幾十次工具，用它練習比較安心
DEFAULT_MODEL = "claude-haiku-5-5"

# effort 的合法值。不給就用模型預設：Haiku 5.5 / Opus 5.5 是 medium，Sonnet 5.5 是 high（Claude Code 自己預設 xhigh）
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
MODEL_DEFAULT_EFFORT = {
    "claude-haiku-5-5": "medium",
    "claude-opus-5-5": "medium",
    "claude-sonnet-5-5": "high",
}

# 對話中途改 effort 的兩種做法：
#   1. 改頂層 output_config.effort：整段 messages 的快取失效，下一次請求全額重寫
#   2. 在 messages 裡塞一則 {"role": "system", "content": [], "output_config": {"effort": ...}}：
#      從下一個 user turn 起生效，接在已快取的前綴後面，快取不受影響。這是 beta，5.x 模型且 thinking 開著才支援。
# /effort 優先用第 2 種，模型不支援（或還沒有對話歷史、本來就沒快取）才退回第 1 種。
PER_MESSAGE_EFFORT_BETA = "mid-conversation-output-config-2026-07-01"
PER_MESSAGE_EFFORT_MODELS = (
    "claude-haiku-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-fable-5-1",
)

# 對話中途換模型（/model）的代價：
#   - 快取是綁模型的，官方明說沒有逃生口，換模型後下一次請求整段重寫一次
#   - thinking block 也是綁模型的：Haiku 5.5 的只有 Opus 5.5 / Sonnet 5.5 讀得懂，Opus 5.5 的只有 Fable 5.1 讀得懂，
#     其餘情況 API 會靜默丟掉（不計費、不報錯），對話照常接下去，只是新模型看不到舊模型的推理過程
#   - /effort 塞進歷史的 system 訊息：新模型也支援那個 beta 就留著；不支援（送了會 400）才移除，
#     而從歷史中間刪訊息會讓之後的 thinking block 前綴對不上，所以連 thinking block 一起拿掉（見 drop_effort_messages）

# 被安全分類器拒絕（stop_reason refusal）時，讓 API 在同一次請求裡自動改用備援模型重跑（beta）。
# "default" 依拒絕的類別挑 Anthropic 建議的模型，不用自己維護清單。
# Haiku 5.5 沒有伺服器端備援（"default" 照樣拒絕，給模型清單會 400），所以不在名單裡
SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"
SERVER_FALLBACK_MODELS = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1")

# 單次回應的輸出上限。5.x 模型最高 128K；thinking 也算在裡面，被截斷就得整段丟掉重來（見 run_turn），
# 串流沒有逾時的顧慮、上限本身也不用錢，所以給寬一點；真的被截斷時會自動加倍重送
DEFAULT_MAX_TOKENS = 64_000
MAX_TOKENS_CEILING = 128_000

# 5.x 模型的 context window 是 1M token，但 token 越多越貴也越慢，
# 這裡用一個保守的門檻：單次請求的輸入超過這個數就把舊對話摘要掉。
# Haiku 5.5 有兩種費率：prompt 超過 100K token，整個請求改用 5 倍的價格（$0.50 / $2.50），所以門檻壓在 100K 以下；
# 留一段餘量，是因為檢查用的是上一次請求的大小，下一次還會多出上一次的輸出與工具結果
DEFAULT_COMPACT_THRESHOLD = 150_000
MODEL_COMPACT_THRESHOLD = {"claude-haiku-5-5": 80_000}

# 工具輸出的上限（照 Claude Code 的做法：read 最多 2000 行、每行最多 2000 字、其他輸出最多 30000 字）
MAX_READ_LINES = 2000
MAX_LINE_CHARS = 2000
MAX_TOOL_OUTPUT_CHARS = 30_000
MAX_GLOB_RESULTS = 500
BASH_TIMEOUT_SECONDS = 120

# 對話存檔目錄：每輪結束自動存一份，--resume 可以接著聊
SESSION_DIR = Path.home() / ".mini_claude_code" / "sessions"

# Skills 目錄（跟 Claude Code 一樣）：專案的 .claude/skills 與家目錄的 ~/.claude/skills，每個 skill 一個資料夾、裡面放 SKILL.md
SKILLS_SUBDIR = Path(".claude") / "skills"
# 內建 skills：跟著這支檔案放在旁邊的 skills/，不管在哪個目錄執行都看得到（類似 Claude Code 自帶的 skill）
BUNDLED_SKILLS_DIR = Path(__file__).resolve().parent / "skills"

# 唯讀工具可以並行執行、不用問人；其他工具依序執行、執行前先問。skill 只是讀一份 markdown，也算唯讀
READ_ONLY_TOOLS = {"read_file", "glob", "grep", "skill"}

# Anthropic 伺服器端的網路工具（--web 開啟）：搜尋與抓網頁都在 API 那邊跑，不經過本機的 Tools.run，也不用問人。
# _20260209 起內建「動態過濾」：模型先寫一小段程式在 Anthropic 的沙盒裡呼叫 web_search 並篩選結果，
# 只把有用的部分放進 context，所以回應裡除了 server_tool_use(web_search) 還會看到 server_tool_use(code_execution)
# 與 code_execution_tool_result；_20260318 是目前最新版，5.x 模型都支援（Models API 的 capabilities.server_tools 查得到）。
# web_fetch 只能抓對話裡已經出現過的網址（使用者訊息、工具結果、先前的搜尋結果），防止模型把資料外送。
# 計費：搜尋每千次 10 美元（不含搜尋結果佔的 token），抓網頁只算 token。max_uses 限制單次請求最多用幾次。
# 其他可選參數：allowed_domains / blocked_domains 限制網域、web_fetch 的 max_content_tokens 限制單頁長度、
# response_inclusion="excluded" 不把動態過濾中間的搜尋結果回傳（省 token，但畫面上就看不到搜到了什麼）。
WEB_TOOL_MAX_USES = 5
SERVER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "web_search_20260318",
        "name": "web_search",
        "max_uses": WEB_TOOL_MAX_USES,
    },
    {"type": "web_fetch_20260318", "name": "web_fetch", "max_uses": WEB_TOOL_MAX_USES},
]

# 下面兩組 pattern 命中的操作「一律詢問」：按過 [a] 或開了 --yolo 也照問，只有 --dangerously-skip-permissions 會跳過。
# 正規表達式只是多一層保險，不是 sandbox（變數展開、base64 之類都能繞過），真要跑不信任的指令請用容器。
DANGEROUS_COMMAND_PATTERNS = [
    (
        re.compile(r"\brm\s+(-[a-zA-Z]*[rRf][a-zA-Z]*|--recursive|--force)\b"),
        "刪除檔案或目錄",
    ),
    (
        re.compile(
            r"\bgit\s+(push\b.*(\s-f\b|--force)|reset\s+--hard|clean\s+-[a-zA-Z]*f|checkout\s+--\s|restore\b|branch\s+-D)"
        ),
        "會丟掉 git 變更",
    ),
    (re.compile(r"\b(curl|wget)\b.*\|\s*(ba|z)?sh\b"), "下載後直接執行"),
    (re.compile(r"\b(sudo|mkfs|dd\s+if=|shutdown|reboot|poweroff)\b"), "系統層操作"),
    (
        re.compile(r"\bchmod\s+(-R\s+)?777\b|>\s*/dev/sd[a-z]"),
        "開放所有權限或寫入磁碟裝置",
    ),
    (re.compile(r"\b(pkill|killall)\s+-9\b"), "強制殺掉同名的所有程序"),
]
# 可能含有金鑰或憑證的路徑；對 read_file 的 path 和 bash 的指令字串都會檢查
SENSITIVE_PATH_PATTERNS = [
    re.compile(r"(?<![\w-])\.env(?!\.example\b)(?![\w-])"),
    re.compile(
        r"(?<![\w-])\.(ssh|aws|gnupg|kube|netrc|pypirc|npmrc|git-credentials)(?![\w-])"
    ),
    re.compile(r"(?<![\w-])(id_rsa|id_ed25519|id_ecdsa|credentials(\.json)?)(?![\w-])"),
    re.compile(r"\.(pem|key|p12|pfx)\b"),
]


# ========== 工具定義（給模型看的 JSON Schema）==========

TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read a file from the local filesystem. Returns the content with line numbers (cat -n style). "
            f"Reads up to {MAX_READ_LINES} lines by default; use offset/limit for large files. "
            "You must read a file before you can edit it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (absolute or relative to cwd)",
                },
                "offset": {
                    "type": "integer",
                    "description": "1-based line number to start from (default 1)",
                },
                "limit": {
                    "type": "integer",
                    "description": f"Number of lines to read (default {MAX_READ_LINES})",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file, creating it (and parent directories) if needed, overwriting if it exists. "
            "Prefer edit_file for changing existing files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Full content to write"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact string in a file. old_string must match exactly (including indentation) and "
            "must be unique in the file unless replace_all is true. You must read the file first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"},
                "old_string": {
                    "type": "string",
                    "description": "Exact text to replace",
                },
                "new_string": {"type": "string", "description": "Replacement text"},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every occurrence (default false)",
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
    },
    {
        "name": "bash",
        "description": (
            "Run a shell command in the working directory and return its stdout/stderr. "
            f"Times out after {BASH_TIMEOUT_SECONDS}s by default. Use for git, tests, builds, package managers. "
            "Do not use it to read or search files; use read_file / grep / glob instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The command to run"},
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (max 600)",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "glob",
        "description": (
            "Find files by glob pattern (e.g. '**/*.py', 'src/**/*.ts'). Respects .gitignore. "
            "Returns paths sorted by modification time, newest first."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern"},
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default cwd)",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": (
            "Search file contents with a regular expression (ripgrep syntax). "
            "output_mode 'content' returns matching lines with file:line prefixes; 'files' returns only file paths."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression to search for",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search (default cwd)",
                },
                "glob": {
                    "type": "string",
                    "description": "Only search files matching this glob, e.g. '*.py'",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files"],
                    "description": "Default 'content'",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "skill",
        "description": (
            "Load the full instructions of a skill listed in the system prompt. "
            "Call it before starting a task that matches a skill's description, then follow the returned instructions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name exactly as listed in the system prompt",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "agent",
        "description": (
            "Delegate a self-contained task to a subagent: a fresh instance of this agent with its own empty context "
            "that runs its own tool loop and returns only its final report. Use it when the intermediate work "
            "(reading many files, long search or command output) would clutter your context, or for an independent subtask. "
            "The subagent sees nothing of this conversation and cannot ask the user anything, so the prompt must be "
            "self-contained: the goal, relevant paths, what you already know, and exactly what to report back. "
            "Set read_only=true for exploration or research so it can only read, glob and grep; "
            "the web tools (web_search / web_fetch), when available, still work in read-only mode."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The complete task for the subagent",
                },
                "description": {
                    "type": "string",
                    "description": "3-5 word summary of the task, shown to the user",
                },
                "read_only": {
                    "type": "boolean",
                    "description": "Only allow read_file / glob / grep, plus web_search / web_fetch when available (default false)",
                },
            },
            "required": ["prompt", "description"],
        },
    },
]


# ========== Skills ==========
# Claude Code 的 skill 就是一個資料夾裡的 SKILL.md：frontmatter 有 name 與 description，正文是給模型的指示。
# 重點是「漸進式載入」：system prompt 只放名稱與描述（幾十個 skill 也只佔幾百 token），
# 模型判斷用得上才呼叫 skill 工具拿全文；使用者也可以打 /名稱 直接觸發。


@dataclass
class Skill:
    name: str
    description: str
    path: Path  # SKILL.md 的路徑；正文每次要用時才讀，對話中改了檔案立刻生效


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """最簡版 YAML frontmatter：只認一層 key: value（引號去掉、縮排的續行併進上一個值），不值得為此多裝 pyyaml"""
    match = re.match(r"---[ \t]*\n(.*?)\n---[ \t]*\n?(.*)", text, re.S)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    key = None
    for line in match.group(1).splitlines():
        if line[:1].isspace() and key:
            meta[key] = (meta[key] + " " + line.strip()).strip()
        elif ":" in line and not line.startswith("#"):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value in (">", "|", ">-", "|-"):  # 多行值的標記，內容在接下來的縮排行
                value = ""
            meta[key] = value.strip("\"'")
    return meta, match.group(2)


def load_skills(cwd: Path) -> dict[str, Skill]:
    """內建先載、家目錄其次、專案最後，同名時後載的蓋掉先載的"""
    skills: dict[str, Skill] = {}
    for root in (BUNDLED_SKILLS_DIR, Path.home() / SKILLS_SUBDIR, cwd / SKILLS_SUBDIR):
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*/SKILL.md")):
            try:
                meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
            except OSError:
                continue
            name = meta.get("name") or path.parent.name
            skills[name] = Skill(
                name, meta.get("description") or "(no description)", path
            )
    return skills


def render_skill(skill: Skill, args: str = "") -> str:
    """給模型看的 skill 全文。$ARGUMENTS 換成使用者打在 /名稱 後面的字（跟 Claude Code 一樣），沒有佔位就附在後面"""
    _, body = parse_frontmatter(skill.path.read_text(encoding="utf-8"))
    body = body.strip()
    if "$ARGUMENTS" in body:
        body = body.replace("$ARGUMENTS", args)
    elif args:
        body += f"\n\nArguments: {args}"
    return (
        f'<skill name="{skill.name}" dir="{skill.path.parent}">\n{body}\n</skill>\n'
        f"Relative paths mentioned in the skill (scripts/, references/, ...) are under {skill.path.parent}."
    )


# ========== 工具實作 ==========


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    """太長的輸出保留頭尾，中間用一行標記省略；不然幾輪工具呼叫就會把 context 撐爆"""
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return (
        f"{text[:half]}\n\n... [{omitted} characters truncated] ...\n\n{text[-half:]}"
    )


class Tools:
    """工具的實作（agent 例外，它要再開一個 MiniClaudeCode，所以放在那個 class 裡）。
    read_files 記錄「讀過哪些檔案、當時的 mtime」，edit_file 會檢查它。"""

    def __init__(self, cwd: Path, skills: dict[str, Skill]):
        self.cwd = cwd
        self.skills = skills
        self.read_files: dict[Path, float] = {}
        self.rg = shutil.which("rg")

    def resolve(self, path: str) -> Path:
        return (self.cwd / Path(path).expanduser()).resolve()

    def display(self, path: Path) -> str:
        """回給模型與印給人看的路徑：在工作目錄底下就用相對路徑，省 token 也好讀"""
        return (
            str(path.relative_to(self.cwd))
            if path.is_relative_to(self.cwd)
            else str(path)
        )

    def read_file(self, path: str, offset: int = 1, limit: int = MAX_READ_LINES) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {self.display(target)}")
        data = target.read_bytes()
        if b"\x00" in data[:8192]:  # 二進位檔解出來是亂碼，白白浪費 token
            raise ValueError(
                f"{self.display(target)} looks like a binary file; read_file only reads text files."
            )
        lines = data.decode("utf-8", errors="replace").splitlines()
        start = max(offset, 1) - 1
        limit = min(max(limit, 1), MAX_READ_LINES)
        selected = lines[start : start + limit]
        self.read_files[target] = target.stat().st_mtime
        if not selected:
            return f"(file has {len(lines)} lines; nothing at offset {offset})"
        numbered = []
        for i, line in enumerate(selected, start=start + 1):
            if len(line) > MAX_LINE_CHARS:
                line = line[:MAX_LINE_CHARS] + " ... [line truncated]"
            numbered.append(f"{i:6}\t{line}")
        footer = ""
        if start + len(selected) < len(lines):
            footer = f"\n\n(showing lines {start + 1}-{start + len(selected)} of {len(lines)}; use offset to read more)"
        return "\n".join(numbered) + footer

    def write_file(self, path: str, content: str) -> str:
        target = self.resolve(path)
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.read_files[target] = target.stat().st_mtime
        verb = "Overwrote" if existed else "Created"
        return f"{verb} {self.display(target)} ({len(content.splitlines())} lines)"

    def edit_file(
        self, path: str, old_string: str, new_string: str, replace_all: bool = False
    ) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {self.display(target)}")
        if target not in self.read_files:
            raise ValueError("You must read_file this file before editing it.")
        if target.stat().st_mtime != self.read_files[target]:
            raise ValueError(
                "File changed on disk since you read it. Read it again before editing."
            )
        if not old_string:  # str.count("") 等於長度 + 1，搭配 replace_all 會把 new_string 塞進每個字元之間
            raise ValueError(
                "old_string must not be empty. Use write_file to create or rewrite a file."
            )
        if old_string == new_string:
            raise ValueError("old_string and new_string are identical.")
        text = target.read_text(encoding="utf-8")
        count = text.count(old_string)
        if count == 0:
            raise ValueError(
                "old_string not found in file. Make sure it matches exactly, including whitespace."
            )
        if count > 1 and not replace_all:
            raise ValueError(
                f"old_string appears {count} times. Add more context to make it unique, or set replace_all."
            )
        new_text = (
            text.replace(old_string, new_string)
            if replace_all
            else text.replace(old_string, new_string, 1)
        )
        target.write_text(new_text, encoding="utf-8")
        self.read_files[target] = target.stat().st_mtime
        return f"Edited {self.display(target)}: replaced {count if replace_all else 1} occurrence(s)."

    def bash(self, command: str, timeout: int = BASH_TIMEOUT_SECONDS) -> str:
        # 不把金鑰交給模型執行的指令：load_dotenv() 把 .env 裡所有金鑰都載進了環境變數，不能只拿掉 ANTHROPIC_API_KEY
        env = {k: v for k, v in os.environ.items() if not is_secret_env(k)}
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.cwd,
                env=env,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=min(max(timeout, 1), 600),
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out after {timeout}s: {command}")
        output = proc.stdout
        if proc.stderr:
            output += ("\n" if output else "") + proc.stderr
        if proc.returncode != 0:
            output += f"\n(exit code {proc.returncode})"
        return output.strip() or "(no output)"

    def glob(self, pattern: str, path: str = ".") -> str:
        root = self.resolve(path)
        if self.rg:
            # rg --files 會自動尊重 .gitignore，比 pathlib.rglob 快也乾淨
            proc = subprocess.run(
                [self.rg, "--files", "--glob", pattern, str(root)],
                capture_output=True,
                text=True,
                errors="replace",
            )
            paths = [Path(p) for p in proc.stdout.splitlines()]
        else:
            paths = [
                p for p in root.glob(pattern) if p.is_file() and ".git" not in p.parts
            ]
        paths.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
        if not paths:
            return "No files found."
        shown = [self.display(p) for p in paths[:MAX_GLOB_RESULTS]]
        footer = (
            f"\n(showing first {MAX_GLOB_RESULTS} of {len(paths)} results)"
            if len(paths) > MAX_GLOB_RESULTS
            else ""
        )
        return "\n".join(shown) + footer

    def grep(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        output_mode: str = "content",
    ) -> str:
        root = self.resolve(path)
        if self.rg:
            cmd = [self.rg, "--no-heading", "--color", "never"]
            cmd += (
                ["--files-with-matches"]
                if output_mode == "files"
                else ["--line-number"]
            )
            if glob:
                cmd += ["--glob", glob]
            cmd += ["--regexp", pattern, str(root)]
        else:
            cmd = ["grep", "-rIE", "--exclude-dir=.git"]
            cmd += ["-l"] if output_mode == "files" else ["-n"]
            if glob:
                cmd += [f"--include={glob}"]
            cmd += ["--", pattern, str(root)]
        proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
        if proc.returncode == 1:
            return "No matches found."
        if proc.returncode not in (0, 1):
            raise RuntimeError(
                proc.stderr.strip() or f"search failed (exit {proc.returncode})"
            )
        prefix = str(root) + os.sep if root.is_dir() else ""
        return proc.stdout.replace(prefix, "").strip() or "No matches found."

    def skill(self, name: str) -> str:
        skill = self.skills.get(name)
        if skill is None:
            raise ValueError(
                f"Unknown skill: {name}. Available: {', '.join(self.skills) or '(none)'}"
            )
        return render_skill(skill)

    def run(self, name: str, args: dict[str, Any]) -> str:
        fn = getattr(self, name, None)
        if fn is None or name not in {t["name"] for t in TOOLS}:
            raise ValueError(f"Unknown tool: {name}")
        return truncate(fn(**args))


# ========== System prompt 與環境資訊 ==========

SYSTEM_PROMPT = """\
You are a mini Claude Code: an interactive CLI coding agent that helps the user with software engineering tasks in the current working directory.

Guidelines:
- Understand before changing: read the relevant files and search the codebase before editing. Never guess at file contents.
- Prefer edit_file for modifications; use write_file only for new files or full rewrites.
- After changing code, verify it (run the tests or the program with bash) when a reasonable way to do so exists.
- Keep answers concise. Report what you did and what you verified; don't restate the code you wrote.
- Do not perform destructive actions (deleting files, git push, force operations, rm -rf) unless the user explicitly asks.
- Respond in the same language the user writes in.
"""

# --web 時加在 system prompt 最後：說清楚什麼時候該上網、什麼時候不該，否則模型會連 repo 裡找得到的東西都去搜
WEB_PROMPT = """\
You can search the web with web_search and read a page whose URL already appears in the conversation with web_fetch. \
Use them for what the repository cannot tell you and what may be newer than your training data: library versions, \
API changes, documentation, unfamiliar error messages. Do not search for things you can find in the codebase. \
When an answer relies on a search result, cite the source URL."""

# 子代理多拿到的這一段：它看不到主代理的對話、也不能問使用者，最後那段文字就是唯一會回去的東西
SUBAGENT_PROMPT = """\
You are a subagent spawned by the main agent to carry out one delegated task. You cannot talk to the user: never ask \
questions; make reasonable assumptions and state them. When finished, reply with a concise final report containing \
everything the main agent needs (findings with file paths and line numbers, what you changed, what you verified). \
Only that final report is passed back; your tool calls and intermediate output are not."""


def git_context(cwd: Path) -> str:
    """git 分支與狀態會一直變，所以放在第一則 user 訊息，不放進 system prompt（才不會每輪都打破快取）"""
    try:
        # 用 --show-current 而不是 rev-parse HEAD：剛 git init 還沒 commit 的 repo 也抓得到分支名
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if branch.returncode != 0:
            return ""
        status = subprocess.run(
            ["git", "status", "--short"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        status_lines = status.stdout.splitlines()
        shown = "\n".join(status_lines[:20]) or "(clean)"
        if len(status_lines) > 20:
            shown += f"\n... and {len(status_lines) - 20} more"
        return f"Git branch: {branch.stdout.strip() or '(detached HEAD)'}\nGit status:\n{shown}"
    except (OSError, subprocess.TimeoutExpired):
        return ""


def build_system_prompt(cwd: Path, skills: dict[str, Skill]) -> str:
    parts = [SYSTEM_PROMPT]
    parts.append(
        "Environment:\n"
        f"- Working directory: {cwd}\n"
        f"- Platform: {platform.system()} {platform.release()}\n"
        # 今天的日期不放這裡：隔天 --resume 時 system prompt 就變了，見 run_turn 的 <context>
        f"- ripgrep available: {'yes' if shutil.which('rg') else 'no'}"
    )
    # 專案的 CLAUDE.md：跟 Claude Code 一樣，整份放進 system prompt 當作專案規範
    for claude_md in (cwd / "CLAUDE.md", cwd / ".claude" / "CLAUDE.md"):
        if claude_md.is_file():
            parts.append(
                f"Project instructions from {claude_md.name}:\n\n{claude_md.read_text(encoding='utf-8')}"
            )
            break
    if skills:  # 只列名稱與描述，全文等模型呼叫 skill 工具才給
        listing = "\n".join(f"- {s.name}: {s.description}" for s in skills.values())
        parts.append(
            "Skills (specialised instructions you can load on demand). When a task matches one of these descriptions, "
            f"call the `skill` tool with its name first and follow the loaded instructions:\n{listing}"
        )
    return "\n\n".join(parts)


# ========== 畫面輸出 ==========
# 版面參考 Claude Code：模型說的話前面是 ●，工具呼叫是 ⚙，工具結果縮排用 ⎿ 接在下面。
# 來自模型或工具的文字一律 markup=False：rich 會把 list[str]、[/tmp] 這種方括號當成自己的標記，
# 輕則把 [str] 吃掉，重則 [/tmp] 直接炸 MarkupError。

TOOL_RESULT_LINES = 6  # 工具結果最多印幾行（完整內容仍會回給模型）
TOOL_RESULT_LINE_CHARS = 160
TOOL_ARG_LINES = 12  # bash 指令、edit_file 的 diff 最多印幾行


def assistant_block(markdown: str, first: bool, indent: str = "") -> Table:
    """把一段 Markdown 排成「● 內容」，續接的段落只縮排、不再畫 ●；indent 是子代理用的額外縮排"""
    grid = Table.grid(padding=0)
    grid.add_column(width=len(indent) + 2)
    grid.add_column(ratio=1)
    grid.add_row(f"{indent}●" if first else "", Markdown(markdown))
    return grid


class StreamingText:
    """邊串流邊用 Markdown 渲染一個 text block。

    rich 的 Live 只能重畫還留在螢幕上的那幾行，內容超過一個螢幕高度後每次重畫都會多印一份。
    所以暫存的文字一旦夠長、又停在段落邊界（空行，且不在程式碼區塊裡），就把前面的段落定稿印出，
    Live 只負責重畫最後一小段。Markdown 的段落彼此獨立，分開渲染結果一樣（有序清單會保留起始編號）。
    """

    def __init__(self, indent: str = ""):
        self.pending = ""
        self.first = True
        self.indent = indent
        self.flush_lines = max(8, console.height // 2)
        self.live = self.new_live()

    def new_live(self) -> Live:
        live = Live(
            assistant_block(self.pending, self.first, self.indent),
            console=console,
            refresh_per_second=12,
            vertical_overflow="visible",
        )
        live.start()
        return live

    def feed(self, text: str) -> None:
        self.pending += text
        lines = self.pending.count("\n")
        if lines >= self.flush_lines:
            # 優先切在空行；一長串清單沒有空行時，累積到兩倍長度就退而求其次切在換行
            head, sep, tail = self.pending.rpartition("\n\n")
            if not sep and lines >= self.flush_lines * 2:
                head, sep, tail = self.pending.rpartition("\n")
            if sep and head.count("```") % 2 == 0:  # 不切在程式碼區塊中間
                self.live.update(assistant_block(head, self.first, self.indent))
                self.live.stop()  # stop 會把最後一次 update 留在畫面上
                self.first = False
                self.pending = tail
                self.live = self.new_live()
                return
        self.live.update(assistant_block(self.pending, self.first, self.indent))

    def close(self) -> None:
        self.live.update(assistant_block(self.pending, self.first, self.indent))
        self.live.stop()


def describe_tool_call(name: str, args: dict[str, Any]) -> list[tuple[str, str]]:
    """工具呼叫的參數摘要，依工具客製（bash 印完整指令、edit_file 印 diff），回傳 (文字, 樣式) 的列表"""

    def clip(lines: list[str], style: str) -> list[tuple[str, str]]:
        shown = [(line, style) for line in lines[:TOOL_ARG_LINES]]
        if len(lines) > TOOL_ARG_LINES:
            shown.append((f"… 共 {len(lines)} 行", "dim"))
        return shown

    if name == "bash":
        return clip(args.get("command", "").splitlines(), "yellow")
    if name == "read_file":
        parts = [args.get("path", "")]
        if args.get("offset", 1) != 1:
            parts.append(f"第 {args['offset']} 行起")
        if args.get("limit"):
            parts.append(f"最多 {args['limit']} 行")
        return [("  ".join(parts), "yellow")]
    if name == "write_file":
        return [
            (
                f"{args.get('path', '')}  ({args.get('content', '').count(chr(10)) + 1} 行)",
                "yellow",
            )
        ]
    if name == "edit_file":
        lines = [
            (
                args.get("path", "")
                + ("  (replace_all)" if args.get("replace_all") else ""),
                "yellow",
            )
        ]
        lines += clip(
            [f"- {line}" for line in args.get("old_string", "").splitlines()], "red"
        )
        lines += clip(
            [f"+ {line}" for line in args.get("new_string", "").splitlines()], "green"
        )
        return lines
    if name in ("glob", "grep"):
        parts = [args.get("pattern", "")]
        if args.get("path", ".") != ".":
            parts.append(f"in {args['path']}")
        if args.get("glob"):
            parts.append(f"glob {args['glob']}")
        if args.get("output_mode", "content") != "content":
            parts.append(args["output_mode"])
        return [("  ".join(parts), "yellow")]
    if name == "agent":
        lines = [
            (
                args.get("description", "")
                + ("  (read-only)" if args.get("read_only") else ""),
                "yellow",
            )
        ]
        return lines + clip(args.get("prompt", "").splitlines(), "dim")
    return [
        (f"{k}={str(v)[:TOOL_RESULT_LINE_CHARS]}", "yellow") for k, v in args.items()
    ]


def print_tool_call(name: str, args: dict[str, Any], indent: str = "") -> None:
    console.print(f"{indent}⚙ {name}", style="bold cyan", markup=False)
    for text, style in describe_tool_call(name, args):
        console.print(
            f"{indent}  {text}",
            style=style,
            markup=False,
            highlight=False,
            soft_wrap=True,
        )


def print_tool_result(output: str, is_error: bool, indent: str = "") -> None:
    style = "red" if is_error else "dim"
    lines = output.splitlines() or ["(沒有輸出)"]
    for i, line in enumerate(lines[:TOOL_RESULT_LINES]):
        prefix = "⎿ " if i == 0 else "  "
        console.print(
            f"{indent}  {prefix}{line[:TOOL_RESULT_LINE_CHARS]}",
            style=style,
            markup=False,
            highlight=False,
        )
    if len(lines) > TOOL_RESULT_LINES:
        console.print(f"{indent}    … 共 {len(lines)} 行", style=style)


def print_server_tool_block(block: Any, indent: str = "") -> None:
    """--web 時回應裡的伺服器端工具 block：呼叫印成 🌐、結果印成 ⎿，跟本機工具的畫面對齊。
    這些 block 是串流到 content_block_stop 才完整，所以在那時印"""
    data = block.model_dump(exclude_none=True)
    if block.type == "server_tool_use":
        args = data.get("input", {})
        console.print(f"{indent}🌐 {block.name}", style="bold cyan", markup=False)
        if block.name == "code_execution":  # 動態過濾：模型寫來篩選搜尋結果的小程式
            lines = args.get("code", "").splitlines()
            shown = lines[:TOOL_RESULT_LINES] + (
                [f"… 共 {len(lines)} 行"] if len(lines) > TOOL_RESULT_LINES else []
            )
            style = "dim"
        elif all(
            isinstance(v, str) for v in args.values()
        ):  # web_search 的 query、web_fetch 的 url
            shown = [v[:TOOL_RESULT_LINE_CHARS] for v in args.values()]
            style = "yellow"
        else:  # 參數格式不對（動態過濾的程式呼叫錯了會這樣），印整個 JSON 才看得出原因
            shown = [json.dumps(args, ensure_ascii=False)[:TOOL_RESULT_LINE_CHARS]]
            style = "yellow"
        for line in shown:
            console.print(
                f"{indent}  {line}",
                style=style,
                markup=False,
                highlight=False,
                soft_wrap=True,
            )
        return
    content = data.get("content")
    if block.type == "web_search_tool_result":
        if isinstance(
            content, list
        ):  # 成功是清單；失敗是 {"type": "web_search_tool_result_error", "error_code": ...}
            output = f"找到 {len(content)} 筆\n" + "\n".join(
                f"{r.get('title', '')}  {r.get('url', '')}" for r in content
            )
            print_tool_result(output.strip(), False, indent)
        else:
            print_tool_result(
                f"搜尋失敗：{(content or {}).get('error_code', '?')}", True, indent
            )
    elif block.type == "web_fetch_tool_result":
        if isinstance(content, dict) and content.get("type") == "web_fetch_result":
            title = (content.get("content") or {}).get("title") or ""
            print_tool_result(
                f"已抓取 {content.get('url', '')}  {title}".rstrip(), False, indent
            )
        else:
            print_tool_result(
                f"抓取失敗：{(content or {}).get('error_code', '?')}", True, indent
            )
    elif block.type == "code_execution_tool_result":
        if isinstance(content, dict) and "return_code" in content:
            failed = content["return_code"] != 0
            output = content.get("stderr" if failed else "stdout") or ""
            print_tool_result(
                output.strip() or f"(exit {content['return_code']})", failed, indent
            )
        else:
            print_tool_result(
                f"執行失敗：{(content or {}).get('error_code', '?')}", True, indent
            )


def print_sources(response: anthropic.types.beta.BetaMessage, indent: str = "") -> None:
    """回答裡引用到的網頁（text block 的 citations）去重後列在回答下面"""
    seen: dict[str, str] = {}
    for block in response.content:
        if block.type != "text":
            continue
        for c in block.citations or []:
            url = getattr(c, "url", None)
            if url and url not in seen:
                seen[url] = getattr(c, "title", "") or ""
    if not seen:
        return
    console.print()  # 串流文字停在行尾
    console.print(f"{indent}  來源：", style="dim")
    for url, title in seen.items():
        console.print(
            f"{indent}    {title}  {url}".rstrip(),
            style="dim",
            markup=False,
            highlight=False,
            soft_wrap=True,
        )


# ========== Agent 本體 ==========


class MiniClaudeCode:
    def __init__(
        self,
        cwd: Path,
        model: str = DEFAULT_MODEL,
        effort: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        compact_threshold: int | None = None,
        auto_approve: bool = False,
        skip_permissions: bool = False,
        show_thinking: bool = False,
        subagent_of: MiniClaudeCode | None = None,
        read_only: bool = False,
        web: bool = False,
    ):
        self.parent = subagent_of
        self.indent = (subagent_of.indent + "    ") if subagent_of else ""
        self.cwd = cwd
        self.model = model
        self.effort = effort  # 目前生效的 effort；None 表示模型預設
        self.base_effort = effort  # 送在頂層 output_config 的值；/effort 中途改的放在 messages 裡，見 set_effort
        self.max_tokens = max_tokens
        # None 表示依模型決定，見 compact_threshold
        self.compact_threshold_setting = compact_threshold
        self.auto_approve = auto_approve or skip_permissions
        self.skip_permissions = skip_permissions  # 連危險指令、敏感檔案都不問
        self.show_thinking = show_thinking
        self.messages: list[dict[str, Any]] = []
        self.interrupted = False  # 這一輪有沒有被 Ctrl+C 打斷；子代理用它告訴主代理「不是做完，是被打斷」
        # 這一輪為什麼沒有結果（API 錯誤、被拒絕）；子代理用它回報給主代理
        self.error: str | None = None
        self.resumed = False  # 剛 --resume：下一則 user 訊息要重新附上日期與 git 狀態
        self.last_context_tokens = 0
        self.session_path = SESSION_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
        if subagent_of:
            # 子代理共用：API client、[a] 放行過的工具、token 統計（/cost 才看得到全部花費）；
            # 不共用：對話歷史、read_files（子代理讀過不代表主代理讀過，mtime 檢查各管各的）
            self.client = subagent_of.client
            self.tools = Tools(cwd, subagent_of.tools.skills)
            self.system_prompt = subagent_of.system_prompt + "\n\n" + SUBAGENT_PROMPT
            self.always_allow = subagent_of.always_allow
            self.usage = subagent_of.usage
        else:
            self.client = anthropic.Anthropic()
            self.tools = Tools(cwd, load_skills(cwd))
            self.system_prompt = build_system_prompt(cwd, self.tools.skills)
            self.always_allow: set[str] = set()
            self.usage: dict[
                str, dict[str, int]
            ] = {}  # 按模型分開記，/model 切換後 /cost 才看得出各自花了多少
        # 這個實例給模型的工具清單：沒有 skill 就不給 skill 工具；子代理不能再開子代理；唯讀子代理只有讀的工具
        allowed = {t["name"] for t in TOOLS}
        if not self.tools.skills:
            allowed.discard("skill")
        if subagent_of:
            allowed.discard("agent")
        if read_only:
            allowed &= READ_ONLY_TOOLS
            self.system_prompt += (
                "\n\nThis subagent is read-only: it can only read_file, glob and grep"
                + (" (and search the web). " if web else ". ")
                + "Do not try to change anything."
            )
        self.tool_specs = [t for t in TOOLS if t["name"] in allowed]
        # 伺服器端工具不在 allowed 的名單裡：它們不經過 Tools.run，唯讀子代理也可以用（上網查資料本來就是唯讀）
        self.web = False
        if web:
            self.enable_web()

    def enable_web(self) -> None:
        """把 web_search / web_fetch 加進工具清單。tools 是快取前綴的一部分，所以整個 session 固定開著，
        不提供中途切換（--resume 時依存檔還原）。子代理拿到的 system prompt 是主代理的，已經含 WEB_PROMPT"""
        if self.web:
            return
        self.web = True
        self.tool_specs = self.tool_specs + SERVER_TOOLS
        if not self.parent:
            self.system_prompt += "\n\n" + WEB_PROMPT

    @property
    def compact_threshold(self) -> int:
        """命令列有給就用命令列的；沒給就依目前的模型（/model 切換後跟著變）"""
        return self.compact_threshold_setting or MODEL_COMPACT_THRESHOLD.get(
            self.model, DEFAULT_COMPACT_THRESHOLD
        )

    # ---------- 核心迴圈 ----------

    def run_turn(self, user_text: str) -> str | None:
        """一輪對話：使用者說一句話，模型可能連續呼叫多次工具，直到它決定停下來。
        回傳模型最後一段回覆的文字（子代理靠這個把報告交回主代理）；被中斷、被拒絕或 API 出錯回傳 None"""
        self.interrupted = False
        self.error = None
        final: str | None = None
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        # 日期與 git 狀態放在第一則 user 訊息；--resume 之後的第一則再附一次（兩者都可能變了），
        # 用 append 的方式補上新資訊，不去改 system prompt（見 load_session）
        if not self.has_user_messages() or self.resumed:
            self.resumed = False
            ctx = "\n".join(
                filter(
                    None,
                    [f"Today's date: {datetime.now():%Y-%m-%d}", git_context(self.cwd)],
                )
            )
            content.insert(0, {"type": "text", "text": f"<context>\n{ctx}\n</context>"})
        user_message = {"role": "user", "content": content}
        self.messages.append(user_message)

        max_tokens = self.max_tokens
        stop_compacting = (
            False  # 這一輪不再自動摘要（摘要失敗過，或摘要了也降不到門檻以下）
        )
        just_compacted = False  # 上一個請求是摘要後的第一個請求
        while True:
            response = self.call_model(max_tokens)
            if (
                response is None
            ):  # 被中斷或 API 出錯，對話歷史停在 user 訊息，下一輪可以直接接著問
                break

            stop = response.stop_reason
            self.record_usage(response.usage)

            # stop_reason 可能的值：end_turn / tool_use / max_tokens / stop_sequence / refusal /
            # pause_turn / model_context_window_exceeded
            if stop == "max_tokens":
                # 回應被截斷，裡面可能有半截的 tool_use，不能接著用；丟掉、把 max_tokens 加倍重送
                if max_tokens >= MAX_TOKENS_CEILING:
                    console.print("[red]回應超過輸出上限，請把任務拆小一點。[/red]")
                    final = self.text_of(response)
                    self.messages.append(
                        {"role": "assistant", "content": self.dump_content(response)}
                    )
                    # 截斷的回應裡若有 tool_use，一定要補 tool_result，否則下一輪 API 會因為孤兒 tool_use 拒絕
                    orphans = [b for b in response.content if b.type == "tool_use"]
                    if orphans:
                        self.messages.append(
                            {
                                "role": "user",
                                "content": [
                                    self.tool_result(
                                        b.id,
                                        "Response was truncated at the output limit; this tool call was not executed.",
                                        is_error=True,
                                    )
                                    for b in orphans
                                ],
                            }
                        )
                    break
                max_tokens = min(max_tokens * 2, MAX_TOKENS_CEILING)
                console.print(
                    f"[yellow]回應被 max_tokens 截斷，改用 {max_tokens:,} 重送…[/yellow]"
                )
                continue

            if stop == "model_context_window_exceeded":
                console.print(
                    "[yellow]context window 已滿，先摘要舊對話再繼續…[/yellow]"
                )
                if not self.compact():
                    # 摘要失敗就停在這裡，歷史仍以 user 訊息結尾，下一輪可以再試
                    self.error = "The context window filled up and compaction failed."
                    break
                just_compacted = True
                continue  # compact() 留下的歷史以 user 訊息（摘要）結尾，可以直接再呼叫

            if stop == "refusal":
                # 被安全分類器拒絕。回應可能已經串流出半截，文件要求整段丟掉：不存進歷史，也不當成結果回傳
                # （子代理若把半截文字交回去，主代理會以為那是完整的報告）
                details = response.stop_details
                category = details.category if details else None
                self.error = "The model refused this request" + (
                    f" (category: {category})." if category else "."
                )
                console.print(
                    f"[red]模型拒絕回應這個請求{f'（{category}）' if category else ''}，已丟掉這次的輸出。[/red]"
                )
                if self.messages and self.messages[-1] is user_message:
                    # 這一輪還沒跑過工具：剛送出的訊息也撤回，換個說法重問時才不會又帶著被拒的內容
                    self.messages.pop()
                elif not self.parent:
                    console.print(
                        "[dim]被拒的內容可能在這一輪的工具結果裡，還留在對話中；換個方向再問，或 /clear 重新開始。[/dim]"
                    )
                break

            # assistant 的內容整塊存起來（含 thinking block 與 signature），下一輪原樣送回
            self.messages.append(
                {"role": "assistant", "content": self.dump_content(response)}
            )

            if stop == "tool_use":
                tool_uses = [b for b in response.content if b.type == "tool_use"]
                results, interrupted = self.run_tools(tool_uses)
                self.messages.append({"role": "user", "content": results})
                if interrupted:
                    break
                # 一輪裡連續呼叫幾十次工具，context 會一路長下去，不能等整輪結束才摘要（超過 1M 就是 400，摘要也救不回來）。
                # 這時 tool_result 已經補齊，摘要不會切斷 tool_use / tool_result 的配對
                if (
                    not stop_compacting
                    and self.last_context_tokens > self.compact_threshold
                ):
                    if just_compacted:
                        # 摘要完的第一個請求就超過門檻：門檻比 system prompt、tools 加上摘要還小，
                        # 再摘要也降不下來，只會每圈白送一次（模型還會重讀檔案）
                        console.print(
                            "[yellow]摘要後 context 仍超過門檻，這一輪不再自動摘要。[/yellow]"
                        )
                        stop_compacting = True
                    else:
                        console.print(
                            f"[yellow]context 已達 {self.last_context_tokens:,} tokens，先摘要舊對話再繼續…[/yellow]"
                        )
                        just_compacted = self.compact()
                        # 失敗一次就不在這一輪重試，免得每圈都白送一次
                        stop_compacting = not just_compacted
                        continue
                just_compacted = False
                continue

            if stop == "pause_turn":  # 伺服器端工具暫停，直接再呼叫一次即可
                continue

            final = self.text_of(response)
            break  # end_turn / stop_sequence

        # 子代理跑完整個實例就丟掉了，摘要它的歷史只是白花一次整段 context 的請求；
        # 這一輪已經判定摘要沒用（或剛摘要完）也不再摘要
        if (
            not self.parent
            and not stop_compacting
            and not just_compacted
            and self.last_context_tokens > self.compact_threshold
        ):
            console.print(
                f"[yellow]context 已達 {self.last_context_tokens:,} tokens，自動摘要舊對話…[/yellow]"
            )
            self.compact()
        self.save_session()
        return final

    def run_skill(self, name: str, args: str) -> None:
        """/名稱 指令：不經過模型判斷，直接把 skill 全文當成這一輪的 user 訊息"""
        self.run_turn(
            f"{render_skill(self.tools.skills[name], args)}\n\n"
            f"The user invoked /{name}. Follow the skill's instructions above."
        )

    def request_kwargs(
        self, max_tokens: int, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """每一次請求共用的參數。快取前綴的順序是 tools → system → messages，
        所以 compaction 的摘要請求也要用同一組 tools 與 system，才能讀到前面幾輪建立的快取"""
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            # 顯式斷點：system prompt 整個 session 都不變，標起來讓每一輪都讀快取
            system=[
                {
                    "type": "text",
                    "text": self.system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=self.tool_specs,
            messages=messages,
            # 頂層 cache_control = 自動快取：斷點自動放在最後一個可快取的 block，並隨對話增長往前移
            cache_control={"type": "ephemeral"},
            betas=[
                PER_MESSAGE_EFFORT_BETA
            ],  # /effort 用的 effort-only system 訊息需要這個 beta
        )
        if self.model.startswith(SERVER_FALLBACK_MODELS):
            kwargs["fallbacks"] = "default"
            kwargs["betas"].append(SERVER_FALLBACK_BETA)
        if self.base_effort:
            kwargs["output_config"] = {"effort": self.base_effort}
        if self.show_thinking:
            # 5.x 的 thinking 預設 display 是 omitted（文字為空，只剩簽章），要明確要求摘要才會串流出思考內容
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        return kwargs

    def call_model(self, max_tokens: int) -> anthropic.types.beta.BetaMessage | None:
        """呼叫一次 Messages API（串流），邊收邊印；回傳完整的 Message，被中斷或出錯回傳 None

        畫面上：thinking 只轉一個會消失的 spinner（--show-thinking 才印摘要），text 用 Markdown 邊收邊渲染，
        tool_use 在串流時只轉 spinner，等 run_tools 真的執行時才連同結果一起印，多個工具時頭尾才不會分家。
        """
        kwargs = self.request_kwargs(max_tokens, self.messages)

        text: StreamingText | None = None
        status = None  # 「思考中」「產生參數中」的 spinner；一有東西要印就收掉，畫面上不留痕跡
        thinking_started = False
        thinking_held = (
            ""  # 思考摘要結尾的換行先扣住，下一段再印；block 結束時丟掉，才不會多出空行
        )

        def finish() -> None:
            nonlocal text, status
            if status is not None:
                status.stop()
                status = None
            if text is not None:
                text.close()
                text = None

        try:
            with self.client.beta.messages.stream(**kwargs) as stream:
                for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "text":
                            # 有引用（--web）時回答會被切成好幾個 text block，接著同一個 StreamingText 印才不會變成好幾個 ●
                            if text is None:
                                finish()
                                console.print()
                                text = StreamingText(self.indent)
                            continue
                        finish()
                        if block.type == "tool_use":
                            status = console.status(
                                f"[cyan]⚙ {block.name}[/cyan] [dim]產生參數中…[/dim]"
                            )
                            status.start()
                        elif block.type == "server_tool_use":
                            status = console.status(
                                f"[cyan]🌐 {block.name}[/cyan] [dim]產生參數中…[/dim]"
                            )
                            status.start()
                        elif block.type == "thinking" and not self.show_thinking:
                            status = console.status("[dim]💭 思考中…[/dim]")
                            status.start()
                    elif event.type == "text" and text is not None:
                        text.feed(event.text)
                    elif (
                        event.type == "thinking"
                        and self.show_thinking
                        and event.thinking
                    ):
                        if not thinking_started:
                            console.print(f"\n{self.indent}💭 ", style="dim", end="")
                            thinking_started = True
                        chunk = thinking_held + event.thinking
                        shown = chunk.rstrip("\n")
                        thinking_held = chunk[len(shown) :]
                        console.print(
                            shown,
                            style="dim",
                            end="",
                            markup=False,
                            highlight=False,
                            soft_wrap=True,
                        )
                    elif event.type == "content_block_stop":
                        block = event.content_block
                        if (
                            block.type != "text"
                        ):  # text 等下一個非 text block 或串流結束再收
                            finish()
                        if thinking_started:
                            console.print()
                            thinking_started = False
                            thinking_held = ""
                        if block.type == "server_tool_use":
                            # 參數齊了才印得出 query；接著 API 那邊真的在搜，轉個 spinner 等結果 block 進來
                            console.print()
                            print_server_tool_block(block, self.indent)
                            status = console.status(
                                f"[dim]🌐 {block.name} 執行中…[/dim]"
                            )
                            status.start()
                        elif block.type.endswith("_tool_result"):
                            print_server_tool_block(block, self.indent)
                        elif block.type == "fallback":
                            # 原本的模型被拒、API 改用備援模型接著產生；前面已經印出的半截文字也會留在 content 裡
                            console.print(
                                f"\n{self.indent}⚠ {block.from_.model} 拒絕了這個請求，改由 {block.to.model} 接手",
                                style="yellow",
                                markup=False,
                            )
                response = stream.get_final_message()
                finish()
                if self.web:
                    print_sources(response, self.indent)
                return response
        except KeyboardInterrupt:
            finish()
            self.interrupted = True
            console.print("\n[yellow]已中斷這次回應。[/yellow]")
            return None
        except anthropic.APIError as e:
            finish()
            self.error = f"API error: {e}"
            console.print(f"\nAPI 錯誤：{e}", style="red", markup=False)
            return None
        finally:
            finish()

    def has_user_messages(self) -> bool:
        return any(m["role"] == "user" for m in self.messages)

    def describe_effort(self) -> str:
        if self.effort:
            return self.effort
        default = MODEL_DEFAULT_EFFORT.get(self.model)
        return f"模型預設（{default}）" if default else "模型預設"

    def set_effort(self, level: str) -> None:
        """對話中途切換 effort。有歷史且模型支援時，用 effort-only 的 system 訊息接在後面，快取不失效"""
        if level == self.effort:
            console.print(f"[dim]effort 已經是 {level}。[/dim]")
            return
        before = self.describe_effort()
        if self.has_user_messages() and self.model.startswith(
            PER_MESSAGE_EFFORT_MODELS
        ):
            self.messages.append(
                {"role": "system", "content": [], "output_config": {"effort": level}}
            )
            how = "以 system 訊息接在歷史後面，快取不受影響"
        else:
            # 沒有歷史就沒有快取可保；模型不支援 beta 就只能改頂層，下一次請求會重建快取
            self.base_effort = level
            how = "改頂層 output_config" + (
                "，下一次請求會重建快取" if self.has_user_messages() else ""
            )
        self.effort = level
        console.print(f"[green]effort {before} → {level}（{how}）。[/green]")

    def drop_effort_messages(self) -> bool:
        """換到不支援 per-message effort 的模型時用：把 /effort 塞進歷史的 system 訊息拿掉，目前 effort 改回頂層帶。
        從歷史中間刪訊息，之後每個 thinking block 記錄的前綴都對不上了（5.5 系列會檢查，新帳號直接 400），
        所以 thinking block 也全部拿掉（文件給的一次性復原做法，text 與 tool_use 留著），之後換回 5.5 模型才不會出錯。
        回傳有沒有動到歷史"""
        self.base_effort = self.effort
        if not any(m["role"] == "system" for m in self.messages):
            return False
        kept: list[dict[str, Any]] = []
        for m in self.messages:
            if m["role"] == "system":
                continue
            if m["role"] == "assistant":
                content = [
                    b
                    for b in m["content"]
                    if b["type"] not in ("thinking", "redacted_thinking")
                ]
                if not content:  # 只有 thinking 的回合（例如被截斷的回應），整則拿掉；前後兩則 user 訊息 API 會自動合併
                    continue
                m = {**m, "content": content}
            kept.append(m)
        self.messages = kept
        return True

    def set_model(self, model: str) -> bool:
        """對話中途換模型。先用 Models API 驗證 ID（不耗 token），再切換；回傳是否有切"""
        if model == self.model:
            console.print(f"[dim]模型已經是 {model}。[/dim]")
            return False
        try:
            self.client.models.retrieve(model)
        except anthropic.NotFoundError:
            console.print(f"[red]找不到模型 {model}。[/red]")
            return False
        except anthropic.APIError as e:
            console.print(f"[red]驗證模型失敗：{e}[/red]")
            return False
        before = self.model
        self.model = model
        notes = [f"effort {self.describe_effort()}"]
        # 新模型也支援 per-message effort 就什麼都不動：歷史只 append 不改，thinking block 才不會失效
        if (
            not model.startswith(PER_MESSAGE_EFFORT_MODELS)
            and self.drop_effort_messages()
        ):
            notes.append(
                "新模型不支援 /effort 的 system 訊息，已連同 thinking block 從歷史移除"
            )
        elif self.has_user_messages():
            notes.append("舊模型的 thinking block 新模型讀不懂會被靜默丟掉")
        if self.has_user_messages():
            notes.append("快取綁模型，下一次請求會整段重建")
        console.print(f"[green]模型 {before} → {model}（{'；'.join(notes)}）。[/green]")
        return True

    @staticmethod
    def dump_content(
        response: anthropic.types.beta.BetaMessage,
    ) -> list[dict[str, Any]]:
        # by_alias：fallback block 的 from 欄位在 SDK 裡叫 from_，存回去要用 API 的名字，不然下一輪會 400
        return [
            block.model_dump(by_alias=True, exclude_none=True)
            for block in response.content
        ]

    @staticmethod
    def text_of(response: anthropic.types.beta.BetaMessage) -> str:
        return "".join(b.text for b in response.content if b.type == "text").strip()

    # ---------- 工具執行與權限 ----------

    def run_tools(self, tool_uses: list[Any]) -> tuple[list[dict[str, Any]], bool]:
        """執行這一輪所有 tool_use；全部是唯讀工具就並行，否則依序。回傳 (tool_results, 是否被中斷)"""
        results: list[dict[str, Any]] = []
        console.print()
        try:
            # 要問人的工具不能並行（input() 會打架），唯讀但碰到敏感路徑的也算要問；
            # 子代理一路都在印東西、它的工具也可能問人，所以也不能跟別的工具同時跑
            if (
                len(tool_uses) > 1
                and not any(t.name == "agent" for t in tool_uses)
                and not any(self.needs_approval(t.name, t.input) for t in tool_uses)
            ):
                with ThreadPoolExecutor(max_workers=min(len(tool_uses), 8)) as pool:
                    outcomes = list(pool.map(self.run_one_tool, tool_uses))
                for t, (result, output, is_error) in zip(
                    tool_uses, outcomes
                ):  # 跑完再印，thread 裡印會交錯
                    print_tool_call(t.name, t.input, self.indent)
                    print_tool_result(output, is_error, self.indent)
                    results.append(result)
            else:
                for t in tool_uses:
                    print_tool_call(t.name, t.input, self.indent)
                    result, output, is_error = self.run_one_tool(t)
                    print_tool_result(output, is_error, self.indent)
                    results.append(result)
        except KeyboardInterrupt:
            self.interrupted = True
            console.print("\n[yellow]已中斷工具執行。[/yellow]")
        # 每個 tool_use 都必須有 tool_result，被中斷的補一個錯誤結果，對話歷史才合法
        done = {r["tool_use_id"] for r in results}
        for t in tool_uses:
            if t.id not in done:
                results.append(
                    self.tool_result(
                        t.id, "Interrupted by user before this tool ran.", is_error=True
                    )
                )
        return results, len(done) < len(tool_uses)

    def run_one_tool(self, tool_use: Any) -> tuple[dict[str, Any], str, bool]:
        """執行一個工具，回傳 (給模型的 tool_result, 給人看的輸出, 是否失敗)；不在這裡印，因為可能跑在 thread 裡"""
        if tool_use.name not in {t["name"] for t in self.tool_specs}:
            # 模型偶爾會呼叫清單外的工具（system prompt 裡提過 write_file、bash 這些名字），
            # 唯讀子代理「只能讀」、子代理「不能再開子代理」不能只靠模型自律
            result = self.tool_result(
                tool_use.id,
                f"Tool {tool_use.name} is not available to you. Use only the tools you were given.",
                is_error=True,
            )
            return result, f"✗ 這裡沒有 {tool_use.name} 工具", True
        if self.needs_approval(
            tool_use.name, tool_use.input
        ) and not self.ask_permission(tool_use.name, tool_use.input):
            result = self.tool_result(
                tool_use.id,
                "The user denied this action. Ask them how to proceed.",
                is_error=True,
            )
            return result, "✗ 已拒絕", True
        try:
            if tool_use.name == "agent":
                output = self.spawn_agent(**tool_use.input)
            else:
                output = self.tools.run(tool_use.name, tool_use.input)
            is_error = False
        except Exception as e:  # 工具失敗不是程式 bug，要把原因回給模型讓它自己修正
            output, is_error = f"{type(e).__name__}: {e}", True
        return self.tool_result(tool_use.id, output, is_error), output, is_error

    def spawn_agent(
        self, prompt: str, description: str = "子代理", read_only: bool = False
    ) -> str:
        """agent 工具：用同一個 class 再開一個獨立 context 的迴圈，跑完只把最後那段文字回傳當 tool_result。
        子代理不存 session、不能再開子代理；它印的東西全部縮排一層，Ctrl+C 會一路傳回主代理"""
        child = MiniClaudeCode(
            cwd=self.cwd,
            model=self.model,
            effort=self.effort,
            max_tokens=self.max_tokens,
            compact_threshold=self.compact_threshold_setting,
            auto_approve=self.auto_approve,
            skip_permissions=self.skip_permissions,
            show_thinking=self.show_thinking,
            subagent_of=self,
            read_only=read_only,
            web=self.web,
        )
        requests_before = sum(u["requests"] for u in self.usage.values())
        console.print(
            f"{child.indent}┌ 子代理開始：{description}",
            style="bold magenta",
            markup=False,
        )
        report = child.run_turn(prompt)
        requests = sum(u["requests"] for u in self.usage.values()) - requests_before
        console.print()  # 子代理的串流文字停在行尾，先換行
        console.print(
            f"{child.indent}└ 子代理結束（{requests} 次請求）", style="bold magenta"
        )
        if child.interrupted:  # 使用者按的 Ctrl+C 是要停掉整件事，不是只停子代理；丟回主代理的 run_tools 處理
            raise KeyboardInterrupt
        if report is None:
            raise RuntimeError(
                f"The subagent stopped without a report: {child.error or 'unknown error'}"
            )
        return report or "(The subagent finished without a final report.)"

    @staticmethod
    def tool_result(
        tool_use_id: str, content: str, is_error: bool = False
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if is_error:
            result["is_error"] = True
        return result

    def risk_reason(self, tool_name: str, args: dict[str, Any]) -> str | None:
        """這個呼叫為什麼要「一律詢問」；回傳 None 表示走一般規則（唯讀不問、其他問一次、[a] 可放行）"""
        if tool_name == "bash":
            command = args.get("command", "")
            for pattern, reason in DANGEROUS_COMMAND_PATTERNS:
                if pattern.search(command):
                    return f"危險指令：{reason}"
            if any(p.search(command) for p in SENSITIVE_PATH_PATTERNS):
                return "指令碰到可能含有金鑰或憑證的檔案"
            return None
        if args.get("path"):
            target = self.tools.resolve(args["path"])
            if any(p.search(str(target)) for p in SENSITIVE_PATH_PATTERNS):
                return "可能含有金鑰或憑證的檔案"
            # 寫到工作目錄之外才算高風險；唯讀工具讀外面的檔案見 outside_cwd（一般詢問，[a] / --yolo 可放行）
            if tool_name not in READ_ONLY_TOOLS and not target.is_relative_to(self.cwd):
                return f"寫入工作目錄之外的路徑：{target}"
        return None

    def outside_cwd(self, tool_name: str, args: dict[str, Any]) -> Path | None:
        """唯讀工具碰到工作目錄之外的路徑：要問一次讓人知道，但 [a] 或 --yolo 可放行。
        bash 在 --yolo 下本來就能 cat 任何檔案，對 read_file 一律詢問擋不住什麼，只會一直煩人"""
        if tool_name in READ_ONLY_TOOLS and args.get("path"):
            target = self.tools.resolve(args["path"])
            if not target.is_relative_to(self.cwd):
                return target
        return None

    def needs_approval(self, tool_name: str, args: dict[str, Any]) -> bool:
        """ask_permission 會不會真的跳出詢問；run_tools 用它決定能不能並行（input() 不能同時跑）"""
        if (
            self.skip_permissions or tool_name == "agent"
        ):  # 開子代理本身不問，子代理用到的工具自己會問
            return False
        if self.risk_reason(tool_name, args) is not None:
            return True
        if self.auto_approve or tool_name in self.always_allow:
            return False
        return (
            tool_name not in READ_ONLY_TOOLS
            or self.outside_cwd(tool_name, args) is not None
        )

    def ask_permission(self, tool_name: str, args: dict[str, Any]) -> bool:
        if self.skip_permissions:
            return True
        risk = self.risk_reason(tool_name, args)
        if risk is None and (self.auto_approve or tool_name in self.always_allow):
            return True
        who = "子代理" if self.parent else ""
        console.print(
            f"{self.indent}  [bold yellow]{who}要執行 {tool_name} 嗎？[/bold yellow]"
        )  # 指令與 diff 已經印在上面
        if risk:
            console.print(
                f"{self.indent}  [bold red]⚠ {risk}[/bold red] [dim](這類操作不受 \\[a] 與 --yolo 影響，每次都會問)[/dim]"
            )
        elif outside := self.outside_cwd(tool_name, args):
            console.print(
                f"{self.indent}  [yellow]路徑在工作目錄之外：{outside}[/yellow] [dim](按 \\[a] 之後這個工具讀外面的檔案也不再問)[/dim]"
            )
        options = "[y] 好 / [n] 不要" + ("" if risk else " / [a] 這個工具以後都不用問")
        try:
            answer = input(f"{self.indent}  {options} > ").strip().lower()
        except EOFError:
            return False
        if answer == "a":
            if risk is None:  # 按過 [a] 只放行一般操作，危險指令與敏感檔案還是會問
                self.always_allow.add(tool_name)
            return True
        return answer in ("y", "yes")

    # ---------- Context 管理 ----------

    def record_usage(self, usage: anthropic.types.beta.BetaUsage) -> None:
        u = self.usage.setdefault(
            self.model,
            {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0, "requests": 0},
        )
        u["requests"] += 1
        u["input"] += usage.input_tokens
        u["output"] += usage.output_tokens
        u["cache_write"] += usage.cache_creation_input_tokens or 0
        u["cache_read"] += usage.cache_read_input_tokens or 0
        if (
            usage.server_tool_use
        ):  # --web 時的搜尋次數，另外計費；舊存檔沒有這個 key，用 get
            u["web_searches"] = (
                u.get("web_searches", 0) + usage.server_tool_use.web_search_requests
            )
            u["web_fetches"] = (
                u.get("web_fetches", 0) + usage.server_tool_use.web_fetch_requests
            )
        # 這次請求的 context 大小 = 沒快取的輸入 + 寫入快取 + 讀取快取
        self.last_context_tokens = (
            usage.input_tokens
            + (usage.cache_creation_input_tokens or 0)
            + (usage.cache_read_input_tokens or 0)
        )

    def compact(self) -> bool:
        """把目前整段對話請模型摘要成一則訊息，取代原本的歷史（Claude Code 的 /compact 做的就是這件事）。
        回傳是否成功；失敗時保留原本的歷史。"""
        if not self.messages:
            console.print("[dim]沒有對話可以摘要。[/dim]")
            return False
        instruction = (
            "The conversation is being compacted to free up context. Write a summary for your future self so you can "
            "continue the work seamlessly. Include: the user's goals and constraints, what has been done so far "
            "(files read/changed, commands run and their results), key findings about the codebase, and what remains "
            "to be done. Be specific about file paths and function names. Do not call any tools; output only the summary."
        )
        # 用跟平常一樣的 tools / system / cache_control：請求前綴相同，整段歷史才讀得到快取（不然 15 萬 token 全額計費）。
        # 不用 tool_choice=none 擋工具：文檔說改 tool_choice 會讓 messages 的快取失效，只能靠 prompt 叫它別呼叫工具。
        kwargs = self.request_kwargs(
            self.max_tokens, self.messages + [{"role": "user", "content": instruction}]
        )
        try:
            with (
                console.status("[dim]摘要對話中…[/dim]"),
                self.client.beta.messages.stream(**kwargs) as stream,
            ):
                response = stream.get_final_message()
        except KeyboardInterrupt:
            console.print("\n[yellow]已取消摘要，保留原本的對話。[/yellow]")
            return False
        except anthropic.APIError as e:
            console.print(f"\n[red]摘要失敗，保留原本的對話：{e}[/red]")
            return False
        self.record_usage(response.usage)
        if response.stop_reason != "end_turn":
            # 模型還是呼叫了工具（只拿到「我先讀一下檔案」這種開場白）、被 max_tokens 截斷、或被拒絕，都不是完整的摘要；
            # 拿它取代歷史等於把整段對話丟掉（存檔也會被覆蓋），寧可保留原本的
            console.print(
                f"[red]摘要沒有正常結束（stop_reason: {response.stop_reason}），保留原本的對話。[/red]"
            )
            return False
        summary = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not summary:
            console.print("[red]模型沒有產出摘要，保留原本的對話。[/red]")
            return False
        # 只留一則 user 訊息、不補 assistant 的「收到」：4.6 以後的模型不接受以 assistant 結尾的請求（視為 prefill，直接 400）。
        # 之後使用者的下一句話會變成連續兩則 user 訊息，API 會自動合併成一個 turn，這是文檔明寫的行為。
        self.messages = [
            {
                "role": "user",
                "content": (
                    "<summary>\n"
                    f"{summary}\n"
                    "</summary>\n"
                    "The above is a summary of our conversation so far; it replaces the original history. "
                    "Files you read before are no longer in context: read them again before editing. Continue from here."
                ),
            },
        ]
        self.tools.read_files.clear()  # 摘要後模型看不到檔案內容了，edit 前要重新 read
        self.base_effort = self.effort  # 歷史已重建（effort 訊息也沒了）、快取反正要重寫，中途改過的 effort 直接改回頂層帶
        self.last_context_tokens = 0
        console.print(f"[green]已摘要（{len(summary)} 字）。[/green]")
        return True

    def print_cost(self) -> None:
        console.print()
        if not self.usage:
            console.print("[bold]Token 用量[/bold]：還沒送出任何請求")
        for model, u in self.usage.items():  # /model 切換過就會有好幾段
            console.print(
                f"[bold]Token 用量[/bold]（{u['requests']} 次請求，模型 {model}）\n"
                f"  輸入（未快取）：{u['input']:>10,}\n"
                f"  寫入快取：      {u['cache_write']:>10,}\n"
                f"  讀取快取：      {u['cache_read']:>10,}\n"
                f"  輸出：          {u['output']:>10,}"
                + (
                    f"\n  網路搜尋：      {u['web_searches']:>10,} 次（每千次 10 美元）"
                    if u.get("web_searches")
                    else ""
                )
                + (
                    f"\n  網頁抓取：      {u['web_fetches']:>10,} 次"
                    if u.get("web_fetches")
                    else ""
                )
            )
        console.print(
            f"  目前 context：  {self.last_context_tokens:>10,}（超過 {self.compact_threshold:,} 會自動摘要）"
        )

    # ---------- Session 存檔 ----------

    def save_session(self) -> None:
        if self.parent:  # 子代理的對話不值得存：--resume 接的是主代理
            return
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        data = {
            "model": self.model,
            "effort": self.effort,
            "base_effort": self.base_effort,
            "cwd": str(self.cwd),
            "web": self.web,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "usage": self.usage,
            # thinking block 綁定的前綴包含 system 與 tools，--resume 要原樣還原，見 load_session
            "system_prompt": self.system_prompt,
            "tool_specs": self.tool_specs,
            "messages": self.messages,
        }
        self.session_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def load_session(self, path: Path, restore_model: bool = True) -> None:
        """restore_model=False 表示命令列有明確給 --model，以命令列為準"""
        data = json.loads(path.read_text(encoding="utf-8"))
        self.messages = data["messages"]
        self.effort = data.get("effort", self.effort)
        self.base_effort = data.get("base_effort", self.base_effort)
        saved_model = data.get("model", self.model)
        usage = data.get("usage", {})
        if "requests" in usage:  # 舊存檔是單一模型的平面格式
            usage = {saved_model: usage}
        self.usage = usage
        self.session_path = path
        self.resumed = True
        if "system_prompt" in data:
            # 原樣還原存檔時的 system prompt 與工具清單，不用這次啟動重新產生的：thinking block 記錄了產生它時的
            # system、tools 與之前的 messages，CLAUDE.md、skills 或 --web 只要有一點不同，5.5 系列就判定前綴被改過
            # （新帳號直接 400，連 /compact 都會失敗），快取也全部失效。日期與 git 狀態改在下一則 user 訊息補上
            if self.web and not data.get("web"):
                console.print(
                    "[yellow]這個對話存檔時沒開 --web；tools 是前綴的一部分，中途加上會讓歷史失效，這次不開。[/yellow]"
                )
            self.system_prompt = data["system_prompt"]
            self.tool_specs = data["tool_specs"]
            self.web = bool(data.get("web"))
        elif data.get("web"):
            # 舊格式的存檔：歷史裡有 server_tool_use block，工具清單也要跟著有
            self.enable_web()
        note = ""
        if restore_model:
            self.model = saved_model
        elif self.model != saved_model:
            note = f"，存檔是 {saved_model}，依 --model 改用 {self.model}"
            # 跟 /model 一樣：新模型不支援 /effort 的 system 訊息才從歷史移除
            if (
                not self.model.startswith(PER_MESSAGE_EFFORT_MODELS)
                and self.drop_effort_messages()
            ):
                note += "，已移除歷史裡的 /effort 訊息與 thinking block"
        turns = sum(1 for m in self.messages if m["role"] == "user")
        console.print(
            f"[green]已載入 {path.name}（{turns} 則使用者訊息，模型 {self.model}{note}，effort {self.describe_effort()}）[/green]"
        )


def latest_session(cwd: Path) -> Path | None:
    """找同一個目錄最近的一次對話存檔"""
    if not SESSION_DIR.is_dir():
        return None
    for path in sorted(SESSION_DIR.glob("*.json"), reverse=True):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("cwd") == str(cwd):
                return path
        except (OSError, json.JSONDecodeError):
            continue
    return None


# ========== REPL ==========

HELP = """\
指令：
  /help      顯示這段說明
  /cost      顯示 token 用量
  /effort    顯示目前的 effort；/effort high 切換（low / medium / high / xhigh / max）
  /model     顯示目前的模型；/model claude-sonnet-5-5 切換（快取會重建一次）
  /compact   手動把對話摘要成一則訊息
  /clear     清空對話（開新 session）
  /exit      離開（Ctrl+D 也可以）
  /<skill>   直接執行一個 skill，後面可接參數（找到的 skills 列在下面）
輸入其他文字就是交給模型的任務。模型要寫檔、改檔或執行指令時會先問你。
打 / 會彈出指令與 skill 的選單；/effort 和 /model 後面接著彈出等級與模型 ID，上下鍵選、Tab 或 Enter 補完。"""

COMMANDS = {
    "/help": "顯示說明",
    "/cost": "顯示 token 用量",
    "/effort": "顯示或切換 effort",
    "/model": "顯示或切換模型",
    "/compact": "把對話摘要成一則訊息",
    "/clear": "清空對話",
    "/exit": "離開",
}


# /effort 選單右邊的註解
MODEL_DEFAULT_EFFORT_NOTE = {
    "low": "最快最省",
    "medium": "Haiku / Opus 預設",
    "high": "Sonnet 預設",
    "xhigh": "Claude Code 預設",
    "max": "最深",
}


class SlashCompleter(Completer):
    """打 / 就彈出選單（prompt_toolkit 的 complete_while_typing）：指令與 skill 名稱帶描述，
    /effort 後接等級，/model 後接模型 ID（第一次用才呼叫 Models API，之後用快取）"""

    def __init__(self, agent: MiniClaudeCode):
        self.agent = agent
        self.models: list[str] = []

    def model_ids(self) -> list[str]:
        if not self.models:
            try:
                self.models = [m.id for m in self.agent.client.models.list(limit=100)]
            except anthropic.APIError:
                pass
        return self.models

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        command, _, arg = text.partition(" ")
        if not _:  # 還在打指令名稱：列出內建指令與 skills
            options = [(name, desc) for name, desc in COMMANDS.items()]
            options += [
                (f"/{s.name}", s.description) for s in self.agent.tools.skills.values()
            ]
            for name, desc in options:
                if name.startswith(text):
                    # 補完直接帶一個空格，參數的選單（/effort 的等級、/model 的模型）會接著彈出來
                    yield Completion(
                        name + " ",
                        start_position=-len(text),
                        display=name,
                        display_meta=desc,
                    )
            return
        if " " in arg:  # 參數只有一個，第二個字之後不再提示
            return
        if command == "/effort":
            candidates = [
                (x, MODEL_DEFAULT_EFFORT_NOTE.get(x, "")) for x in EFFORT_LEVELS
            ]
        elif command == "/model":
            candidates = [(x, "") for x in self.model_ids()]
        else:
            return
        for value, desc in candidates:
            if value.startswith(arg):
                yield Completion(value, start_position=-len(arg), display_meta=desc)


def make_prompt(agent: MiniClaudeCode):
    """互動模式的輸入：終端機用 prompt_toolkit（上下鍵歷史、行編輯、/ 選單），stdin 被重導時退回 input()"""
    if not sys.stdin.isatty():
        return lambda: input("> ")
    session: PromptSession[str] = PromptSession(
        # ThreadedCompleter：/model 第一次要打 Models API，放到 thread 才不會卡住輸入
        completer=ThreadedCompleter(SlashCompleter(agent)),
        complete_while_typing=True,
        # 預設只留 8 行給選單，指令加 skills 超過就得捲動；依數量保留（上限 16），多的才捲
        reserve_space_for_menu=min(len(COMMANDS) + len(agent.tools.skills) + 1, 16),
    )
    return lambda: session.prompt("> ")


def repl(agent: MiniClaudeCode) -> None:
    read_line = make_prompt(agent)
    console.print(
        f"[bold]mini claude code[/bold]  模型 {agent.model}"
        f"  effort {agent.describe_effort()}"
        f"{'  [cyan]--web：可上網搜尋[/cyan]' if agent.web else ''}"
        f"{'  [bold red]--dangerously-skip-permissions：什麼都不問[/bold red]' if agent.skip_permissions else '  [red]--yolo：不詢問直接執行[/red]' if agent.auto_approve else ''}\n"
        f"[dim]{agent.cwd}  ·  /help 看指令"
        f"{f'  ·  {len(agent.tools.skills)} 個 skills' if agent.tools.skills else ''}"
        "  ·  Ctrl+C 中斷回應[/dim]"
    )
    while True:
        try:
            console.print()
            text = read_line().strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            console.print()
            continue
        if not text:
            continue
        if text in ("/exit", "/quit"):
            break
        if text == "/help":
            console.print(HELP)
            if agent.tools.skills:
                console.print("\nSkills：")
                for s in (
                    agent.tools.skills.values()
                ):  # 描述來自檔案，可能含方括號，不能交給 rich 當標記
                    console.print(
                        f"  /{s.name:<12} {s.description}", end="", markup=False
                    )
                    console.print(f"  ({s.path.parent})", style="dim")
        elif text == "/cost":
            agent.print_cost()
        elif text == "/effort":
            console.print(
                f"目前 effort：{agent.describe_effort()}（/effort <等級> 切換，可選 {' / '.join(EFFORT_LEVELS)}）"
            )
        elif text.startswith("/effort "):
            level = text.split(maxsplit=1)[1].strip().lower()
            if level in EFFORT_LEVELS:
                agent.set_effort(level)
                agent.save_session()
            else:
                console.print(f"[red]effort 只能是 {' / '.join(EFFORT_LEVELS)}。[/red]")
        elif text == "/model":
            console.print(
                f"目前模型：{agent.model}（/model <模型 ID> 切換，按 Tab 列出可用模型）"
            )
        elif text.startswith("/model "):
            if agent.set_model(text.split(maxsplit=1)[1].strip()):
                agent.save_session()
        elif text == "/compact":
            agent.compact()
            agent.save_session()
        elif text == "/clear":
            agent.messages.clear()
            agent.tools.read_files.clear()
            agent.last_context_tokens = 0
            agent.base_effort = agent.effort
            agent.session_path = SESSION_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
            console.print("[green]已清空對話。[/green]")
        elif (
            text.startswith("/")
            and (parts := text[1:].split(maxsplit=1))[0] in agent.tools.skills
        ):
            agent.run_skill(parts[0], parts[1] if len(parts) > 1 else "")
        elif text.startswith("/"):
            console.print(f"[red]不認識的指令 {text}，/help 看可用指令。[/red]")
        else:
            agent.run_turn(text)
    agent.print_cost()
    console.print(
        f"[dim]對話已存到 {agent.session_path}，下次用 --resume 接著聊。[/dim]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="迷你 Claude Code：只用 anthropic SDK 的 coding agent"
    )
    parser.add_argument(
        "prompt", nargs="?", help="單次模式：給一個任務，跑完就退出；不給就進入互動模式"
    )
    parser.add_argument(
        "--model",
        help=f"模型（預設 {DEFAULT_MODEL}；--resume 時不給就沿用存檔的模型；對話中可用 /model 切換）",
    )
    parser.add_argument(
        "--effort",
        choices=EFFORT_LEVELS,
        help="思考深度（不給就用模型預設；對話中可用 /effort 切換）",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"單次回應輸出上限（預設 {DEFAULT_MAX_TOKENS}）",
    )
    parser.add_argument(
        "--compact-threshold",
        type=int,
        help=f"context 超過多少 token 自動摘要（預設 Haiku 5.5 為 {MODEL_COMPACT_THRESHOLD['claude-haiku-5-5']:,}，其他 {DEFAULT_COMPACT_THRESHOLD:,}）",
    )
    parser.add_argument("--cwd", default=".", help="工作目錄（預設目前目錄）")
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="不詢問，直接執行工具（危險指令、敏感檔案、寫到工作目錄外仍會問）",
    )
    parser.add_argument(
        "--dangerously-skip-permissions",
        action="store_true",
        help="連危險指令、敏感檔案都不問，跟 Claude Code 的同名選項一樣；只適合在容器裡跑",
    )
    parser.add_argument(
        "--show-thinking", action="store_true", help="把模型的思考摘要也印出來"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="開放模型用 Anthropic 伺服器端的 web_search / web_fetch 上網查資料（子代理也能用；搜尋另外計費，每千次 10 美元，抓網頁只算 token）",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        metavar="PATH",
        help="接續對話：給存檔路徑，或不給參數接續這個目錄最近一次",
    )
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit(
            "請在 .env 或環境變數設定 ANTHROPIC_API_KEY（https://platform.claude.com/settings/keys）"
        )

    cwd = Path(args.cwd).resolve()
    if not cwd.is_dir():
        sys.exit(f"找不到目錄：{cwd}")

    agent = MiniClaudeCode(
        cwd=cwd,
        model=args.model or DEFAULT_MODEL,
        effort=args.effort,
        max_tokens=args.max_tokens,
        compact_threshold=args.compact_threshold,
        auto_approve=args.yolo,
        skip_permissions=args.dangerously_skip_permissions,
        show_thinking=args.show_thinking,
        web=args.web,
    )

    if args.resume:
        path = latest_session(cwd) if args.resume == "latest" else Path(args.resume)
        if path is None or not path.is_file():
            sys.exit("找不到可接續的對話存檔。")
        agent.load_session(path, restore_model=args.model is None)

    if args.prompt:
        agent.run_turn(args.prompt)
        agent.print_cost()
    else:
        repl(agent)


if __name__ == "__main__":
    main()
