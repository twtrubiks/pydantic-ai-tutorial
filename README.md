# Pydantic AI：讓 LLM 應用開發像 FastAPI 一樣簡單

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Pydantic AI](https://img.shields.io/badge/Pydantic_AI-2.55-E92063.svg)](https://pydantic.dev/docs/ai/overview/)
[![Pydantic](https://img.shields.io/badge/Pydantic-2.x-E92063.svg)](https://docs.pydantic.dev/)

* [Youtube Tutorial - Pydantic AI - AI 開發的 FastAPI 時刻！告別混亂解析 & 燒錢測試](https://youtu.be/mEwA5DeSFqU)

- **官方文檔**：[https://pydantic.dev/docs/ai/overview/](https://pydantic.dev/docs/ai/overview/)

- **GitHub 專案**：[https://github.com/pydantic/pydantic-ai](https://github.com/pydantic/pydantic-ai)

關鍵是要理解它的定位和限制，在合適的場景中發揮其優勢，而不是盲目地將 AI 用於所有任務。

## 引言：LLM 應用的現實困境

想像一下這個場景：你花了幾週開發了一個 AI 客服系統，測試時運行良好，但上線後卻發現：

* AI 有時返回 JSON，有時返回純文字
* 解析回應的程式碼充滿了 try-catch
* 每次測試都要花費 API 費用
* 切換模型供應商需要重寫大量程式碼

這些痛點讓很多 LLM 應用止步於「原型」階段，難以進入生產環境。

**Pydantic AI** 的誕生就是為了解決這些問題，它的設計理念是：

**「讓 AI 應用開發像使用 FastAPI 一樣簡單、類型安全、生產就緒」**。

---

## 安裝

本專案使用 **Pydantic AI v2**

```bash
# 安裝相依套件
pip install -r requirements.txt
```

---

## 環境設定

範例透過 `python-dotenv` 讀取 `.env` 中的金鑰。請先複製範本再填入你自己的金鑰：

```bash
cp .env.example .env
```

接著編輯 `.env`，依你要執行的範例填入對應的金鑰即可（不必全部填寫）：

| 變數 | 用途 | 取得位置 |
|------|------|----------|
| `ANTHROPIC_API_KEY` | Anthropic Claude（`claude_cache_demo.py`、`claude_harness_demo.py`、`mini_claude_code/`、`claude_advisor_demo.py`、`tool_approval_demo.py`、`web_chat_demo.py`、`tools_demo_claude.py`、`error_handling_demo.py`） | <https://platform.claude.com/settings/keys> |
| `GOOGLE_API_KEY` | Google Gemini（`tools_demo.py`、`gemini_thinking_demo.py`、`gemini_embedding_demo.py` 等） | <https://aistudio.google.com/app/apikey> |
| `OPENROUTER_API_KEY` | OpenRouter（`tools_demo_openrouter.py`） | <https://openrouter.ai/keys> |
| `NVIDIA_API_KEY` | NVIDIA NIM（`output_validator_demo.py`、`mcp_client_demo.py`、`streaming_demo.py`、`toolset_filter_demo.py`、`output_mode_demo.py`、`bank_support_agent_demo.py`、`message_history_demo.py`、`evals_demo.py`、`claude_advisor_demo.py` 主題 B） | <https://build.nvidia.com/> |

---

## 從混亂到有序：簡單的結構化資料提取

讓我們從最簡單的例子開始。假設你需要從影評中提取結構化資訊：

### 傳統方式

```python
import openai

# 發送請求
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": "從以下影評中提取電影名稱、評分和評論：'我昨天看了駭客任務, 整體超酷的, 我給七點二分'"
    }]
)

# 痛苦的解析過程
text = response.choices[0].message.content
# 可能得到："電影：駭客任務\n評分：7.2\n評論：整體超酷的"
# 或者："{'movie': '駭客任務', 'score': 7.2, 'review': '整體超酷的'}"
# 甚至："駭客任務這部電影我給七點二分，因為..."

# 需要寫複雜的解析邏輯
import json
import re
try:
    # 嘗試解析 JSON
    data = json.loads(text)
except:
    # 嘗試正則表達式
    movie = re.search(r'電影[:：]\s*(.+)', text)
    # ... 更多解析邏輯
```

### Pydantic AI 方式

參考 [structured_review_extraction.py](structured_review_extraction.py)（使用 Google Gemini）

- **保證格式**：`output_type=FilmCriticReview`，永遠拿到 Pydantic 模型物件
- **自動驗證**：`Field(ge=1, le=10)` 把評分限制在 1-10 之間
- **自動重試**：格式或驗證不合格時，會請模型重新產生

---

## 結構化輸出的三種模式：Output Mode

上一個範例用 `output_type=FilmCriticReview` 就拿到結構化結果，背後其實可以選擇「怎麼讓模型吐出 JSON」。

參考 [output_mode_demo.py](output_mode_demo.py)（使用 NVIDIA NIM）

- **ToolOutput（預設）**：用 tool calling 回傳結構化資料，相容性最廣，需要多工具混合呼叫時用它
- **NativeOutput**：schema 走 API 的 `response_format`，端點支援原生結構化輸出時優先用它
- **PromptedOutput**：把 schema 寫進 prompt，請模型自己回 JSON，任何模型都能用，但可靠度略低

不論哪種模式，最後都會用 Pydantic 驗證，不合格就重試，重試用完才拋 `UnexpectedModelBehavior`。

**NVIDIA 的坑**：NativeOutput 在 NVIDIA NIM 上不會把 schema 給模型看，欄位容易亂填，要用 `NativeOutput(Recipe, template=...)` 把 schema 放進 prompt（Claude 不需要）

---

## 工具的威力：骰子遊戲範例

用 `@agent.tool` 裝飾器就能把函式註冊成工具，模型會依使用者輸入自己決定何時呼叫。函式的 docstring 就是工具描述，參數會透過 Pydantic 自動驗證。

以下三個範例內容相同，只是模型不同：

- [tools_demo.py](tools_demo.py) - Google Gemini
- [tools_demo_openrouter.py](tools_demo_openrouter.py) - OpenRouter 免費模型
- [tools_demo_claude.py](tools_demo_claude.py) - Anthropic Claude Haiku

**模型差異**：OpenRouter 免費模型常被限流或改為付費，失效時到 <https://openrouter.ai/models> 篩選「免費 + 支援 tools」的模型替換；NVIDIA Nemotron 呼叫無參數工具（如 `roll_dice()`）會自己編參數而失敗，所以 `tools_demo_claude.py` 和 `error_handling_demo.py` 改用 Claude

---

## 生產級系統：銀行客服支援

把依賴注入、動態系統提示、工具和結構化輸出組在一起的完整範例。

參考 [bank_support_agent_demo.py](bank_support_agent_demo.py)（使用 NVIDIA NIM）

- **依賴注入**：`RunContext[SupportDependencies]` 型別安全地傳遞客戶 ID 和資料庫連線
- **動態系統提示**：`@agent.system_prompt` 依客戶資料組出提示，例如 VIP 身份
- **結構化輸出**：`SupportOutput` 包含建議、是否封卡、風險等級

---

## 錯誤處理與重試機制

在工具裡 `raise ModelRetry('提示')`，模型會帶著提示重新呼叫工具，例如查「張小」時引導它改用完整姓名「張小華」。

參考 [error_handling_demo.py](error_handling_demo.py)（使用 Anthropic Claude）

- **重試次數**：`@agent.tool(retries=2)` 設定單一工具，`Agent(..., retries=2)` 設定整體
- **使用上限**：`UsageLimits(request_limit=10)` 防止無限重試燒 token
- **分開捕捉**：重試用完拋 `UnexpectedModelBehavior`，超過使用上限拋 `UsageLimitExceeded`
- **看重試軌跡**：從 `result.all_messages()` 找 `RetryPromptPart`，確認模型真的收到回饋後重試

---

## 輸出驗證層：output_validator

`ModelRetry` 也能用在最終輸出。`@agent.output_validator` 在模型組好結構化輸出後再檢查一次，處理 Pydantic `Field` 做不到的跨欄位邏輯，例如「有 email 就必須有 name」。

參考 [output_validator_demo.py](output_validator_demo.py)（使用 NVIDIA NIM）

- **工具層 vs 輸出層**：工具層驗證單一工具的輸入，輸出層驗證整包結果
- **不通過就重試**：`raise ModelRetry`，模型會重新產生整包輸出，次數由 `Agent(..., retries=2)` 控制

---

## 架構擴展：多代理協作

研究代理產生結構化結果，交給審查代理檢查，代理之間用 Pydantic 模型傳遞資料。

參考 [simple_multi_agent.py](simple_multi_agent.py)（使用 Google Gemini）

- **結構化傳遞**：`ResearchResult`、`ReviewResult` 用 `Field` 限制欄位範圍，傳給下一個代理前就驗證過
- **並行處理**：`asyncio.gather()` 同時處理多個主題
- **自動重試 429**：`GoogleProvider(retry_options=HttpRetryOptions(...))` 遇到限流自動等待重試，Gemini 免費 key 並行時很容易碰到；跨 provider 的做法見 [transport retries](https://pydantic.dev/docs/ai/core-concepts/retries/#transport-retries)

---

## 連接外部工具：MCP Client（Odoo）

Pydantic AI 可以當 MCP Client，把 MCP server 提供的工具直接掛到 Agent 上，Agent 不需要知道背後是 Odoo。

參考 [mcp_client_demo.py](mcp_client_demo.py)（使用 NVIDIA NIM）

- **掛上 server**：`Agent(model, toolsets=[MCPToolset("http://localhost:8000/mcp")])`
- **開關連線**：在 `async with agent:` 區塊內執行 `agent.run()`
- **多來源混用**：本地工具和多個 MCP server 會合併成一份清單，撞名用 `.prefixed("odoo")`

> 執行前要先把 [Odoo MCP server](https://github.com/twtrubiks/odoo19-mcp-server) 用 HTTP transport 跑起來。

---

## 安全護欄：Toolset 過濾（filtered）

用 `filtered()` 從完整工具集衍生出唯讀版，寫入工具不會出現在送給模型的清單裡。模型根本無從呼叫，比在 prompt 叮嚀「不要刪資料」可靠。

參考 [toolset_filter_demo.py](toolset_filter_demo.py)（使用 NVIDIA NIM）

- **用 metadata 標記**：`@toolset.tool(metadata={"access": "read"})`，過濾時不必寫死工具名稱
- **依標記過濾**：`toolset.filtered(lambda ctx, tool_def: (tool_def.metadata or {}).get("access") == "read")`

---

## 安全護欄：工具核准（Human-in-the-Loop）

有些操作該讓模型做，但執行前要先問過人，例如大額轉帳、關閉帳戶。模型照常呼叫工具，Pydantic AI 先攔下來，核准才執行，拒絕就把原因交給模型。

參考 [tool_approval_demo.py](tool_approval_demo.py)（使用 Anthropic Claude Haiku，執行時會在終端機問你 y/N）

- **每次都要核准**：`Tool(close_account, requires_approval=True)`
- **依參數決定**：在工具裡 `raise ApprovalRequired(...)`，例如轉帳超過 1,000 元才要核准
- **暫停式**：`output_type=[str, DeferredToolRequests]`，run 先結束，存下 `result.conversation`，審核後用 `agent.run(conversation=..., deferred_tool_results=...)` 接著跑，適合核准的人不在線上
- **中途核准**：`capabilities=[HandleDeferredToolCalls(handler)]`，run 進行中直接問人，適合核准的人就在程式旁邊

**注意**：核准防的是模型自作主張，不是權限控管，敏感操作在工具裡仍要自己檢查權限

---

## 網頁聊天介面：agent.to_web()

`agent.to_web()` 一行把 agent 變成瀏覽器聊天室，有串流回覆、工具呼叫紀錄、模型切換，需要核准的工具會變成 Approve / Deny 按鈕。範例沿用上一節的銀行轉帳 agent。

參考 [web_chat_demo.py](web_chat_demo.py)（使用 Claude Haiku，有 `GOOGLE_API_KEY` 可切換 Gemini），執行後打開 <http://127.0.0.1:7932>

- **核准按鈕**：agent 的 `output_type` 必須包含 `DeferredToolRequests`，否則遇到要核准的工具會報錯
- **只適合本機開發**：正式產品請用 [UI Event Stream](https://pydantic.dev/docs/ai/integrations/ui/overview/)（Vercel AI / AG-UI）接自己的前端

---

## 省錢開關：Gemini Thinking 控制

Gemini 3 預設會思考，會多花 token 和時間。範例對比 HIGH 與 MINIMAL 的 token 數和耗時。

參考 [gemini_thinking_demo.py](gemini_thinking_demo.py)（使用 Google Gemini）

- **調整等級**：`GoogleModelSettings(google_thinking_config={"thinking_level": "MINIMAL"})`，由低到高為 MINIMAL／LOW／MEDIUM／HIGH
- **看思考 token**：`result.usage.details["thoughts_tokens"]`
- **無法完全關閉**：Gemini 3 最低只到 MINIMAL，要完全關閉得改用 `gemini-2.5-flash` 搭配 `thinking_budget=0`

---

## 語意向量：Embeddings 與相似度搜尋

Embedding 把文字轉成向量，語意接近的句子向量也接近，是語意搜尋和 RAG 的基礎。

參考 [gemini_embedding_demo.py](gemini_embedding_demo.py)（使用 Google Gemini）

- **統一介面**：`Embedder(GoogleEmbeddingModel(...))`，換 provider 不用改呼叫方式
- **查詢 vs 內容**：查詢用 `embed_query()`，被檢索的內容用 `embed_documents()`
- **調維度**：`EmbeddingSettings(dimensions=768)`，越大越精細，索引也越大

---

## 即時體驗：串流輸出（Streaming）

不用等模型講完，第一個字出來就能顯示。範例示範文字串流和結構化資料串流。

參考 [streaming_demo.py](streaming_demo.py)（使用 NVIDIA NIM）

- **開啟串流**：`async with agent.run_stream(prompt) as result:`
- **取輸出**：`result.stream_output()` 每次給累積到目前的完整輸出；`result.stream_text(delta=True)` 只給新增的片段
- **結構化串流**：用 TypedDict + `NotRequired`，欄位在串流途中可以先缺著

---

## 多輪對話：訊息歷史與記憶（Message History）

每次 `agent.run()` 預設都是全新對話，要讓模型記得前文，就要把先前的對話傳回去。

參考 [message_history_demo.py](message_history_demo.py)（使用 NVIDIA NIM）

- **接續對話**：`agent.run(prompt, conversation=result.conversation)`，會一起帶上訊息、usage 和 `conversation_id`
- **存檔／讀回**：`ConversationTypeAdapter.dump_json()` / `validate_json()`，下次執行可以接著聊
- **省 token**：`capabilities=[ProcessHistory(fn)]` 只保留最近 N 則；裁切時 tool call 和它的 return 要成對保留

---

## Claude 專屬：Prompt Caching 與快取診斷

多輪對話每一輪都會把工具定義、instructions、先前對話整包重送。Anthropic 預設不快取，要標記快取斷點才會生效，快取讀取只收 0.1 倍輸入價（部分模型更低）。

參考 [claude_cache_demo.py](claude_cache_demo.py)（使用 Anthropic Claude）

- **開快取**：`Agent(model, capabilities=[Caching()])`，斷點自動放好
- **看效果**：`result.usage.cache_write_tokens`、`cache_read_tokens`；`result.usage.cost` 會算出美金
- **快取診斷**：`AnthropicModelSettings(anthropic_cache_diagnostics=True)`，沒命中時會告訴你原因（換模型、改 system、改工具等）
- **用美金設上限**：`UsageLimits(cost_limit=Decimal('0.01'))`

**注意**：前綴太短（目前的模型低於 512 token）不會被快取，也不會報錯；instructions 裡放時間戳、中途增減工具都會讓快取失效

---

## Claude Code 同款零件：Pydantic AI Harness

[pydantic-ai-harness](https://pydantic.dev/docs/ai/harness/) 把 Claude Code 這類 coding agent 的零件做成 capability，掛到 `Agent(capabilities=[...])` 就能用。範例在暫存目錄建一個有 bug 的迷你專案，讓 agent 自己修好。

參考 [claude_harness_demo.py](claude_harness_demo.py)（使用 Anthropic Claude Haiku）

- **Coder**：讀寫檔案、grep、shell、子代理
- **Skills**：讀 `SKILL.md`，模型一開始只看到名稱和描述，需要時才載入內文
- **RepoContext**：自動把 `CLAUDE.md` / `AGENTS.md` 放進 instructions
- **SystemReminders**：長任務中途用 `<system-reminder>` 重申規則

**注意**：skill 描述要寫清楚「什麼時候用」，因為模型只看得到描述；Coder 的 shell 沒有指令白名單，要在暫存目錄或 sandbox 裡跑

---

## 不靠框架：用 anthropic SDK 手寫迷你 Claude Code

上一節的 Coder 一行就能用，但看不到裡面怎麼跑。這個範例只用 `anthropic` SDK，把 coding agent 的核心迴圈、六個工具、prompt caching、compaction、工具核准、Skills、子代理與網路搜尋全部手寫一遍，是一個真的能在你的專案目錄裡工作的 CLI。

參考 [mini_claude_code/](mini_claude_code/)（使用 Anthropic Claude Haiku），完整說明見資料夾裡的 [README](mini_claude_code/README.md)

```bash
python mini_claude_code/mini_claude_code.py   # 以目前目錄為工作目錄啟動互動模式，/help 看指令
```

- **核心迴圈**：呼叫模型 → 看 `stop_reason` → `tool_use` 就執行工具、接回 `tool_result` 再呼叫，直到 `end_turn`，從 `run_turn` 開始讀
- **對照框架**：上一節 pydantic-ai 幫你做掉的事（`tool_use` / `tool_result` 配對、各種 `stop_reason`、快取斷點、對話歷史）在這裡都要自己寫

**注意**：bash 工具沒有白名單，會在目前目錄執行任意指令；預設寫檔、改檔、shell 執行前都會先問，`--yolo` 跳過詢問

---

## Claude 專屬：Advisor 顧問模式（Haiku 做事、Opus 指點）

便宜的模型負責做事，遇到關鍵判斷再請教更強的模型。Anthropic 的 [advisor tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool)（beta）在伺服器上完成這件事，同一次 API 請求內就拿到建議。範例算跨年的出貨日，Haiku 單獨會答錯，加了 Opus 顧問就答對。

參考 [claude_advisor_demo.py](claude_advisor_demo.py)（執行者 Claude Haiku 5.5、顧問 Claude Opus 5.5；主題 B 另需 `NVIDIA_API_KEY`）

```python
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import AdvisorTool

agent = Agent(
    'anthropic:claude-haiku-5-5',
    capabilities=[NativeTool(AdvisorTool(model='claude-opus-5-5', max_tokens=2048))],
)
```

- **費用要自己算**：顧問的 token 不算進 `usage.cost`，要從 `usage.details` 的 `advisor_*` 換算
- **不一定比較省**：顧問每次都讀完整段對話，短任務直接用 Opus 反而便宜；適合大部分步驟很機械、少數判斷要很準的長任務
- **Haiku 容易忘記問**：instructions 要寫清楚什麼時候呼叫顧問
- **跨供應商**：pydantic-ai-harness 的 `Advisor(...)` 讓其他供應商（範例用 NVIDIA）也能用 Claude 當顧問

---

## 開發體驗：測試與除錯

### 零成本測試

用 `agent.override(model=...)` 換成假模型，不呼叫真實 API 也能測試 agent 的流程。

參考 [zero_cost_testing.py](zero_cost_testing.py)

```python
from pydantic_ai.models.test import TestModel

with agent.override(model=TestModel()):
    result = await agent.run('擲骰子', deps='玩家')
```

- **TestModel**：自動呼叫所有工具並產生假輸出，`custom_output_text=` 可以指定回覆
- **FunctionModel**：自己寫函式決定每一步要呼叫哪個工具、回什麼文字
- **不花錢、結果一致**：可以用 `asyncio.gather` 並行跑大量案例，適合放進 CI

### 模型切換

```python
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel

# 字串格式：<provider>:<model>
agent = Agent('anthropic:claude-sonnet-5-5')
agent = Agent('google:gemini-3-flash-preview')
agent = Agent('openai:gpt-4o')  # Responses API；要用 Chat Completions 請寫 openai-chat:gpt-4o

# FallbackModel：第一個模型失敗就換下一個
agent = Agent(FallbackModel(OpenAIChatModel('gpt-4o'), AnthropicModel('claude-sonnet-5-5')))
```

---

## 品質驗證：評估框架（Pydantic Evals）

LLM 的回答品質常是程度問題，沒辦法用 `assert ==` 判斷。Pydantic Evals 把測試案例和評分標準整理成資料集，跑一輪後產出評分報告。TestModel 驗證流程對不對，Evals 驗證回答好不好。

參考 [evals_demo.py](evals_demo.py)（使用 NVIDIA NIM）

- **Case / Dataset**：`Case(inputs=..., expected_output=...)` 組成 `Dataset`
- **評估器**：`IsInstance` 檢查型別、自訂 `Evaluator` 做確定性檢查、`LLMJudge` 用 LLM 依 `rubric` 評主觀品質
- **執行**：`dataset.evaluate_sync(task)`，再用 `report.print()` 印出報告

**注意**：每個 LLMJudge 都會多打一次 API，case 和裁判越多就越貴；有標準答案的題目用程式檢查就夠了

---

## 框架對比

| 特性 | Pydantic AI | LangChain | Instructor |
|------|-------------|-----------|------------|
| **學習曲線** | 簡單（類似 FastAPI） | 陡峭（抽象層多） | 簡單（專注單一功能） |
| **類型安全** | 完整（Pydantic 驅動） | 有限 | 完整（基於 Pydantic） |
| **測試支援** | TestModel、FunctionModel | 需自行 mock | 基本支援 |
| **生產就緒** | 是（設計初衷） | 需大量配置 | 功能有限 |
| **適用場景** | 生產級結構化 AI 應用 | 原型開發、實驗 | 純結構化輸出 |

---

## 使用建議

### 適合場景

- 生產級應用需要穩定的 AI 功能
- 需要保證結構化輸出格式
- 重視類型安全和程式碼品質
- 需要在不同 LLM 間靈活切換

### 不適合場景

- 複雜對話鏈（LangChain 更適合）
- 純實驗性項目
- 需要精確計算或實時回應
- 高頻交易或遊戲邏輯

---

## 重要限制

- 每次執行都要呼叫 LLM，成本隨用量增加，延遲通常 1-5 秒
- 相同輸入可能產生不同輸出，決策過程不透明，除錯比較困難
- 依賴第三方 AI 服務，受速率限制，難以保證 99.9% SLA

**AI 是增強而非取代**：簡單邏輯用傳統程式處理，AI 留給低頻但高價值的任務。

---

## Donation

文章都是我自己研究內化後原創，如果有幫助到您，也想鼓勵我的話，歡迎請我喝一杯咖啡 :laughing:

綠界科技 ECPay ( 不需註冊會員 )

![alt tag](https://payment.ecpay.com.tw/Upload/QRCode/201906/QRCode_672351b8-5ab3-42dd-9c7c-c24c3e6a10a0.png)

[贊助者付款](http://bit.ly/2F7Jrha)

歐付寶 (需註冊會員)

![alt tag](https://i.imgur.com/LRct9xa.png)

[贊助者付款](https://payment.opay.tw/Broadcaster/Donate/9E47FDEF85ABE383A0F5FC6A218606F8)

## 贊助名單

[贊助名單](https://github.com/twtrubiks/Thank-you-for-donate)

## License

MIT license
