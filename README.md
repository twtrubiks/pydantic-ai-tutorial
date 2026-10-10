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
| `ANTHROPIC_API_KEY` | Anthropic Claude（`claude_cache_demo.py`、`claude_harness_demo.py`、`tools_demo_claude.py`、`error_handling_demo.py`） | <https://platform.claude.com/settings/keys> |
| `GOOGLE_API_KEY` | Google Gemini（`tools_demo.py`、`gemini_thinking_demo.py`、`gemini_embedding_demo.py` 等） | <https://aistudio.google.com/app/apikey> |
| `OPENROUTER_API_KEY` | OpenRouter（`tools_demo_openrouter.py`） | <https://openrouter.ai/keys> |
| `NVIDIA_API_KEY` | NVIDIA NIM（`output_validator_demo.py`、`mcp_client_demo.py`、`streaming_demo.py`、`toolset_filter_demo.py`、`output_mode_demo.py`、`bank_support_agent_demo.py`、`message_history_demo.py`、`evals_demo.py`） | <https://build.nvidia.com/> |

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

參考 [structured_review_extraction.py](structured_review_extraction.py)

這個範例展示如何使用 PydanticAI 將非結構化的影評文字轉換為結構化的 Python 物件。主要特色包括：

**保證格式**：永遠返回 `FilmCriticReview` 物件

**自動驗證**：評分自動限制在 1-10 之間

**自動重試**：如果格式錯誤，AI 會自動重新生成(ModelRetry)

**結構化輸出**：AI 輸出自動轉換為 Pydantic 模型物件

---

## 結構化輸出的三種模式：Output Mode

上一個範例用 `output_type=FilmCriticReview` 就拿到結構化結果，但它背後其實有「怎麼讓模型吐出 JSON」的選擇。Pydantic AI 提供三種 output mode，差別在 schema 走哪條路、以及對端點的要求。

參考 [output_mode_demo.py](output_mode_demo.py)（使用 NVIDIA NIM）

**三種模式**：

**ToolOutput（預設）**：用 tool calling 回傳結構化資料，相容性最廣（前面範例就是它）

**NativeOutput**：schema 走 API 頂層 `response_format` / JSON schema（**預設不進 prompt**），只有端點支援原生結構化輸出時才會被「強制」

**PromptedOutput**：把 JSON schema 塞進 prompt，請模型「自己」回 JSON，**任何模型都能用**，但純靠提示約束、可靠度略低

**核心做法**：

**指定模式**：`output_type=NativeOutput(Recipe)` 或 `output_type=PromptedOutput(Recipe)`，不指定則為預設 ToolOutput

**同一結構、兩種模式**：範例用同一個 `Recipe` 資料結構分別跑 NativeOutput 與 PromptedOutput，證明兩者都能得到同樣型別的結構化結果

**都有安全網**：兩種模式模型「都可能不遵守」（Native 端點不支援會被忽略、Prompted 模型可能回非預期格式），但不論哪種，pydantic_ai 最後都會用 Pydantic 驗證回傳值，不合就重試，重試用完才拋 `UnexpectedModelBehavior`——不會默默把壞資料丟給你

**NVIDIA 端點的坑**：NVIDIA NIM 會依 schema 強制 JSON 格式，卻不把 schema 給模型看（輸入只有約 69 token）。Nemotron 不知道欄位意思只能硬填，實測會出現料理名稱帶「(2 人份)」、食材清單混進 `':材料:'`、`'- 番茄…'` 這類雜訊。只改 `Field(description=...)` 沒用，因為模型根本看不到。解法是用 `NativeOutput(Recipe, template="請依照這個 JSON schema 回答：\n{schema}")` 把 schema 也放進 prompt，pydantic-ai 的 vLLM provider 也是這樣處理。同樣的 `Recipe` 換成 Claude Haiku 則不需要 template，Anthropic 會自己把 schema 交給模型

**怎麼選**：

- 端點支援原生結構化輸出（`response_format` json_schema）→ 優先 **NativeOutput**
- 端點不支援 tool / 原生 JSON、或想最大相容 → **PromptedOutput**
- 需要多工具混合呼叫的流程 → 維持預設 **ToolOutput**

---

## 工具的威力：骰子遊戲範例

傳統的 Function Calling 實現起來很繁瑣，但 Pydantic AI 讓它變得優雅：

以下範例都是一樣的, 只是模型不同

[tools_demo.py](tools_demo.py) - 基本骰子遊戲工具展示

[tools_demo_openrouter.py](tools_demo_openrouter.py) - 使用 OpenRouter 的工具範例

[tools_demo_claude.py](tools_demo_claude.py) - 使用 Anthropic Claude（Haiku）的工具範例

**工具調用支援說明**：

**OpenAI 模型**：Function Calling 原生支援，表現最穩定（推薦）

**Google Gemini**：`gemini-3-flash-preview` 工具調用表現不夠穩定

**OpenRouter 免費模型**：變動頻繁，常見狀況有被上游限流（429）、改為付費（404）、只開放給特定應用（403）。範例目前使用 `cohere/north-mini-code:free`，失效時可到 <https://openrouter.ai/models> 篩選「免費 + 支援 tools」的模型替換

**NVIDIA Nemotron**：呼叫無參數工具（如 `roll_dice()`）時會自己編參數，被參數驗證擋下後超過重試上限，所以 `tools_demo_claude.py` 和 `error_handling_demo.py` 改用 Claude

**Anthropic Claude**：`claude-haiku-5-5` 便宜，工具呼叫（含無參數工具）穩定，收到 ModelRetry 提示也會照著修正

**核心特色**：

**工具定義簡單**：用 `@agent.tool` 裝飾器就能註冊函數為工具

**AI 自主決策**：模型根據用戶輸入自動判斷何時調用哪個工具

**文檔字符串重要**：函數的 docstring 成為工具描述，幫助 AI 理解工具用途

**參數自動驗證**：工具參數透過 Pydantic 自動驗證和轉換

---

## 生產級系統：銀行客服支援

這個完整的銀行客服範例展示了 Pydantic AI 如何整合所有核心功能，建構一個真正的生產級 AI 系統。

參考 [bank_support_agent_demo.py](bank_support_agent_demo.py)（使用 NVIDIA NIM）

**技術架構解析**：

**依賴注入**：使用 `SupportDependencies` 類別封裝客戶 ID 和 `DatabaseConn`，透過 `RunContext[SupportDependencies]` 型別安全地傳遞

**多層動態提示**：兩個系統提示函數自動組合 - `add_customer_name()` 查詢客戶姓名、`add_customer_status()` 識別 VIP 身份

**智能工具調用**：`customer_balance()` 工具讓 AI 主動查詢餘額，支援 `include_pending` 參數控制是否包含待處理交易

**結構化輸出**：`SupportOutput` 模型定義回應格式，包含 `support_advice`（建議）、`block_card`（封卡決策）和 `risk`（0-10 風險等級）

---

## 錯誤處理與重試機制

展示 Pydantic AI 的 ModelRetry 功能，包括智能處理不完整輸入和自動重試機制。

參考 [error_handling_demo.py](error_handling_demo.py)（使用 Anthropic Claude）

**多層重試架構**：

**工具層級重試**：`@user_agent.tool(retries=2)` 針對 `get_user_by_name()` 設定重試次數，`@user_agent.tool(retries=3)` 針對 `validate_external_api()` 設定 API 重試

**Agent 全域重試**：`Agent(..., retries=2)` 設定整體對話的重試次數上限

**使用限制保護**：`UsageLimits(request_limit=10)` 防止無限重試消耗過多 token

**智能錯誤引導（學習機制）**：

**兩個字找三個字**：當查詢「張小」這種兩個字姓名時，系統會使用 `ModelRetry` 找到「張小華」並引導 AI 使用完整姓名重試

**不存在用戶處理**：查詢「小明」等不存在用戶時，直接返回 None 並在 message 中說明用戶不存在

**模擬 API 失敗**：`validate_api_service()` 前兩次故意失敗，第三次成功，展示自動重試直到成功的機制

**自我修正軌跡（`print_retry_trace()`）**：走訪 `result.all_messages()` 把 `ToolCallPart`／`RetryPromptPart`／`ToolReturnPart` 印出來，用「工具被呼叫 > 1 次」證明 AI 真的收到回饋後重試；場景 1（不存在用戶）只呼叫 1 次，剛好是「沒有觸發重試」的對照。

---

## 輸出驗證層：output_validator

`ModelRetry` 不只能用在工具層，也能用在「最終輸出」這一層。`@agent.output_validator` 會在 AI 組裝出**結構化輸出之後**再做一次語意驗證，專門處理 Pydantic `Field` 驗證做不到的「跨欄位邏輯」。

參考 [output_validator_demo.py](output_validator_demo.py)（使用 NVIDIA NIM）

**與工具層 ModelRetry 的差異**：

**工具層**：驗證單一工具的輸入／執行（如 `error_handling_demo.py` 的 `@tool(retries=2)`）

**輸出層**：驗證 AI 組裝出來的最終結果，做跨欄位一致性檢查

**範例的兩條驗證規則**：

**規則 1（自相矛盾）**：輸出有 `email` 卻沒有 `name` → `raise ModelRetry` 要求補上完整姓名

**規則 2（跨欄位一致）**：查到用戶時，`message` 必須包含其 `email` → 否則 `raise ModelRetry` 引導 AI 把 email 寫進 message

**運作重點**：

- 驗證不通過時 `raise ModelRetry`，AI 會帶著提示重新生成「整包輸出」，最多重試次數由 `Agent(..., retries=2)` 控制
- 若模型第一次輸出就符合規則，validator 直接放行（安全閘的正確行為，不會強制重試）
- 可透過 `result.all_messages()` 觀察 `RetryPromptPart`，看見輸出層自我修正的完整軌跡

**NVIDIA NIM 接入方式**：透過 `OpenAIChatModel` + `OpenAIProvider(base_url="https://integrate.api.nvidia.com/v1", api_key=...)`，以 OpenAI 相容介面呼叫 Nemotron 等模型。

---

## 架構擴展：多代理協作

簡化的多代理協作範例，展示 PydanticAI 的核心優勢：結構化輸出驗證、代理間資料傳遞和異步處理能力。

參考 [simple_multi_agent.py](simple_multi_agent.py)

**核心特性展示**：

- **結構化輸出與自動驗證**：
  - `ResearchResult`：包含主題、關鍵發現（1-5個）、信心分數（0-1）和審查需求
  - `ReviewResult`：包含審查結果、回饋意見和改進建議
  - 使用 Pydantic Field 驗證（`min_length`, `max_length`, `ge`, `le`）確保資料品質

- **兩階段工作流程**：
  - **研究代理**：分析主題並產生結構化研究結果，配備 `calculate_confidence()` 和 `get_current_time()` 工具函數
  - **審查代理**：基於研究結果進行品質審查，提供建設性回饋（僅在需要時執行）

**實作優勢**：

- **簡潔的協調邏輯**：`coordinate_agents()` 函數展示清晰的代理間資料流
- **異步批次處理**：使用 `asyncio.gather()` 平行處理多個主題，提升效能
- **自動錯誤處理**：每個階段都有 try-catch 區塊，確保部分失敗不影響整體流程
- **類型安全**：所有資料傳遞都有明確的類型定義，減少執行時錯誤
- **自動重試限流（429）**：`GoogleProvider(retry_options=HttpRetryOptions(...))` 讓 google-genai SDK 遇到 429 / 5xx 時自動等待再重試（指數退避，最長等 60 秒）

**免費額度提醒**：Gemini 免費 key 有每分鐘請求數上限，範例 2 用 `asyncio.gather()` 一次送出多個請求，很容易收到 429。範例已加上 `retry_options`，實測免費 key 的結果是 10 個請求成功、8 次 429 自動重試，約 1.5 分鐘跑完，3 個主題全部成功。付費 key 不會碰到 429，執行速度不受影響。更通用、跨 provider 的做法可參考官方文件的 [transport retries](https://pydantic.dev/docs/ai/core-concepts/retries/#transport-retries)。

---

## 連接外部工具：MCP Client（Odoo）

Pydantic AI 可以作為 **MCP Client**，把「已經獨立跑起來」的 MCP server 當成一組工具掛上 Agent。server 暴露的工具會自動變成 LLM 可呼叫的工具，Agent 完全不需要知道背後是 Odoo / XML-RPC。

參考 [mcp_client_demo.py](mcp_client_demo.py)（使用 NVIDIA NIM；連線 Odoo MCP server）

**核心做法**：

**掛上 MCP server**：用 `MCPToolset("http://localhost:8000/mcp")` 連到 Streamable HTTP 端點，再放進 `Agent(model, toolsets=[odoo_mcp])`

**自動開關連線**：所有 `agent.run(...)` 都要在 `async with odoo_agent:` 區塊內執行，進入時開啟、離開時自動關閉 MCP 連線

**多來源混用**：本地函式工具（`tools=[...]`／`@agent.tool`）與多個 MCP server（`toolsets=[a_mcp, b_mcp]`）會合併成同一份工具清單；撞名用 `.prefixed("odoo")`，工具太多用 `.filtered()`／`.prepared()` 收斂

**呼叫軌跡**：透過 `result.all_messages()` 觀察 `ToolCallPart`／`ToolReturnPart`，看 LLM 實際呼叫了哪些 MCP 工具

> **執行前提**：本範例只是 client 端，需先自行把 Odoo MCP server 用 HTTP transport 跑起來（server 來源：<https://github.com/twtrubiks/odoo19-mcp-server>）。Odoo 連線資訊設在 server 端，client 端只需 `ODOO_MCP_URL` 端點（預設 `http://localhost:8000/mcp`）。

---

## 安全護欄：Toolset 過濾（filtered）

把工具過濾當成「安全護欄」——用 `filtered()` 從完整工具集衍生出「唯讀版」，危險的寫入工具直接不會出現在送給模型的工具清單裡。模型不是被勸退，而是「無從呼叫」，比在 system prompt 叮嚀「不要刪資料」可靠得多。

參考 [toolset_filter_demo.py](toolset_filter_demo.py)（使用 NVIDIA NIM）

**核心做法**：

**衍生受限工具集**：`employee_toolset.filtered(lambda ctx, tool_def: (tool_def.metadata or {}).get("access") == "read")`，只放行唯讀工具

**用 metadata 標記**：`@toolset.tool(metadata={"access": "read"/"write"})` 標記工具屬性，過濾邏輯就不必硬編工具名稱

**護欄在工具集層**：被擋掉的工具不會進入送給模型的清單，模型根本看不到、也就無從呼叫

**驗證生效**：`await toolset.get_tools(ctx)` 印出模型「真正看得到」的工具——場景 A（完整工具集）看得到刪除工具、資料真的被刪；場景 B（filtered）看不到、資料安然無恙

---

## 省錢開關：Gemini Thinking 控制

Gemini 3 系列預設會「思考」（thinking），思考會額外消耗 token、增加延遲。這個範例直接對比「高思考（HIGH）」與「最省思考（MINIMAL）」，用 token 數與耗時證明降低思考等級後更省、更快。

參考 [gemini_thinking_demo.py](gemini_thinking_demo.py)

**核心做法**：

**省錢開關**：用 `GoogleModelSettings(google_thinking_config={"thinking_level": "MINIMAL"})` 把思考降到最低

**思考等級**：`thinking_level` 由低到高為 `MINIMAL < LOW < MEDIUM < HIGH`，等級越高思考越多、越貴

**思考摘要**：`include_thoughts=True` 會把思考內容一併回傳，方便在 `result.all_messages()` 軌跡裡看到 `ThinkingPart`

**省錢證據**：透過 `result.usage.details["thoughts_tokens"]` 觀察思考 token 在 MINIMAL 時大幅下降，搭配耗時對比一目了然

**模型差異**：本範例使用 `gemini-3-flash-preview`。Gemini 3 系列以 `thinking_level`（MINIMAL／LOW／MEDIUM／HIGH）控制思考強度，**無法完全關閉**思考（沒有 `budget=0`），MINIMAL 即為最省等級。若需要「完全關閉」思考，要改用 `gemini-2.5-flash` 搭配 `thinking_budget=0`。

---

## 語意向量：Embeddings 與相似度搜尋

Embedding 把文字轉成向量，讓「語意接近」變成「向量接近」，是語意搜尋、推薦、分群、RAG 的基礎。這個範例用 Pydantic AI 的 `Embedder` 把句子向量化，再用 cosine 相似度從候選句子裡找出與查詢語意最接近的。

參考 [gemini_embedding_demo.py](gemini_embedding_demo.py)

**核心做法**：

**統一介面**：`Embedder(GoogleEmbeddingModel("gemini-embedding-2", provider=GoogleProvider(api_key=...)))`，換 provider 不用改呼叫方式

**查詢 vs 內容**：`embed_query()` 給查詢、`embed_documents()` 給被檢索的內容；某些模型會分別最佳化，分開呼叫語意更準

**取向量**：`result.embeddings[i]` 就是第 i 句的向量（一串 float）

**調維度**：`EmbeddingSettings(dimensions=768)` 控制向量維度——越大越精細，索引也越大、越慢

**相似度**：用 cosine 相似度（1 ≈ 語意相同、0 ≈ 不相關）排序，即得語意搜尋結果

---

## 即時體驗：串流輸出（Streaming）

不必等模型「整段講完」才看到結果——串流讓第一個字一出來就能顯示，對長文、即時對話與進度感很有幫助。這個範例展示兩種串流：文字／Markdown 串流（打字機效果），以及結構化資料串流（一邊生成、一邊做部分驗證，表格即時長出來）。

參考 [streaming_demo.py](streaming_demo.py)（使用 NVIDIA NIM）

**核心做法**：

**開啟串流**：`async with agent.run_stream(prompt) as result:`

**累積輸出**：`result.stream_output()` 每次給「累積到目前」的完整輸出（文字或結構化）；`result.stream_output(debounce_by=0.01)` 控制畫面更新頻率

**只取新增片段**：`result.stream_text(delta=True)` 只回傳「這次新增」的文字片段

**部分驗證**：`output_type=list[Whale]`（TypedDict + `NotRequired`）讓結構化串流途中欄位可「先缺著」，取值用 `.get()` 並給預設值

**小提醒**：`usage`（用量）要等串流結束才會是完整數字

> 本範例用 `rich` 的 `Live` 即時渲染（pydantic-ai 的 cli extra 已帶入 rich，`requirements.txt` 亦明列）。

---

## 多輪對話：訊息歷史與記憶（Message History）

預設每次 `agent.run()` 都是「全新對話」——模型不會記得上一輪講過什麼。要做出能連續聊天的應用，關鍵就是把先前的對話用 `message_history=` 餵回去。這個範例展示三件事：① 無記憶 vs 有記憶的對比、② 把對話序列化成 JSON 存檔、下次再讀回來接著聊（跨 session 持久化）、③ 用 History Processor 控制送出的訊息量來省 token。

參考 [message_history_demo.py](message_history_demo.py)（使用 NVIDIA NIM）

**核心做法**：

**延續對話**：`agent.run(prompt, message_history=previous_messages)`，傳入歷史後,系統提示沿用歷史裡那份、不重新生成

**取出訊息**：`result.all_messages()` 取完整歷史（含先前所有 run）；`result.new_messages()` 只取本輪新增

**存檔／讀回**：`ModelMessagesTypeAdapter.dump_json(history)` 序列化成 JSON bytes，`ModelMessagesTypeAdapter.validate_json(data)` 還原，即可跨 session 接續對話

**對話關聯**：同一段對話的多次 run 共享同一個 `result.conversation_id`，即使存檔讀回也不變，方便在 Logfire 等工具追蹤

**省 token（V2 特性）**：`Agent(model, capabilities=[ProcessHistory(fn)])` 在每次送出前處理歷史；範例用「只保留最近 N 則」，避免 input token 隨對話無限膨脹（注意 V2 改用 `capabilities=`，V1 的 `history_processors=` 參數已移除）

**小提醒**：裁切或摘要歷史時，要確保 tool call 與其 return 成對出現，否則模型可能報錯（純文字對話無此問題）

---

## Claude 專屬：Prompt Caching 與快取診斷

多輪對話每一輪都會把「工具定義 + instructions + 先前對話」整包重送，手冊、程式碼這類長前綴每輪都要重新付費。Prompt caching 讓重複的前綴從快取讀回，只收 0.1 倍輸入價（部分模型更低）。OpenAI、Gemini 會自動快取，**Anthropic 預設什麼都不快取**，要在請求裡標記快取斷點才會生效。這個範例展示三件事：① 同一段對話開快取 vs 沒開的 token 帳單與美金對比、② 用 Claude API 專屬的 cache diagnostics 找出「快取為什麼沒命中」、③ 用 `cost_limit` 直接以美金設預算上限。

參考 [claude_cache_demo.py](claude_cache_demo.py)（使用 Anthropic Claude，需要 pydantic-ai >= 2.55）

**核心做法**：

**一行開快取**：`Agent(model, capabilities=[Caching()])`，等同 `model_settings={'cache': True}`；Pydantic AI 會自動在工具定義、instructions、對話尾端放斷點（Anthropic 上限 4 個，自動管理）

**計價**：5 分鐘快取寫入 1.25 倍、讀取 0.1 倍（Sonnet／Opus 5.5 為 0.05 倍、Fable 5.1 為 0.025 倍，範例依 `MODEL_NAME` 自動套用），被讀回一次就回本，對話越長越省；`Caching(retention='1h')` 寫入 2 倍，要讀回兩次才回本

**看效果**：`result.usage.cache_write_tokens`、`cache_read_tokens`、`cache_hit_ratio`；注意 `input_tokens` 已包含快取讀寫

**最小長度**：前綴低於模型的最小可快取長度（Haiku／Sonnet／Opus 5.5 與 Fable 5.1 為 512 token，較舊的模型為 1,024～4,096 token）就不會被快取，讀寫都是 0，而且不會報錯

**快取診斷（V2.55 新增）**：`AnthropicModelSettings(anthropic_cache_diagnostics=True)`，沒命中時 `ModelResponse.provider_details['cache_diagnostics']` 會寫出原因（`model_changed` / `system_changed` / `tools_changed` / `messages_changed`）與少讀了多少 token；免費、不影響快取，只在 Claude API 有效

**快取殺手**：instructions 放到秒的時間戳、對話中途增減工具、中途換模型，都會讓前綴改變；Claude 5.x 開著 thinking 時改前綴，還可能直接回 400（thinking block 綁定前綴）

**只快取固定前綴**：大量一次性問答共用長 instructions 時，用 `Caching(messages=False)`，避免每段對話都付寫入費卻沒被讀回

**費用追蹤（V2 內建）**：`result.usage.cost` 是用 [genai-prices](https://github.com/pydantic/genai-prices) 價格表估算的美金（`Decimal`），快取讀寫、輸出 token 都算進去。實測兩輪合計：沒開快取 $0.001689、開快取 $0.001135，省約 33%，跟等價 token 的 32% 吻合。查不到價格的模型（例如 NVIDIA NIM）會是 `None`，不是 0

**用美金設預算**：`UsageLimits(cost_limit=Decimal('0.01'))`，超過就丟 `UsageLimitExceeded`。預設在**回應後**才檢查，請求已經送出、已經計費；加上 `count_tokens_before_request=True` 會先呼叫 token 計數 API 估算輸入費用，超過就不送出（錯誤訊息會變成 `The next request would exceed ...`）。這只是估算，要搭配 `request_limit` 或供應商後台的花費上限

---

## Claude Code 同款零件：Pydantic AI Harness

[pydantic-ai-harness](https://pydantic.dev/docs/ai/harness/) 是 Pydantic 官方的另一個套件，把 Claude Code 這類 coding agent 的零件做成一個個 capability，掛到 `Agent(capabilities=[...])` 就能用，等於用 pydantic-ai 組出一個簡化版 Claude Code。範例會在暫存目錄建一個迷你專案（有 `CLAUDE.md`、兩個 skill、埋了兩個 bug 的 `calc.py` 和測試檔），讓 agent 在裡面工作，跑完自動刪除。

參考 [claude_harness_demo.py](claude_harness_demo.py)（使用 Anthropic Claude Haiku，需要 `pip install "pydantic-ai-harness[coder,skills]"`）

**四個零件**：

**Coder**：提供 `read_file` / `write_file` / `edit_file` / `list_files` / `grep` / `shell` / `delegate_task`（子代理），本身由 FileSystem + Shell + RepoContext + SubAgents + 壓縮、快取等零件組成，也預設開啟 prompt caching

**Skills**：讀 `SKILL.md`（跟 Claude Code skills 同一個 [agentskills.io](https://agentskills.io/specification) 標準）。模型一開始只看到每個 skill 的名稱和描述，需要時才用 `load_capability` 載入內文；不會自動搜尋 `.claude/skills`，要用 `Skills('.claude/skills')` 明確指定

**RepoContext**：run 開始時自動讀 `CLAUDE.md` / `AGENTS.md` 放進 instructions；`expose_inventory_tool=True` 會多一個工具，列出 `.claude` 底下的 skills、agents、hooks（Coder 已內含 RepoContext）

**SystemReminders**：照 Claude Code 的做法，用 `<system-reminder>` 標籤在長任務中途重申規則（例如 `Reminder('只能修改 calc.py', interval=3)`）。只放在每次請求的尾端，不寫進對話歷史、不打破快取；要用 `@agent.on_event(ReminderFiredEvent)` 才看得到它何時觸發

**兩個主題**：

**A. RepoContext + Skills（唯讀）**：不給 shell，問 agent 這個專案有哪些 coding-assistant 設定。實測它會呼叫 `inventory_agent_context()`，正確說出 CLAUDE.md 的規則和兩個 skill，且不載入 skill 內文

**B. Coder + Skills + SystemReminders**：交代「`test_calc.py` 跑不過，請找出原因並修好，最後寫一份修正說明」。實測 agent 會自己讀檔、跑測試、用 `edit_file` 修好兩個 bug、再跑一次測試，並載入 `release-notes` skill 照格式寫說明；範例最後會自己重跑測試、確認測試檔沒被改，不只相信 agent 的說法。整個主題約 3～4 次請求，大部分輸入從快取讀回

**Skill 描述要寫清楚「什麼時候用」**：實測描述寫成「撰寫修正說明時使用」，Haiku 會直接用自己的格式寫；改成「撰寫修正說明前，一定要先載入這個 skill」才穩定載入。模型只看得到描述，描述就是觸發條件

**安全性**：Coder 的 shell 沒有指令白名單，務必讓它在暫存目錄或 sandbox（Modal / E2B）裡跑。`LocalWorkspace` 只把 `PATH`、`HOME`、`LANG` 傳給 shell，不會把 API Key 交給模型執行的指令；不要用 `env=os.environ`

---

## 開發體驗：測試與除錯

### 零成本測試

展示如何使用 TestModel 和 FunctionModel 進行單元測試，完全不需要真實 API 調用。

參考完整範例：[zero_cost_testing.py](zero_cost_testing.py)

**三個核心測試場景：**

#### 1. TestModel - 自動模擬工具調用

```python
from pydantic_ai.models.test import TestModel

# TestModel 會自動返回工具調用的 JSON 結果
test_model = TestModel()
with agent.override(model=test_model):
    result = await agent.run('測試輸入', deps='玩家')
    # 輸出: {"tool1":"結果1","tool2":"結果2"}

    # 驗證工具註冊
    tools = test_model.last_model_request_parameters.function_tools
    print([t.name for t in tools])  # ['roll_dice', 'get_player_name', 'check_guess']

# 自定義輸出
custom_model = TestModel(custom_output_text="你贏了！🎉")
with agent.override(model=custom_model):
    result = await agent.run('測試')
    # 輸出: 你贏了！🎉
```

#### 2. FunctionModel - 完全控制回應

```python
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart

async def simulate_flow(messages, info):
    # 第一次調用：模擬工具調用
    if len(messages) == 1:
        return ModelResponse(parts=[
            ToolCallPart('get_player_name', {}),
            ToolCallPart('roll_dice', {})
        ])
    # 第二次調用：基於工具結果回應
    else:
        return ModelResponse(parts=[
            TextPart("骰子結果是 4！你猜對了！")
        ])

with agent.override(model=FunctionModel(simulate_flow)):
    result = await agent.run('擲骰子')
```

#### 3. 批次測試 - 零成本效能展示

```python
# 並行測試 10 個案例
test_cases = [("玩家1", "猜1"), ("玩家2", "猜2"), ...]

with agent.override(model=TestModel()):
    tasks = [agent.run(guess, deps=player) for player, guess in test_cases]
    results = await asyncio.gather(*tasks)

```

**測試優勢總結：**

- **TestModel**：自動模擬工具調用 + 自定義輸出
- **FunctionModel**：完全控制回應流程和工具調用
- **並行測試**：
- **零成本**：不消耗任何 API 額度
- **可預測**：結果一致，適合 CI/CD

### 模型切換

```python
# 字串格式：<provider>:<model>
agent = Agent('openai:gpt-4o')
# 注意：裸 openai: 前綴預設使用 Responses API；
# 若要使用 Chat Completions，請改寫成 openai-chat:gpt-4o
agent = Agent('anthropic:claude-sonnet-5-5')
agent = Agent('google:gemini-3-flash-preview')
agent = Agent('groq:llama-3.3-70b-versatile')
agent = Agent('mistral:mistral-large-latest')
agent = Agent('bedrock:us.anthropic.claude-haiku-5-5')

# 明確實例化（更多控制）
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key='...'))
agent = Agent(model)

# FallbackModel 自動切換
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.fallback import FallbackModel
fallback = FallbackModel(
    OpenAIChatModel('gpt-4o'),
    AnthropicModel('claude-sonnet-5-5')
)
agent = Agent(fallback)
```

**開發優勢**：

- **零測試成本**：TestModel 和 FunctionModel 不呼叫實際 API
- **模型無關**：支援 OpenAI、Anthropic、Google、Groq、Mistral、Bedrock、HuggingFace 等
- **測試隔離**：使用 `Agent.override()` 和 pytest fixture 管理測試環境
- **訊息追蹤**：`capture_run_messages()` 記錄完整對話過程

---

## 品質驗證：評估框架（Pydantic Evals）

單元測試問的是「給定輸入，程式有沒有回傳正確的輸出？」——確定性、非對即錯。但 LLM 是機率性的，回答品質往往是「程度」問題，沒辦法用 `assert ==` 一刀切。Pydantic Evals 就是為此而生：把「測試案例 + 評分標準」組織成資料集，對 AI 系統跑一輪實驗並產出評分報告。它和上面的零成本測試互補——TestModel 驗證「程式流程」對不對，Evals 驗證「回答品質」好不好。

參考 [evals_demo.py](evals_demo.py)（使用 NVIDIA NIM）

**核心做法**：

**Case / Dataset**：用 `Case`（單一測試案例：`inputs`、`expected_output`、`metadata`）組成 `Dataset`（一整批案例），把測試集中管理

**待評估的 Task**：被打分的對象就是一個普通的 pydantic-ai Agent；用 `async def answer_question(q)` 包起來交給 Evals 呼叫

**三類評估器（Evaluator）互補**：

- **IsInstance（內建）**：零成本的格式 / 型別檢查（輸出是不是 `str`）
- **ContainsExpected（自訂）**：繼承 `Evaluator` 自己寫的零成本確定性檢查（輸出有沒有包含標準答案）
- **LLMJudge**：用 LLM 當裁判，靠 `rubric` 評「切題、語氣」等主觀品質；`include_input=True` 讓裁判同時看到「問題」與「回答」

**評估器的兩種層級**：資料集層級評估器對「每個」case 都跑；Case 層級評估器（`Case(..., evaluators=[...])`）只在該案例跑

**為什麼開放題只能靠 LLMJudge**：事實題（如「台灣最高的山」）有標準答案，便宜的 `ContainsExpected` 就夠；開放題（如「寫一句歡迎詞」）沒有標準答案可比對，確定性檢查無能為力，只能用 LLMJudge 評主觀品質

**跑實驗、出報告**：`dataset.evaluate_sync(task)` 對每個 case 執行 Task 再套用所有評估器，回傳的 report 用 `report.print()` 印出每個案例的輸入、輸出與各評估器分數；report 物件還可序列化存檔，或送進 Logfire 做視覺化與版本比較

**會花幾次 LLM 呼叫**：每個 case 會打兩次 API——一次問 Agent 拿答案（Task）、一次給 LLMJudge 打分；`IsInstance`、`ContainsExpected` 是純程式檢查，不花 LLM。所以本範例一輪共 `3 case ×（1 Task + 1 Judge）= 6 次`。公式是「case 數 ×（1 + 用到幾個 LLM 評估器）」，疊越多 LLM 裁判、case 越多就越貴。

> **安裝與裁判提醒**：`pydantic-evals` 需單獨安裝（已列入 `requirements.txt`）。

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

### 成本與性能

- 每次執行都需要 LLM 調用，成本隨使用量增長
- 延遲通常 1-5 秒，無法達到毫秒級回應
- 建議使用快取（`@lru_cache`）和更便宜的模型（如 `gemini-3-flash-preview`）

### 技術限制

- 依賴第三方 AI 服務，難以保證 99.9% SLA
- 相同輸入可能產生不同輸出
- AI 決策過程不透明，除錯困難
- 高流量場景成本激增，受 API 速率限制

### 解決方案

```python
# 混合架構：簡單邏輯用傳統程式，複雜問題用 AI
def smart_processing(input_data):
    if is_simple_case(input_data):
        return traditional_processing(input_data)  # 毫秒級
    else:
        return ai_agent.run(input_data)  # 秒級但更智能

# 批量處理降低成本
async def batch_process(inputs: list[str]):
    results = await asyncio.gather(*[
        agent.run(text) for text in inputs
    ])
    return results
```

### 核心原則

**AI 是增強而非取代**：針對低頻但高價值的任務使用 AI，簡單問題用傳統程式處理。

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
