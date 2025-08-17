# Pydantic AI：讓 LLM 應用開發像 FastAPI 一樣簡單

* [Youtube Tutorial - Pydantic AI - AI 開發的 FastAPI 時刻！告別混亂解析 & 燒錢測試](https://youtu.be/mEwA5DeSFqU)

- **官方文檔**：[https://ai.pydantic.dev/](https://ai.pydantic.dev/)

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

## 工具的威力：骰子遊戲範例

傳統的 Function Calling 實現起來很繁瑣，但 Pydantic AI 讓它變得優雅：

以下範例都是一樣的, 只是模型不同

[tools_demo.py](tools_demo.py) - 基本骰子遊戲工具展示

[tools_demo_openrouter.py](tools_demo_openrouter.py) - 使用 OpenRouter 的工具範例

[tools_demo_github.py](tools_demo_github.py) - GitHub API 整合工具範例

**工具調用支援說明**：

**OpenAI 模型**：Function Calling 原生支援，表現最穩定（推薦）

**Google Gemini**：`gemini-2.0-flash-exp` 工具調用表現不夠穩定

**免費模型**：OpenRouter 不確定是否為免費額度, 導致對話都只有一次就結束

**核心特色**：

**工具定義簡單**：用 `@agent.tool` 裝飾器就能註冊函數為工具

**AI 自主決策**：模型根據用戶輸入自動判斷何時調用哪個工具

**文檔字符串重要**：函數的 docstring 成為工具描述，幫助 AI 理解工具用途

**參數自動驗證**：工具參數透過 Pydantic 自動驗證和轉換

---

## 生產級系統：銀行客服支援

這個完整的銀行客服範例展示了 Pydantic AI 如何整合所有核心功能，建構一個真正的生產級 AI 系統。

參考 [bank_support_agent_demo.py](bank_support_agent_demo.py)

**技術架構解析**：

**依賴注入**：使用 `SupportDependencies` 類別封裝客戶 ID 和 `DatabaseConn`，透過 `RunContext[SupportDependencies]` 型別安全地傳遞

**多層動態提示**：兩個系統提示函數自動組合 - `add_customer_name()` 查詢客戶姓名、`add_customer_status()` 識別 VIP 身份

**智能工具調用**：`customer_balance()` 工具讓 AI 主動查詢餘額，支援 `include_pending` 參數控制是否包含待處理交易

**結構化輸出**：`SupportOutput` 模型定義回應格式，包含 `support_advice`（建議）、`block_card`（封卡決策）和 `risk`（0-10 風險等級）

---

## 錯誤處理與重試機制

展示 Pydantic AI 的 ModelRetry 功能，包括智能處理不完整輸入和自動重試機制。

參考 [error_handling_demo.py](error_handling_demo.py)

**多層重試架構**：

**工具層級重試**：`@user_agent.tool(retries=2)` 針對 `get_user_by_name()` 設定重試次數，`@user_agent.tool(retries=3)` 針對 `validate_external_api()` 設定 API 重試

**Agent 全域重試**：`Agent(..., retries=2)` 設定整體對話的重試次數上限

**使用限制保護**：`UsageLimits(request_limit=10)` 防止無限重試消耗過多 token

**智能錯誤引導（學習機制）**：

**兩個字找三個字**：當查詢「張小」這種兩個字姓名時，系統會使用 `ModelRetry` 找到「張小華」並引導 AI 使用完整姓名重試

**不存在用戶處理**：查詢「小明」等不存在用戶時，直接返回 None 並在 message 中說明用戶不存在

**模擬 API 失敗**：`validate_api_service()` 前兩次故意失敗，第三次成功，展示自動重試直到成功的機制

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
agent = Agent('anthropic:claude-3-5-sonnet-latest')
agent = Agent('google:gemini-1.5-flash')
agent = Agent('groq:llama-3.3-70b-versatile')
agent = Agent('mistral:mistral-large-latest')
agent = Agent('bedrock:anthropic.claude-3-sonnet-20240229-v1:0')

# 明確實例化（更多控制）
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
model = OpenAIModel('gpt-4o', provider=OpenAIProvider(api_key='...'))
agent = Agent(model)

# FallbackModel 自動切換
from pydantic_ai.models.fallback import FallbackModel
fallback = FallbackModel(
    OpenAIModel('gpt-4o'),
    AnthropicModel('claude-3-5-sonnet-latest')
)
agent = Agent(fallback)
```

**開發優勢**：

- **零測試成本**：TestModel 和 FunctionModel 不呼叫實際 API
- **模型無關**：支援 OpenAI、Anthropic、Google、Groq、Mistral、Bedrock、HuggingFace 等
- **測試隔離**：使用 `Agent.override()` 和 pytest fixture 管理測試環境
- **訊息追蹤**：`capture_run_messages()` 記錄完整對話過程

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
- 建議使用快取（`@lru_cache`）和更便宜的模型（如 `gemini-2.0-flash-exp`）

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
