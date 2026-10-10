"""
PydanticAI 評估框架示範 - Pydantic Evals (使用 NVIDIA NIM)
展示如何系統化地「驗證 AI 回答品質」

單元測試問：「給定輸入，程式是否回傳正確的輸出？」（確定性、非對即錯）
但 LLM 是機率性的，回答品質往往是「程度」問題。Pydantic Evals 就是為此而生：
把「測試案例 + 評分標準」組織成資料集，對 AI 系統跑一輪實驗並產出報告。

這個範例展示了：
1. Case / Dataset - 把測試案例組成資料集
2. 待評估的 Task - 一個真實的 pydantic-ai Agent
3. 三類評估器（Evaluator）的互補：
   - IsInstance（內建）：零成本的格式 / 型別檢查
   - ContainsExpected（自訂）：零成本的確定性檢查（答案是否包含標準答案）
   - LLMJudge：用 LLM 當裁判，評估「切題、語氣」等主觀品質
4. 為什麼開放式問題只能靠 LLMJudge - 沒有標準答案可比對
5. evaluate_sync() 跑實驗並用 report.print() 產出評分報告

Pydantic Evals 文檔：https://ai.pydantic.dev/evals/
"""

import os

from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, IsInstance, LLMJudge

# 載入 .env 檔案中的環境變數
load_dotenv()

# 設置 NVIDIA API Key
NVIDIA_API_KEY = os.getenv('NVIDIA_API_KEY')
model_name = "nvidia/nemotron-3-super-120b-a12b"

# 檢查 API Key
if not NVIDIA_API_KEY:
    print("⚠️ 警告：請在 .env 檔案中設置 NVIDIA_API_KEY")
    print("💡 執行步驟：")
    print("   1. 前往 https://build.nvidia.com/ 註冊並獲取 API Key")
    print("   2. 在 .env 檔案中添加：NVIDIA_API_KEY=nvapi-...")
    exit(1)

# 建立 AI 模型（透過 OpenAIProvider 指向 NVIDIA NIM 的 OpenAI 相容端點）
model = OpenAIChatModel(
    model_name,
    provider=OpenAIProvider(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    ),
)


# ========== 1. 待評估的 Task：一個台灣在地小幫手 Agent ==========

# 這就是我們要「打分數」的對象。它是一個普通的 pydantic-ai Agent。
qa_agent = Agent(
    model,
    system_prompt="你是台灣在地小幫手，請用繁體中文、友善有禮地簡短回答問題。",
)


async def answer_question(question: str) -> str:
    """Task 函式：接收問題、回傳 Agent 的回答。Evals 會對它的輸出評分。"""
    result = await qa_agent.run(question)
    return result.output


# ========== 2. 自訂評估器：零成本的確定性檢查 ==========

class ContainsExpected(Evaluator[str, str]):
    """檢查模型輸出是否包含標準答案（純字串比對，不需呼叫 LLM，零成本）

    適合有明確標準答案的事實題；開放式問題沒有 expected_output，
    就改用下方的 LLMJudge 來評分。
    """

    def evaluate(self, ctx: EvaluatorContext[str, str]) -> float:
        if ctx.expected_output and ctx.expected_output in ctx.output:
            return 1.0  # 命中標準答案
        return 0.0


# ========== 3. 建立資料集（Dataset）與測試案例（Case）==========

dataset = Dataset(
    name="taiwan_qa",
    cases=[
        # 事實題：有明確標準答案 → 用確定性評估器（ContainsExpected）就夠了，便宜又準
        Case(
            name="最高的山",
            inputs="台灣最高的山是哪座？",
            expected_output="玉山",
            metadata={"type": "fact"},
            evaluators=[ContainsExpected()],  # case 專屬評估器：只在這個案例跑
        ),
        Case(
            name="101樓層",
            inputs="台北 101 大樓地上有幾層樓？",
            expected_output="101",
            metadata={"type": "fact"},
            evaluators=[ContainsExpected()],
        ),
        # 開放題：沒有標準答案 → 確定性比對無能為力，只能靠 LLMJudge 評主觀品質
        Case(
            name="歡迎詞",
            inputs="幫我用一句話歡迎新朋友加入我們的社群。",
            metadata={"type": "open"},
        ),
    ],
    # 資料集層級的評估器：對「每一個」case 都會執行
    evaluators=[
        # 內建評估器：檢查輸出是不是字串（格式 / 型別檢查，零成本）
        IsInstance(type_name="str"),
        # LLM 裁判：用 LLM 評估主觀品質。include_input=True 讓裁判同時看到「問題」與「回答」
        # 這裡指定用同一個 NVIDIA NIM 模型當裁判（預設會用 OpenAI，需另設金鑰）
        LLMJudge(
            rubric="回答必須切題、誠實，且使用繁體中文，語氣友善有禮。",
            include_input=True,
            model=model,
        ),
    ],
)


# ========== 4. 執行實驗並產出報告 ==========

def main():
    """跑一輪評估實驗：對資料集裡每個 case 執行 Task，再用評估器打分。"""
    print("📊 PydanticAI 評估框架展示 - Pydantic Evals (NVIDIA NIM)")
    print(f"🚀 待評估模型：{model_name}")
    print("⏳ 正在對 3 個案例執行 Task 並評分（含 LLM 裁判，需呼叫 API）...\n")

    # evaluate_sync 會對所有 case 執行 answer_question，並套用所有評估器，回傳報告物件
    # （task 是 async 函式也沒問題，evaluate_sync 會自行處理）
    report = dataset.evaluate_sync(answer_question)

    # 印出評分報告：每個 case 的輸入、輸出、各評估器分數與 assertion 結果
    report.print(include_input=True, include_output=True, include_durations=False)

    print("\n" + "=" * 60)
    print("💡 重點回顧:")
    print("   - 事實題（有標準答案）：ContainsExpected 這種零成本的確定性檢查就夠了")
    print("   - 開放題（無標準答案）：確定性檢查無能為力，需要 LLMJudge 評主觀品質")
    print("   - IsInstance / 自訂評估器是『便宜的程式檢查』，LLMJudge 是『較貴的 LLM 評分』")
    print("   - 資料集層級評估器套用到每個 case；Case 層級評估器只套用到該 case")
    print("   - report 物件可序列化存檔，或送進 Logfire 做視覺化與版本比較")


if __name__ == "__main__":
    main()
