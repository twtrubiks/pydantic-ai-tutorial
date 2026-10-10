"""
Claude Prompt Caching 範例（聚焦：快取省錢 + 找出快取為什麼沒命中）

主題：
    A. Prompt Caching：同一份長員工手冊連問兩題，對比「沒開快取」與「開快取」的 token 帳單與美金
    B. Cache Diagnostics：故意在對話中途多加一個工具把快取打破，讓 Anthropic 直接告訴我們原因
    C. cost_limit：用美金設預算上限，對比「回應後才檢查」與「送出前先估算」

為什麼是 Claude 專屬：
    OpenAI、Gemini 會自動快取；Anthropic 預設什麼都不快取，要在請求裡標記「快取斷點」才會生效。
    Pydantic AI 的 Caching() capability 一行就幫你把斷點標好（工具定義、instructions、對話各一個）。
    Cache diagnostics 則是 Claude API 才有的功能：快取沒命中時，回應裡會直接寫出原因
    （model_changed / system_changed / tools_changed / messages_changed）。

計價（Anthropic 5 分鐘快取）：
    寫入快取 = 1.25 倍輸入價；讀取快取 = 0.1 倍輸入價（Sonnet／Opus 5.5 為 0.05 倍、Fable 5.1 為 0.025 倍）
    → 只要被讀回一次就回本，對話越長越省。

文檔參考：
    - Caching：https://pydantic.dev/docs/ai/capabilities/caching/
    - Anthropic prompt caching：https://pydantic.dev/docs/ai/models/anthropic/#prompt-caching
    - Anthropic cache diagnostics：https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics

執行前：
    1. 前往 https://platform.claude.com/settings/keys 取得 API Key（格式 sk-ant-...）
    2. 在 .env 檔案中設置 ANTHROPIC_API_KEY=sk-ant-...
    3. 需要 pydantic-ai >= 2.55（anthropic_cache_diagnostics 在 2.55 才加入）
    4. python claude_cache_demo.py
"""

import asyncio
import os
from datetime import date
from decimal import Decimal

from dotenv import load_dotenv
from pydantic_ai import Agent, UsageLimitExceeded, UsageLimits
from pydantic_ai.capabilities import Caching
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider

# 載入 .env 檔案中的環境變數
load_dotenv()

# ========== 模型設定 ==========

# 設置 Anthropic API Key（前往 https://platform.claude.com/settings/keys 取得，格式 sk-ant-...）
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Haiku 5.5：最便宜，適合反覆執行的教學範例；想換更強的模型可改成 claude-sonnet-5-5
MODEL_NAME = "claude-haiku-5-5"

# Haiku 5.5 預設會開 adaptive thinking，這裡關掉：一來省 token，二來讓帳單只反映快取的影響。
# max_tokens 不設的話 Pydantic AI 會送出模型上限（128,000），回答只要一兩句，設小一點即可。
BASE_SETTINGS = AnthropicModelSettings(thinking=False, max_tokens=1024)

# 5 分鐘快取的計價倍率（相對於一般輸入價）
# 寫入倍率各模型相同；讀取倍率依模型不同，沒列出的模型都是 0.1 倍
# 參考：https://platform.claude.com/docs/en/about-claude/pricing
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIERS = {
    "claude-haiku-5-5": 0.1,
    "claude-sonnet-5-5": 0.05,
    "claude-opus-5-5": 0.05,
    "claude-fable-5-1": 0.025,
}
CACHE_READ_MULTIPLIER = CACHE_READ_MULTIPLIERS.get(MODEL_NAME, 0.1)


def build_model() -> AnthropicModel:
    """建立 Claude 模型（延後到確認有 API Key 後才建，避免 import 即報錯）"""
    return AnthropicModel(
        MODEL_NAME,
        provider=AnthropicProvider(api_key=ANTHROPIC_API_KEY),
    )


# ========== 共用資料：一份夠長的員工手冊 ==========

POLICIES = """\
# 星河科技員工手冊

## 一、特別休假
- 到職滿 6 個月未滿 1 年：3 天。
- 滿 1 年未滿 2 年：7 天；滿 2 年未滿 3 年：10 天；滿 3 年未滿 5 年：每年 14 天。
- 滿 5 年未滿 10 年：每年 15 天；滿 10 年以上，每滿 1 年加給 1 天，最多 30 天。
- 年度未休完的特休，可遞延至次年度使用，或於年底依日薪折算工資。

## 二、加班
- 平日延長工時：前 2 小時按時薪 1.34 倍、第 3、4 小時按 1.67 倍計給。
- 休息日（週六）出勤：前 2 小時 1.34 倍，第 3 至 8 小時 1.67 倍，第 9 小時起 2.67 倍。
- 國定假日出勤：當日工資加倍發給，或經員工同意改為補休。
- 加班需事先在 HR 系統提出申請，經直屬主管核准。

## 三、遠端工作
- 每週最多 2 天遠端，需在前一週五前登記。
- 遠端日須於 10:00～16:00 保持線上，並可即時回覆訊息。
- 新進員工試用期（前 3 個月）不適用遠端工作。

## 四、出差與報帳
- 國內出差日支費 800 元，國外依地區另訂。
- 高鐵、台鐵可實報實銷；計程車需附收據並註明事由。
- 報帳須於費用發生後 30 天內提出，逾期不受理。

## 五、教育訓練補助
- 每人每年 20,000 元，可用於課程、證照考試與技術書籍。
- 單筆超過 5,000 元需事先申請；補助後一年內離職須按比例歸還。

## 六、設備
- 新進員工可選 MacBook Pro 或 ThinkPad，每 3 年可申請汰換。
- 居家辦公椅補助上限 6,000 元，限到職滿 1 年者申請一次。
"""

# Anthropic 有「最小可快取長度」，前綴太短就不會被快取：Haiku／Sonnet／Opus 5.5 與 Fable 5.1 為 512 token，
# 較舊的模型為 1,024～4,096 token（見 https://platform.claude.com/docs/en/build-with-claude/prompt-caching）。
# 真實情境的手冊、程式碼庫、長篇文件都很容易超過門檻；這裡用自動產生的據點通訊錄把長度撐過去。
CITIES = [
    ("台北", "TP", "02"),
    ("新北", "NT", "02"),
    ("桃園", "TY", "03"),
    ("新竹", "HC", "03"),
    ("台中", "TC", "04"),
    ("彰化", "CH", "04"),
    ("嘉義", "CY", "05"),
    ("台南", "TN", "06"),
    ("高雄", "KH", "07"),
    ("屏東", "PT", "08"),
    ("宜蘭", "IL", "03"),
    ("花蓮", "HL", "03"),
]


def build_handbook() -> str:
    """組出完整員工手冊：規章 + 附錄據點通訊錄"""
    rows = ["| 城市 | 據點代碼 | 地址 | 電話 | HR 分機 |", "|---|---|---|---|---|"]
    for i in range(1, 151):
        city, code, area = CITIES[i % len(CITIES)]
        rows.append(
            f"| {city} | {code}-{i:03d} | {city}市中正路 {i * 7} 號 {i % 12 + 1} 樓 "
            f"| ({area}) {2000 + i * 13:04d}-{1000 + i * 37 % 9000:04d} | {100 + i} |"
        )
    return POLICIES + "\n## 附錄：全台據點通訊錄\n" + "\n".join(rows)


INSTRUCTIONS = (
    "你是星河科技的人資助理，只根據下面的員工手冊回答，"
    "請用繁體中文、一兩句話作答。\n\n" + build_handbook()
)


# ========== 共用工具函式 ==========


def bill(result) -> dict:
    """整理一次 run 的 token 帳單，並換算成「等價輸入 token」方便比較成本"""
    usage = result.usage
    # input_tokens 已包含快取讀取與寫入，扣掉兩者才是一般計價的部分
    uncached = usage.input_tokens - usage.cache_read_tokens - usage.cache_write_tokens
    equivalent = (
        uncached
        + usage.cache_write_tokens * CACHE_WRITE_MULTIPLIER
        + usage.cache_read_tokens * CACHE_READ_MULTIPLIER
    )
    return {
        "input": usage.input_tokens,
        "uncached": uncached,
        "write": usage.cache_write_tokens,
        "read": usage.cache_read_tokens,
        "hit_ratio": usage.cache_hit_ratio,
        "equivalent": equivalent,
        # V2 內建費用估算（用 genai-prices 的價格表算，含輸出 token）；查不到價格時是 None
        "cost": usage.cost,
    }


def format_usd(cost: Decimal | None) -> str:
    """把美金金額印成固定小數位；None 代表 genai-prices 查不到這個模型的價格"""
    return "（查無價格）" if cost is None else f"${cost:.6f}"


def print_bill(label: str, b: dict):
    """印出單次 run 的 token 帳單"""
    print(f"\n  📊 {label}")
    print(f"     輸入 token：{b['input']}")
    print(f"       ├─ 一般計價（1x）：{b['uncached']}")
    print(f"       ├─ 寫入快取（1.25x）：{b['write']}")
    print(f"       └─ 讀取快取（{CACHE_READ_MULTIPLIER:g}x）：{b['read']}")
    print(f"     快取命中率：{b['hit_ratio']:.0%}")
    print(f"     等價輸入 token：{b['equivalent']:.0f}")
    print(f"     估算費用（usage.cost）：{format_usd(b['cost'])}")


# ========== 主題 A：Prompt Caching（同一段對話，開快取 vs 沒開）==========


async def demo_prompt_caching(model: AnthropicModel):
    """同樣兩輪對話跑兩次：沒開快取 vs 開快取，比較等價輸入 token"""
    print("\n💾 主題 A：Prompt Caching（開快取 vs 沒開）")
    print("=" * 60)

    questions = ["新進員工第一年有幾天特休？", "那年資滿五年之後呢？"]

    # 兩個 agent 只差在有沒有 Caching()
    agents = {
        "沒開快取": Agent(model, instructions=INSTRUCTIONS, model_settings=BASE_SETTINGS),
        "開快取（Caching()）": Agent(
            model,
            instructions=INSTRUCTIONS,
            model_settings=BASE_SETTINGS,
            capabilities=[Caching()],  # 等同 model_settings={'cache': True}
        ),
    }

    totals = {}
    costs = {}
    for label, agent in agents.items():
        print(f"\n📌 {label}")
        print("-" * 40)
        history = []
        total = 0.0
        cost = Decimal(0)
        for turn, question in enumerate(questions, start=1):
            result = await agent.run(question, message_history=history)
            history = result.all_messages()  # 帶著完整歷史繼續問，前綴才會一樣
            b = bill(result)
            total += b["equivalent"]
            cost += b["cost"] or 0
            print(f"\n  👤 第 {turn} 輪：{question}")
            print(f"  🤖 {result.output.strip()}")
            print_bill(f"第 {turn} 輪帳單", b)

            if label != "沒開快取" and turn == 1 and b["write"] == 0 and b["read"] == 0:
                print("  ⚠️  沒有寫入也沒有讀取快取：前綴可能低於模型的最小可快取長度")
        totals[label] = total
        costs[label] = cost

    no_cache, with_cache = totals.values()
    print("\n  💰 對比結論（兩輪合計的等價輸入 token）：")
    print(f"     沒開快取：{no_cache:.0f}")
    print(f"     開快取：  {with_cache:.0f}")
    if no_cache:
        print(f"     ⇒ 省下約 {1 - with_cache / no_cache:.0%}；對話輪數越多，省得越多。")
    print(f"     （第 1 輪多付 25% 寫入費，第 2 輪起手冊只收 {CACHE_READ_MULTIPLIER * 100:g}%）")

    # 等價 token 只算輸入；usage.cost 直接換成美金，連輸出 token 一起算
    no_cache_usd, with_cache_usd = costs.values()
    print("\n  💵 換算成美金（usage.cost 兩輪合計，含輸出 token）：")
    print(f"     沒開快取：{format_usd(no_cache_usd)}")
    print(f"     開快取：  {format_usd(with_cache_usd)}")
    if no_cache_usd:
        print(f"     ⇒ 省下約 {1 - with_cache_usd / no_cache_usd:.0%}")


# ========== 主題 B：Cache Diagnostics（快取為什麼沒命中？）==========


def get_today() -> str:
    """取得今天的日期（ISO 格式）"""
    # 只回傳到「日」：instructions 或工具結果放到秒的時間戳，是快取失效的常見元兇
    return date.today().isoformat()


def print_diagnostics(result):
    """印出這一輪每個模型回應的 cache diagnostics"""
    for message in result.new_messages():
        if not isinstance(message, ModelResponse):
            continue
        details = message.provider_details or {}
        if "cache_diagnostics" not in details:
            # key 不存在 = Anthropic 比對後沒發現前綴有分歧
            print("  🔎 diagnostics：前綴與上一個請求一致，沒有分歧 ✅")
            continue
        reason = details["cache_diagnostics"].get("cache_miss_reason")
        if reason is None:
            print("  🔎 diagnostics：Anthropic 還沒比對完（cache_miss_reason=None）")
        else:
            missed = reason.get("cache_missed_input_tokens")
            print(f"  🔎 diagnostics：快取沒命中 ❌ 原因 = {reason['type']}，少讀了 {missed} 個 token")


async def demo_cache_diagnostics(model: AnthropicModel):
    """先正常追問一次（前綴一致），再換一個多了工具的 agent 追問（前綴被打破）"""
    print("\n\n🩺 主題 B：Cache Diagnostics（快取為什麼沒命中？）")
    print("=" * 60)

    # anthropic_cache_diagnostics 不屬於 anthropic_cache* 系列，不會蓋掉 Caching() 的統一設定
    settings = AnthropicModelSettings(**BASE_SETTINGS, anthropic_cache_diagnostics=True)

    # 同樣的 instructions，差別只在第二個多了一個工具
    stable_agent = Agent(
        model, instructions=INSTRUCTIONS, model_settings=settings, capabilities=[Caching()]
    )
    changed_agent = Agent(
        model,
        instructions=INSTRUCTIONS,
        model_settings=settings,
        capabilities=[Caching()],
        tools=[get_today],
    )

    # 第 1 輪：沒有上一個回應可比，這一輪只是「報名」讓下一輪有比較基準
    print("\n📌 第 1 輪：建立比較基準")
    print("-" * 40)
    result = await stable_agent.run("週六來加班，前兩小時怎麼算？")
    print(f"  🤖 {result.output.strip()}")
    history = result.all_messages()

    # 第 2 輪：同一個 agent 追問，前綴（工具 + instructions + 先前對話）完全一致
    print("\n📌 第 2 輪：同一個 agent 追問（前綴不變）")
    print("-" * 40)
    result = await stable_agent.run("那第三小時開始呢？", message_history=history)
    print(f"  🤖 {result.output.strip()}")
    print_diagnostics(result)
    print(f"  📊 快取讀取：{result.usage.cache_read_tokens} token")
    history = result.all_messages()

    # 第 3 輪：換成多了 get_today 工具的 agent。Anthropic 前綴順序是「工具 → system → 對話」，
    # 工具一變，後面全部都得重算，等於整份手冊和對話重新付全額
    print("\n📌 第 3 輪：中途多加一個工具（前綴被打破）")
    print("-" * 40)
    result = await changed_agent.run("國定假日來上班呢？", message_history=history)
    print(f"  🤖 {result.output.strip()}")
    print_diagnostics(result)
    print(f"  📊 快取讀取：{result.usage.cache_read_tokens} token，寫入：{result.usage.cache_write_tokens} token")


# ========== 主題 C：cost_limit（直接用美金設預算）==========


async def demo_cost_limit(model: AnthropicModel):
    """預算故意設得比一次請求還低，對比「回應後才檢查」與「送出前先估算」"""
    print("\n\n💵 主題 C：cost_limit（直接用美金設預算）")
    print("=" * 60)

    # 不開快取：每次都要付整份手冊（約 8k token）的錢，大約 $0.0008，預算只給一半多一點
    agent = Agent(model, instructions=INSTRUCTIONS, model_settings=BASE_SETTINGS)
    budget = Decimal("0.0005")

    cases = {
        "只設 cost_limit（回應後才檢查）": UsageLimits(cost_limit=budget),
        # 先呼叫 Anthropic 的 token 計數 API（免費）估算輸入費用，超過就不送出
        "加上 count_tokens_before_request=True（送出前先估算）": UsageLimits(
            cost_limit=budget, count_tokens_before_request=True
        ),
    }
    for label, limits in cases.items():
        print(f"\n📌 {label}，預算 ${budget}")
        print("-" * 40)
        try:
            result = await agent.run("出差日支費多少？", usage_limits=limits)
            print(f"  🤖 {result.output.strip()}（{format_usd(result.usage.cost)}，在預算內）")
        except UsageLimitExceeded as e:
            print(f"  ⛔ UsageLimitExceeded：{str(e).split('. Consider')[0]}")

    print("\n  💡 第一種：請求已經送出、也已經計費，回應回來才發現超支")
    print("     第二種：訊息是 'The next request would exceed'，請求根本沒送出")


# ========== 說明 ==========


def explain():
    """整理重點"""
    print("\n\n" + "=" * 60)
    print("💡 重點整理")
    print("=" * 60)

    print("\n1. Prompt Caching（Anthropic 要手動開）：")
    print("   • capabilities=[Caching()] 或 model_settings={'cache': True}，一行開啟")
    print("   • 會在工具定義、instructions、對話尾端放快取斷點（最多 4 個，Pydantic AI 會自動管理）")
    print("   • 一次性問答很多、共用長 instructions 時，用 Caching(messages=False) 只快取固定前綴")
    print("   • 對話常隔超過 5 分鐘才繼續，可考慮 Caching(retention='1h')（寫入 2 倍，要讀回兩次才回本）")
    print("   • 看效果：usage.cache_write_tokens / cache_read_tokens / cache_hit_ratio")

    print("\n2. Cache Diagnostics（Claude API 專屬）：")
    print("   • AnthropicModelSettings(anthropic_cache_diagnostics=True) 開啟，免費、不影響快取")
    print("   • 結果在 ModelResponse.provider_details['cache_diagnostics']")
    print("   • 原因類型：model_changed / system_changed / tools_changed / messages_changed")
    print("   • 只在 Claude API 有效，Bedrock / Vertex AI / Foundry 會忽略這個設定")

    print("\n3. 常見的快取殺手（對話中途改了前綴）：")
    print("   • instructions 裡放到分鐘、秒的時間戳 → 改成只放日期")
    print("   • 對話中途增減、重新排序工具（例如動態開關 MCP server 的工具）")
    print("   • 對話中途換模型或 provider（每個模型的快取各自獨立）")
    print("   • Claude 5.x 會把 thinking block 綁在產生它的前綴上：開著 thinking 時改前綴，")
    print("     不只是快取沒命中，還可能直接回 400；本範例關掉 thinking 才避開這個問題")

    print("\n4. 費用追蹤與預算（V2 內建）：")
    print("   • result.usage.cost：用 genai-prices 價格表估算的美金（Decimal），快取讀寫、輸出都算進去")
    print("   • 查不到價格的模型（例如 NVIDIA NIM）會是 None，不是 0")
    print("   • UsageLimits(cost_limit=Decimal('0.01'))：超過就丟 UsageLimitExceeded")
    print("   • 預設在回應後才檢查（已經計費）；加 count_tokens_before_request=True 可在送出前擋下")
    print("   • 只是估算，不能當帳單保證，要搭配 request_limit 或供應商後台的花費上限")


# ========== 主程式 ==========


async def main():
    """主程式入口"""
    if not ANTHROPIC_API_KEY:
        print("⚠️  警告：請在 .env 檔案中設置 ANTHROPIC_API_KEY")
        print("💡 前往 https://platform.claude.com/settings/keys 取得 API Key（格式 sk-ant-...）")
        return

    try:
        # 確認有 Key 後才建立模型
        model = build_model()
        print(f"🚀 使用 Claude 模型：{MODEL_NAME}")

        # 主題 A：快取省多少
        await demo_prompt_caching(model)

        # 主題 B：快取為什麼沒命中
        await demo_cache_diagnostics(model)

        # 主題 C：用美金設預算
        await demo_cost_limit(model)

        # 重點整理
        explain()

    except Exception as e:
        print(f"\n❌ 執行出錯：{e}")
        print("💡 請檢查 ANTHROPIC_API_KEY 是否正確設置在 .env 檔案中，以及模型名稱是否可用")


if __name__ == "__main__":
    # 執行異步主程式
    asyncio.run(main())
