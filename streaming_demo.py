"""
串流輸出（Streaming）範例（聚焦：即時顯示 AI 的生成過程）

這個範例展示了 Pydantic AI 的兩種串流模式：
1. 串流純文字 / Markdown（邊生成邊渲染，像打字機效果）
2. 串流「結構化資料」（一邊生成、一邊驗證，即時更新表格）

對應官方範例（本檔案改寫自這兩支）：
- 串流 Markdown：examples/stream_markdown.py
- 串流結構化資料：examples/stream_whales.py

核心 API：
- agent.run_stream(prompt)        → 開啟串流（async context manager）
- result.stream_output()          → 逐步吐出「累積到目前」的輸出（文字或結構化）
- result.stream_text(delta=True)  → 只想拿「每次新增的片段」時可改用這個

文檔參考：https://ai.pydantic.dev/output/#streaming-structured-output
"""

import asyncio
import os
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from rich.console import Console, ConsoleOptions, RenderResult
from rich.live import Live
from rich.markdown import CodeBlock, Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from typing_extensions import NotRequired, TypedDict

# 載入 .env 檔案中的環境變數
load_dotenv()

# ========== 資料結構定義（結構化串流用）==========


class Whale(TypedDict):
    """單一鯨魚的資料結構（給場景 2 的結構化串流使用）

    用 NotRequired 標註的欄位，模型在串流途中還沒生成時會「先缺著」，
    這正是觀察「部分驗證（partial validation）」的好材料。
    """

    name: str
    length: Annotated[float, Field(description="成年鯨魚的平均長度（公尺）")]
    weight: NotRequired[
        Annotated[float, Field(description="成年鯨魚的平均體重（公斤）", ge=50)]
    ]
    ocean: NotRequired[Annotated[str, Field(description="主要棲息海域")]]
    description: NotRequired[Annotated[str, Field(description="簡短介紹")]]


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

# ========== Agent 建立 ==========

# 場景 1（文字串流）用：未指定 output_type，預設輸出為 str
markdown_agent = Agent(model)

# 場景 2（結構化串流）用：output_type=list[Whale]，串流途中會持續做部分驗證
whale_agent = Agent(model, output_type=list[Whale])


# ========== 場景 1：串流 Markdown 文字 ==========


def prettier_code_blocks():
    """讓 rich 的程式碼區塊更漂亮、更好複製。

    取自官方範例 stream_markdown.py
    （原始來源：https://github.com/samuelcolvin/aicli）
    """

    class SimpleCodeBlock(CodeBlock):
        def __rich_console__(
            self, console: Console, options: ConsoleOptions
        ) -> RenderResult:
            code = str(self.text).rstrip()
            yield Text(self.lexer_name, style="dim")
            yield Syntax(
                code,
                self.lexer_name,
                theme=self.theme,
                background_color="default",
                word_wrap=True,
            )
            yield Text(f"/{self.lexer_name}", style="dim")

    Markdown.elements["fence"] = SimpleCodeBlock


async def stream_markdown_demo():
    """場景 1：串流 Markdown 文字（邊生成邊即時渲染，像打字機）"""
    print("\n📌 場景 1：串流 Markdown 文字")
    print("-" * 40)

    prettier_code_blocks()
    console = Console()
    prompt = "用 Python 寫一個使用 Pydantic 的簡短範例，並用 Markdown 格式簡要說明。"
    console.log(f"提問：{prompt}", style="cyan")

    # run_stream 開啟串流；stream_output() 每次吐出的是「累積到目前」的完整文字，
    # 所以每收到一段就用最新內容整個重畫 Markdown，畫面就會像逐字長出來。
    with Live("", console=console, vertical_overflow="visible") as live:
        async with markdown_agent.run_stream(prompt) as result:
            async for message in result.stream_output():
                live.update(Markdown(message))

    # 注意：usage 要等串流結束才會是完整數字
    console.log(result.usage)


# ========== 場景 2：串流結構化資料 ==========


def build_whale_table(whales) -> Table:
    """根據「目前已串流到的」鯨魚資料即時建立表格。

    串流途中欄位可能還沒生成，所以一律用 .get() 取值並提供預設 '…'，
    避免讀到尚未出現的欄位而 KeyError。
    """
    table = Table(
        title="🐳 鯨魚物種",
        caption="從 NVIDIA 模型串流而來的結構化輸出",
        width=120,
    )
    table.add_column("ID", justify="right")
    table.add_column("名稱")
    table.add_column("平均長度 (m)", justify="right")
    table.add_column("平均體重 (kg)", justify="right")
    table.add_column("海域")
    table.add_column("簡介")

    for wid, whale in enumerate(whales, start=1):
        table.add_row(
            str(wid),
            whale.get("name") or "…",
            f"{length:0.0f}" if (length := whale.get("length")) else "…",
            f"{weight:0.0f}" if (weight := whale.get("weight")) else "…",
            whale.get("ocean") or "…",
            whale.get("description") or "…",
        )
    return table


async def stream_whales_demo():
    """場景 2：串流結構化資料（一邊生成、一邊驗證，表格即時長出來）"""
    print("\n\n📌 場景 2：串流結構化資料（即時更新表格）")
    print("-" * 40)

    console = Console()
    # 預先墊 12 行空白，避免表格冒出來時畫面跳動
    with Live("\n" * 12, console=console) as live:
        console.print("請求資料中…", style="cyan")
        prompt = "請提供 5 種鯨魚的詳細資料（名稱、平均長度、平均體重、棲息海域、簡短介紹）。"
        async with whale_agent.run_stream(prompt) as result:
            console.print("回應：", style="green")
            # debounce_by：每 0.01 秒最多更新一次畫面，避免刷新過於頻繁而閃爍
            async for whales in result.stream_output(debounce_by=0.01):
                live.update(build_whale_table(whales))

    console.log(result.usage)


# ========== 說明 ==========


def explain_streaming():
    """說明串流機制"""
    print("\n\n" + "=" * 60)
    print("💡 Streaming 串流輸出機制說明")
    print("=" * 60)

    print("\n1. 為什麼要串流？")
    print("   • 不必等模型「整段講完」才看到結果，第一個字出來就能顯示")
    print("   • 對長文、即時對話、進度感很有幫助")

    print("\n2. 兩種串流模式：")
    print("   • 文字串流（場景 1）：適合文章、Markdown、程式碼")
    print("   • 結構化串流（場景 2）：output_type 為結構時，邊生成邊做『部分驗證』")

    print("\n3. 關鍵 API：")
    print("   • async with agent.run_stream(prompt) as result:")
    print("   • result.stream_output()         → 每次給『累積到目前』的完整輸出")
    print("   • result.stream_output(debounce_by=0.01) → 控制畫面更新頻率")
    print("   • result.stream_text(delta=True) → 只要『這次新增的片段』時改用它")

    print("\n4. 小提醒：")
    print("   • usage（用量）要等串流結束才會是完整數字")
    print("   • 結構化串流途中欄位可能還沒齊，取值請用 .get() 並給預設值")


# ========== 主程式 ==========


async def main():
    """主程式入口"""
    try:
        # 場景 1：串流 Markdown 文字
        await stream_markdown_demo()

        # 場景 2：串流結構化資料
        # await stream_whales_demo()

        # 說明串流機制
        explain_streaming()

    except Exception as e:
        print(f"\n❌ 執行出錯：{e}")
        print("💡 請檢查 NVIDIA_API_KEY 是否正確設置在 .env 檔案中")


if __name__ == "__main__":
    # 執行異步主程式
    asyncio.run(main())
