"""
網頁聊天介面範例：agent.to_web() 一行把 agent 變成瀏覽器聊天室

直接沿用 tool_approval_demo.py 的銀行轉帳 agent（同樣的工具、同樣的核准規則），在網頁上可以看到：
    1. 串流回覆，以及每次工具呼叫的參數與回傳值
    2. 需要核准的工具會變成「核准 / 拒絕」按鈕（終端機的 y/N 變成按鈕）
    3. 切換模型（有設定 GOOGLE_API_KEY 才會多出 Gemini）

注意：
    - 只適合本機開發、除錯；正式產品請用 UI Event Stream（Vercel AI / AG-UI）接自己的前端
    - 預設只接受 localhost 或 IP 的 Host（防 DNS rebinding）；核准按鈕不是權限控管，不要對外公開
    - 聊天介面的 HTML 第一次會從 CDN 下載，之後快取在本機

文檔參考：https://pydantic.dev/docs/ai/guides/web/

執行前：
    1. 在 .env 檔案中設置 ANTHROPIC_API_KEY（GOOGLE_API_KEY 選填）
    2. python web_chat_demo.py
    3. 瀏覽器打開 http://127.0.0.1:7932
"""

import os

import uvicorn
from dotenv import load_dotenv
from pydantic_ai import DeferredToolRequests
from pydantic_ai.models.google import GoogleModel

from tool_approval_demo import MODEL_NAME, TRANSFER_LIMIT, Bank, build_agent, build_model

# 載入 .env 檔案中的環境變數
load_dotenv()

HOST = "127.0.0.1"
PORT = 7932


def build_app():
    """建立 Starlette app：agent 預設用 Claude Haiku，有 Gemini key 就加進模型選單"""
    haiku = build_model()
    # output_type 一定要加 DeferredToolRequests：網頁的核准按鈕走的是「暫停式」流程
    # （run 先結束、按下按鈕後再接續）。沒加的話，一遇到要核准的工具就會直接報錯
    agent = build_agent(haiku, output_type=[str, DeferredToolRequests])

    # 模型選單：key 是畫面上顯示的名稱
    models = {"Claude Haiku 5.5": haiku}
    if os.getenv("GOOGLE_API_KEY"):
        models["Gemini 3 Flash"] = GoogleModel("gemini-3-flash-preview")

    # deps 會給所有請求共用：伺服器沒關之前，餘額會一直延續（重開伺服器才回到 50,000）
    return agent.to_web(models=models, deps=Bank())


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  警告：請在 .env 檔案中設置 ANTHROPIC_API_KEY")
        print("💡 前往 https://platform.claude.com/settings/keys 取得 API Key（格式 sk-ant-...）")
        return

    app = build_app()
    print(f"🚀 預設模型：{MODEL_NAME}")
    print(f"🏦 初始餘額 50,000 元；單筆轉帳超過 {TRANSFER_LIMIT:,} 元、關閉帳戶都要按核准")
    print(f"🌐 打開瀏覽器：http://{HOST}:{PORT}（Ctrl+C 結束）")
    print("💬 可以試試：「轉 500 給媽媽，再轉 3,000 給房東」")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
