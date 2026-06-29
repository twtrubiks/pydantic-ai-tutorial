"""
簡化的多代理協作範例
展示 PydanticAI 的核心優勢：
1. 結構化輸出與自動驗證
2. 代理間的資料傳遞
3. 工具函數的實際應用
"""

import asyncio
from typing import List
from datetime import datetime
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from dotenv import load_dotenv

load_dotenv()

# === 定義結構化輸出模型 (展示 Pydantic 的驗證優勢) ===

class ResearchResult(BaseModel):
    """研究結果的結構化輸出"""
    topic: str = Field(description="研究主題")
    key_findings: List[str] = Field(
        description="主要發現",
        min_length=1,  # 自動驗證：至少要有一個發現
        max_length=5   # 自動驗證：最多五個發現
    )
    confidence_score: float = Field(
        description="信心分數",
        ge=0.0,  # 自動驗證：必須 >= 0
        le=1.0   # 自動驗證：必須 <= 1
    )
    needs_review: bool = Field(description="是否需要審查")


class ReviewResult(BaseModel):
    """審查結果的結構化輸出"""
    approved: bool = Field(description="是否通過審查")
    feedback: str = Field(description="審查意見")
    suggestions: List[str] = Field(
        default_factory=list,
        description="改進建議"
    )


# === 初始化模型 ===
model = GoogleModel('gemini-3-flash-preview')


# === 研究代理 (展示工具使用) ===
research_agent = Agent(
    model,
    output_type=ResearchResult,
    system_prompt=(
        "你是一個研究助手。"
        "請分析給定的主題，提供結構化的研究結果。"
        "回應請用繁體中文。"
    )
)

@research_agent.tool
async def calculate_confidence(ctx: RunContext, findings_count: int, has_sources: bool) -> float:
    """
    實際的工具函數：根據發現數量和是否有來源計算信心分數
    展示 PydanticAI 如何整合 Python 函數作為工具
    """
    base_score = min(findings_count * 0.2, 0.8)
    if has_sources:
        base_score += 0.2
    return min(base_score, 1.0)


@research_agent.tool_plain
async def get_current_time() -> str:
    """獲取當前時間戳"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# === 審查代理 (展示代理間協作) ===
review_agent = Agent(
    model,
    output_type=ReviewResult,
    system_prompt=(
        "你是一個品質審查專家。"
        "請審查研究結果並提供建設性的回饋。"
        "回應請用繁體中文。"
    )
)


# === 協調器 (展示代理間的資料流) ===
async def coordinate_agents(topic: str) -> dict:
    """
    協調多個代理完成任務
    展示 PydanticAI 的優勢：
    1. 自動結構化輸出
    2. 類型安全的資料傳遞
    3. 簡潔的錯誤處理
    """
    print(f"\n🔍 開始研究主題: {topic}")
    print("=" * 40)

    # Step 1: 研究階段
    print("\n📚 階段 1: 研究")
    try:
        research_result = await research_agent.run(
            f"請研究以下主題並提供分析：{topic}"
        )

        # PydanticAI 自動確保輸出符合 ResearchResult 模型
        print(f"  ✓ 找到 {len(research_result.output.key_findings)} 個關鍵發現")
        print(f"  ✓ 信心分數: {research_result.output.confidence_score:.2f}")

        # 展示結構化資料的便利性
        for i, finding in enumerate(research_result.output.key_findings, 1):
            print(f"    {i}. {finding[:50]}...")

    except Exception as e:
        print(f"  ✗ 研究失敗: {e}")
        return {"error": str(e)}

    # Step 2: 審查階段 (基於研究結果)
    if research_result.output.needs_review:
        print("\n✅ 階段 2: 審查")

        # 將研究結果轉換為審查輸入
        review_input = f"""
        請審查以下研究結果：
        主題: {research_result.output.topic}
        發現: {', '.join(research_result.output.key_findings)}
        信心分數: {research_result.output.confidence_score}
        """

        try:
            review_result = await review_agent.run(review_input)

            print(f"  ✓ 審查結果: {'通過' if review_result.output.approved else '需要改進'}")
            print(f"  ✓ 回饋: {review_result.output.feedback[:100]}...")

            if review_result.output.suggestions:
                print(f"  ✓ 建議數量: {len(review_result.output.suggestions)}")

        except Exception as e:
            print(f"  ✗ 審查失敗: {e}")
            review_result = None
    else:
        print("\n✅ 階段 2: 跳過審查 (信心分數足夠高)")
        review_result = None

    # 返回結構化結果
    return {
        "research": research_result.output.model_dump() if research_result else None,
        "review": review_result.output.model_dump() if review_result else None,
        "timestamp": datetime.now().isoformat()
    }


# === 批次處理 (展示異步優勢) ===
async def batch_process(topics: List[str]) -> List[dict]:
    """
    平行處理多個主題
    展示 PydanticAI 的異步處理能力
    """
    print(f"\n🚀 開始批次處理 {len(topics)} 個主題...")

    # 使用 asyncio.gather 平行執行
    tasks = [coordinate_agents(topic) for topic in topics]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 統計結果
    successful = sum(1 for r in results if not isinstance(r, Exception) and "error" not in r)
    print(f"\n📊 批次處理完成: {successful}/{len(topics)} 成功")

    return results


# === 主程式 ===
async def main():
    """主程式示例"""

    # 範例 1: 單一主題處理
    print("\n" + "="*50)
    print("範例 1: 單一主題處理")
    print("="*50)

    result = await coordinate_agents("Python 異步編程最佳實踐")

    if "research" in result and result["research"]:
        print("\n📋 最終結果摘要:")
        print(f"  主題: {result['research']['topic']}")
        print(f"  信心分數: {result['research']['confidence_score']}")

    # 範例 2: 批次處理多個主題
    print("\n" + "="*50)
    print("範例 2: 批次處理")
    print("="*50)

    topics = [
        "FastAPI vs Flask 比較",
        "Docker 容器化策略",
        "微服務架構設計"
    ]

    batch_results = await batch_process(topics)

    # 顯示批次處理統計
    print("\n📈 批次處理統計:")
    for i, (topic, result) in enumerate(zip(topics, batch_results), 1):
        if isinstance(result, dict) and "research" in result:
            status = "✓ 成功" if result["research"] else "✗ 失敗"
            print(f"  {i}. {topic}: {status}")


if __name__ == "__main__":
    asyncio.run(main())