from abc import ABC, abstractmethod
from typing import List
from .schemas import TraceData


class BaseScorer(ABC):
    """评分器基类：定义统一接口"""

    @abstractmethod
    def calculate(self, trace: TraceData) -> float:
        """返回该维度的得分"""
        pass


# ----------------------------------------------------------------
# 1. 代码产出评分 (关注 Productivity)
# ----------------------------------------------------------------
class CodeProductionScorer(BaseScorer):
    def calculate(self, trace: TraceData) -> float:
        lines = trace.metrics.get('gemini_cli.lines.changed', 0)
        # 逻辑：每行代码 0.5 分，上限 20 分
        return min(lines * 0.5, 20.0)


# ----------------------------------------------------------------
# 2. 推理深度评分 (关注 Reasoning)
# ----------------------------------------------------------------
class ReasoningDepthScorer(BaseScorer):
    def calculate(self, trace: TraceData) -> float:
        responses = [e for e in trace.events if e['name'] == 'gemini_cli.api_response']
        if not responses: return 0.0

        total_thoughts = sum(r.get('attributes', {}).get('thoughts_token_count', 0) for r in responses)
        total_tokens = sum(r.get('attributes', {}).get('output_token_count', 0) for r in responses)

        ratio = (total_thoughts / total_tokens) if total_tokens > 0 else 0.0
        # 逻辑：推理占比越高越好，满分 20 分
        return ratio * 20.0


# ----------------------------------------------------------------
# 3. 工具多样性评分 (关注 Tool Capability)
# ----------------------------------------------------------------
class ToolDiversityScorer(BaseScorer):
    def calculate(self, trace: TraceData) -> float:
        tool_calls = [e for e in trace.events if e['name'] == 'gemini_cli.tool_call']
        if not tool_calls: return 0.0

        unique_tools = set(
            t.get('attributes', {}).get('function_name')
            for t in tool_calls
            if t.get('attributes', {}).get('function_name')
        )

        # 逻辑：每多用一种工具加 5 分，上限 15 分
        return min(len(unique_tools) * 5.0, 15.0)


# ----------------------------------------------------------------
# 4. 工具成功率评分 (关注 Reliability)
# ----------------------------------------------------------------
class ToolSuccessScorer(BaseScorer):
    def calculate(self, trace: TraceData) -> float:
        tool_calls = [e for e in trace.events if e['name'] == 'gemini_cli.tool_call']
        if not tool_calls: return 0.0

        success_cnt = sum(1 for t in tool_calls if t.get('attributes', {}).get('success'))
        rate = success_cnt / len(tool_calls)

        # 逻辑：成功率 * 30 分
        return rate * 30.0


# ----------------------------------------------------------------
# 5. [新增] 步数效率评分 (关注 Efficiency)
# ----------------------------------------------------------------
class TurnEfficiencyScorer(BaseScorer):
    """
    效率评分：在任务完成的前提下，轮数越少分越高。
    """

    def calculate(self, trace: TraceData) -> float:
        turns = trace.metrics.get('gemini_cli.agent.turns', 0)

        # 极端情况处理
        if turns < 2:
            return 0.0  # 过于简单，不给效率分（可能只是打招呼）

        # 核心逻辑：
        # 设定一个“完美区间” (2-5轮)，给予满分奖励 (e.g., 15分)
        # 超过这个区间后，每多一轮扣分 (Decay)

        MAX_SCORE = 15.0
        OPTIMAL_TURN_MAX = 5
        PENALTY_PER_EXTRA_TURN = 2.0

        if turns <= OPTIMAL_TURN_MAX:
            return MAX_SCORE
        else:
            # 计算溢出的轮数
            extra_turns = turns - OPTIMAL_TURN_MAX
            # 计算扣分后的得分，最低不低于 -10 (避免过度惩罚长任务)
            score = MAX_SCORE - (extra_turns * PENALTY_PER_EXTRA_TURN)
            return max(score, -10.0)


# ================================================================
# 配置区域：动态配置生效的评分逻辑
# ================================================================

# 你可以在这里注释掉你不想要的评分维度
ACTIVE_SCORERS: List[BaseScorer] = [
    CodeProductionScorer(),  # 有代码产出 +20
    ReasoningDepthScorer(),  # 深度思考 +20
    ToolDiversityScorer(),  # 多种工具 +15
    ToolSuccessScorer(),  # 工具成功 +30
    TurnEfficiencyScorer()  # 高效解决 +15 (随着轮数增加而减少)
]
