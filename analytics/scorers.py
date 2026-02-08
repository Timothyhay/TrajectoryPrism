from abc import ABC, abstractmethod
from typing import Optional
from .schemas import TraceData


class BaseScorer(ABC):
    @abstractmethod
    def calculate(self, trace: TraceData) -> float:
        pass


# ----------------------------------------------------------------
# 1. 代码产出评分
# ----------------------------------------------------------------
class CodeProductionScorer(BaseScorer):
    def __init__(self, weight_per_line: float = 0.5, max_score: float = 20.0):
        self.weight = weight_per_line
        self.max_score = max_score

    def calculate(self, trace: TraceData) -> float:
        lines = trace.metrics.get('gemini_cli.lines.changed', 0)
        return min(lines * self.weight, self.max_score)


# ----------------------------------------------------------------
# 2. 推理深度评分
# ----------------------------------------------------------------
class ReasoningDepthScorer(BaseScorer):
    def __init__(self, max_score: float = 20.0):
        self.max_score = max_score

    def calculate(self, trace: TraceData) -> float:
        responses = [e for e in trace.events if e['name'] == 'gemini_cli.api_response']
        if not responses: return 0.0

        total_thoughts = sum(r.get('attributes', {}).get('thoughts_token_count', 0) for r in responses)
        total_tokens = sum(r.get('attributes', {}).get('output_token_count', 0) for r in responses)

        ratio = (total_thoughts / total_tokens) if total_tokens > 0 else 0.0
        return ratio * self.max_score


# ----------------------------------------------------------------
# 3. 工具多样性评分
# ----------------------------------------------------------------
class ToolDiversityScorer(BaseScorer):
    def __init__(self, weight_per_tool: float = 5.0, max_score: float = 15.0):
        self.weight = weight_per_tool
        self.max_score = max_score

    def calculate(self, trace: TraceData) -> float:
        tool_calls = [e for e in trace.events if e['name'] == 'gemini_cli.tool_call']
        if not tool_calls: return 0.0

        unique_tools = set(
            t.get('attributes', {}).get('function_name')
            for t in tool_calls
            if t.get('attributes', {}).get('function_name')
        )
        return min(len(unique_tools) * self.weight, self.max_score)


# ----------------------------------------------------------------
# 4. 工具成功率评分
# ----------------------------------------------------------------
class ToolSuccessScorer(BaseScorer):
    def __init__(self, max_score: float = 30.0):
        self.max_score = max_score

    def calculate(self, trace: TraceData) -> float:
        tool_calls = [e for e in trace.events if e['name'] == 'gemini_cli.tool_call']
        if not tool_calls: return 0.0

        success_cnt = sum(1 for t in tool_calls if t.get('attributes', {}).get('success'))
        rate = success_cnt / len(tool_calls)
        return rate * self.max_score


# ----------------------------------------------------------------
# 5. 步数效率评分
# ----------------------------------------------------------------
class TurnEfficiencyScorer(BaseScorer):
    def __init__(self, max_score: float = 15.0, optimal_turns: int = 5, penalty_per_turn: float = 2.0):
        self.max_score = max_score
        self.optimal_turns = optimal_turns
        self.penalty = penalty_per_turn

    def calculate(self, trace: TraceData) -> float:
        turns = trace.metrics.get('gemini_cli.agent.turns', 0)

        if turns < 2: return 0.0

        if turns <= self.optimal_turns:
            return self.max_score
        else:
            extra = turns - self.optimal_turns
            score = self.max_score - (extra * self.penalty)
            # 最低分不低于 -10，防止单个维度毁掉总分
            return max(score, -10.0)