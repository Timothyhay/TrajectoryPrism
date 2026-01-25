from .schemas import TraceData


class QualityScorer:
    def __init__(self):
        # 权重配置
        self.weights = {
            "code_output": 0.5,  # 每行代码的分数
            "code_cap": 20.0,  # 代码产出分上限
            "reasoning": 20.0,  # 推理密度满分
            "interaction": 15.0,  # 交互区间奖励
            "success": 30.0,  # 工具成功率满分
            "efficiency": 10.0  # Token效率奖励
        }

    def _calculate_thought_density(self, events) -> float:
        responses = [e for e in events if e['name'] == 'gemini_cli.api_response']
        if not responses: return 0.0

        total_thoughts = sum(r.get('attributes', {}).get('thoughts_token_count', 0) for r in responses)
        total_tokens = sum(r.get('attributes', {}).get('output_token_count', 0) for r in responses)
        return (total_thoughts / total_tokens) if total_tokens > 0 else 0.0

    def evaluate(self, trace: TraceData) -> float:
        score = 0.0
        metrics = trace.metrics

        # 1. 代码产出分
        lines = metrics.get('gemini_cli.lines.changed', 0)
        score += min(lines * self.weights['code_output'], self.weights['code_cap'])

        # 2. 推理密度分
        density = self._calculate_thought_density(trace.events)
        score += density * self.weights['reasoning']

        # 3. 交互结构分
        turns = metrics.get('gemini_cli.agent.turns', 0)
        if 3 <= turns <= 15:
            score += self.weights['interaction']
        elif turns > 20:
            score -= 10  # 惩罚

        # 4. 工具成功率
        tool_calls = [e for e in trace.events if e['name'] == 'gemini_cli.tool_call']
        if tool_calls:
            success_cnt = sum(1 for t in tool_calls if t.get('attributes', {}).get('success'))
            score += (success_cnt / len(tool_calls)) * self.weights['success']

        return round(score, 2)