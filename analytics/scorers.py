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
            "efficiency": 10.0,  # Token效率奖励
            "tool_diversity": 5.0,  # 每个不同工具的奖励分
            "tool_div_cap": 15.0  # 工具多样性奖励上限
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

        # 2. 推理深度分
        density = self._calculate_thought_density(trace.events)
        score += density * self.weights['reasoning']

        # 3. 交互结构分
        turns = metrics.get('gemini_cli.agent.turns', 0)
        if 3 <= turns <= 15:
            score += self.weights['interaction']
        elif turns > 20:
            score -= 10  # 惩罚

        # 4. 工具使用评估
        # - 26/01/31 增加多样性计算
        tool_calls = [e for e in trace.events if e['name'] == 'gemini_cli.tool_call']
        if tool_calls:
            # 4.1 成功率
            success_cnt = sum(1 for t in tool_calls if t.get('attributes', {}).get('success'))
            score += (success_cnt / len(tool_calls)) * self.weights['success']

            # 4.2 [新增] 工具多样性 (Tool Diversity)
            # 统计有多少个不重复的 function_name
            unique_tools = set()
            for t in tool_calls:
                fname = t.get('attributes', {}).get('function_name')
                if fname:
                    unique_tools.add(fname)

            # 计算分数：每多用一种工具加 5 分，上限 15 分
            # 例如：只用了 read_file -> 5分
            # 用了 read_file + search -> 10分
            # 用了 read_file + search + write_file -> 15分 (满)
            diversity_score = len(unique_tools) * self.weights['tool_diversity']
            score += min(diversity_score, self.weights['tool_div_cap'])

        return round(score, 2)