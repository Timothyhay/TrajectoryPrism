import json
import logging
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DatasetType(Enum):
    SFT = "sft"  # 监督微调：完美轨迹
    RLHF = "rlhf"  # 强化学习：包含自我修正但最终成功的轨迹
    REJECTED = "rejected"  # 拒绝：失败、低质量或死循环


@dataclass
class EvaluationResult:
    score: float
    dataset_type: DatasetType
    reason: List[str]
    openai_messages: Optional[List[Dict[str, Any]]] = None


class TraceAnalyzer:
    def __init__(self, metrics: Dict[str, Any], events: List[Dict[str, Any]]):
        """
        初始化分析器
        :param metrics: 聚合的Counter/Histogram指标 (key-value)
        :param events: 按时间排序的事件列表
        """
        self.metrics = metrics
        self.events = events
        self.attributes = self._extract_attributes()

    def _extract_attributes(self):
        """辅助函数：从事件流中提取关键配置"""
        config = next((e for e in self.events if e['name'] == 'gemini_cli.config'), {})
        return config.get('attributes', {})

    def _calculate_thought_density(self) -> float:
        """计算思维链密度: 思考Token / 总Token"""
        responses = [e for e in self.events if e['name'] == 'gemini_cli.api_response']
        if not responses:
            return 0.0

        total_thoughts = sum(r['attributes'].get('thoughts_token_count', 0) for r in responses)
        total_tokens = sum(r['attributes'].get('output_token_count', 0) for r in responses)

        return (total_thoughts / total_tokens) if total_tokens > 0 else 0.0

    def validate_hard_filters(self) -> List[str]:
        """
        维度一 & 四：硬性过滤规则
        返回拒绝原因列表，为空表示通过
        """
        reasons = []

        # 1. 检查致命错误 (Counter指标)
        if self.metrics.get('gemini_cli.exit.fail.count', 0) > 0:
            reasons.append("CLI_EXIT_FAILURE")

        if self.metrics.get('gemini_cli.chat.content_retry_failure.count', 0) > 0:
            reasons.append("CONTENT_RETRY_FAILURE")

        # 2. 检查API状态 (Event字段)
        api_errors = [e for e in self.events if e['name'] == 'gemini_cli.api_error']
        if api_errors:
            reasons.append(f"API_ERROR_FOUND: {len(api_errors)}")

        # 3. 检查截断 (Event字段)
        truncated = [e for e in self.events if e['name'] == 'gemini_cli.tool_output_truncated']
        if truncated:
            # 如果截断比例过高，视为废弃
            reasons.append("TOOL_OUTPUT_TRUNCATED")

        # 4. 检查产出有效性 (补充规则一)
        # 如果进行了文件写操作，但changed lines为0，视为无效
        file_ops = self.metrics.get('gemini_cli.file.operation.count', 0)
        lines_changed = self.metrics.get('gemini_cli.lines.changed', 0)
        if file_ops > 0 and lines_changed == 0:
            reasons.append("INEFFECTIVE_FILE_OPERATION")

        return reasons

    def calculate_score(self) -> float:
        """
        维度二 & 三：质量打分 (0-100+)
        """
        score = 0.0

        # 1. 基础分：代码/文件修改价值
        lines_changed = self.metrics.get('gemini_cli.lines.changed', 0)
        # 每行代码给0.5分，上限20分（避免copy-paste大文件刷分）
        score += min(lines_changed * 0.5, 20)

        # 2. 推理深度奖励
        thought_density = self._calculate_thought_density()
        # 密度越高分越高，满分20
        score += thought_density * 20

        # 3. 交互复杂度奖励
        turns = self.metrics.get('gemini_cli.agent.turns', 0)
        if 3 <= turns <= 15:
            score += 15  # 黄金交互区间
        elif turns > 20:
            score -= 10  # 惩罚啰嗦/死循环

        # 4. 工具使用成功率
        tool_calls = [e for e in self.events if e['name'] == 'gemini_cli.tool_call']
        if tool_calls:
            success_count = sum(1 for t in tool_calls if t['attributes'].get('success', False))
            success_rate = success_count / len(tool_calls)
            score += success_rate * 30  # 满分30

        # 5. Token效率奖励
        # 假设有一个预设的基准，这里简化处理
        total_tokens = self.metrics.get('gemini_cli.token.usage', 0)
        if lines_changed > 0 and total_tokens < 5000:
            score += 10  # 高效产出

        return round(score, 2)

    def determine_dataset_type(self) -> DatasetType:
        """
        补充规则二：SFT vs RLHF 分流
        """
        recovery_attempts = self.metrics.get('gemini_cli.agent.recovery_attempt.count', 0)
        content_retries = self.metrics.get('gemini_cli.chat.content_retry.count', 0)

        # 如果有恢复尝试或重试，但最终没有失败(在hard_filters中已拦截)，
        # 说明这很好的 RLHF/DPO 样本（负样本->正样本的修正过程）
        if recovery_attempts > 0 or content_retries > 0:
            return DatasetType.RLHF

        return DatasetType.SFT

    def convert_to_openai_format(self) -> List[Dict[str, Any]]:
        """
        将 Event 序列重构为 OpenAI 训练数据格式
        """
        messages = []

        # 添加 System Message (基于 Config)
        messages.append({
            "role": "system",
            "content": f"You are an agent with tools: {self.attributes.get('core_tools_enabled', 'default')}."
        })

        # 遍历事件重建对话
        # 注意：这里简化了逻辑，实际需要根据 prompt_id 串联
        for event in self.events:
            attrs = event.get('attributes', {})

            if event['name'] == 'gemini_cli.user_prompt':
                # 在训练数据中，通常需要具体的 prompt 内容。
                # 如果 'prompt' 字段因隐私未记录，这里可能需要回填或跳过
                content = attrs.get('prompt', '<PROMPT_CONTENT>')
                messages.append({"role": "user", "content": content})

            elif event['name'] == 'gemini_cli.api_response':
                # 提取模型回复
                text = attrs.get('response_text', '')
                # 如果有 thoughts，通常 SFT 训练会将其包含在 content 中或特定字段
                # 这里假设直接拼接
                if attrs.get('thoughts_token_count', 0) > 0:
                    # 注意：实际数据中需要从 response_text 里解析或者 event 里有 separate field
                    pass
                messages.append({"role": "assistant", "content": text})

            elif event['name'] == 'gemini_cli.tool_call':
                # 模拟工具调用
                tool_call = {
                    "id": f"call_{attrs.get('function_name')}",
                    "type": "function",
                    "function": {
                        "name": attrs.get('function_name'),
                        "arguments": json.dumps(attrs.get('function_args'))
                    }
                }
                # OpenAI 格式通常 Tool Calls 挂在 Assistant message 下，
                # 这里为了简化，假设是一个单独的步骤展示
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })

                # 模拟工具返回 (根据 tool_call 的 success 状态)
                # 注意：实际轨迹中需要 tool_result 事件，这里用 tool_call 的结果模拟
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call['id'],
                    "content": "Success" if attrs.get('success') else f"Error: {attrs.get('error')}"
                })

        return messages

    def analyze(self) -> EvaluationResult:
        # 1. 硬过滤
        fail_reasons = self.validate_hard_filters()
        if fail_reasons:
            return EvaluationResult(
                score=0,
                dataset_type=DatasetType.REJECTED,
                reason=fail_reasons
            )

        # 2. 打分
        score = self.calculate_score()

        # 3. 分类
        ds_type = self.determine_dataset_type()

        # 4. 格式转换
        openai_msgs = self.convert_to_openai_format()

        return EvaluationResult(
            score=score,
            dataset_type=ds_type,
            reason=["PASSED"],
            openai_messages=openai_msgs
        )


# ==========================================
# 模拟数据生成与测试
# ==========================================

def get_mock_high_quality_trace():
    """模拟一个高质量的编码任务轨迹"""
    metrics = {
        "gemini_cli.exit.fail.count": 0,
        "gemini_cli.lines.changed": 45,  # 有效产出
        "gemini_cli.agent.turns": 6,  # 适中的轮数
        "gemini_cli.file.operation.count": 2,
        "gemini_cli.token.usage": 3200,
        "gemini_cli.tool.call.count": 2,
        "gemini_cli.agent.recovery_attempt.count": 0  # 一次成功
    }
    events = [
        {"name": "gemini_cli.config", "attributes": {"core_tools_enabled": "file_editor, search"}},
        {"name": "gemini_cli.user_prompt", "attributes": {"prompt_id": "p1", "prompt": "Refactor the login function."}},
        {"name": "gemini_cli.api_response",
         "attributes": {"prompt_id": "p1", "response_text": "I will read the file first.", "thoughts_token_count": 120,
                        "output_token_count": 200}},
        {"name": "gemini_cli.tool_call",
         "attributes": {"function_name": "read_file", "function_args": {"path": "auth.py"}, "success": True}},
        {"name": "gemini_cli.api_response",
         "attributes": {"prompt_id": "p1", "response_text": "Now applying changes.", "thoughts_token_count": 150,
                        "output_token_count": 300}},
        {"name": "gemini_cli.tool_call",
         "attributes": {"function_name": "update_file", "function_args": {"path": "auth.py"}, "success": True}},
    ]
    return metrics, events


def get_mock_recovery_trace():
    """模拟一个发生错误但自我修正的轨迹 (RLHF样本)"""
    metrics = {
        "gemini_cli.exit.fail.count": 0,
        "gemini_cli.lines.changed": 10,
        "gemini_cli.agent.turns": 8,
        "gemini_cli.agent.recovery_attempt.count": 1,  # 关键指标
        "gemini_cli.chat.content_retry.count": 1
    }
    events = [
        {"name": "gemini_cli.config", "attributes": {}},
        {"name": "gemini_cli.user_prompt", "attributes": {"prompt": "Fix bug"}},
        {"name": "gemini_cli.tool_call",
         "attributes": {"function_name": "test_run", "success": False, "error": "Syntax Error"}},  # 失败
        {"name": "gemini_cli.api_response",
         "attributes": {"response_text": "I made a syntax error, fixing it now...", "thoughts_token_count": 200,
                        "output_token_count": 300}},  # 思考修正
        {"name": "gemini_cli.tool_call", "attributes": {"function_name": "test_run", "success": True}},  # 成功
    ]
    return metrics, events


# ==========================================
# 主执行流程
# ==========================================

if __name__ == "__main__":
    # 1. 测试高质量 SFT 样本
    print("--- Analyzing High Quality Trace ---")
    m1, e1 = get_mock_high_quality_trace()
    analyzer1 = TraceAnalyzer(m1, e1)
    result1 = analyzer1.analyze()
    print(f"Type: {result1.dataset_type.value}")
    print(f"Score: {result1.score}")
    print(f"Messages: {len(result1.openai_messages)}")

    print("\n--- Analyzing Recovery Trace (RLHF) ---")
    m2, e2 = get_mock_recovery_trace()
    analyzer2 = TraceAnalyzer(m2, e2)
    result2 = analyzer2.analyze()
    print(f"Type: {result2.dataset_type.value}")
    print(f"Score: {result2.score}")
    print(f"Reason: {result2.reason}")

    import pandas as pd
    import json


    # 假设这是通过 TraceAnalyzer 计算出的一批结果
    # 结构：[EvaluationResult, EvaluationResult, ...]
    # 我们需要把 EvaluationResult 转换成更易读的字典列表

    def generate_leaderboard_report(results):
        """
        生成轨迹排行榜数据
        """
        report_data = []

        for res in results:
            # 提取轨迹摘要（第一句用户Prompt）
            first_prompt = "N/A"
            formatted_trace = []

            if res.openai_messages:
                # 找到第一个用户消息作为标题
                for msg in res.openai_messages:
                    if msg['role'] == 'user':
                        first_prompt = msg['content'][:50] + "..."
                        break

                # 格式化完整的轨迹文本用于展示
                for msg in res.openai_messages:
                    role = msg['role'].upper()
                    content = msg.get('content', '')

                    # 处理工具调用显示
                    tool_calls = msg.get('tool_calls')
                    if tool_calls:
                        content = f"[Calling Tool]: {json.dumps(tool_calls[0]['function'], indent=2)}"

                    formatted_trace.append(f"[{role}]:\n{content}\n{'-' * 20}")

            report_data.append({
                "Score": res.score,
                "Type": res.dataset_type.value,
                "Summary": first_prompt,
                "Status": "PASS" if not res.reason or res.reason == ["PASSED"] else "FAIL",
                "Reasons": ", ".join(res.reason),
                "Full_Trace_Text": "\n".join(formatted_trace)  # 将完整轨迹拼接成文本块
            })

        # 1. 转换为 DataFrame
        df = pd.read_json(json.dumps(report_data))

        # 2. 按分数降序排列
        df_ranked = df.sort_values(by="Score", ascending=False).reset_index(drop=True)

        return df_ranked


    # ==========================================
    # 模拟运行
    # ==========================================

    # 假设 analyzer1 和 analyzer2 是上一轮对话中生成的实例
    # 这里的 result1 是高分样本，result2 是RLHF样本
    results = [result1, result2]

    df_rank = generate_leaderboard_report(results)

    # 打印排名摘要（终端查看）
    print("=== Top Trajectories ===")
    print(df_rank[['Score', 'Type', 'Summary', 'Status']])


    # ==========================================
    # 技巧：导出为 HTML 以便人工审查
    # ==========================================
    # 我们可以利用 pandas 的 style 功能生成一个带颜色、可展开的 HTML 表格
    # 这对于人工审核数据质量非常有用
    def save_html_view(df, filename="trajectory_rank.html"):
        # 简单的 CSS 样式，处理换行符以便阅读长文本
        pd.set_option('display.max_colwidth', None)

        html = df.to_html(
            index=True,
            escape=False,  # 允许 HTML 渲染（小心 XSS，如果是内部数据没关系）
            formatters={
                # 将换行符转换为 HTML换行，方便阅读对话
                'Full_Trace_Text': lambda
                    x: f"<pre style='background:#f4f4f4; padding:10px; border-radius:5px; white-space: pre-wrap;'>{x}</pre>"
            }
        )

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"<html><body><h1>Trajectory Leaderboard</h1>{html}</body></html>")
        print(f"\nReport saved to {filename}")


    save_html_view(df_rank)