import json
import logging
from typing import List, Dict, Any
from .schemas import TraceData

'''
仅凭 Trajectories, 我们彻底丢失了以下系统级维度，分析时必须忽略或设置为默认值：
OTel 指标名	状态	原因	处理策略
gemini_cli.exit.fail.count	❌ 丢失	无法知道进程是否崩溃，只能看到对话在哪里结束。	默认为 0
gemini_cli.chat.invalid_chunk	❌ 丢失	传输层的错误在最终 JSON 中不可见。	默认为 0
gemini_cli.memory.usage / cpu	❌ 丢失	无系统资源数据。	忽略
gemini_cli.tool.call.latency	⚠️ 估算/丢失	除非 JSON 带时间戳，否则无法计算耗时。	默认为 0
gemini_cli.model_routing.failure	❌ 丢失	路由逻辑通常在 Agent 内部，外部不可见。	忽略
gemini_cli.token.usage	⚠️ 估算	API 返回通常带 usage，如果只有 messages，需用 Tiktoken 估算。	使用估算值
'''

try:
    import tiktoken

    ENCODER = tiktoken.get_encoding("cl100k_base")


    def count_tokens(text: str) -> int:
        return len(ENCODER.encode(text or ""))
except ImportError:
    def count_tokens(text: str) -> int:
        return len(text or "") // 4


class OpenAIAdapter:
    """
    将原始 OpenAI 格式对话转换为内部 TraceData 对象。
    通过‘逆向工程’从文本中提取指标。
    """

    @staticmethod
    def infer_lines_changed(function_name: str, args: Dict) -> int:
        """从工具参数中推断代码变更行数"""
        # 常见的文件写入/修改工具名
        write_tools = ['write_file', 'create_file', 'update_file', 'apply_diff', 'replace_string']

        if any(t in function_name.lower() for t in write_tools):
            content = args.get('content') or args.get('code') or args.get('diff') or ""
            if isinstance(content, str):
                return len(content.splitlines())
        return 0

    @staticmethod
    def to_trace_data(trace_id: str, messages: List[Dict[str, Any]]) -> TraceData:
        metrics = {
            "gemini_cli.lines.changed": 0,
            "gemini_cli.file.operation.count": 0,
            "gemini_cli.agent.turns": 0,
            "gemini_cli.tool.call.count": 0,
            "gemini_cli.agent.recovery_attempt.count": 0,
            "gemini_cli.exit.fail.count": 0,  # 无法得知，默认为0
        }

        events = []

        # 模拟 Config 事件
        events.append({
            "name": "gemini_cli.config",
            "attributes": {"core_tools_enabled": "inferred_from_trace"}
        })

        for i, msg in enumerate(messages):
            role = msg.get('role')
            content = msg.get('content')

            # 1. User Prompt
            if role == 'user':
                events.append({
                    "name": "gemini_cli.user_prompt",
                    "attributes": {
                        "prompt": content,
                        "prompt_length": len(content or "")
                    }
                })

            # 2. Assistant (Model Response)
            elif role == 'assistant':
                metrics["gemini_cli.agent.turns"] += 1

                # 尝试提取思维链 (如果是 <thought> 格式)
                thoughts_tokens = 0
                if content and "<thought>" in content:
                    # 极其简化的提取逻辑，实际需正则
                    pass

                events.append({
                    "name": "gemini_cli.api_response",
                    "attributes": {
                        "response_text": content,
                        "output_token_count": count_tokens(content),
                        "thoughts_token_count": thoughts_tokens  # 可能为0
                    }
                })

                # 处理 Tool Calls
                tool_calls = msg.get('tool_calls', [])
                for tc in tool_calls:
                    func = tc.get('function', {})
                    fname = func.get('name')
                    try:
                        fargs = json.loads(func.get('arguments', '{}'))
                    except:
                        fargs = {}

                    metrics["gemini_cli.tool.call.count"] += 1

                    # 推断文件操作 metrics
                    lines = OpenAIAdapter.infer_lines_changed(fname, fargs)
                    if lines > 0:
                        metrics["gemini_cli.lines.changed"] += lines
                        metrics["gemini_cli.file.operation.count"] += 1

                    events.append({
                        "name": "gemini_cli.tool_call",
                        "attributes": {
                            "function_name": fname,
                            "function_args": fargs,
                            # 暂时假设调用发起是成功的，具体结果看 tool message
                            "tool_call_id": tc.get('id')
                        }
                    })

            # 3. Tool Output
            elif role == 'tool':
                # 寻找对应的 tool call 事件来回填 success 状态
                call_id = msg.get('tool_call_id')
                is_error = False

                # 简单的错误检测逻辑
                content_lower = str(content).lower()[:200]  # 只看开头
                if "error" in content_lower or "exception" in content_lower or "failed" in content_lower:
                    is_error = True
                    metrics["gemini_cli.agent.recovery_attempt.count"] += 1  # 视为发生了一次错误，需要恢复

                # 回溯更新最近的一个匹配的 tool_call event (这是一个简化的处理)
                # 在真实场景中，应该建立 id 映射
                for event in reversed(events):
                    if event['name'] == 'gemini_cli.tool_call' and \
                            event['attributes'].get('tool_call_id') == call_id:
                        event['attributes']['success'] = not is_error
                        if is_error:
                            event['attributes']['error'] = str(content)[:100]
                        break

        return TraceData(trace_id=trace_id, metrics=metrics, events=events)