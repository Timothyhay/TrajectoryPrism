import json
import pandas as pd
from typing import List, Dict, Any
from .schemas import AnalysisResult, TraceData


class OpenAIConverter:
    """将内部 Trace 数据转换为 OpenAI 聊天格式"""

    @staticmethod
    def convert(trace: TraceData) -> List[Dict[str, Any]]:
        messages = []

        # System Message
        tools = trace.config.get('core_tools_enabled', 'standard tools')
        messages.append({"role": "system", "content": f"Agent tools: {tools}"})

        # Reconstruct Conversation
        for event in trace.events:
            attrs = event.get('attributes', {})
            name = event['name']

            if name == 'gemini_cli.user_prompt':
                messages.append({"role": "user", "content": attrs.get('prompt', '<REDACTED>')})

            elif name == 'gemini_cli.api_response':
                content = attrs.get('response_text', '')
                # 可选：如果你想保留思维链作为独立部分，可在此处理
                messages.append({"role": "assistant", "content": content})

            elif name == 'gemini_cli.tool_call':
                # 构造 Tool Call
                call_id = f"call_{attrs.get('function_name')}_{hash(str(attrs))}"[:10]
                tool_msg = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": attrs.get('function_name'),
                            "arguments": json.dumps(attrs.get('function_args'))
                        }
                    }]
                }
                messages.append(tool_msg)

                # 构造 Tool Output (模拟)
                output_content = "Success" if attrs.get('success') else f"Error: {attrs.get('error')}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": output_content
                })

        return messages


class ReportGenerator:
    """生成 HTML 排行榜报告"""

    @staticmethod
    def generate_html(results: List[AnalysisResult], filename="report.html"):
        data = []
        for res in results:
            # 提取第一句 Prompt 作为摘要
            summary = "No content"
            full_text = ""

            if res.openai_messages:
                # 找 User Prompt
                users = [m['content'] for m in res.openai_messages if m['role'] == 'user']
                if users: summary = users[0][:60] + "..."

                # 格式化完整文本
                for msg in res.openai_messages:
                    role = msg['role'].upper()
                    content = msg.get('content') or json.dumps(msg.get('tool_calls'), indent=2)
                    full_text += f"[{role}]:\n{content}\n{'-' * 20}\n"

            data.append({
                "ID": res.trace_id,
                "Score": res.score,
                "Type": res.dataset_type.value,
                "Status": "PASS" if not res.reasons else "FAIL",
                "Summary": summary,
                "Full Trace": full_text
            })

        df = pd.DataFrame(data).sort_values(by="Score", ascending=False)

        # 简单的 HTML 样式
        pd.set_option('display.max_colwidth', None)
        html = df.to_html(escape=False, formatters={
            'Full Trace': lambda
                x: f"<pre style='max-height:200px; overflow-y:auto; background:#f4f4f4; padding:5px;'>{x}</pre>"
        })

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(
                f"<html><head><title>Agent Trace Analysis</title></head><body><h1>Trace Leaderboard</h1>{html}</body></html>")
        print(f"Report saved to {filename}")