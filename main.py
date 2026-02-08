from analytics.pipeline import TracePipeline
from analytics.utils import get_mock_data


def main():
    # 场景 A: 分析普通代码生成任务
    print("\n=== Running General Coding Scenario ===")
    pipeline_coding = TracePipeline(scenario_name="default")

    # 场景 B: 分析 SWE-bench 任务
    print("\n=== Running SWE-bench Scenario ===")
    pipeline_swe = TracePipeline(scenario_name="swe_bench")

    # 模拟数据
    mock_swe_trace = {
        "metrics": {
            "gemini_cli.lines.changed": 1,  # 只改了1行 (对 default 场景分很低，对 swe 场景分还可以)
            "gemini_cli.agent.turns": 25,  # 跑了25轮 (对 default 场景会扣烂分，对 swe 场景可容忍)
            "gemini_cli.exit.fail.count": 0,
            "gemini_cli.file.operation.count": 1
        },
        "events": [
            {"name": "gemini_cli.user_prompt",
             "attributes": {"prompt": "Fix complex race condition", "prompt_length": 50}},
            {"name": "gemini_cli.api_response",
             "attributes": {"response_text": "Thinking...", "thoughts_token_count": 500, "output_token_count": 600}},
            # 深度思考
            {"name": "gemini_cli.tool_call", "attributes": {"function_name": "patch", "success": True}}
        ]
    }

    # 1. 用通用标准跑
    res1 = pipeline_coding.process_trace("swe_task_001", mock_swe_trace['metrics'], mock_swe_trace['events'])
    print(f"[General] Score: {res1.score} (Status: {res1.reasons or 'Pass'})")
    # 预期：分数较低，因为 turns 太高被惩罚，lines 太少分不高

    # 2. 用 SWE-bench 标准跑
    res2 = pipeline_swe.process_trace("swe_task_001", mock_swe_trace['metrics'], mock_swe_trace['events'])
    print(f"[SWE-Bench] Score: {res2.score} (Status: {res2.reasons or 'Pass'})")
    # 预期：分数较高，因为 turns 惩罚很轻，Reasoning 权重很高，且没有被 ProductivityFilter 拦截


if __name__ == "__main__":
    main()