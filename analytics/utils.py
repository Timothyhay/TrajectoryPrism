# gemini_analytics/utils.py

def get_mock_data():
    """返回 (trace_id, metrics, events) 的生成器"""

    # ==========================================================
    # 样本 1: 高质量 SFT (修复：添加 prompt_length)
    # ==========================================================
    prompt_text_1 = "Write a fast sort function in Python"
    yield "trace_001", {
        "gemini_cli.exit.fail.count": 0,
        "gemini_cli.lines.changed": 50,
        "gemini_cli.agent.turns": 5,
        "gemini_cli.file.operation.count": 1
    }, [
        {"name": "gemini_cli.config", "attributes": {"core_tools_enabled": "code_edit"}},
        {
            "name": "gemini_cli.user_prompt",
            "attributes": {
                "prompt": prompt_text_1,
                "prompt_length": len(prompt_text_1)  # <--- 关键修复：必须有这个字段且 > 10
            }
        },
        {"name": "gemini_cli.api_response",
         "attributes": {"response_text": "Sure.", "thoughts_token_count": 100, "output_token_count": 200}},
        {"name": "gemini_cli.tool_call",
         "attributes": {"function_name": "write_file", "function_args": {"file": "sort.py"}, "success": True}}
    ]

    # ==========================================================
    # 样本 2: 失败样本 (预期被 Reject，保持原样即可，或者也加上 length 以便观察是因何 Reject)
    # ==========================================================
    yield "trace_002_fail", {
        "gemini_cli.exit.fail.count": 1,
        "gemini_cli.lines.changed": 0
    }, [
        # 即使这里加了 length，也会因为 exit.fail.count 被 IntegrityFilter 拦截
        {"name": "gemini_cli.user_prompt", "attributes": {"prompt": "Run", "prompt_length": 3}}
    ]

    # ==========================================================
    # 样本 3: RLHF 修正样本 (修复：加长 Prompt 并添加 length)
    # ==========================================================
    # 原来的 "Fix logic" 只有 9 个字符，会被视为无效输入，改为长一点的
    prompt_text_3 = "Fix the logic error in the calculation module"
    yield "trace_003_rlhf", {
        "gemini_cli.exit.fail.count": 0,
        "gemini_cli.lines.changed": 10,
        "gemini_cli.agent.turns": 8,
        "gemini_cli.agent.recovery_attempt.count": 1
    }, [
        {"name": "gemini_cli.config", "attributes": {}},
        {
            "name": "gemini_cli.user_prompt",
            "attributes": {
                "prompt": prompt_text_3,
                "prompt_length": len(prompt_text_3)  # <--- 关键修复
            }
        },
        {"name": "gemini_cli.tool_call", "attributes": {"function_name": "test", "success": False, "error": "Fail"}},
        {"name": "gemini_cli.api_response",
         "attributes": {"response_text": "My bad, fixing it.", "thoughts_token_count": 200, "output_token_count": 300}},
        {"name": "gemini_cli.tool_call", "attributes": {"function_name": "test", "success": True}}
    ]