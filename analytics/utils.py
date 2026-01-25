def get_mock_data():
    """返回 (trace_id, metrics, events) 的生成器"""
    # 样本 1: 高质量 SFT
    yield "trace_001", {
        "gemini_cli.exit.fail.count": 0,
        "gemini_cli.lines.changed": 50,
        "gemini_cli.agent.turns": 5,
        "gemini_cli.file.operation.count": 1
    }, [
        {"name": "gemini_cli.config", "attributes": {"core_tools_enabled": "code_edit"}},
        {"name": "gemini_cli.user_prompt", "attributes": {"prompt": "Write a fast sort function"}},
        {"name": "gemini_cli.api_response",
         "attributes": {"response_text": "Sure.", "thoughts_token_count": 100, "output_token_count": 200}},
        {"name": "gemini_cli.tool_call",
         "attributes": {"function_name": "write_file", "function_args": {"file": "sort.py"}, "success": True}}
    ]

    # 样本 2: 失败样本
    yield "trace_002_fail", {
        "gemini_cli.exit.fail.count": 1,  # 致命错误
        "gemini_cli.lines.changed": 0
    }, []

    # 样本 3: RLHF 修正样本
    yield "trace_003_rlhf", {
        "gemini_cli.exit.fail.count": 0,
        "gemini_cli.lines.changed": 10,
        "gemini_cli.agent.turns": 8,
        "gemini_cli.agent.recovery_attempt.count": 1
    }, [
        {"name": "gemini_cli.config", "attributes": {}},
        {"name": "gemini_cli.user_prompt", "attributes": {"prompt": "Fix logic"}},
        {"name": "gemini_cli.tool_call", "attributes": {"function_name": "test", "success": False, "error": "Fail"}},
        {"name": "gemini_cli.api_response",
         "attributes": {"response_text": "My bad, fixing it.", "thoughts_token_count": 200, "output_token_count": 300}},
        {"name": "gemini_cli.tool_call", "attributes": {"function_name": "test", "success": True}}
    ]