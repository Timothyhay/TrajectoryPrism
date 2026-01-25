from analytics.pipeline import TracePipeline
from analytics.converters import ReportGenerator
from analytics.utils import get_mock_data
from analytics.schemas import DatasetType


mock_openai_data = [
    {
        "role": "user",
        "content": "Create a hello world python script."
    },
    {
        "role": "assistant",
        "tool_calls": [{
            "id": "call_123",
            "function": {
                "name": "write_file",
                "arguments": "{\"filename\": \"hello.py\", \"content\": \"print('hello')\\nprint('world')\"}"
            }
        }]
    },
    {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": "File created successfully."
    }
]



def main():
    pipeline = TracePipeline()
    results = []

    print("🚀 Starting Analysis Pipeline...")

    print("--- 1. Analyzing OTel Data ---")

    # 1. 模拟遍历数据源
    for trace_id, metrics, events in get_mock_data():
        result = pipeline.process_trace(trace_id, metrics, events)
        results.append(result)

        status_icon = "✅" if result.dataset_type != DatasetType.REJECTED else "❌"
        print(f"{status_icon} Processed {trace_id}: Score={result.score} Type={result.dataset_type.value}")

    # 2. 生成 HTML 报告
    ReportGenerator.generate_html(results, "final_analysis_report.html")

    # 3. 导出 SFT 数据集 (JSONL)
    sft_data = [r.openai_messages for r in results if r.dataset_type == DatasetType.SFT]
    print(f"\n📦 Extracted {len(sft_data)} SFT traces for fine-tuning.")

    print("\n--- 2. Analyzing Raw OpenAI Traj Data ---")
    # 适配器会自动：
    # 1. 发现 write_file 工具
    # 2. 解析 content 参数，计算出 lines.changed = 2
    # 3. 统计 turns = 1
    # 4. 检测 tool output 没有 "error"，标记 success = True

    result = pipeline.process_openai_trace("raw_trace_001", mock_openai_data)

    print(f"Result: {result.dataset_type.value}")
    print(f"Score:  {result.score}")
    # 分数计算：
    # + 2行代码 * 0.5 = 1分
    # + 交互区间(1轮) = 不加分 (假设区间是3-15)
    # + 工具成功率 100% = 30分
    # 总分约 31分

    print(f"Metrics (Inferred): {result.metadata}")  # 你可以在 _analyze 里把 trace.metrics 塞进 metadata 查看


if __name__ == "__main__":
    main()