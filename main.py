from analytics.pipeline import TracePipeline
from analytics.converters import ReportGenerator
from analytics.utils import get_mock_data
from analytics.schemas import DatasetType


def main():
    pipeline = TracePipeline()
    results = []

    print("🚀 Starting Analysis Pipeline...")

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


if __name__ == "__main__":
    main()