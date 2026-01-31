from typing import Dict, Any, List
from .schemas import TraceData, AnalysisResult, DatasetType
from .filters import ACTIVE_FILTERS
from .scorers import ACTIVE_SCORERS
from .converters import OpenAIConverter
from .adapters import OpenAIAdapter


class TracePipeline:
    def process_trace(self, trace_id: str, metrics: Dict, events: List) -> AnalysisResult:
        trace = TraceData(trace_id=trace_id, metrics=metrics, events=events)
        return self._analyze(trace)

    def process_openai_trace(self, trace_id: str, messages: List[Dict]) -> AnalysisResult:
        trace = OpenAIAdapter.to_trace_data(trace_id, messages)
        return self._analyze(trace)

    def _analyze(self, trace: TraceData) -> AnalysisResult:
        """
        核心分析逻辑：过滤 -> 打分 -> 分类 -> 格式化
        """
        # 1. 运行过滤器 (Filters)
        reasons = []
        for f in ACTIVE_FILTERS:
            error = f.check(trace)
            if error: reasons.append(error)

        if reasons:
            return AnalysisResult(
                trace_id=trace.trace_id,
                score=0.0,
                dataset_type=DatasetType.REJECTED,
                reasons=reasons,
                metadata=trace.metrics
            )

        # 2. Scorer
        # 动态遍历所有激活的评分逻辑
        total_score = 0.0
        for scorer in ACTIVE_SCORERS:
            total_score += scorer.calculate(trace)

        # 保留两位小数
        total_score = round(total_score, 2)

        # 3. Classification
        # 数据集分类 (Classification: SFT, RLHF)
        # 检查是否发生过需要修正的错误
        # OpenAIAdapter 会尝试从文本中推断这些计数，如果无法推断则为 0
        recovery_cnt = trace.metrics.get('gemini_cli.agent.recovery_attempt.count', 0)
        retry_cnt = trace.metrics.get('gemini_cli.chat.content_retry.count', 0)

        # 只要发生过 1 次以上的恢复尝试或内容重试，且通过了上面的 Hard Filters（意味着最终成功了），
        # 则该轨迹非常适合用于 RLHF/DPO 的 Preference 训练。
        dataset_type = DatasetType.RLHF if (recovery_cnt > 0 or retry_cnt > 0) else DatasetType.SFT

        # 4. Conversion
        # 将内部 Event 结构转回标准的 OpenAI 轨迹格式
        # 即使输入就是 OpenAI 格式，这一步也能起到清洗和标准化的作用（如统一 System Prompt）。
        openai_msgs = OpenAIConverter.convert(trace)

        return AnalysisResult(
            trace_id=trace.trace_id,
            score=total_score,
            dataset_type=dataset_type,
            reasons=[],
            openai_messages=openai_msgs,
            metadata=trace.metrics
        )

