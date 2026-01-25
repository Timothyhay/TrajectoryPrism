from typing import Dict, Any, List
from .schemas import TraceData, AnalysisResult, DatasetType
from .filters import ACTIVE_FILTERS
from .scorers import QualityScorer
from .converters import OpenAIConverter


class TracePipeline:
    def __init__(self):
        self.scorer = QualityScorer()

    def process_trace(self, trace_id: str, metrics: Dict, events: List) -> AnalysisResult:
        # 1. 构建对象
        trace = TraceData(trace_id=trace_id, metrics=metrics, events=events)

        # 2. 运行硬性过滤器
        reasons = []
        for f in ACTIVE_FILTERS:
            error = f.check(trace)
            if error:
                reasons.append(error)

        # 如果过滤失败，提前返回
        if reasons:
            return AnalysisResult(
                trace_id=trace_id,
                score=0.0,
                dataset_type=DatasetType.REJECTED,
                reasons=reasons
            )

        # 3. 计算得分
        score = self.scorer.evaluate(trace)

        # 4. 分类 (SFT vs RLHF)
        # 逻辑：有恢复尝试或重试记录，且最终成功（已通过过滤器） -> RLHF
        is_recovery = (trace.metrics.get('gemini_cli.agent.recovery_attempt.count', 0) > 0 or
                       trace.metrics.get('gemini_cli.chat.content_retry.count', 0) > 0)

        ds_type = DatasetType.RLHF if is_recovery else DatasetType.SFT

        # 5. 格式转换
        openai_msgs = OpenAIConverter.convert(trace)

        return AnalysisResult(
            trace_id=trace_id,
            score=score,
            dataset_type=ds_type,
            reasons=[],
            openai_messages=openai_msgs
        )