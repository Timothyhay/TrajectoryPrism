from typing import Dict, Any, List
from .schemas import TraceData, AnalysisResult, DatasetType
from .filters import ACTIVE_FILTERS
from .scorers import QualityScorer
from .converters import OpenAIConverter
from .adapters import OpenAIAdapter


class TracePipeline:
    def __init__(self):
        self.scorer = QualityScorer()

    def process_trace(self, trace_id: str, metrics: Dict, events: List) -> AnalysisResult:
        """入口 1: 处理来自 OpenTelemetry 的原始数据"""
        trace = TraceData(trace_id=trace_id, metrics=metrics, events=events)
        return self._analyze(trace)

    def process_openai_trace(self, trace_id: str, messages: List[Dict]) -> AnalysisResult:
        """入口 2: 处理来自 OpenAI JSONL 的文本数据"""
        # 使用 Adapter 将文本逆向解析为标准 TraceData
        trace = OpenAIAdapter.to_trace_data(trace_id, messages)
        return self._analyze(trace)

    def _analyze(self, trace: TraceData) -> AnalysisResult:
        """
        核心分析逻辑：过滤 -> 打分 -> 分类 -> 格式化
        """

        # -------------------------------------------------------
        # 1. 硬性过滤 (Hard Filters)
        # -------------------------------------------------------
        reasons = []
        for filter_instance in ACTIVE_FILTERS:
            # check 方法返回 None 表示通过，返回 string 表示拒绝原因
            fail_reason = filter_instance.check(trace)
            if fail_reason:
                reasons.append(fail_reason)

        # 如果存在任何拒绝原因，直接标记为 REJECTED 并返回
        if reasons:
            return AnalysisResult(
                trace_id=trace.trace_id,
                score=0.0,
                dataset_type=DatasetType.REJECTED,
                reasons=reasons,
                openai_messages=None,  # 被拒绝的数据通常不需要转换消息体，节省资源
                metadata=trace.metrics
            )

        # -------------------------------------------------------
        # 2. 质量打分 (Scoring)
        # -------------------------------------------------------
        # Scorer 内部会处理缺失指标的情况（默认为0）
        score = self.scorer.evaluate(trace)

        # -------------------------------------------------------
        # 3. 数据集分类 (Classification: SFT vs RLHF)
        # -------------------------------------------------------
        # 检查是否发生过需要修正的错误
        # 注意：OpenAIAdapter 会尝试从文本中推断这些计数，如果无法推断则为 0
        recovery_cnt = trace.metrics.get('gemini_cli.agent.recovery_attempt.count', 0)
        retry_cnt = trace.metrics.get('gemini_cli.chat.content_retry.count', 0)

        # 逻辑：只要发生过 1 次以上的恢复尝试或内容重试，
        # 且通过了上面的 Hard Filters（意味着最终成功了），
        # 则该轨迹非常适合用于 RLHF/DPO 的 Preference 训练。
        if recovery_cnt > 0 or retry_cnt > 0:
            dataset_type = DatasetType.RLHF
        else:
            # 没有任何错误修正记录，是一条完美轨迹，适合 SFT
            dataset_type = DatasetType.SFT

        # -------------------------------------------------------
        # 4. 格式转换 (Standardization)
        # -------------------------------------------------------
        # 将内部 Event 结构转回标准的 OpenAI 训练格式。
        # 即使输入就是 OpenAI 格式，这一步也能起到清洗和标准化的作用（如统一 System Prompt）。
        openai_msgs = OpenAIConverter.convert(trace)

        # -------------------------------------------------------
        # 5. 构建最终结果
        # -------------------------------------------------------
        return AnalysisResult(
            trace_id=trace.trace_id,
            score=score,
            dataset_type=dataset_type,
            reasons=[],  # 空列表代表通过
            openai_messages=openai_msgs,
            metadata=trace.metrics  # 保留原始指标以便调试
        )