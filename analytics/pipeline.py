from typing import Dict, List, Optional
from .schemas import TraceData, AnalysisResult, DatasetType
from .scenarios import get_scenario, ScenarioConfig
from .adapters import OpenAIAdapter
from .converters import OpenAIConverter


class TracePipeline:
    def __init__(self, scenario_name: str = "default"):
        """
        初始化 Pipeline，加载指定场景配置
        :param scenario_name: 'default', 'swe_bench', 'qa'
        """
        self.config: ScenarioConfig = get_scenario(scenario_name)
        print(f"🔧 Pipeline initialized with scenario: {self.config.name}")
        print(f"   - Active Filters: {len(self.config.filters)}")
        print(f"   - Active Scorers: {len(self.config.scorers)}")

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
        # 1. 使用配置中的 Filters
        reasons = []
        for f in self.config.filters:
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

        # 2. 使用配置中的 Scorers
        total_score = 0.0
        for scorer in self.config.scorers:
            total_score += scorer.calculate(trace)

        total_score = round(total_score, 2)

        # 3. 分类 (逻辑通用)
        # 数据集分类 (Classification: SFT, RLHF)
        # 检查是否发生过需要修正的错误
        # OpenAIAdapter 会尝试从文本中推断这些计数，如果无法推断则为 0
        is_recovery = (trace.metrics.get('gemini_cli.agent.recovery_attempt.count', 0) > 0 or
                       trace.metrics.get('gemini_cli.chat.content_retry.count', 0) > 0)
        ds_type = DatasetType.RLHF if is_recovery else DatasetType.SFT

        # 4. 转换
        openai_msgs = OpenAIConverter.convert(trace)

        return AnalysisResult(
            trace_id=trace.trace_id,
            score=total_score,
            dataset_type=ds_type,
            reasons=[],
            openai_messages=openai_msgs,
            metadata=trace.metrics
        )