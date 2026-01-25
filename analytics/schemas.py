from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class DatasetType(Enum):
    SFT = "sft"  # 监督微调：完美轨迹
    RLHF = "rlhf"  # 强化学习：包含自我修正
    REJECTED = "rejected"  # 拒绝：低质量


@dataclass
class TraceData:
    """原始轨迹数据容器"""
    trace_id: str
    metrics: Dict[str, Any]
    events: List[Dict[str, Any]]

    @property
    def config(self) -> Dict[str, Any]:
        """提取配置信息的便捷属性"""
        return next((e.get('attributes', {}) for e in self.events if e['name'] == 'gemini_cli.config'), {})


@dataclass
class AnalysisResult:
    """分析结果容器"""
    trace_id: str
    score: float
    dataset_type: DatasetType
    reasons: List[str]
    openai_messages: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)  # 用于存储额外统计，如token数