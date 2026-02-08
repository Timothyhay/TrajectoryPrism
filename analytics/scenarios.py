from dataclasses import dataclass, field
from typing import List, Dict

# 导入具体的实现类
from .filters import (
    BaseFilter,
    IntegrityFilter,
    ProductivityFilter,
    ContextTruncationFilter,
    PromptRichnessFilter
)
from .scorers import (
    BaseScorer,
    CodeProductionScorer,
    ReasoningDepthScorer,
    ToolDiversityScorer,
    ToolSuccessScorer,
    TurnEfficiencyScorer
)


@dataclass
class ScenarioConfig:
    """定义一个场景所需的过滤器组合和评分器组合"""
    name: str
    description: str
    filters: List[BaseFilter]
    scorers: List[BaseScorer]


# =========================================================================
# 场景 1: 通用 Agent，允许调用各种工具，也可以用作编码助手 (Default)
# 特点：均衡，奖励写代码，奖励效率
# =========================================================================
SCENARIO_GENERAL_CODING = ScenarioConfig(
    name="general_coding",
    description="Standard coding tasks. Rewards efficiency and code production.",
    filters=[
        IntegrityFilter(),
        ProductivityFilter(),  # 必须有产出
        ContextTruncationFilter(),
        PromptRichnessFilter()  # 必须有丰富的 Prompt
    ],
    scorers=[
        CodeProductionScorer(weight_per_line=0.5, max_score=20),
        ReasoningDepthScorer(max_score=20),
        ToolDiversityScorer(max_score=15),
        ToolSuccessScorer(max_score=30),
        TurnEfficiencyScorer(max_score=15, penalty_per_turn=2.0)  # 对慢任务惩罚较重
    ]
)

# =========================================================================
# 场景 2: SWE-bench (复杂仓库级问题)
# 特点：问题很难，可能需要几十轮；改动可能只是一行配置；不看重效率，只看重是否成功解决
# =========================================================================
SCENARIO_SWE_BENCH = ScenarioConfig(
    name="swe_bench",
    description="Repository level bug fixing. Tolerates long turns and small diffs.",
    filters=[
        IntegrityFilter(),
        ContextTruncationFilter(),
    ],
    scorers=[
        # [调整] 代码行数权重极大降低，只要有一点产出就给满分(因为可能是改一行配置)
        CodeProductionScorer(weight_per_line=5.0, max_score=10),

        # [调整] 推理深度极其重要，提高权重
        ReasoningDepthScorer(max_score=40),

        # [调整] 工具成功率依然重要
        ToolSuccessScorer(max_score=40),

        # [调整] 效率几乎不重要，甚至不扣分，只要做出来就行
        TurnEfficiencyScorer(max_score=10, optimal_turns=20, penalty_per_turn=0.1)
    ]
)

# =========================================================================
# 场景 3: 纯聊天/问答 (QA)
# 特点：不需要写文件，不需要代码产出
# =========================================================================
SCENARIO_CHAT_QA = ScenarioConfig(
    name="chat_qa",
    description="Pure logic reasoning or QA. No file operations required.",
    filters=[
        IntegrityFilter(),
        PromptRichnessFilter()
        # 禁用了 ProductivityFilter (不写文件)
    ],
    scorers=[
        # 禁用了 CodeProductionScorer
        ReasoningDepthScorer(max_score=50),  # 思考最重要
        TurnEfficiencyScorer(max_score=50)  # 简洁最重要
    ]
)

# 注册表
SCENARIO_REGISTRY = {
    "default": SCENARIO_GENERAL_CODING,
    "swe_bench": SCENARIO_SWE_BENCH,
    "qa": SCENARIO_CHAT_QA
}


def get_scenario(name: str) -> ScenarioConfig:
    return SCENARIO_REGISTRY.get(name, SCENARIO_GENERAL_CODING)