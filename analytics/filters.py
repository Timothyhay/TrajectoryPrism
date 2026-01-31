from abc import ABC, abstractmethod
from typing import List, Optional
from .schemas import TraceData


class BaseFilter(ABC):
    @abstractmethod
    def check(self, trace: TraceData) -> Optional[str]:
        """返回 None 表示通过，返回字符串表示拒绝原因"""
        pass


class IntegrityFilter(BaseFilter):
    """检查系统层面的完整性"""

    def check(self, trace: TraceData) -> Optional[str]:
        if trace.metrics.get('gemini_cli.exit.fail.count', 0) > 0:
            return "CLI_EXIT_FAILURE"
        if trace.metrics.get('gemini_cli.chat.content_retry_failure.count', 0) > 0:
            return "CONTENT_RETRY_FAILURE"
        return None


class ProductivityFilter(BaseFilter):
    """检查是否有有效产出"""

    def check(self, trace: TraceData) -> Optional[str]:
        file_ops = trace.metrics.get('gemini_cli.file.operation.count', 0)
        lines_changed = trace.metrics.get('gemini_cli.lines.changed', 0)

        # 如果做了文件操作但没有行数变更，视为无效操作
        if file_ops > 0 and lines_changed == 0:
            return "INEFFECTIVE_FILE_OPERATION"
        return None


class ContextTruncationFilter(BaseFilter):
    """检查上下文是否完整"""

    def check(self, trace: TraceData) -> Optional[str]:
        truncated_events = [e for e in trace.events if e['name'] == 'gemini_cli.tool_output_truncated']
        if truncated_events:
            return "TOOL_OUTPUT_TRUNCATED"
        return None


class PromptRichnessFilter(BaseFilter):
    """
    [26/01/31 新增] 提示词丰富度过滤
    规则：剔除 prompt_length 过短的轨迹，这些通常是无效的测试或寒暄。
    """

    def check(self, trace: TraceData) -> Optional[str]:
        # 寻找第一个用户 Prompt 事件
        user_prompt_event = next(
            (e for e in trace.events if e['name'] == 'gemini_cli.user_prompt'),
            None
        )

        # 如果连 Prompt 事件都没有，视为数据缺失
        if not user_prompt_event:
            return "MISSING_USER_PROMPT"

        # 获取长度 (兼容 OTel 字段和 Adapter 生成的字段)
        length = user_prompt_event['attributes'].get('prompt_length', 0)

        # 阈值设定：少于 10 个字符通常无法构成一个复杂的 Agent 任务
        # 例如 "fix bug" (7 chars) -> Reject
        # "Fix the login bug in auth.py" (28 chars) -> Pass
        if length < 10:
            return f"PROMPT_TOO_SHORT (len={length})"

        return None

# 注册所有活跃的过滤器
ACTIVE_FILTERS = [
    IntegrityFilter(),
    ProductivityFilter(),
    ContextTruncationFilter(),
    PromptRichnessFilter()
]