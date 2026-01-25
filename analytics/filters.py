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


# 注册所有活跃的过滤器
ACTIVE_FILTERS = [
    IntegrityFilter(),
    ProductivityFilter(),
    ContextTruncationFilter()
]