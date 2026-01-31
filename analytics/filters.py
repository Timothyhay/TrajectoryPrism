from abc import ABC, abstractmethod
from typing import List, Optional
from .schemas import TraceData
import tiktoken

# 使用通用的 cl100k_base (GPT-4/3.5/Gemini 通用近似)
_TOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
HAS_TIKTOKEN = True
# True:  如果缺少必要的打点字段，直接忽略该过滤器（让轨迹通过）。
# False: 如果缺少必要的打点字段，视为不合规，拒绝该轨迹（严格模式）。
IGNORE_MISSING_FIELDS = True

class BaseFilter(ABC):
    def handle_missing_data(self, field_name: str) -> Optional[str]:
        """
        统一处理数据缺失的逻辑
        :param field_name: 缺失的字段名，用于记录日志
        :return: None 表示忽略/通过，String 表示拒绝原因
        """
        if IGNORE_MISSING_FIELDS:
            # 宽容模式：缺失数据不作为拒绝理由，跳过此过滤器
            # print(f"WARNING: Missing field {field_name}")
            return None
        else:
            # 严格模式：缺失数据直接拒绝
            return f"MISSING_REQUIRED_FIELD: {field_name}"

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
        if 'gemini_cli.file.operation.count' not in trace.metrics:
            return self.handle_missing_data("metric: file.operation.count")

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
        # 寻找 User Prompt 事件
        user_prompt_event = next(
            (e for e in trace.events if e['name'] == 'gemini_cli.user_prompt'),
            None
        )

        # 连 prompt 事件都没打点
        if not user_prompt_event:
            return self.handle_missing_data("event: gemini_cli.user_prompt")

        attrs = user_prompt_event.get('attributes', {})

        # 有事件，但没有 prompt_length 属性 (旧数据或未计算)
        if 'prompt_length' not in attrs:
            # 如果有 'prompt' 文本，现场算一个
            if 'prompt' in attrs and isinstance(attrs['prompt'], str):
                length = len(_TOKEN_ENCODING.encode(attrs['prompt']))
            else:
                return self.handle_missing_data("attribute: prompt_length")
        else:
            length = attrs['prompt_length']

        # 2. 只有在数据存在的情况下，才执行真正的业务逻辑校验
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