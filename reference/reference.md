为了从 `gemini_cli` 生成的轨迹中筛选出高质量数据用于训练（如 SFT 或 RLHF），我们需要关注数据的**完整性**、**复杂度**、**正确性**和**推理能力**。

基于你提供的事件打点字段，我为你提取并设计了以下四个维度的关键指标。你可以利用这些指标构建评分系统或过滤规则。

---

### 维度一：任务完成度与稳定性 (Correctness & Stability)
*用于剔除失败、报错或无效的低质量轨迹。*

1.  **API 交互成功率**
    *   **来源:** `gemini_cli.api_response`, `gemini_cli.api_error`
    *   **逻辑:** 统计 `status_code` 为 200 的比例。
    *   **筛选建议:** **100%**。训练数据应尽量避免包含底层 API 错误的轨迹，除非你是为了训练模型的容错能力。

2.  **工具执行成功率 (Tool Execution Success Rate)**
    *   **来源:** `gemini_cli.tool_call` (`success`)
    *   **逻辑:** $\frac{\text{Count}(success=true)}{\text{Total Tool Calls}}$
    *   **筛选建议:** **> 90%**。如果一个轨迹中工具调用频繁失败（`success=false`），说明模型无法正确使用工具或参数生成错误，属于负样本。

3.  **无截断内容比例**
    *   **来源:** `gemini_cli.tool_output_truncated`
    *   **逻辑:** 检查是否存在该事件。
    *   **筛选建议:** **排除**或**降权**包含大量截断事件的轨迹。工具输出被截断意味着模型丢失了部分上下文，后续的推理可能是基于不完整信息进行的，容易产生幻觉。

4.  **完成原因合规性**
    *   **来源:** `gemini_cli.api_response` (`finish_reasons`)
    *   **逻辑:** 检查是否为 `STOP`。
    *   **筛选建议:** 剔除 `MAX_TOKENS`（输出被硬截断）或 `SAFETY`（被安全拦截）结束的轨迹。

---

### 维度二：任务复杂度与信息量 (Complexity & Richness)
*用于筛选出有深度、包含多步推理的高价值数据。*

5.  **推理深度 (Reasoning Density)**
    *   **来源:** `gemini_cli.api_response` (`thoughts_token_count`, `output_token_count`)
    *   **逻辑:** $\frac{\text{thoughts\_token\_count}}{\text{output\_token\_count}}$
    *   **分析:** 该比例越高，说明模型在输出最终结果前进行了越多的“思考”。对于训练 CoT（思维链）能力至关重要。
    *   **筛选建议:** 优先保留有显式思考过程的轨迹。

6.  **会话多轮交互数 (Multi-turn Depth)**
    *   **来源:** 聚合 `gemini_cli.user_prompt` 和 `gemini_cli.api_response`
    *   **逻辑:** 单个 `prompt_id` 对应的 API 响应次数或工具调用轮数。
    *   **筛选建议:** **> 1**。单轮对话通常较简单，多轮工具调用（Agent loop）更能体现 Agent 解决复杂问题的能力。

7.  **提示词丰富度**
    *   **来源:** `gemini_cli.user_prompt` (`prompt_length`)
    *   **筛选建议:** 剔除极短的 Prompt（如 "hi", "test"）。保留长度适中、意图明确的 Prompt。

8.  **代码/文件操作密度**
    *   **来源:** `gemini_cli.file_operation`
    *   **逻辑:** 统计 `operation` 为 `create` 或 `update` 的次数。
    *   **分析:** 实际产生代码或修改文件的轨迹通常比仅进行 `read` 操作的轨迹更有价值，代表模型具有“生产力”。

---

### 维度三：工具使用能力 (Tool Usage Capability)
*用于评估模型作为 Agent 的核心能力。*

9.  **工具多样性 (Tool Diversity)**
    *   **来源:** `gemini_cli.tool_call` (`function_name`)
    *   **逻辑:** 单个轨迹中不重复的 `function_name` 数量。
    *   **筛选建议:** 使用多种工具（例如：既搜索了信息，又修改了文件）的轨迹价值高于重复使用单一工具的轨迹。

10. **参数构建复杂度**
    *   **来源:** `gemini_cli.tool_call` (`function_args`)
    *   **逻辑:** 分析 JSON 参数的嵌套层级或长度。
    *   **筛选建议:** 能够正确构造复杂参数（如复杂的 SQL 查询或多参数函数调用）的轨迹是高质量的 SFT 样本。

11. **MCP 跨服务器协作**
    *   **来源:** `gemini_cli.tool_call` (`mcp_server_name`)
    *   **逻辑:** 检查是否调用了来自不同 `mcp_server_name` 的工具。
    *   **分析:** 跨服务器调用代表模型能够整合不同领域的工具能力。

---

### 维度四：效率与性能 (Efficiency)
*用于筛选“聪明且高效”的轨迹，避免冗余。*

12. **任务解决步数比 (Token Efficiency)**
    *   **来源:** `gemini_cli.api_response` (`total_token_count`), `gemini_cli.user_prompt`
    *   **逻辑:** 完成任务的总 Token 数与任务复杂度的比值。
    *   **筛选建议:** 在任务结果正确的前提下，优先选择 Token 消耗较少的轨迹（剔除死循环或啰嗦的轨迹）。

13. **无效路由尝试**
    *   **来源:** `gemini_cli.model_routing` (`failed`, `decision_source`)
    *   **逻辑:** 统计 `failed=true` 的次数。
    *   **筛选建议:** 剔除路由频繁失败的轨迹，这通常意味着系统在纠结使用哪个模型，导致上下文混乱。

---

### 建议的 SQL/Pandas 聚合逻辑示例

你可以按照 `prompt_id`（会话/任务ID）进行聚合，计算如下综合得分：

```python
# 伪代码逻辑
def calculate_quality_score(events):
    score = 0
    
    # 1. 基础过滤：必须成功
    if any(e.status_code != 200 for e in api_responses): return 0
    if any(e.success == False for e in tool_calls): score -= 50 # 严厉惩罚工具错误
    
    # 2. 奖励复杂推理
    avg_thought_ratio = sum(e.thoughts_token_count for e in api_responses) / sum(e.total_token_count)
    score += avg_thought_ratio * 100 
    
    # 3. 奖励实际产出 (Coding Agent)
    file_writes = count(e for e in file_ops if e.operation in ['create', 'update'])
    score += file_writes * 10
    
    # 4. 奖励多轮交互
    turns = len(tool_calls)
    if 2 <= turns <= 10: # 黄金区间
        score += 20
    elif turns > 20: # 可能死循环
        score -= 20
        
    return score
```

### 总结
如果你的目标是**微调一个强大的 Agent 模型**，建议按优先级提取：
1.  **`tool_call.success` == True** （必须正确执行）
2.  **`file_operation` 有实际产出** （有实际代码贡献）
3.  **`thoughts_token_count` 较高** （包含思维链）
4.  **无 `tool_output_truncated`** （上下文完整）