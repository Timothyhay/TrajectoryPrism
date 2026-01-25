# 💎 TrajectoryPrism

> *Separate the signal from the noise. Refract the raw into the refined.*

**Turn your Agent's past execution logs into its future training textbooks.**

**TrajectoryPrism** is an intelligent ETL pipeline designed to mine high-value training samples from OpenTelemetry data. Just as a prism splits white light into colors, this tool analyzes raw agent trajectories and separates them based on quality and intent:

*   ✨ **Perfect Traces → SFT:** High-scoring, clean executions for Supervised Fine-Tuning.
*   🔧 **Correction Traces → DPO/RLHF:** Trajectories containing successful self-corrections for Preference Optimization.
*   🗑️ **Noise → Discarded:** Automatically filters out hallucinations, errors, and low-density thoughts.

**Key Features:**
*   **Hard Filters:** Security checks & status validation.
*   **Smart Scoring:** Weighted evaluation based on CoT density, code changes, and tool complexity.
*   **Format Converter:** Exports directly to OpenAI-compatible Chat format.



