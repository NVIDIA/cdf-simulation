## Description: <br>
Component-owned guidance for the cdf-simulation half of the Airflow-triggered PM prompt templates (DIG Flow Catalog) — golden THT board, THT pin-geometry variation, good/clean SMT images, structural pose defects, and solder texture defects — that turns one matched prompt into one validated `cdf.CDFJob` build and its compiled JobOrder payload, then hands that to the SDG workflow-operations skill without executing a render itself. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers running Airflow-triggered, non-interactive DIG synthetic-data pipelines use this skill to map a fixed catalog prompt or SQA reference flow onto a validated Cosmos Data Factory (CDF) PCB-AOI job and compiled JobOrder payload for golden boards, pin-geometry variations, and structural or solder defects, applying documented defaults instead of asking clarifying questions. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Fixed defaults](references/fixed-defaults.md) <br>
- [Payload contract](references/payload-contract.md) <br>
- [Reference flows](references/reference-flows.md) <br>
- [Class filter taxonomy](references/class-filter-taxonomy.md) <br>
- [Verification status](references/verification-status.md) <br>
- [Interactive CDF skill (skills/cdf)](../cdf/SKILL.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Configuration instructions, Analysis] <br>
**Output Format:** [Python `cdf.CDFJob` build script plus a compiled `JobOrderPlan` serialized as JSON, with a Markdown resolved-inputs summary] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [One variant per payload; compiles and hands off only, never renders] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
8 evaluation tasks (8 positive), 1 attempt per task, each run in an isolated sandbox pod (`k8s-sandbox`), with a three-tier evaluation (static validation, semantic deduplication, live agent evaluation) comparing skill-assisted runs against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from the `security` signal. <br>
- Correctness: Is the answer correct? Scored from the `accuracy` signal. <br>
- Discoverability: Was the right skill loaded when needed? Scored from the `skill_execution` signal. <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of `goal_accuracy` and `behavior_check`. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% `skill_efficiency` (tool-call productivity) and 50% `token_efficiency`. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 88.5% — baseline ran, but no comparable score was available; uplift unavailable | 73.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 75.0% → 100.0% (+25.0 points) | 87.5% → 100.0% (+12.5 points) |
| Correctness | 22.5% → 87.5% (+65.0 points) | 17.5% → 57.5% (+40.0 points) |
| Discoverability | 91.3% — baseline ran, but no comparable score was available; uplift unavailable | 76.3% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 39.2% → 76.0% (+36.8 points) | 24.8% → 39.6% (+14.8 points) |
| Efficiency | 87.9% — baseline ran, but no comparable score was available; uplift unavailable | 94.7% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
