## Description: <br>
Create, modify, or explain Python scripts using the CDF Simulation CDFJob API for PCB-AOI operations, pin variation, solder, annotations, capture, and JobOrders. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers writing, modifying, explaining, or validating Python authoring scripts for the declarative CDFJob API. They use it to produce correct, runnable scripts that generate labeled synthetic PCB-AOI datasets and integrate the API with caller-owned code. <br>

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
- [CDF authoring API specification](../../cdf/API_SPEC.md) <br>
- [CDF supported surfaces and capabilities](../../cdf/CAPABILITIES.md) <br>
- [CDF cookbook](../../COOKBOOK.md) <br>
- [cdf-simulation skill](../cdf-simulation/SKILL.md) <br>
- [PCB-AOI configuration guide](../../pcb_aoi/config/README.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code] <br>
**Output Format:** [Python scripts with brief Markdown explanation] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (7 positive), one attempt per task, each run in its own isolated k8s-sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use — free of unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and activated when needed, avoiding decoys. <br>
- Effectiveness: Whether the skill helped complete the user's goal and followed the expected workflow. <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 83.8% | 81.7% |
| Security | 57.1% → 71.4% (+14.3 points) | 42.9% → 85.7% (+42.8 points) |
| Correctness | 31.4% → 97.1% (+65.7 points) | 34.3% → 85.7% (+51.4 points) |
| Discoverability | 91.4% | 77.1% |
| Effectiveness | 26.6% → 85.4% (+58.8 points) | 36.2% → 76.0% (+39.8 points) |
| Efficiency | 73.5% | 84.1% |

## Skill Version(s): <br>
1.2.0 (source: pyproject.toml, README) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
