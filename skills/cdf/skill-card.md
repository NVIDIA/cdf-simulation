## Description: <br>
Create and validate declarative CDFJob Python for PCB-AOI operations, pin variation, solder, annotations, capture, and JobOrders. Not for YAML/Docker workflows. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers generating synthetic PCB-AOI training data by authoring declarative CDFJob Python scripts for board scene configuration, defect injection, pin variation, solder generation, annotation, and tiled capture. <br>

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
- [CDF API Specification](../../cdf/API_SPEC.md) <br>
- [CDF Capabilities](../../cdf/CAPABILITIES.md) <br>
- [Repository Cookbook](../../COOKBOOK.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Shell commands] <br>
**Output Format:** [Python scripts with argparse CLI] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Compile-only by default; Isaac Sim execution opt-in via --execute flag] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
7 evaluation tasks (7 positive), each run in an isolated k8s-sandbox pod with 1 attempt per task. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill is safe to use: checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded when needed. <br>
- Effectiveness: Whether the skill helped complete the task (50% goal completion + 50% expected workflow adherence). <br>
- Efficiency: Whether the skill avoided wasted tool calls and token usage (50% tool-call productivity + 50% token efficiency). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool-call productivity (routing is scored under Discoverability). <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 85.8% | 80.5% |
| Security | 71.4% → 100.0% (+28.6 points) | 57.1% → 100.0% (+42.9 points) |
| Correctness | 37.1% → 88.6% (+51.5 points) | 48.6% → 88.6% (+40.0 points) |
| Discoverability | 86.4% | 70.0% |
| Effectiveness | 24.3% → 78.2% (+53.9 points) | 37.2% → 62.6% (+25.4 points) |
| Efficiency | 75.9% | 81.2% |

## Skill Version(s): <br>
1.2.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
