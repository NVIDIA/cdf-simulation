# Skill Benchmark: cdf-simulation

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-simulation`
- Evaluation date: 2026-10-20
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 8 evaluation tasks (8 positive)
- Dataset digest: `sha256:5ca0180bb6dd1a78f6b5f12b11b51d3e5e9971c0a73a9d9ed0a801f7a716dfb6` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 87.8% — baseline ran, but no comparable score was available; uplift unavailable | 75.6% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 75.0% → 100.0% (+25.0 points) | 75.0% → 100.0% (+25.0 points) |
| Correctness | 30.0% → 90.0% (+60.0 points) | 15.0% → 65.0% (+50.0 points) |
| Discoverability | 90.4% — baseline ran, but no comparable score was available; uplift unavailable | 74.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 42.5% → 69.7% (+27.2 points) | 22.3% → 45.8% (+23.5 points) |
| Efficiency | 88.7% — baseline ran, but no comparable score was available; uplift unavailable | 92.6% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,877,513 | 1,876,191 | +1,322 | +0.07% | skill 8/8; base 8/8 |
| claude-code | 1 | 324,466 | 220,120 | +104,346 | +47.40% | skill 1/1; base 1/1 |
| claude-code | 2 | 265,084 | 223,788 | +41,296 | +18.45% | skill 1/1; base 1/1 |
| claude-code | 3 | 256,863 | 246,155 | +10,708 | +4.35% | skill 1/1; base 1/1 |
| claude-code | 4 | 75,823 | 567,772 | -491,949 | -86.65% | skill 1/1; base 1/1 |
| claude-code | 5 | 77,635 | 156,106 | -78,471 | -50.27% | skill 1/1; base 1/1 |
| claude-code | 6 | 227,474 | 29,389 | +198,085 | +674.01% | skill 1/1; base 1/1 |
| claude-code | 7 | 77,633 | 248,105 | -170,472 | -68.71% | skill 1/1; base 1/1 |
| claude-code | 8 | 572,535 | 184,756 | +387,779 | +209.89% | skill 1/1; base 1/1 |
| codex | All cases | 2,007,398 | 1,103,187 | +904,211 | +81.96% | skill 8/8; base 8/8 |
| codex | 1 | 299,772 | 88,071 | +211,701 | +240.38% | skill 1/1; base 1/1 |
| codex | 2 | 176,004 | 228,008 | -52,004 | -22.81% | skill 1/1; base 1/1 |
| codex | 3 | 1,006,658 | 97,729 | +908,929 | +930.05% | skill 1/1; base 1/1 |
| codex | 4 | 30,251 | 169,408 | -139,157 | -82.14% | skill 1/1; base 1/1 |
| codex | 5 | 71,288 | 71,072 | +216 | +0.30% | skill 1/1; base 1/1 |
| codex | 6 | 13,526 | 13,336 | +190 | +1.42% | skill 1/1; base 1/1 |
| codex | 7 | 71,252 | 98,198 | -26,946 | -27.44% | skill 1/1; base 1/1 |
| codex | 8 | 338,647 | 337,365 | +1,282 | +0.38% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 3,884,911 | 2,979,378 | +905,533 | +30.39% | skill 16/16; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Large skill (10826 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/cdf-simulation/SKILL.md`)
- **MEDIUM** SCHEMA/line_count: SKILL.md has 655 lines (limit: 500) (`skills/cdf-simulation/SKILL.md`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/cdf/SKILL.md (`SKILL.md:39`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/cdf/SKILL.md (`SKILL.md:177`)
- **MEDIUM** SECURITY/Skill Enumeration (AS3): Agent Snooping: skills/cdf/SKILL.md (`SKILL.md:237`)
- 7 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
