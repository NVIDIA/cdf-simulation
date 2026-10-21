# Skill Benchmark: cdf-simulation

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-simulation`
- Evaluation date: 2026-10-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 8 evaluation tasks (8 positive)
- Dataset digest: `sha256:2e5e11881e77f0623a60210ec155e7d091f33ef15aa5dd0ec3a558c1188ab83c` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 87.4% — baseline ran, but no comparable score was available; uplift unavailable | 76.9% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 87.5% → 100.0% (+12.5 points) | 100.0% → 100.0% (±0.0 points) |
| Correctness | 7.5% → 90.0% (+82.5 points) | 0.0% → 60.0% (+60.0 points) |
| Discoverability | 95.0% — baseline ran, but no comparable score was available; uplift unavailable | 87.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 28.1% → 64.0% (+35.9 points) | 22.7% → 41.7% (+19.0 points) |
| Efficiency | 88.2% — baseline ran, but no comparable score was available; uplift unavailable | 95.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 2,733,289 | 1,732,238 | +1,001,051 | +57.79% | skill 8/8; base 8/8 |
| claude-code | 1 | 542,856 | 158,201 | +384,655 | +243.14% | skill 1/1; base 1/1 |
| claude-code | 2 | 911,893 | 287,270 | +624,623 | +217.43% | skill 1/1; base 1/1 |
| claude-code | 3 | 397,515 | 151,136 | +246,379 | +163.02% | skill 1/1; base 1/1 |
| claude-code | 4 | 73,529 | 156,564 | -83,035 | -53.04% | skill 1/1; base 1/1 |
| claude-code | 5 | 74,296 | 189,913 | -115,617 | -60.88% | skill 1/1; base 1/1 |
| claude-code | 6 | 347,894 | 154,046 | +193,848 | +125.84% | skill 1/1; base 1/1 |
| claude-code | 7 | 73,403 | 418,986 | -345,583 | -82.48% | skill 1/1; base 1/1 |
| claude-code | 8 | 311,903 | 216,122 | +95,781 | +44.32% | skill 1/1; base 1/1 |
| codex | All cases | 1,301,096 | 801,442 | +499,654 | +62.34% | skill 8/8; base 8/8 |
| codex | 1 | 165,175 | 88,149 | +77,026 | +87.38% | skill 1/1; base 1/1 |
| codex | 2 | 174,009 | 133,193 | +40,816 | +30.64% | skill 1/1; base 1/1 |
| codex | 3 | 556,838 | 112,666 | +444,172 | +394.24% | skill 1/1; base 1/1 |
| codex | 4 | 50,287 | 80,381 | -30,094 | -37.44% | skill 1/1; base 1/1 |
| codex | 5 | 50,526 | 68,808 | -18,282 | -26.57% | skill 1/1; base 1/1 |
| codex | 6 | 164,393 | 86,300 | +78,093 | +90.49% | skill 1/1; base 1/1 |
| codex | 7 | 50,477 | 84,285 | -33,808 | -40.11% | skill 1/1; base 1/1 |
| codex | 8 | 89,391 | 147,660 | -58,269 | -39.46% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 4,034,385 | 2,533,680 | +1,500,705 | +59.23% | skill 16/16; base 16/16 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 6 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 8 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Large skill (9503 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/cdf-simulation/SKILL.md`)
- **MEDIUM** SCHEMA/line_count: SKILL.md has 587 lines (limit: 500) (`skills/cdf-simulation/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: Description very long (656 chars, recommend 50-150) (`skills/cdf-simulation/SKILL.md`)
- **LOW** QUALITY/quality_discoverability: No '## Purpose' section (`skills/cdf-simulation/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No prerequisites/requirements documented (`skills/cdf-simulation/SKILL.md`)
- 1 additional finding(s) are available in the full evaluation artifacts.

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
