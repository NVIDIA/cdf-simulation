# Skill Benchmark: cdf-simulation-api

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-simulation-api`
- Evaluation date: 2026-10-21
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (7 positive)
- Dataset digest: `sha256:1d5c2e719d753565680a66a3f17d6eda58745bd5ad57842e46c74d413728c279` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 83.8% — baseline ran, but no comparable score was available; uplift unavailable | 81.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 57.1% → 71.4% (+14.3 points) | 42.9% → 85.7% (+42.8 points) |
| Correctness | 31.4% → 97.1% (+65.7 points) | 34.3% → 85.7% (+51.4 points) |
| Discoverability | 91.4% — baseline ran, but no comparable score was available; uplift unavailable | 77.1% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 26.6% → 85.4% (+58.8 points) | 36.2% → 76.0% (+39.8 points) |
| Efficiency | 73.5% — baseline ran, but no comparable score was available; uplift unavailable | 84.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 13,435,605 | 4,175,467 | +9,260,138 | +221.77% | skill 7/7; base 7/7 |
| claude-code | 1 | 2,015,629 | 1,587,590 | +428,039 | +26.96% | skill 1/1; base 1/1 |
| claude-code | 2 | 2,142,193 | 503,992 | +1,638,201 | +325.05% | skill 1/1; base 1/1 |
| claude-code | 3 | 1,530,352 | 381,448 | +1,148,904 | +301.20% | skill 1/1; base 1/1 |
| claude-code | 4 | 2,436,513 | 594,532 | +1,841,981 | +309.82% | skill 1/1; base 1/1 |
| claude-code | 5 | 1,692,189 | 298,709 | +1,393,480 | +466.50% | skill 1/1; base 1/1 |
| claude-code | 6 | 1,298,483 | 591,883 | +706,600 | +119.38% | skill 1/1; base 1/1 |
| claude-code | 8 | 2,320,246 | 217,313 | +2,102,933 | +967.70% | skill 1/1; base 1/1 |
| codex | All cases | 3,445,537 | 1,778,284 | +1,667,253 | +93.76% | skill 7/7; base 7/7 |
| codex | 1 | 357,411 | 447,389 | -89,978 | -20.11% | skill 1/1; base 1/1 |
| codex | 2 | 474,491 | 94,467 | +380,024 | +402.28% | skill 1/1; base 1/1 |
| codex | 3 | 284,421 | 577,034 | -292,613 | -50.71% | skill 1/1; base 1/1 |
| codex | 4 | 837,429 | 84,202 | +753,227 | +894.55% | skill 1/1; base 1/1 |
| codex | 5 | 754,644 | 252,996 | +501,648 | +198.28% | skill 1/1; base 1/1 |
| codex | 6 | 357,365 | 208,237 | +149,128 | +71.61% | skill 1/1; base 1/1 |
| codex | 8 | 379,776 | 113,959 | +265,817 | +233.26% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 16,881,142 | 5,953,751 | +10,927,391 | +183.54% | skill 14/14; base 14/14 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 11 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/cdf-simulation-api/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/cdf-simulation-api/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/cdf-simulation-api/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/cdf-simulation-api/SKILL.md`)
- **MEDIUM** SCHEMA/author_missing: Author not specified in metadata (`skills/cdf-simulation-api/SKILL.md`)
- 6 additional finding(s) are available in the full evaluation artifacts.

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
