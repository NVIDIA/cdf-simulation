# Skill Benchmark: cdf

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf`
- Evaluation date: 2026-10-13
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 7 evaluation tasks (7 positive)
- Dataset digest: `sha256:e01db4d8c4b5a8af5144688e73dbf66eb8fc75fd338e786c4c960191f65a6558` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 88.1% — baseline ran, but no comparable score was available; uplift unavailable | 76.4% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 85.7% → 100.0% (+14.3 points) | 78.6% → 71.4% (-7.2 points) |
| Correctness | 37.1% → 94.3% (+57.2 points) | 45.7% → 91.4% (+45.7 points) |
| Discoverability | 92.1% — baseline ran, but no comparable score was available; uplift unavailable | 70.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 32.5% → 81.0% (+48.5 points) | 32.0% → 64.2% (+32.2 points) |
| Efficiency | 72.9% — baseline ran, but no comparable score was available; uplift unavailable | 85.1% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 9,963,667 | 3,139,398 | +6,824,269 | +217.38% | skill 7/7; base 7/7 |
| claude-code | 1 | 1,868,449 | 494,115 | +1,374,334 | +278.14% | skill 1/1; base 1/1 |
| claude-code | 2 | 1,780,985 | 334,845 | +1,446,140 | +431.88% | skill 1/1; base 1/1 |
| claude-code | 3 | 1,150,564 | 580,427 | +570,137 | +98.23% | skill 1/1; base 1/1 |
| claude-code | 4 | 1,554,860 | 587,646 | +967,214 | +164.59% | skill 1/1; base 1/1 |
| claude-code | 5 | 1,001,119 | 268,679 | +732,440 | +272.61% | skill 1/1; base 1/1 |
| claude-code | 6 | 1,025,108 | 653,650 | +371,458 | +56.83% | skill 1/1; base 1/1 |
| claude-code | 8 | 1,582,582 | 220,036 | +1,362,546 | +619.24% | skill 1/1; base 1/1 |
| codex | All cases | 3,018,668 | 1,994,329 | +1,024,339 | +51.36% | skill 7/7; base 7/7 |
| codex | 1 | 598,632 | 271,115 | +327,517 | +120.80% | skill 1/1; base 1/1 |
| codex | 2 | 320,652 | 137,294 | +183,358 | +133.55% | skill 1/1; base 1/1 |
| codex | 3 | 466,441 | 246,714 | +219,727 | +89.06% | skill 1/1; base 1/1 |
| codex | 4 | 389,635 | 447,364 | -57,729 | -12.90% | skill 1/1; base 1/1 |
| codex | 5 | 486,342 | 346,345 | +139,997 | +40.42% | skill 1/1; base 1/1 |
| codex | 6 | 570,227 | 237,803 | +332,424 | +139.79% | skill 1/1; base 1/1 |
| codex | 8 | 186,739 | 307,694 | -120,955 | -39.31% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 12,982,335 | 5,133,727 | +7,848,608 | +152.88% | skill 14/14; base 14/14 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 12 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 7 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.author' (`skills/cdf/SKILL.md`)
- **MEDIUM** QUALITY/quality_correctness: SKILL_SPEC recommended field missing: 'metadata.tags' (`skills/cdf/SKILL.md`)
- **MEDIUM** QUALITY/quality_discoverability: Skill name very short: 'cdf' (`skills/cdf/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Instructions' (`skills/cdf/SKILL.md`)
- **MEDIUM** SCHEMA/body_recommended_section: Missing recommended section: '## Examples' (`skills/cdf/SKILL.md`)
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
