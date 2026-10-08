# Output records

All commands write UTF-8 JSONL: one JSON object per item or decision unit.
Original item identifiers remain available throughout the workflow.

## Clean evaluation

An `EvaluationRecord` contains `item_id`, the expanded `unit`, its `result`,
and `initially_correct`. A `DecisionResult` records the selected branch,
normalized option probabilities, model revision, request hash, latency, and
input-token count when the provider supplies it.

## Context construction

An `OptimizationResult` contains the fixed `unit_id`, `target_option`,
`target_branch`, `clean_result`, `config`, and the complete `attempts` list.
`target_option` is an original option key; on a membership unit its evaluated
`target_branch` is `true`.

Each attempt records its parent, additions, proposal, deterministic checks,
semantic checker judgment, rejection reason, and rendered source hash.
Accepted target evaluations additionally record `result`, `target_probability`,
`margin`, `targeted_flip`, and a one-based `target_call_index`.

`proposal_calls` counts every proposal attempt, including rejections and
duplicates. `target_calls` counts accepted candidate evaluations and excludes
the original eligibility evaluation. `targeted_flip` means that at least one
accepted evaluation selects the fixed wrong branch. `threshold_reached` also
requires the configured probability threshold, which controls stopping.

`best_*` fields select the highest-margin evaluated attempt. The separate
`select_representative_attempt` function prioritizes successful contexts meeting
the stopping threshold, then shorter contexts, for frozen-context transfer.
The repeatability tool instead preselects the successful attempt with the
highest target probability. These selection rules serve different analyses.

An `IndependentRootResult` uses the same attempt fields and adds
`slots_completed` and `stop_reason`. Its configuration separately records
`target_budget`, `proposal_budget`, and `batch_size`. An exhausted slot can
contain only rejected attempts; such a slot consumes no target call. Parse this
record with `IndependentRootResult`, rather than `OptimizationResult`.

The CLI summary reports a fractional `targeted_flip_rate` (for example, `0.614`
means 61.4%). A run with no eligible decisions reports `null` for this field.

## Repeatability

A `RepeatabilityResult` contains the source attempt identifier, exact additions,
original and augmented source hashes, model revision, all `clean_results` and
`augmented_results`, and the hit counts. With the default ten repetitions:

- `flip_confirmed` requires at least eight selections of the fixed wrong target
  on the augmented input.
- `clean_and_flip_confirmed` additionally requires at least eight correct
  selections on the original input.

Every completed case keeps all ten outcomes, including failures. The API bypasses
local `CachedTarget` wrappers; service-internal execution and caching remain
provider-controlled. Request failures and revision changes raise errors and
are not recorded as unsuccessful predictions.

## Aggregate paper results

[`assets/results_summary.json`](../assets/results_summary.json) is a separate
export of paper-level counts and settings. It contains primary and one-shot
results, budget curves, the Jev independent-generation comparison, separate
repeatability samples, reported human-validation aggregates, and the matched
learned-proposer results. It contains no individual questions or trajectories.
