# Experiment settings

This document records the settings used in the revised
[JevOut paper](https://arxiv.org/abs/2609.30243). The reusable public package
implements context construction, evaluation, and the additional controls.
[results.md](results.md) reports the outcomes and validation; the paper appendices
provide full prompts and task-specific details.

## Tasks and populations

Seven datasets contribute 50 development items and 100 held-out items each:
350 development and 700 held-out source items. Sampling uses seed `20260921`,
SHA-256 ranks of item identifiers, and balancing over native categories.
Development and evaluation item identifiers are disjoint.

- MMLU-Pro and SuperGPQA become question-only decisions, with an insertion
  boundary before the original question and the original option order.
- MuSR and ToMBench retain their narrative or social-inference passages.
  LAR-ECHR combines case facts and prior arguments for next-argument selection.
  Passage boundaries follow sentence endings.
- SATA-Bench becomes one binary membership decision per option. Construction
  retains the complete gold-answer set and targets one initially rejected
  absent option per source item.
- BFCL V4 supplies tool-routing checkpoints. Function schemas form the choices;
  additions are permitted only inside the final user message, preserving earlier
  turns and completed calls. Use explicit `insertion_boundaries` in the public
  JSONL format to enforce this restriction.

Each target sees 1,531 expanded held-out decision units. Construction retains at
most one initially correct unit per source item and system, giving 508 Jev,
328 OpenSourceJev, 328 Von, and 285 plain-Qwen cases. For single-answer items,
the target is the highest-probability wrong option on the original input. For
SATA, it is the clean-correct absent option with the highest membership probability.
The selected unit and wrong target remain fixed throughout construction.

LAR-ECHR's official item splits share some underlying legal cases. The paper
includes a learned-generation sensitivity analysis excluding associated test
items, in addition to the item-disjoint headline evaluation.

## Models and generation

The targets are hosted Jev `jev-1.13.0`, OpenSourceJev using Qwen3-1.7B Q8
weights, Von `wfzyx/von-1.0` (395M), and a plain numbered-option Qwen scorer
using the same Qwen3-1.7B Q8 weights as OpenSourceJev. They expose different
probability constructions through a common selected-branch/distribution interface.
The public [adapter guide](target_adapters.md) explains how to connect targets.

The frozen Gemma4-12B proposer uses temperature 0.8, top-p 0.95, and up to 160
output tokens per single-addition proposal. A separate invocation checks the
completed augmented input at temperature 0 and up to 160 tokens. The checker
does not see the fixed target or target feedback. The public prompts are
[`generator.txt`](../context_optimization/prompts/generator.txt),
[`generator_neutral.txt`](../context_optimization/prompts/generator_neutral.txt),
and [`checker.txt`](../context_optimization/prompts/checker.txt).

Mechanical checks enforce a valid insertion boundary, nonempty text,
nonduplication, and absence of explicit answer-selection phrases. The checker
assesses natural fit, preservation of the gold-answer set, direct decision
cues, and decisive answer-changing evidence. Accepted rendered inputs are
deduplicated before target evaluation. The renderer copies original characters
and inserts additions at offsets defined on the original source.

## Probability-guided context optimization

For fixed wrong branch t, a candidate's objective is
`log(p(t) + 1e-12) - log(max(p(c) for c != t) + 1e-12)`.
The archive stores accepted contexts and their margins. Later iterations
allocate parents from a temperature-scaled distribution over margins mixed
with a uniform component, alongside scheduled original-input restarts.

The formal configuration uses 16 slots for up to four iterations, allocation
temperature 1.0, exploration mixture 0.1, restart fraction 0.25, one retry per
rejected slot, and seed `20260921`. After the first iteration, four of the
16 slots restart from the original input. Each extension adds one sentence,
so a completed context has at most four additions. Rejected slots need not be
refilled beyond their retry; accepted target calls are capped at 64.

Stopping is checked after an entire iteration, when an accepted context selects
the target with probability at least 0.7. Any target selection counts as a flip,
even below that stopping probability. The original eligibility evaluation is
outside the construction budget.

## One-shot controls and feedback comparison

Formal neutral and target-aware controls each receive exactly one proposal
opportunity from the original input. The former does not see the wrong target;
the latter does. Both retain the primary eligible denominator, assigning no
success to rejected proposals. In the public CLI, use `--proposal-retries 0`
with either one-shot strategy to match this one-proposal protocol. The shared
CLI retry default is one, as in the adaptive optimizer.

The feedback audit uses 20 initially correct Jev decisions per dataset (140
total). Probability-only allocation uses margins; label-only allocation uses
an indicator of target selection. Neither supplies numerical history to the
proposer. The same restarts, acceptance checks, budgets, and stopping threshold
apply. The public release preserves the main optimizer; feedback-ablation
outcomes are provided as aggregate results.

## Independent-generation control

The completed control uses the same 508 Jev units, original distributions,
and fixed wrong targets as primary optimization. Every proposal starts from
the unaugmented input and supplies one sentence. It receives no previous
proposals, additions, or target-feedback history. The proposer, target-blind
checker, deterministic filters, and rendering rules are shared.

Each case has a ceiling of 64 accepted target evaluations and 128 proposal
attempts. One retry is allowed per rejected slot; exhausted slots are refilled
while both ceilings permit. Distinct accepted contexts consume target calls.
After each block of 16 slots, construction stops if any context has selected
the target with probability at least 0.7. Otherwise it continues to a ceiling.
All original 508 decisions remain in the denominator.

`sample_independent_context` implements this policy. Pass a saved primary result
as `reference` to reuse its unit, clean distribution, and target without
rerunning eligibility selection. Its model revision is checked against new
target responses. This control's realized calls differ from primary optimization
because its refill and stopping outcomes differ.

## Transfer, repeatability, and learned generation

Transfer freezes a source-selected context and wrong target, then evaluates the
exact selected unit on a destination. Matching requires the destination to
answer that original unit correctly. Source failures remain in the matched
denominator. `transfer_context` and `targeted_transfer_rate` implement this
protocol.

The primary repeatability audit samples 100 successful Jev cases proportionally
across datasets and preselects a successful context by original target
probability. Ten original-input and ten augmented-input evaluations assess
both clean correctness and fixed-target reproduction. `retest_context`
implements this preselected-context evaluation; details of the separate
baseline audit and its fallback rule are in [results.md](results.md).

The supporting learned-generation study trains on development trajectories.
V1 uses margin-weighted transitions; V2 uses complete successful contexts with
item-balanced supervision. Their matched held-out comparison generates one to
four additions per proposal and evaluates one generation and best-of-four on
the same 508 Jev decisions. The source splits, construction prompts, and
complete-context protocol are recorded in the paper; training artifacts are
not included in this package.

## Metrics and aggregate export

TFR is the proportion of initially correct decisions with at least one
accepted fixed-target flip within the specified accepted-call prefix. Rejected
or unsuccessful attempts do not remove decisions from that denominator.
High-probability TFR requires target selection and the probability threshold
in the same evaluated context. Transfer uses its matched initially correct
population. Binomial intervals in the paper use 95% Wilson intervals; feedback
differences use paired bootstrap resamples grouped by source item and stratified
by dataset.

The [aggregate JSON](../assets/results_summary.json) contains counts, settings,
budget curves, and reported validation outcomes, with source-file SHA-256 hashes.
It supports checking the reported arithmetic; it is not a bundle of raw model
requests or sampled dataset records.
