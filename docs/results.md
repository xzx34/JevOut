# Paper results and validation

The paper studies natural-context redirection: an accepted addition makes a
decision system select a wrong option fixed in advance while the original
task and gold answer remain unchanged. All discovery rates below retain each
system's initially correct population. Settings are in
[paper_protocol.md](paper_protocol.md); counts are also available in
[`results_summary.json`](../assets/results_summary.json).

## Primary results

| System | Eligible decisions | Targeted flips | TFR | Flips with target probability >= 0.7 |
| :--- | ---: | ---: | ---: | ---: |
| Jev | 508 | 312 | 61.4% | 229 |
| OpenSourceJev | 328 | 238 | 72.6% | 211 |
| Von | 328 | 240 | 73.2% | 177 |
| Plain Qwen | 285 | 185 | 64.9% | 172 |

The primary trajectories contain 68,912 proposals and 60,685 accepted target
evaluations across 1,449 model-item pairs. A maximum of 64 accepted target
evaluations is available per case. Across systems, redirection can affect
initially confident decisions, produce high-probability wrong choices, and
transfer to other decision interfaces.

## Independent generation

Repeated independent target-aware generation reaches 244/508 flips (48.0%) on
the exact Jev units and targets used by context optimization, compared with
312/508 (61.4%) for optimization. Both have a ceiling of 64 accepted target calls.
The paired outcomes are 237 successes under both procedures, 75 only under
optimization, seven only under independent generation, and 189 under neither.

Independent proposals uncover much of the phenomenon, while iterative
optimization extends coverage. At the first 16 accepted calls, the procedures
have similar coverage (201 and 205 cases); the difference grows at later
prefixes. The control uses 27,143 proposals and 24,860 accepted calls. Its
single-sentence proposals, refill policy, and stopping rule are specified in
[the protocol](paper_protocol.md#independent-generation-control).

## Probability feedback

On 140 matched Jev decisions, probability-only parent allocation uncovers
89/140 flips (63.6%), compared with 79/140 (56.4%) for label-only allocation,
within 64 accepted evaluations. Numerical history is hidden from the proposer
in both conditions. The comparison measures the benefit of graded feedback for
allocating parents in the implemented optimization procedure.

## Independent human validation

Three independent annotators assessed 250 successful contexts mixed with 100
accepted but unsuccessful controls. Sampling was stratified across datasets,
balanced across the four target systems, and deduplicated. Annotators did not
see the fixed wrong target, model prediction, or flip outcome.

For each augmented item, annotators judged naturalness, preservation of the
original gold answer, and whether the added detail supplied decisive evidence
that changed the answer. Majority judgments classify 229/250 successes (91.6%)
and 93/100 controls (93.0%) as natural, gold-preserving, and non-decisive.
Reported Fleiss' kappa is 0.72 for naturalness, 0.79 for gold preservation,
and 0.75 for decisive-evidence judgments.

These pooled sample results are reported alongside the system-specific TFRs.
The release includes aggregate counts and reported agreement, rather than
individual annotations.

## Repeatability

The primary audit samples 100 of the 312 successful Jev cases proportionally
across all seven datasets. Each case's successful context is selected by its
original target probability before any new evaluations. The context reproduces
the fixed-target flip in at least eight of ten repeats on 97/100 cases; 93/100
reproduce it on all ten. Requiring the original input also to remain correct
in at least eight repeats yields 94/100 cases.

A separate audit covers 101 successful cases from an intermediate independent-
generation snapshot. Its first preselected contexts reproduce at least eight
of ten times on 95/101 cases. Under the recorded fallback protocol, which
checks other previously successful contexts until one reproduces ten of ten
or the archive is exhausted, 96/101 cases meet the eight-of-ten criterion.
The samples and their selection rules remain separate in the aggregate export.

The public `retest` command implements the preselected-context check with clean
controls. It does not switch contexts after seeing repeat outcomes. Identical-
input repeatability and human semantic validity answer different questions.

## Learned generation

The supporting adaptation study trains proposers on development trajectories
and evaluates complete contexts on held-out items. In the matched full-context
protocol, one generation emits one to four additions. On 508 Jev decisions:

| Proposer | One generation | Best of four generations |
| :--- | ---: | ---: |
| Base | 82/508 (16.1%) | 148/508 (29.1%) |
| V1: transition-trained | 95/508 (18.7%) | 167/508 (32.9%) |
| V2: context-trained | 111/508 (21.9%) | 174/508 (34.3%) |

Best-of-four uses target evaluations to choose among accepted contexts. These
complete-context outputs differ from the one-sentence formal one-shot controls.
The adaptation study supports learned context generation relative to its matched
base proposer. Training scripts and weights are outside this software release.
