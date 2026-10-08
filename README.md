# JevOut: Natural Context Can Flip Decision Models

[Paper](https://arxiv.org/abs/2609.30243) ·
[Project page](https://xzx34.github.io/jevout/) ·
[Quickstart](docs/quickstart.md) ·
[Experiment settings](docs/paper_protocol.md)

**Short, natural-looking context can redirect a correct decision toward a
chosen wrong answer, even when the underlying task remains unchanged.**
We study this behavior in four decision systems across seven datasets spanning
knowledge, reasoning, and tool routing. Context additions supply background or
procedural details while preserving the original text, question, choices, and
correct answer.

![A natural context addition redirects a climate-ethics decision; the original question and choices remain fixed.](assets/natural_context_redirection.png)

*Figure 1 from the paper. The example asks for a definition: the added detail
raises a related intergenerational issue without changing that definition.
Jev moves from the correct answer at probability 0.97 to the fixed wrong target
at probability 0.54.*

## Main findings

Within **64 accepted target evaluations per decision**, probability-guided
context optimization uncovers targeted flips in a majority of each system's
initially correct decisions:

| Decision system | Initially correct decisions | Neutral one-shot | Target-aware one-shot | Context optimization |
| :--- | ---: | ---: | ---: | ---: |
| Jev | 508 | 2.2% | 16.9% | **61.4%** |
| OpenSourceJev | 328 | 6.4% | 16.2% | **72.6%** |
| Von | 328 | 7.6% | 21.6% | **73.2%** |
| Plain Qwen | 285 | 8.4% | 18.6% | **64.9%** |

TFR counts a decision only when an accepted context makes the model select the
wrong option fixed before construction. Each row uses that system's initially
correct population. One-shot controls generate one sentence; optimization
uses the stated call budget.

The V2 evaluation also includes a budget-matched independent-generation
control on Jev (48.0% TFR), blinded human validation (229/250 sampled successful
contexts judged valid), and repeated evaluations (97/100 sampled primary Jev
successes reproduce at least 8/10 times with a context selected before retesting).
See [results and validation](docs/results.md) for protocols and
[machine-readable aggregates](assets/results_summary.json) for counts.

## Try it offline

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/xzx34/JevOut.git
cd JevOut
uv sync --frozen
uv run context-opt demo --output-dir outputs/demo
```

The demo runs clean evaluation, context optimization, independent generation,
and repeatability checks without credentials or network requests. It writes
JSONL records and a summary using a deterministic synthetic target; the demo
probabilities are illustrative rather than model measurements.

## Use a decision model

Provide decision items in the [JSONL input format](docs/input_format.md), a
target adapter, and an OpenAI-compatible proposer/checker endpoint:

```bash
uv run context-opt validate --input examples/toy_choices.jsonl

uv run context-opt optimize \
  --input decisions.jsonl --output outputs/optimized.jsonl \
  --target jev \
  --proposer-url http://127.0.0.1:8000/v1 \
  --proposer-model your-model
```

Jev reads `TYPESAFE_API_KEY` from the environment. The
[quickstart](docs/quickstart.md) includes a smaller-budget run, one-shot
controls, the independent-root baseline, and fixed-context retesting.
[Target adapters](docs/target_adapters.md) cover Jev, generic HTTP services,
and in-process Python models.

The Python API exposes `optimize_context`, `sample_independent_context`,
`retest_context`, clean evaluation, and frozen-context transfer. The package
records proposals, acceptance decisions, target probabilities, margins, and
call counts. Rejected or duplicate proposals consume proposal attempts but
not accepted target evaluations.

## Paper and software versions

Version **0.2.0** accompanies the revised arXiv manuscript and adds the
independent-generation control, repeatability tools, offline demo, and V2
documentation. See the [changelog](CHANGELOG.md) and
[output format](docs/output_format.md).

This repository releases the reusable evaluation code, prompts, a paper figure,
and aggregate results. Experimental source data, raw trajectories, individual
human ratings, proposer fine-tuning code, and model weights are not bundled.
The [experiment settings](docs/paper_protocol.md) describe the paper's models,
task adaptations, budgets, and training-based supporting analysis.

## Citation

Zixiang Xu, Zirui Song, Chiyu Zhang, Xiuying Chen, Xi Liu, Xiyang Hu, and Yue Zhao.
Corresponding author: Yue Zhao ([yue.z@usc.edu](mailto:yue.z@usc.edu)).

```bibtex
@article{xu2026jevout,
  title   = {JevOut: Natural Context Can Flip Decision Models},
  author  = {Xu, Zixiang and Song, Zirui and Zhang, Chiyu and Chen, Xiuying and Liu, Xi and Hu, Xiyang and Zhao, Yue},
  journal = {arXiv preprint arXiv:2609.30243},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.30243}
}
```

Licensed under Apache-2.0. See [SECURITY.md](SECURITY.md) for private security reports.

## Development

```bash
uv sync --frozen --extra dev
uv run pytest
uv run ruff check .
uv build --wheel
```
