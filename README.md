<h1 align="center">JevOut</h1>

<p align="center">
  <strong>Natural Context Can Flip Decision Models</strong><br>
  <sub><a href="https://www.agenthon.net/#call-for-papers">Agenthon @ NeurIPS 2026</a> · Poster</sub>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.30243"><kbd>&nbsp; Paper &nbsp;</kbd></a>
  &nbsp;&nbsp;
  <a href="https://xzx34.github.io/jevout/"><kbd>&nbsp; Project Page &nbsp;</kbd></a>
  &nbsp;&nbsp;
  <a href="docs/quickstart.md"><kbd>&nbsp; Quickstart &nbsp;</kbd></a>
</p>

## News

- **[10/08/2026]** 🎉 JevOut has been accepted to **Agenthon @ NeurIPS 2026** as a **poster**! See you in Atlanta!
- **[10/08/2026]** Code [v0.2.0](https://github.com/xzx34/JevOut/releases/tag/v0.2.0) is available, with an offline demo, independent-generation baseline, repeatability tools, and updated experiment documentation.
- **[09/24/2026]** Initial code release.

## Introduction

Decision models turn language into choices for routers, evaluators, and agents.
But can they tell useful context from a distraction? We find that short,
natural-looking additions can redirect a correct decision toward a chosen
wrong answer, without changing the original task, choices, or correct answer.
The additions read as ordinary background or procedural details rather than
instructions to change the decision.

JevOut studies this phenomenon across four decision systems and seven datasets.
We use probability-guided context optimization to construct answer-preserving
additions and measure **Targeted Flip Rate (TFR)**: how often an initially
correct decision moves to the wrong option fixed in advance. With a budget of
up to 64 accepted target evaluations per decision:

| Decision system | Initially correct decisions | TFR |
| :--- | ---: | ---: |
| Jev | 508 | **61.4%** |
| OpenSourceJev | 328 | **72.6%** |
| Von | 328 | **73.2%** |
| Plain Qwen | 285 | **64.9%** |

The study also examines cross-model transfer, independent human validation,
repeatability, and learned context generation. See the
[results](docs/results.md) and [experiment settings](docs/paper_protocol.md)
for the full comparisons.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/xzx34/JevOut.git
cd JevOut
uv sync --frozen
uv run context-opt demo --output-dir outputs/demo
```

The demo uses a deterministic synthetic model and needs no API keys or model
service. It runs context optimization, independent generation, and repeatability
checks, then writes the results to `outputs/demo/`.

To use Jev or your own decision model, follow the [quickstart guide](docs/quickstart.md).
The package supports Jev, HTTP endpoints, and Python callbacks through a common
[target interface](docs/target_adapters.md).

## Repository layout

```text
context_optimization/   evaluation, context optimization, controls, and prompts
docs/                   usage guides, experiment settings, and result summaries
examples/               sample decision items and a Python target adapter
tests/                  offline tests
```

**Usage:** [Quickstart](docs/quickstart.md) · [Input format](docs/input_format.md) ·
[Output format](docs/output_format.md) · [Target adapters](docs/target_adapters.md)

**Experiments:** [Settings](docs/paper_protocol.md) · [Results](docs/results.md) ·
[Aggregate JSON](docs/results_summary.json)

## Citation

If you use this work, please cite:

```bibtex
@article{xu2026jevout,
  title   = {JevOut: Natural Context Can Flip Decision Models},
  author  = {Xu, Zixiang and Song, Zirui and Zhang, Chiyu and Chen, Xiuying and Liu, Xi and Hu, Xiyang and Zhao, Yue},
  journal = {arXiv preprint arXiv:2609.30243},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.30243}
}
```

Corresponding author: Yue Zhao ([yue.z@usc.edu](mailto:yue.z@usc.edu)).
Code is licensed under [Apache-2.0](LICENSE).
