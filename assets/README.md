# Paper assets

`natural_context_redirection.png` is a web rendering of Figure 1 from the
revised JevOut manuscript; the matching vector PDF is included. Its example is
MMLU-Pro item 11233 from the frozen
[TIGER-Lab/MMLU-Pro source](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/b189ec765aa7ed75c8acfea42df31fdae71f97be)
(MIT license). The question and choices are quoted within the original research
figure. Dataset paper: [MMLU-Pro](https://arxiv.org/abs/2406.01574).

`results_summary.json` contains paper-level aggregate results, experiment
settings, model and dataset revisions, and SHA-256 hashes of the frozen source
summaries. Rates are fractions in [0, 1]. The file is exported mechanically
from the verified paper summaries; it contains no individual annotations,
dataset records, or raw model trajectories. Result definitions and sampling
protocols are explained in [results.md](../docs/results.md).

The software is licensed under Apache-2.0. External datasets and model artifacts
retain their own licenses, recorded in the aggregate file; they are not
redistributed by this release.
