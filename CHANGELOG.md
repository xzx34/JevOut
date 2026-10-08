# Changelog

## 0.2.0 - 2026-10-08

This release accompanies the revised JevOut arXiv manuscript.

- Add independent-root generation with separate accepted-call and proposal
  budgets, batch stopping, and optional reuse of saved units and fixed targets.
- Add fixed-context repeatability evaluation with fresh local requests,
  preselected contexts, clean controls, and per-repeat predictions.
- Add a complete offline demo using a deterministic synthetic target.
- Publish a paper figure and machine-readable aggregate results, including the
  independent-generation comparison, human validation, and repeatability samples.
- Document paper settings, output formats, target integration, and small-budget
  usage. Update the paper citation to the seven-author version.
- Add tests for the new controls, budget accounting, cache bypass, and CLI paths.

The adaptive optimizer and its prompts retain their 0.1.0 implementation.

## 0.1.0 - 2026-09-24

Initial public release: strict JSONL input, deterministic rendering, Jev/HTTP/
callable target adapters, clean evaluation, probability-guided context
optimization, one-shot controls, frozen-context transfer, tests, and CI.
