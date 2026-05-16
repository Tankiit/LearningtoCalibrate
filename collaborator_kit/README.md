# Sieve Prediction via Logit-Space Gap - Collaborator Kit

This kit scopes appendix-only work for the ARR May 25 submission. The main
paper should not depend on these tasks landing. If the appendix tasks finish,
they strengthen reviewer defense; if they slip, the paper still submits.

## TL;DR

1. Read `INTERFACE.md` first. It defines the cache contract and deliverables.
2. Read `REVIEWER_ATTACK_MAP.md` second. It explains what each task defends.
3. Start with `tasks/R1_replication.md`. It is the onboarding task.
4. After R1, work through `tasks/D2_scod_optimal.md`,
   `tasks/G1_selfcheckgpt.md`, and `tasks/W3_rebuttals.md`.

Do not treat this directory as the main paper pipeline. Treat it as a
self-contained replication and appendix workspace.

## Setup

From the repository root:

```bash
pip install -e .
python -m collaborator_kit.data.load_cached_extraction --root collaborator_kit/cached_extractions --check
```

The check should report which expected `(model, dataset)` cache cells are
present and which are missing. If the lead author has not staged the cache yet,
missing cells are expected.

## Tasks and order

1. R1 - independent replication on your cluster, using Qwen2.5-7B-Instruct.
   This onboards you and catches reproducibility bugs.
2. D2 - SCOD-optimal selector comparison across the cached matrix.
   This compares gap against the additive selector and learned 2D upper bound.
3. G1 - SelfCheckGPT baseline on ChaosNLI and TruthfulQA.
   This adds a second LLM-uncertainty baseline in the appendix.
4. W3 - reviewer-attack rebuttal drafting.
   This uses fresh eyes to pressure-test the claims before submission.

R1 comes first. The other tasks can run in parallel after R1 if time permits.

## Compute budget

- R1: about 6 GPU-hours on an A100/H100-class GPU.
- D2: usually CPU-only if cached probes/signals are available; otherwise GPU
  only for re-extraction.
- G1: about 4-8 GPU-hours for SelfCheckGPT NLI or embedding passes.
- W3: no compute.

## In scope

- Appendix tables, appendix prose, and internal notes.
- Running extractions or baselines on your own GPU/cluster when necessary.
- Creating feature branches named `collab/<task_id>` if this is inside a Git
  checkout.

## Out of scope

- Main-text critical-path experiments.
- Debugging the lead author's Modal pipeline.
- Modifying signed pre-registration artifacts.
- Rewriting shared paper scripts. If you need a variant, fork it under a
  task-specific filename.

## Done means done

Each task is complete only when these files exist:

- `results_<task_id>.parquet`
- `appendix_<task_id>.tex`
- `appendix_<task_id>.pdf`
- `NOTES_<task_id>.md`

The lead author owns final prose tightening and paper integration.

