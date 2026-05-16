# Interface Contract

This file is the handoff contract. The collaborator should be able to run the
appendix tasks from cached intermediates without knowing the full codebase.

## Cache Layout

Every `(model, dataset)` directory under `cached_extractions/` should contain:

```text
contrastive_h.pt
contrastive_meta.pt
gen_greedy_h.pt
gen_greedy_meta.pt
judge_greedy.pt
nsamples10.pt          # only required for ChaosNLI and TruthfulQA
```

Expected matrix for the appendix package:

```text
models:
  llama3.1_8b_inst
  mistral_7b_inst_v0.3
  qwen2.5_7b_inst

datasets:
  chaosnli
  pavlick_nli
  ambigqa_kge2
  truthfulqa
  halueval
```

The current repository also contains code paths for `truthfulqa`, `halueval_qa`,
`triviaqa`, `popqa`, and `bioasq`. If you run against the current main pipeline,
use the repository's config names. If you run against this appendix cache, use
the directory names above.

## File Contracts

`contrastive_h.pt`:

```python
{
    "h_pos": torch.Tensor,  # (N, L, 3, D), CPU, fp16 allowed
    "h_neg": torch.Tensor,  # (N, L, 3, D), CPU, fp16 allowed
}
```

`contrastive_meta.pt`:

```python
{
    "records": list[dict],  # each record includes y_expert/expert_reliable,
                            # category, example_id, and task-specific metadata
}
```

`gen_greedy_h.pt`:

```python
{
    "h": torch.Tensor,  # (N_eval, L, 3, D), CPU, fp16 allowed
}
```

`gen_greedy_meta.pt`:

```python
{
    "records": list[dict],
    "text": list[str],
    "logprob": list[float],
    "seq_entropy": list[float],
}
```

`judge_greedy.pt`:

```python
{
    "y_correct": torch.Tensor,  # (N_eval,), int64 or bool
}
```

`nsamples10.pt`, only for ChaosNLI and TruthfulQA:

```python
{
    "samples": list[list[str]],  # len N_eval x 10
    "records": list[dict],
}
```

All arrays are aligned by index. `records[i]` corresponds to row `i` in every
tensor/list for that cache cell.

## Canonical Representation Slice

Use `response_mean` at relative layer `-4` unless a task explicitly says
otherwise:

```python
pool_idx = {"last": 0, "response_mean": 1, "response_max": 2}["response_mean"]
layer_rel = -4
h = contrastive_h["h_pos"][:, layer_rel, pool_idx, :].float().numpy()
```

## Deliverables

For each appendix task, produce:

```text
results_<task_id>.parquet
appendix_<task_id>.tex
appendix_<task_id>.pdf
NOTES_<task_id>.md
```

The parquet should be long format with at least:

```text
task_id
model
dataset
seed
method
metric
value
n
split
notes
```

Add extra columns only when they are task-specific and documented in
`NOTES_<task_id>.md`.

## Guardrails

- Do not modify `cached_extractions/`.
- Do not modify `PRE_REGISTRATION.json`.
- Use seeds `100+` for exploratory work. Seeds `0-4` are reserved for locked
  headline runs.
- Do not rewrite shared paper scripts. Fork task-specific variants instead.
- Report disagreements with pre-registration neutrally in notes:
  `observed X; pre-registered Y; difference W`.

## Sanity Check

Run:

```bash
python -m collaborator_kit.data.load_cached_extraction --root collaborator_kit/cached_extractions --check
```

This validates file presence and basic tensor/list alignment.

