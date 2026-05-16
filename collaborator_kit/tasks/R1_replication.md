# Task R1 - Independent Replication

## Goal

Rerun the headline matrix on independent hardware using
`Qwen2.5-7B-Instruct`, then report whether AURC numbers reproduce within seed
variance.

This is appendix-only, but it is the first task because it forces a clean pass
through setup, cache assumptions, evaluation scripts, and result formatting.

## Why this matters

This defends against the reviewer failure mode: "I could not reproduce the
result on my hardware." It also catches code-path assumptions before submission.

## Inputs

- Repository checkout.
- GPU cluster with A100/H100-class GPU.
- Model access for `Qwen/Qwen2.5-7B-Instruct`.
- Dataset/cache access from `collaborator_kit/cached_extractions/`.
- Existing pipeline scripts in `scripts/`.

## Suggested run path

First validate the cache:

```bash
python -m collaborator_kit.data.load_cached_extraction --root collaborator_kit/cached_extractions --check
```

Then run the current pipeline for the available Qwen config. If a Qwen config
does not exist yet, create a task-local config rather than editing main configs:

```bash
python scripts/run_full_sweep.py \
  --models qwen2_5_7b \
  --datasets chaosnli pavlick_nli ambigqa_kge2 truthfulqa halueval \
  --seeds 100 101 102
```

If the current repository has not yet added those dataset adapters, record the
blocker in `NOTES_R1.md` and run the closest staged cache path. Do not debug the
lead author's Modal extraction pipeline unless explicitly asked.

## Output schema

Write `results_R1.parquet` with rows:

```text
task_id = R1
model
dataset
seed
method
metric = aurc
value
n
split
notes
```

Include at least these methods when available:

- `gap`
- `probe_only`
- `logprob`
- `semantic_entropy`

## Appendix prose

Write `appendix_R1.tex` as a short note:

```tex
\paragraph{Independent reproduction.}
An independent collaborator reran the headline evaluation on [hardware] using
[software stack] and Qwen2.5-7B-Instruct. The headline AURC on [dataset] was
[X] $\pm$ [Y], compared with the locked result [X*] $\pm$ [Y*], a relative
difference of [Z]\%.
```

If a number disagrees, do not smooth it over. Report it neutrally in the notes
and flag the exact cell.

## Success criteria

- `results_R1.parquet` exists and passes schema checks.
- `appendix_R1.tex` exists.
- `appendix_R1.pdf` renders.
- `NOTES_R1.md` lists hardware, software stack, commands, and any failures.

