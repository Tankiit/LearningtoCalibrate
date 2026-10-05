"""E9 (reports/estimand_note_E9_popqa_2026-10-03.md): rerun E2-frozen, E5 (Addendum E/F), E8 and the
motivation nulls on PopQA letter-legend reports, using the existing code with a swapped data source.

  python -m experiments.e9_popqa --part {e2,e5,e8,nulls} [--model MODEL]
Reports: inputs/popqa_letter11/reports/e9full_20261003/<model>/{main,alt}/{fwd,rev}.pt (Run 1 schema).
The 188 alias-collision questions are excluded everywhere.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from mondrian.audit import write_json, source_hash
from . import e2_frozen, e8_transport, e5v2, e5_partition
from .e5_partition import unit

REPORTS = Path("inputs/popqa_letter11/reports/e9full_20261003")
HS = "/Users/tanmoy/research/OOD_detection/OVA_ARR/outputs/step1_extract_variants/token_mode_mean/{m}/popqa/hidden_states.pt"
KEEP = None
_FROZEN_LOAD = e2_frozen.load   # the original; part_e2 swaps e2_frozen.load for load_reports


def keep_mask(ids):
    global KEEP
    if KEEP is None:
        lab = json.load(open("inputs/popqa_letter11/labels.json"))
        KEEP = {r["question_id"]: not r["collision"] for r in lab}
    return np.array([KEEP[q] for q in ids])


def load_reports(model):
    # pass root explicitly: e2_frozen.load binds RUN1 as a default at definition time
    ids, y, V = _FROZEN_LOAD(model, root=str(REPORTS))
    k = keep_mask(ids)
    return ids[k], y[k], {p: v[k] for p, v in V.items()}


def load_popqa_tqa_style(model):
    """(H, y, V) like e5_partition.load_tqa, for PopQA, collisions removed."""
    import torch
    ids, y, V = load_reports(model)
    h = torch.load(HS.format(m=model), map_location="cpu", weights_only=False)
    hid = np.asarray(h["example_ids"]).astype(str); pos = {q: i for i, q in enumerate(hid)}
    k = np.array([pos[q] for q in ids])
    pick = np.array([hashlib.sha256(("p1-e2f|" + q).encode()).digest()[0] & 1 == 0 for q in ids])
    H = unit(np.where(pick[:, None], h["h_pos"].double().numpy()[k], h["h_neg"].double().numpy()[k]))
    assert np.array_equal(pick.astype(int), y)
    return H, y, V


def part_e2(models, out, seeds=500):
    orig = e2_frozen.load
    e2_frozen.load = load_reports
    try:
        res = e2_frozen.run(seeds=range(seeds))
    finally:
        e2_frozen.load = orig
    write_json(Path(out) / "e9a_e2frozen.json", res | {"source_sha256": source_hash()})
    print(res["status"], res["checks"])


def part_e5(models, out, seeds=100):
    budgets = (100, 300, 1000, 2000)
    res = {}
    for model in models:
        H, y, V = load_popqa_tqa_style(model); n = len(y)
        ids = np.array([str(i) for i in range(n)])
        def split(seed, n=n):
            perm = np.random.default_rng(np.random.SeedSequence([seed, 2527])).permutation(n)
            f, c, te = np.array_split(perm, 3); return f, c, te
        full = len(np.array_split(np.arange(n), 3)[1])
        rows, picks, trans, conf = e5v2.run_dataset(f"popqa_{model}", H, y, V, ids, None, range(seeds),
                                                   budgets + (full,), split, False)
        S = e5v2.summarise(rows, picks, [], conf, budgets + (full,), len(np.array_split(np.arange(n), 3)[2]))
        S["informativeness"] = e5v2.informativeness(V, y.astype(float))
        res[model] = S
        write_json(Path(out) / f"e9b_rows_{model}.json", {"rows": rows, "picks": picks, "conformal": conf})
        print("done", model, flush=True)
    write_json(Path(out) / f"e9b_e5_{'_'.join(models)}.json", res | {"source_sha256": source_hash()})


def part_e8(models, out, seeds=100):
    orig = e8_transport.load_tqa
    e8_transport.load_tqa = load_popqa_tqa_style
    try:
        e8_transport.MODELS_ = models
        res = {m: frozen_one(m, seeds) for m in models}
    finally:
        e8_transport.load_tqa = orig
    write_json(Path(out) / f"e9c_e8_{'_'.join(models)}.json", res | {"source_sha256": source_hash()})


def frozen_one(model, seeds=100):
    saved = e8_transport.MODELS
    e8_transport.MODELS = (model,)
    try:
        return e8_transport.frozen(range(seeds))[model]
    finally:
        e8_transport.MODELS = saved


def main():
    pa = argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--part", choices=["e2", "e5", "e8"], required=True)
    pa.add_argument("--model", default=None)
    pa.add_argument("--out", default="runs/p1_e9")
    pa.add_argument("--seeds", type=int, default=None, help="smoke tests only; defaults: e2 500, e5 100, e8 100")
    a = pa.parse_args()
    models = [a.model] if a.model else list(e2_frozen.MODELS)
    kw = {} if a.seeds is None else {"seeds": a.seeds}
    {"e2": part_e2, "e5": part_e5, "e8": part_e8}[a.part](models, a.out, **kw)


if __name__ == "__main__":
    main()
