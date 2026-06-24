#!/usr/bin/env python3
"""run_repspace_deferral.py — rep-space random-set width as deferral signal.

Loads the deferral-paper Modal cache (h_correct/h_wrong contrastive pairs),
builds a random-set frame over hidden-state space, computes credal width two
ways (geometric soft-assign + trained head), and runs the deferral paper's
screen: STC, gap composition (conf + W vs conf), AURC. Logprobs are the
confidence foil.

Revised (see repspace_fixes_skeleton): single split, train-only
standardization, bootstrap-CI verdict instead of sign count, geometric
computed once per (seed, K), and shape/label asserts at cell entry.

Two input modes:
  1. Pre-cached .pt (default): --cache path/to/hidden_states.pt
  2. Live HF model: --model <hf_id> --dataset <name> (uses rsuq.extract
     to collect states on-the-fly via collect_states / logits_from_states)

Run (cached):
  python run_repspace_deferral.py --cache "outputs/llama3_8b/halueval_qa/hidden_states.pt" \
      --k 50 --seeds 5 --head_mode both

Run (live model):
  python run_repspace_deferral.py --model "gpt2" --dataset "triviaqa" \
      --k 50 --seeds 5
"""
from __future__ import annotations
import argparse, json, os, glob
import numpy as np

# ---- rsuq.extract utilities (optional; needed for --model mode) -----------
# These are imported here so the script can call collect_states,
# logits_from_states, and embedding_matrix directly. The import is guarded
# so the script remains runnable standalone if the rsuq package is absent
# (the cached-.pt path doesn't need any of these).
try:
    from rsuq.extract import (collect_states, logits_from_states,
                               embedding_matrix, chunk_texts, StateCache)
    _HAS_EXTRACT = True
except Exception:
    _HAS_EXTRACT = False


# ----------------------------------------------------------------- cache I/O
def load_cache(path: str) -> dict:
    """Deferral-paper OVA_ARR schema: one dict per cell with paired rows.
        h_correct (N,d) -> h+ (right answer, y_model=0 = no error)
        h_wrong   (N,d) -> h- (hallucinated,  y_model=1 = error)
        lp_correct (N,) / lp_wrong (N,)  generion logprobs (confidence foil)
    Returns {(model,dataset): {h_pos, h_neg, lp_pos, lp_neg}}.
    """
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)

    def np_(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    model = str(d.get("model_key", d.get("model", "model")))
    dataset = str(d.get("dataset", os.path.basename(os.path.dirname(path)) or "cell"))
    cell = {"h_pos": np_(d["h_correct"]).astype(np.float32),
            "h_neg": np_(d["h_wrong"]).astype(np.float32)}
    if "lp_correct" in d and "lp_wrong" in d:
        cell["lp_pos"] = np_(d["lp_correct"]).astype(np.float32)
        cell["lp_neg"] = np_(d["lp_wrong"]).astype(np.float32)
    return {(model, dataset): cell}


# --------------------------------------------------------- live-model collection
def load_live_model(model_id: str, device: str = "cuda", dtype: str | None = None):
    """Load a HF AutoModelForCausalLM + tokenizer for live state collection.
    Wraps the transformers import so the cached-.pt path doesn't require it.
    dtype: 'fp16' / 'bf16' / 'fp32' / None (model default)."""
    if not _HAS_EXTRACT:
        raise ImportError("rsuq.extract not available; cannot use --model mode")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch as _torch
    dt_map = {"fp16": _torch.float16, "bf16": _torch.bfloat16,
              "fp32": _torch.float32, None: None}
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dt_map.get(dtype)).to(device)
    return tok, model


def collect_qa_cells(tok, model, texts_correct, texts_wrong,
                     device: str = "cuda", block_size: int = 256,
                     batch_size: int = 32, pool: str = "last",
                     max_blocks: int | None = None,
                     model_key: str = "live", dataset_key: str = "qa"):
    """Build the deferral cell schema {h_pos, h_neg, lp_pos, lp_neg} from
    paired correct/wrong text lists, using rsuq.extract.collect_states and
    logits_from_states.

    h_*  : (N, d)  hidden states from teacher-forced blocks
    lp_* : (N,)    per-token logprob of the gold next-token under the model
    """
    # chunk and collect (h, gold) for each arm
    blocks_c = chunk_texts(tok, texts_correct, block_size, max_blocks)
    blocks_w = chunk_texts(tok, texts_wrong, block_size, max_blocks)
    sc = collect_states(model, blocks_c, batch_size, device, pool)
    sw = collect_states(model, blocks_w, batch_size, device, pool)
    # per-token logprob of the gold token under the model
    def _lp(sc):
        z = logits_from_states(model, sc.h.to(device))
        logp = _torch_log_softmax(z, dim=-1)
        return logp.gather(-1, sc.gold.to(logp.device).unsqueeze(-1)).squeeze(-1).float().cpu().numpy()

    h_pos = sc.h.numpy()
    h_neg = sw.h.numpy()
    lp_pos = _lp(sc)
    lp_neg = _lp(sw)
    return {(model_key, dataset_key): {
        "h_pos": h_pos.astype(np.float32),
        "h_neg": h_neg.astype(np.float32),
        "lp_pos": lp_pos.astype(np.float32),
        "lp_neg": lp_neg.astype(np.float32),
    }}


def _torch_log_softmax(z, dim=-1):
    """Lazy import wrapper to avoid module-level torch dependency."""
    import torch
    return torch.log_softmax(z, dim=dim)


# ----------------------------------------------------------------- rep-space frame
def build_repspace_frame(H_all: np.ndarray, K: int, seed: int):
    from sklearn.cluster import KMeans
    km = KMeans(K, random_state=seed, n_init=5).fit(H_all)
    sizes = np.bincount(km.labels_, minlength=K).astype(float)
    return {"centroids": km.cluster_centers_, "sizes": sizes,
            "width_coef": 1 - 1 / np.maximum(sizes, 1.0)}


def mass_geometric(H, frame, tau=1.0):
    d = np.linalg.norm(H[:, None, :] - frame["centroids"][None], axis=-1)
    e = np.exp(-(d - d.min(-1, keepdims=True)) / tau)
    return e / e.sum(-1, keepdims=True)


def width_from_mass(M, frame):
    return M @ frame["width_coef"]


def train_mass_head(H_tr, frame, mode, y_tr=None, seed=0):
    """H -> mass(N,K).  mode in {supervised, unsupervised}."""
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    geo = mass_geometric(H_tr, frame)
    if mode == "unsupervised":
        reg = MLPRegressor(hidden_layer_sizes=(64,), max_iter=300,
                           random_state=seed).fit(H_tr, geo)
        def f(H):
            r = np.clip(reg.predict(H), 1e-6, None)
            return r / r.sum(-1, keepdims=True)
        return f
    tgt = geo.argmax(-1)
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300,
                        random_state=seed).fit(H_tr, tgt)
    K = frame["sizes"].shape[0]
    def f(H):
        p = clf.predict_proba(H)
        full = np.zeros((len(H), K))
        full[:, clf.classes_] = p
        full = np.clip(full, 1e-6, None)
        return full / full.sum(-1, keepdims=True)
    return f


# ----------------------------------------------------------------- deferral screen
def aurc(unc, err, n=50):
    """Risk-coverage AUC, vectorized: all n prefix-means via a single
    cumsum + gather, instead of a Python list comprehension. Equivalent
    to the old implementation up to float ordering."""
    e = np.asarray(err)[np.argsort(unc)]
    m = len(e)
    cs = np.concatenate(([0.0], np.cumsum(e)))           # (m+1,)
    ks = np.maximum(1, (np.linspace(1, n, n) * m / n).astype(int))
    risks = cs[ks] / ks                                   # (n,)
    covs = np.linspace(1/n, 1, n)
    return float(np.trapezoid(risks, covs))


def stc(W, err):
    from scipy.stats import pointbiserialr
    return float(pointbiserialr(err, (W > np.median(W)).astype(float))[0])


# ---- FIX 5: shape/label sanity asserts (call at top of run_cell) ---------
def assert_cell_ok(h_pos, h_neg, y, lp_pos=None, lp_neg=None):
    assert h_pos.shape == h_neg.shape, "pos/neg must be paired, equal shape"
    assert set(np.unique(y).tolist()) == {0, 1}, "y must be the contrastive split"
    # memo trap: y must come from pos/neg split, never y_expert (==1 on HaluEval)
    N = len(h_pos)
    assert (y[:N] == 0).all() and (y[N:] == 1).all(), "label/order mismatch"
    if lp_pos is not None:
        assert len(lp_pos) == N and len(lp_neg) == N, "logprob foil misaligned"


# ---- FIX 4: compute geometric once; real arms are 3 not 4 ----------------
def arms_for_cell():
    """The distinct measurements. geometric does NOT depend on head mode."""
    return ["geometric", "trained_supervised", "trained_unsupervised"]


# ---- FIX 1 + FIX 2: single split, train-only standardization -------------
def compose_aurc_fixed(sigs_tr, y_tr, sigs_te, y_te):
    """Fit composite on the TRAIN signals/labels, score AURC on TEST.
    sigs_tr / sigs_te: list of 1-D arrays already restricted to that split.
    Returns (aurc_on_test, unc_on_test) so callers can run the bootstrap on
    already-fit scores. Returns (None, None) if train is single-class.
    """
    if len(set(y_tr.tolist())) < 2:
        return None, None
    from sklearn.linear_model import LogisticRegression
    Xtr = np.column_stack(sigs_tr)
    Xte = np.column_stack(sigs_te)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8      # FIX 2: train-only stats
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    lr = LogisticRegression(max_iter=500).fit(Xtr, y_tr)
    unc = lr.predict_proba(Xte)[:, 1]
    return aurc(unc, y_te), unc


# ---- FIX 3: bootstrap CI on the paired AURC gap, not a sign count --------
def aurc_gap_bootstrap(unc_conf_te, unc_confW_te, y_te, B=2000, seed=0):
    """Paired item-bootstrap of  AURC(conf) - AURC(conf+W)  on the test set,
    using *already-fit* per-item uncertainty scores. Positive gap = W reduces
    AURC (helps).

    Design note: we deliberately do NOT refit the logistic inside each
    bootstrap iteration. Refitting on resampled data would (i) be O(B) more
    expensive and (ii) change the estimand — we want the uncertainty about
    *this* fitted composite's test-set gap, not the gap of a refit procedure.
    Bootstrap of the sorted-prefix AUC of fixed scores is the standard
    non-parametric CI for a fixed ranking.
    """
    rng = np.random.default_rng(seed)
    y_te = np.asarray(y_te)
    n = len(y_te)
    gaps = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        a_conf  = aurc(unc_conf_te[idx],  y_te[idx])
        a_confW = aurc(unc_confW_te[idx], y_te[idx])
        gaps[b] = a_conf - a_confW
    lo, hi = np.nanpercentile(gaps, [2.5, 97.5])
    return float(np.nanmean(gaps)), float(lo), float(hi)


def verdict_from_ci(gap_mean, lo, hi, noise_floor=0.02):
    """Honest verdict: helps only if CI excludes 0 AND beats the deferral
    paper's own seed-noise floor (~0.02 AURC)."""
    if lo > 0 and gap_mean > noise_floor:
        return "HELPS (CI>0 and > noise floor)"
    if lo > 0:
        return "SIGNIFICANT-BUT-TINY (CI>0 but < noise floor)"
    return "NULL (CI includes 0)"


# ----------------------------------------------------------------- per-cell run
def run_cell(h_pos, h_neg, K, seed, head_mode, lp_pos=None, lp_neg=None):
    """Compute width signals and the conf-vs-conf+W AURC gap on one cell.

    Returns a dict keyed by arm (geometric, trained_supervised,
    trained_unsupervised). Each arm carries:
        STC, aurc_conf, aurc_conf_plus_W, aurc_W_alone,
        gap_mean, gap_lo, gap_hi, verdict.
    The geometric arm is computed once here regardless of head_mode.
    """
    N, d = h_pos.shape
    H_all = np.vstack([h_pos, h_neg])
    y = np.r_[np.zeros(N), np.ones(N)].astype(int)        # 1 = error (h_neg)
    assert_cell_ok(h_pos, h_neg, y, lp_pos, lp_neg)

    # single split (60/40) — same split used for every arm, so arms are
    # directly comparable on the same test items.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(2 * N)
    n_tr = int(1.2 * N)
    tr, te = idx[:n_tr], idx[n_tr:]

    # confidence foil: real logprobs if provided, else logistic probe on H.
    from sklearn.linear_model import LogisticRegression
    if lp_pos is not None and lp_neg is not None:
        conf = -np.r_[lp_pos, lp_neg]
    else:
        fpred = LogisticRegression(max_iter=500).fit(H_all[tr], y[tr])
        conf = -fpred.decision_function(H_all)

    # --- FIX 4: geometric once ---
    frame = build_repspace_frame(H_all, K, seed)
    W_geo = width_from_mass(mass_geometric(H_all, frame), frame)

    res = {}

    # ---- geometric arm ----
    a_conf, unc_conf = compose_aurc_fixed([conf[tr]], y[tr], [conf[te]], y[te])
    a_cw,  unc_confW = compose_aurc_fixed([conf[tr], W_geo[tr]], y[tr],
                                          [conf[te], W_geo[te]], y[te])
    arm = {"STC": stc(W_geo[te], y[te]),
           "aurc_conf": a_conf,
           "aurc_conf_plus_W": a_cw,
           "aurc_W_alone": aurc(W_geo[te], y[te])}
    if unc_conf is not None and unc_confW is not None:
        g_mean, g_lo, g_hi = aurc_gap_bootstrap(
            unc_conf, unc_confW, y[te], B=2000, seed=seed)
        arm.update({"gap_mean": g_mean, "gap_lo": g_lo, "gap_hi": g_hi,
                    "verdict": verdict_from_ci(g_mean, g_lo, g_hi)})
    else:
        arm.update({"gap_mean": None, "gap_lo": None, "gap_hi": None,
                    "verdict": "NULL (train single-class)"})
    res["geometric"] = arm

    # ---- trained arms (supervised / unsupervised) ----
    modes = (["supervised", "unsupervised"] if head_mode == "both"
             else [head_mode])
    for mode in modes:
        head = train_mass_head(H_all[tr], frame, mode, y_tr=y[tr], seed=seed)
        W_trn = width_from_mass(head(H_all), frame)
        a_conf_m, unc_conf_m = compose_aurc_fixed([conf[tr]], y[tr],
                                                   [conf[te]], y[te])
        a_cw_m,  unc_confW_m = compose_aurc_fixed(
            [conf[tr], W_trn[tr]], y[tr],
            [conf[te], W_trn[te]], y[te])
        arm = {"STC": stc(W_trn[te], y[te]),
               "aurc_conf": a_conf_m,
               "aurc_conf_plus_W": a_cw_m,
               "aurc_W_alone": aurc(W_trn[te], y[te])}
        if unc_conf_m is not None and unc_confW_m is not None:
            g_mean, g_lo, g_hi = aurc_gap_bootstrap(
                unc_conf_m, unc_confW_m, y[te], B=2000, seed=seed)
            arm.update({"gap_mean": g_mean, "gap_lo": g_lo, "gap_hi": g_hi,
                        "verdict": verdict_from_ci(g_mean, g_lo, g_hi)})
        else:
            arm.update({"gap_mean": None, "gap_lo": None, "gap_hi": None,
                        "verdict": "NULL (train single-class)"})
        res[f"trained_{mode}"] = arm

    return res


# ----------------------------------------------------------------- aggregation
def aggregate_arms(seed_rows):
    """Pull each arm across seeds; report bootstrap-gap summary.
    The per-seed verdict is itself a CI verdict; the aggregate verdict below
    uses the across-seed mean gap and the fraction of seeds whose own CI
    excludes zero and clears the noise floor.
    """
    arms = seed_rows[0].keys()
    agg = {}
    for arm in arms:
        rows = [r[arm] for r in seed_rows if arm in r]
        gaps = np.array([r["gap_mean"] for r in rows
                         if r["gap_mean"] is not None])
        stcs = np.array([r["STC"] for r in rows])
        a_conf = np.array([r["aurc_conf"] for r in rows
                           if r["aurc_conf"] is not None])
        a_cw   = np.array([r["aurc_conf_plus_W"] for r in rows
                           if r["aurc_conf_plus_W"] is not None])
        helps_strict = sum(
            1 for r in rows
            if r["verdict"].startswith("HELPS"))
        n_seeds = len(rows)
        agg[arm] = {
            "STC_mean": float(np.mean(stcs)) if len(stcs) else None,
            "STC_std":  float(np.std(stcs))  if len(stcs) else None,
            "aurc_conf_mean":          (float(np.mean(a_conf))
                                        if len(a_conf) else None),
            "aurc_conf_plus_W_mean":   (float(np.mean(a_cw))
                                        if len(a_cw) else None),
            "aurc_W_alone_mean":       float(np.mean(
                [r["aurc_W_alone"] for r in rows])),
            "gap_mean_across_seeds":   (float(np.mean(gaps))
                                        if len(gaps) else None),
            "gap_std_across_seeds":    (float(np.std(gaps))
                                        if len(gaps) else None),
            "n_seeds_gap_positive":    int((gaps > 0).sum()) if len(gaps) else 0,
            "n_seeds_gap_helps":       helps_strict,
            "n_seeds":                 n_seeds,
        }
    return agg


def verdict_aggregate(agg_row, noise_floor=0.02):
    """Cross-seed verdict. Requires both a positive mean gap AND at least
    half of the seeds to individually return a HELPS verdict (so a single
    lucky seed can't carry the cell)."""
    mean_gap = agg_row["gap_mean_across_seeds"]
    n_helps  = agg_row["n_seeds_gap_helps"]
    n_seeds  = agg_row["n_seeds"]
    if mean_gap is None or n_seeds == 0:
        return "NULL (no valid gaps)"
    if mean_gap > noise_floor and n_helps >= max(1, n_seeds // 2):
        return ("DEFER: W adds info beyond logprob conf "
                f"(mean gap {mean_gap:+.4f} > {noise_floor}, "
                f"{n_helps}/{n_seeds} seeds CI-HELPS).")
    if mean_gap > 0:
        return (f"MARGINAL: mean gap {mean_gap:+.4f} > 0 but below noise "
                f"floor or inconsistent ({n_helps}/{n_seeds} seeds CI-HELPS).")
    return ("ABSTAIN: W does not consistently beat logprob alone "
            f"(mean gap {mean_gap:+.4f}).")


# ----------------------------------------------------------------- main
def _parse_k_sweep(raw):
    """Parse --k_sweep: either a comma-separated list ('20,50,100') or
    a Python-style range spec ('20:201:40' -> start,stop,step)."""
    if raw is None:
        return None
    raw = raw.strip()
    if ":" in raw:
        parts = raw.split(":")
        if len(parts) == 2:
            start, stop = int(parts[0]), int(parts[1])
            ks = list(range(start, stop + 1))
        elif len(parts) == 3:
            start, stop, step = (int(p) for p in parts)
            ks = list(range(start, stop + 1, step))
        else:
            raise ValueError(f"bad --k_sweep range spec: {raw!r}")
    else:
        ks = [int(x) for x in raw.split(",") if x.strip()]
    if not ks:
        raise ValueError(f"--k_sweep parsed to empty list: {raw!r}")
    return sorted(set(ks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None,
                    help="path or glob to cache file(s); required unless --model")
    ap.add_argument("--model", default=None,
                    help="HF model id for live state collection (e.g. 'gpt2', "
                         "'meta-llama/Llama-3-8B'). Requires rsuq.extract.")
    ap.add_argument("--dataset", default="live",
                    help="dataset label for the live-model cell")
    ap.add_argument("--device", default="cuda", help="torch device for live mode")
    ap.add_argument("--dtype", default=None,
                    choices=[None, "fp16", "bf16", "fp32"],
                    help="model dtype for live mode")
    ap.add_argument("--pool", default="last", choices=["last", "last4_mean"],
                    help="hidden-state pooling strategy")
    ap.add_argument("--block_size", type=int, default=256,
                    help="token block size for chunking")
    ap.add_argument("--batch_size", type=int, default=32,
                    help="batch size for live forward passes")
    ap.add_argument("--k", type=int, default=50,
                    help="single K (ignored if --k_sweep is given)")
    ap.add_argument("--k_sweep", default=None,
                    help="comma list '20,50,100' or range '20:201:40'; "
                         "supersedes --k")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--head_mode", default="both",
                    choices=["supervised", "unsupervised", "both"])
    ap.add_argument("--out", default="results_repspace_deferral.json")
    args = ap.parse_args()

    if not args.cache and not args.model:
        ap.error("one of --cache or --model is required")

    ks = _parse_k_sweep(args.k_sweep) or [args.k]

    # ---- gather cells: from cache (.pt) or from a live HF model ----
    cells = {}
    if args.model:
        if not _HAS_EXTRACT:
            raise ImportError(
                "rsuq.extract not available — cannot use --model mode. "
                "Run from the rsuq/ directory or pip install -e .")
        print(f"loading live model: {args.model} ({args.dtype or 'default'})")
        tok, model = load_live_model(args.model, args.device, args.dtype)
        # caller must supply texts; placeholder: raise with guidance
        raise NotImplementedError(
            "Live-model collection requires a QA data loader to provide "
            "texts_correct / texts_wrong. Wire collect_qa_cells to your "
            "dataset (e.g. rsuq.qa_data.load_qa_instances) before using "
            "--model. See collect_qa_cells() signature for the contract.")
    else:
        paths = sorted(glob.glob(args.cache))
        if not paths:
            raise FileNotFoundError(f"no cache files match {args.cache!r}")
        for p in paths:
            cells.update(load_cache(p))
        print(f"loaded {len(cells)} (model,dataset) cells from {len(paths)} file(s); "
              f"K sweep = {ks}")

    results = {}
    verdicts = []
    for meta, data in cells.items():
        key = str(meta)
        results[key] = {"by_K": {}}
        h_pos, h_neg = data["h_pos"], data["h_neg"]
        lp_pos = data.get("lp_pos"); lp_neg = data.get("lp_neg")
        N = h_pos.shape[0]
        print(f"\n=== cell {key}  N={N} d={h_pos.shape[1]} ===")

        for K in ks:
            print(f"\n --- K={K} ---")
            seed_rows = []
            for seed in range(args.seeds):
                cell_res = run_cell(h_pos, h_neg, K, seed, args.head_mode,
                                    lp_pos=lp_pos, lp_neg=lp_neg)
                seed_rows.append(cell_res)
                for arm in ("geometric", "trained_supervised",
                            "trained_unsupervised"):
                    if arm not in cell_res:
                        continue
                    a = cell_res[arm]
                    gap_str = (f"{a['gap_mean']:+.4f}"
                               if a["gap_mean"] is not None else "  n/a")
                    print(f"  K={K} seed={seed} arm={arm:<22} "
                          f"STC={a['STC']:+.4f}  gap={gap_str}  "
                          f"{a['verdict']}")

            agg = aggregate_arms(seed_rows)
            results[key]["by_K"][K] = {"per_seed": seed_rows, "agg": agg}
            for arm, a in agg.items():
                v = verdict_aggregate(a)
                print(f"  [K={K}|{arm}] {v}")
                verdicts.append({
                    "cell": key, "K": K, "arm": arm,
                    "verdict": v,
                    "gap_mean_across_seeds": a["gap_mean_across_seeds"],
                    "n_seeds_gap_helps": a["n_seeds_gap_helps"],
                    "n_seeds": a["n_seeds"],
                    "STC_mean": a["STC_mean"],
                })

    # ---- K-sweep summary table ----
    print("\n" + "=" * 78)
    print("K-SWEEP SUMMARY  (gap = AURC_conf - AURC_conf+W;  positive = W helps)")
    print("=" * 78)
    hdr = f"{'cell':<32} {'K':>5} {'arm':<22} {'gap_mean':>9} {'helps':>6} {'STC':>7}  verdict"
    print(hdr)
    print("-" * len(hdr))
    for v in verdicts:
        gap = v["gap_mean_across_seeds"]
        gap_str = f"{gap:+.4f}" if gap is not None else "n/a"
        stc_str = (f"{v['STC_mean']:+.4f}"
                   if v["STC_mean"] is not None else "n/a")
        # short verdict tag
        vtag = ("DEFER" if v["verdict"].startswith("DEFER")
                else "MARG" if v["verdict"].startswith("MARGINAL")
                else "ABST")
        print(f"{v['cell']:<32} {v['K']:>5} {v['arm']:<22} "
              f"{gap_str:>9} {v['n_seeds_gap_helps']:>2}/{v['n_seeds']:<2} "
              f"{stc_str:>7}  {vtag}")

    summary = {
        "meta": {"source": args.cache or args.model,
                 "K_sweep": ks, "seeds": args.seeds,
                 "head_mode": args.head_mode, "n_cells": len(cells),
                 "bootstrap_B": 2000, "noise_floor": 0.02},
        "results": results,
        "verdicts": verdicts,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()