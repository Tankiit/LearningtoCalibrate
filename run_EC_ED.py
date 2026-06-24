#!/usr/bin/env python3
"""run_EC_ED.py — E-C (headline) + E-D (screen), the AAAI main experiments.

E-C: hallucination detection AUROC of sense-routed width vs context-blind
     width, STRATIFIED by position type. Surgical-improvement claim:
         Stratum P (polysemous, measured-separation): ctx-W > fixed-W
         Stratum M (monosemous):                      no difference
     + controls: random-sense, placebo (freq-matched mono), shared/retrained.

E-D: at Stratum P, sense-routed width passes the redundancy screen vs
     max-BetP_R confidence (the parameter-free foil) where fixed-W does not.

Substrate: TriviaQA (primary) / HaluEval (secondary), answer-span tokens,
offset-aligned. Frozen GPT-2 (or any AutoModelForCausalLM via --model).

Run order:
  0. reuse frame from run_EA_gate.py (results_EA_frame.pt)
  1. collect occurrences -> sense inventory (OD-1 measured separation)
  2. build ContextFrame; train mass head on cluster CE (shared across frames)
  3. label answer-span tokens (hallucinated vs correct)
  4. compute fixed-W and ctx-W per answer-span token; stratify by P/M
  5. E-C: paired-bootstrap AUROC delta per stratum + controls
  6. E-D: screen (STC, rho_conf, partial W⊥err|conf) per stratum, both widths

This is a SKELETON with the measurement code wired and the data-loading /
training loops stubbed where they reuse existing RSUQ components. Stubs are
marked TODO and point at the component to reuse.
"""
from __future__ import annotations
import argparse, json, sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rsuq"))

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


# ============================================================ data + labels
from rsuq.qa_data import load_qa_instances   # real loaders (HaluEval/TriviaQA)


@torch.inference_mode()
def score_answer_tokens(model, tokenizer, frame, ctx_frame, mass_head,
                        instances, pool="last4_mean", span_mode="first"):
    """Per answer-span token: fixed-W, ctx-W, max-BetP_R confidence, error
    label, and whether the token is in Stratum P. Uses offset alignment to
    map the answer char span to token positions (rsuq.align)."""
    from rsuq.align import map_span_to_tokens
    from rsuq.core.beliefs import within_cluster_q, ranked_pignistic
    from rsuq.core.signals import credal_width
    model.eval()
    poly_set = set(ctx_frame.poly_tokens.tolist())
    rows = []
    for inst in instances:
        text = inst["question"] + " " + inst["response"]
        enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        ids = enc["input_ids"].to(DEVICE)
        span = inst["answer_char_span"]
        tok_idx = map_span_to_tokens(tokenizer, text, span[0], span[1], enc)
        if not tok_idx:
            continue
        # span_mode: 'first' = first answer-span token (deferral-honest, the
        # per-token correctness convention); 'span' = all answer-span tokens.
        if span_mode == "first":
            tok_idx = tok_idx[:1]
        out = model(ids, output_hidden_states=(pool == "last4_mean"))
        if pool == "last4_mean":
            H = torch.stack(out.hidden_states[-4:], 0).mean(0)[0]
        else:
            H = out.hidden_states[-1][0]
        logits = out.logits[0]
        for t in tok_idx:
            h_t = H[t]
            p_t = torch.softmax(logits[t], -1).cpu()
            m_t = mass_head.masses(h_t.to(DEVICE)).cpu()
            q_t = within_cluster_q(p_t, frame.kappa, frame.K)
            # fixed frame
            W_fix = float(credal_width(m_t, frame.sizes))
            # context frame: per-position kappa_t, sizes_t from h_t
            kap_t, sz_t = ctx_frame.assignments(h_t.to(DEVICE))
            W_ctx = float(credal_width(m_t, sz_t.cpu()))
            bp = ranked_pignistic(m_t, q_t, frame.kappa, frame.sizes, 1.0)
            tok_id = int(ids[0, t].item())
            kap_fix = int(frame.kappa[tok_id])
            kap_ctx = int(kap_t[tok_id]) if kap_t.dim() == 1 else int(kap_t[0, tok_id])
            rows.append({
                "W_fixed": W_fix, "W_ctx": W_ctx,
                "dW": abs(W_ctx - W_fix),
                "routed": int(kap_fix != kap_ctx),
                "size_fixed": float(frame.sizes[kap_fix]),
                "size_ctx": float(sz_t.cpu()[kap_ctx]),
                "conf": float(bp.max()), "error": int(inst["label"]),
                "in_P": int(tok_id in poly_set)})
    return rows


# ================================================================= E-C / E-D
def auroc_delta_ci(a, b, y, B=2000, seed=0):
    from sklearn.metrics import roc_auc_score
    a, b, y = map(np.asarray, (a, b, y))
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, b) - roc_auc_score(y, a)
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() in (0, len(y)):
            continue
        d.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    lo, hi = np.quantile(d, [0.025, 0.975])
    return {"delta": float(base), "ci": [float(lo), float(hi)],
            "significant": bool(lo > 0)}


def screen(W, conf, err):
    from scipy.stats import pearsonr, pointbiserialr
    W, conf, err = map(lambda v: np.asarray(v, float), (W, conf, err))
    Z = np.column_stack([np.ones(len(W)), conf])
    rW = W - Z @ np.linalg.lstsq(Z, W, rcond=None)[0]
    rE = err - Z @ np.linalg.lstsq(Z, err, rcond=None)[0]
    return {"STC": float(pointbiserialr(err, (W > np.median(W)).astype(float))[0]),
            "rho_conf": float(pearsonr(W, conf)[0]),
            "partial_W_err_given_conf": float(pearsonr(rW, rE)[0])}


def routing_diagnostics(rows):
    P = [r for r in rows if r["in_P"] == 1]
    routed = [r for r in P if r.get("routed")]
    d = {
        "n_total": len(rows),
        "n_in_P": len(P),
        "frac_in_P": len(P) / max(len(rows), 1),
        "frac_P_routed": len(routed) / max(len(P), 1),
        "mean_dW_when_routed": float(np.mean([r["dW"] for r in routed])) if routed else 0.0,
        "max_dW_when_routed": float(np.max([r["dW"] for r in routed])) if routed else 0.0,
        "mean_dW_all": float(np.mean([r["dW"] for r in rows])) if rows else 0.0,
        "mean_size_fixed": float(np.mean([r["size_fixed"] for r in rows])) if rows else 0.0,
        "mean_size_ctx_when_routed": float(np.mean([r["size_ctx"] for r in routed])) if routed else 0.0,
    }
    if d["frac_P_routed"] < 0.1:
        d["likely_cause"] = ("ROUTING RARELY FIRES: polysemous tokens seldom get "
                             "sent to a different cluster. Fix: sense->host mapping "
                             "(OD-3) or lower tau / more candidates.")
    elif d["mean_dW_when_routed"] < 0.01:
        d["likely_cause"] = ("ROUTING FIRES BUT W BARELY MOVES: clusters too "
                             "large/similar-sized (K=200). Fix: smaller K.")
    else:
        d["likely_cause"] = ("Routing fires and moves W; if no AUROC gain, the "
                             "moved width just isn't more error-aligned.")
    return d


def analyse(rows):
    R = {s: [r for r in rows if r["in_P"] == (s == "P")] for s in ("P", "M")}
    out = {"routing_diagnostics": routing_diagnostics(rows)}
    for s, rs in R.items():
        if len(rs) < 30:
            out[s] = {"n": len(rs), "note": "too few positions"}
            continue
        Wf = [r["W_fixed"] for r in rs]; Wc = [r["W_ctx"] for r in rs]
        cf = [r["conf"] for r in rs]; er = [r["error"] for r in rs]
        out[s] = {
            "n": len(rs),
            "EC_auroc_delta_ctx_minus_fixed": auroc_delta_ci(Wf, Wc, er),
            "ED_screen_fixed": screen(Wf, cf, er),
            "ED_screen_ctx": screen(Wc, cf, er)}
    out["surgical_claim"] = (
        "PASS" if (out.get("P", {}).get("EC_auroc_delta_ctx_minus_fixed", {})
                   .get("significant") and
                   not out.get("M", {}).get("EC_auroc_delta_ctx_minus_fixed", {})
                   .get("significant", True))
        else "INSPECT: gain not localised to Stratum P as predicted")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--dataset", default="triviaqa")
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--frame", default="results_EA_frame.pt")
    ap.add_argument("--tau", type=float, default=0.5, help="OD-1 separation")
    ap.add_argument("--span_mode", default="first", choices=["first", "span"],
                    help="first answer-span token (deferral-honest) or all")
    ap.add_argument("--n_docs", type=int, default=2000)
    ap.add_argument("--train_blocks", type=int, default=400)
    ap.add_argument("--n_candidates", type=int, default=2000)
    ap.add_argument("--max_occ", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rsuq.core.frame import FixedFrame, ContextFrame
    from rsuq.core.sense_inventory import collect_occurrences, build_inventory
    from rsuq.core.beliefs import MassHead
    from rsuq.extract import chunk_texts

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    frame = (FixedFrame.load(args.frame) if os.path.exists(args.frame)
             else FixedFrame.from_model(model))

    from datasets import load_dataset
    from rsuq.core.sense_inventory import collect_occurrences, build_inventory
    from rsuq.core.beliefs import MassHead
    from rsuq.extract import chunk_texts, collect_states
    from rsuq.train import train_mass_head
    d = model.config.hidden_size

    # 1. corpus blocks (WikiText) for occurrences + head training
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in wt["text"] if len(t.strip()) > 200][:args.n_docs]
    blocks = chunk_texts(tok, texts, block_size=256, max_blocks=args.train_blocks)

    # candidate tokens for senses: the most frequent token ids in the corpus
    flat = blocks.flatten()
    freq = torch.bincount(flat, minlength=model.config.vocab_size)
    candidate_ids = set(torch.topk(freq, args.n_candidates).indices.tolist())

    print("collecting occurrences ...")
    occ = collect_occurrences(model, tok, blocks, candidate_ids, DEVICE,
                              max_per_token=args.max_occ)
    inv = build_inventory(frame, occ, d, tau=args.tau)
    json.dump({str(k): v for k, v in inv["separation_report"].items()},
              open("results_sense_inventory.json", "w"), indent=2)
    print(f"Stratum P: {inv['n_stratum_P']} polysemous tokens "
          f"(tau={args.tau})")
    ctx_frame = ContextFrame(base=frame, poly_tokens=inv["poly_tokens"],
                             sense_centroids=inv["sense_centroids"],
                             sense_cluster=inv["sense_cluster"])

    # 2. train the (shared) mass head on cached cluster-CE
    print("caching states + training mass head ...")
    cache = collect_states(model, blocks, device=DEVICE)
    head = MassHead(d_model=d, K=frame.K)
    train_mass_head(head, cache.h, cache.gold, frame, device=DEVICE,
                    epochs=args.epochs)

    # 3-4. QA instances + per-answer-token scoring
    print(f"loading {args.dataset} + scoring answer tokens ...")
    instances = load_qa_instances(args.dataset, args.n)
    rows = score_answer_tokens(model, tok, frame, ctx_frame, head,
                               instances, span_mode=args.span_mode)

    # 5-6. stratified E-C / E-D
    results = analyse(rows)
    results["meta"] = {"model": args.model, "dataset": args.dataset,
                       "n_rows": len(rows), "span_mode": args.span_mode,
                       "n_stratum_P_tokens": inv["n_stratum_P"]}
    print(json.dumps(results, indent=2))
    json.dump(results, open("results_EC_ED.json", "w"), indent=2)
    rd = results["routing_diagnostics"]
    print("\n=== ROUTING DIAGNOSTICS ===")
    print(f"  polysemous-token positions scored: {rd['n_in_P']} ({100*rd['frac_in_P']:.1f}% of all)")
    print(f"  of those, fraction actually re-routed: {100*rd['frac_P_routed']:.1f}%")
    print(f"  width move when routed (mean / max): {rd['mean_dW_when_routed']:.4f} / {rd['max_dW_when_routed']:.4f}")
    print(f"  mean cluster size (fixed): {rd['mean_size_fixed']:.1f}")
    print(f"  --> {rd['likely_cause']}")
    print("\n" + results["surgical_claim"])


if __name__ == "__main__":
    main()
