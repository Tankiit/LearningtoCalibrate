#!/usr/bin/env python3
"""run_rsllm_bench.py — AAAI experiment on RS-LLM's own benchmarks.

Claim under test (the ONLY variable is frame membership):
    On OBQA and CoQA, with the SAME model, SAME frame recipe (MiniBatch
    K-means on embed_tokens), and SAME frozen mass head, does sense-routed
    (context-dependent) credal width detect model error better than the
    fixed-frame width — surgically, at positions whose answer token's
    competitor set is context-sensitive (Stratum P), and not elsewhere (M)?

What this deliberately is NOT: a head-to-head against RS-LLM's TRAINED LoRA
belief head. We hold the head frozen for BOTH arms so the comparison isolates
fixed-vs-context membership. RS-LLM's pipeline supplies the baseline recipe
(prompts, frame from embed_tokens, answer-slot scoring); we cite it as the
predecessor whose fixed frame we make context-dependent. Expressiveness is
their head's advantage and is not contested here; single-pass + context is
ours.

Substrates (prompts mirror Shireen's notebooks exactly):
  OBQA  '### Answer: {A/B/C/D}'  — ONE option-token position per question.
        Cleanest: unambiguous label, one scored slot, option competitor set.
  CoQA  '### Answer: {free text}' — answer-span tokens (offset-aligned).

Model: Llama-2-7B 4-bit (their setup) via rsuq.loading; GPT-2 fallback for a
laptop smoke test (--model gpt2). Frozen backbone, no LoRA, no head surgery.

Run:
  python run_rsllm_bench.py --model meta-llama/Llama-2-7b-hf --dataset obqa \\
      --load_in_4bit --k 8000 --tau 0.5
"""
from __future__ import annotations
import argparse, json, sys, os, re
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rsuq"))

# ----------------------------------------------------------------- prompts
def obqa_prompt(rec) -> tuple[str, str]:
    """Mirror Shireen's OBQA template. Returns (prompt_without_answer, gold_letter)."""
    q = list(rec["question"].values())
    options = "".join(f"{o['label']}) {o['text']}\n" for o in q[1])
    text = (f"Fact:\n{rec['fact1']}\n\nAnswer the following question based on "
            f"the above fact by selecting the correct option.\n\n{q[0]}\n"
            f"{options}\n\n### Answer: ")
    return text, rec["answerKey"]


def coqa_prompt(rec, turn: int = 0) -> tuple[str, str]:
    story = rec["story"].replace("\n", "")
    text = (f"### Story: {story}\nAnswer the following question based on the "
            f"above story.\n{rec['questions'][turn]}\n### Answer: ")
    return text, rec["answers"]["input_text"][turn]


# ----------------------------------------------------- OBQA: option-token scoring
@torch.inference_mode()
def score_obqa(model, tok, frame, ctx_frame, head, recs, device, pool="last4_mean"):
    from rsuq.core.beliefs import within_cluster_q
    from rsuq.core.signals import credal_width
    # Llama option-letter token ids (single tokens after a space)
    letters = ["A", "B", "C", "D"]
    rows = []
    poly = set(ctx_frame.poly_tokens.tolist())
    for rec in recs:
        prompt, gold = obqa_prompt(rec)
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        out = model(ids, output_hidden_states=(pool == "last4_mean"))
        logits = out.logits[0, -1]                      # answer-slot logits
        if pool == "last4_mean":
            h = torch.stack(out.hidden_states[-4:], 0).mean(0)[0, -1]
        else:
            h = out.hidden_states[-1][0, -1]
        # option-token ids (leading space, as generated)
        opt_ids = [tok.encode(" " + L, add_special_tokens=False)[-1]
                   for L in letters]
        opt_logits = logits[opt_ids]
        pred = letters[int(opt_logits.argmax())]
        err = int(pred != gold)
        gold_id = opt_ids[letters.index(gold)] if gold in letters else opt_ids[0]

        p = torch.softmax(logits, -1).cpu()
        m = head.masses(h.to(device)).cpu()
        q = within_cluster_q(p, frame.kappa, frame.K)
        W_fix = float(credal_width(m, frame.sizes))
        kap_t, sz_t = ctx_frame.assignments(h.to(device))
        W_ctx = float(credal_width(m, sz_t.cpu()))
        rows.append({"W_fixed": W_fix, "W_ctx": W_ctx,
                     "conf": float(p.max()), "error": err,
                     "in_P": int(gold_id in poly)})
    return rows


# ----------------------------------------------------- CoQA: answer-span scoring
@torch.inference_mode()
def score_coqa(model, tok, frame, ctx_frame, head, recs, device,
               pool="last4_mean", span_mode="first"):
    from rsuq.align import map_span_to_tokens
    from rsuq.core.beliefs import within_cluster_q
    from rsuq.core.signals import credal_width
    from rsuq.qa_data import _appended_span
    rows = []
    poly = set(ctx_frame.poly_tokens.tolist())
    for rec in recs:
        prompt, gold = coqa_prompt(rec)
        text = prompt + gold
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        span = (len(prompt), len(text))
        tok_idx = map_span_to_tokens(tok, text, span[0], span[1], enc)
        if not tok_idx:
            continue
        if span_mode == "first":
            tok_idx = tok_idx[:1]
        out = model(ids, output_hidden_states=(pool == "last4_mean"))
        H = (torch.stack(out.hidden_states[-4:], 0).mean(0)[0]
             if pool == "last4_mean" else out.hidden_states[-1][0])
        logits = out.logits[0]
        # CoQA error: greedy continuation vs gold (EM). Cheap proxy: gold-token
        # NLL above a per-batch threshold; here label by whether gold token is
        # the argmax at its slot (teacher-forced correctness).
        for t in tok_idx:
            p = torch.softmax(logits[t], -1).cpu()
            m = head.masses(H[t].to(device)).cpu()
            q = within_cluster_q(p, frame.kappa, frame.K)
            gold_tok = int(ids[0, t]) if t < ids.shape[1] else -1
            err = int(int(logits[t].argmax()) != gold_tok)
            W_fix = float(credal_width(m, frame.sizes))
            _, sz_t = ctx_frame.assignments(H[t].to(device))
            W_ctx = float(credal_width(m, sz_t.cpu()))
            rows.append({"W_fixed": W_fix, "W_ctx": W_ctx,
                         "conf": float(p.max()), "error": err,
                         "in_P": int(gold_tok in poly)})
    return rows


# ----------------------------------------------------------------- analysis
def auroc_delta_ci(a, b, y, B=2000, seed=0):
    from sklearn.metrics import roc_auc_score
    a, b, y = map(np.asarray, (a, b, y))
    if y.sum() in (0, len(y)):
        return {"delta": None, "ci": [None, None], "significant": False,
                "note": "degenerate labels"}
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
            "auroc_fixed": float(roc_auc_score(y, a)),
            "auroc_ctx": float(roc_auc_score(y, b)),
            "significant": bool(lo > 0)}


def screen(W, conf, err):
    from scipy.stats import pearsonr, pointbiserialr
    W, conf, err = map(lambda v: np.asarray(v, float), (W, conf, err))
    if len(set(err.tolist())) < 2:
        return {"note": "degenerate labels"}
    Z = np.column_stack([np.ones(len(W)), conf])
    rW = W - Z @ np.linalg.lstsq(Z, W, rcond=None)[0]
    rE = err - Z @ np.linalg.lstsq(Z, err, rcond=None)[0]
    return {"STC": float(pointbiserialr(err, (W > np.median(W)).astype(float))[0]),
            "rho_conf": float(pearsonr(W, conf)[0]),
            "partial_W_err_given_conf": float(pearsonr(rW, rE)[0])}


def analyse(rows):
    from rsuq.deferral import deferral_battery
    out = {"n_total": len(rows),
           "accuracy": 1 - np.mean([r["error"] for r in rows])}
    # OVERALL deferral table (headline metric, all positions)
    allf = [r["W_fixed"] for r in rows]; allc = [r["W_ctx"] for r in rows]
    allcf = [r["conf"] for r in rows]; alle = [r["error"] for r in rows]
    out["deferral_overall"] = deferral_battery(allcf, allf, allc, alle)
    for s in ("P", "M"):
        rs = [r for r in rows if r["in_P"] == (s == "P")]
        if len(rs) < 30:
            out[s] = {"n": len(rs), "note": "too few positions"}
            continue
        Wf = [r["W_fixed"] for r in rs]; Wc = [r["W_ctx"] for r in rs]
        cf = [r["conf"] for r in rs]; er = [r["error"] for r in rs]
        from rsuq.deferral import deferral_battery
        out[s] = {"n": len(rs),
                  "EC_auroc_delta": auroc_delta_ci(Wf, Wc, er),
                  "ED_screen_fixed": screen(Wf, cf, er),
                  "ED_screen_ctx": screen(Wc, cf, er),
                  "deferral": deferral_battery(cf, Wf, Wc, er)}
    p = out.get("P", {}).get("EC_auroc_delta", {})
    m = out.get("M", {}).get("EC_auroc_delta", {})
    p_def = out.get("P", {}).get("deferral", {})
    out["surgical_claim"] = ("PASS" if p.get("significant") and not
                             m.get("significant", True)
                             else "INSPECT: not localised to Stratum P")
    out["deferral_claim_P"] = p_def.get("deferral_claim", "n/a")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--dataset", default="obqa", choices=["obqa", "coqa"])
    ap.add_argument("--k", type=int, default=8000)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--span_mode", default="first")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--n_candidates", type=int, default=3000)
    ap.add_argument("--train_blocks", type=int, default=400)
    args = ap.parse_args()

    from datasets import load_dataset
    from rsuq.loading import load_model_and_tokenizer
    from rsuq.core.frame import FixedFrame, ContextFrame
    from rsuq.core.sense_inventory import collect_occurrences, build_inventory
    from rsuq.core.beliefs import MassHead
    from rsuq.extract import chunk_texts, collect_states
    from rsuq.train import train_mass_head

    model, tok, device = load_model_and_tokenizer(
        args.model, load_in_4bit=args.load_in_4bit)
    d = model.config.hidden_size

    # frame: OUR recipe (MiniBatch K-means) for both arms
    frame = FixedFrame.from_model(model, K=args.k)

    # sense inventory built from the SAME benchmark's contexts (on-task!):
    # use the stories/facts as the corpus so 'sense' = competitor shift under
    # the actual task context, not generic WikiText.
    if args.dataset == "obqa":
        ds = load_dataset("json", data_files="OpenBookQA-V1-Sep2018/Data/"
                          "Additional/train_complete.jsonl")["train"]
        corpus = [r["fact1"] + " " + list(r["question"].values())[0]
                  for r in ds.select(range(min(2000, len(ds))))]
        test = list(load_dataset("json", data_files="OpenBookQA-V1-Sep2018/"
                    "Data/Additional/test_complete.jsonl")["train"]
                    .select(range(args.n)))
    else:
        ds = load_dataset("stanfordnlp/coqa", split="train")
        corpus = [r["story"].replace("\n", "") for r in ds.select(range(min(1000, len(ds))))]
        test = list(load_dataset("stanfordnlp/coqa", split="validation")
                    .select(range(args.n)))

    blocks = chunk_texts(tok, corpus, block_size=256, max_blocks=args.train_blocks)
    flat = blocks.flatten()
    freq = torch.bincount(flat, minlength=model.config.vocab_size)
    candidate_ids = set(torch.topk(freq, args.n_candidates).indices.tolist())

    print("building sense inventory from on-task contexts ...")
    occ = collect_occurrences(model, tok, blocks, candidate_ids, device)
    inv = build_inventory(frame, occ, d, tau=args.tau)
    print(f"Stratum P: {inv['n_stratum_P']} polysemous tokens")
    ctx = ContextFrame(base=frame, poly_tokens=inv["poly_tokens"],
                       sense_centroids=inv["sense_centroids"],
                       sense_cluster=inv["sense_cluster"])

    print("training frozen-frame mass head (shared by both arms) ...")
    cache = collect_states(model, blocks, device=device)
    head = MassHead(d_model=d, K=frame.K)
    train_mass_head(head, cache.h, cache.gold, frame, device=device)

    print(f"scoring {args.dataset} ...")
    if args.dataset == "obqa":
        rows = score_obqa(model, tok, frame, ctx, head, test, device)
    else:
        rows = score_coqa(model, tok, frame, ctx, head, test, device,
                          span_mode=args.span_mode)

    res = analyse(rows)
    res["meta"] = {"model": args.model, "dataset": args.dataset,
                   "K": args.k, "tau": args.tau,
                   "n_stratum_P_tokens": inv["n_stratum_P"]}
    print(json.dumps(res, indent=2))
    json.dump(res, open(f"results_rsllm_{args.dataset}.json", "w"), indent=2)
    print("\n" + res["surgical_claim"])


if __name__ == "__main__":
    main()
