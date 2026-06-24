#!/usr/bin/env python3
"""run_EB_figure1.py — E-B: the Figure-1 probe (run after E-A passes, on M2).

THE CORRECTED CLAIM (do not state the naive version): it is NOT simply
"fixed-frame width is identical across contexts." That is not guaranteed —
a trained mass head can read different m in different contexts for reasons
unrelated to the sense shift. The honest, sharper claim has THREE quantities
per polysemous token:

    competitor turnover  : Jaccard of top-k next-token competitors across the
                           two sense contexts. LOW => the model's confusion
                           structure genuinely differs by sense.
    fixed-W shift        : |W_fixed(ctxA) - W_fixed(ctxB)|. The fixed frame
                           CANNOT respond to the turnover (membership is
                           frozen), so this shift, whatever its size, is
                           UNCORRELATED with the turnover — it is not tracking
                           the sense change.
    contextual-W shift   : |W_ctx(ctxA) - W_ctx(ctxB)|, where membership is
                           re-routed per context. This SHOULD correlate with
                           the turnover — it tracks the sense change.

Figure 1 = scatter / paired bars showing: across polysemes, contextual-W
shift correlates with competitor turnover while fixed-W shift does not. That
is the well-posedness claim made visible.

This script computes the fixed-W half fully and STUBS the contextual-W half
(needs the ContextFrame sense inventory — E-C pipeline). Run it now for the
fixed-W + turnover columns; fill contextual-W once sense routing is built.

Usage (M2):
    python run_EB_figure1.py --model gpt2 --k 200
"""
from __future__ import annotations
import argparse, json, sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rsuq"))

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")

# (token, contextA, contextB) — A and B select different senses.
PROBES = [
    ("bank",  "I sat by the edge of the river",
              "I deposited the cheque at the"),
    ("spring","The flowers all bloom in early",
              "The mattress is supported by each metal"),
    ("bark",  "The tall oak tree had rough",
              "The angry guard dog began to"),
    ("plant", "She gently watered the potted",
              "They built a large chemical"),
    ("match", "To light the candle he struck a",
              "Our team lost the championship"),
    ("court", "The medieval king summoned his royal",
              "They served an ace on the tennis"),
    ("seal",  "On the arctic ice slept a fat",
              "He stamped the letter with a wax"),
    ("rock",  "The climbers scaled the sheer granite",
              "The crowd loved the loud punk"),
]


@torch.inference_mode()
def fixed_W_and_turnover(model, tokenizer, frame, mass_head=None,
                         topk=10, pool="last4_mean"):
    """Per probe: fixed-frame W in each context + competitor Jaccard.
    If mass_head is None, uses the parameter-free collapsed mass
    (m_k = sum_{w in F_k} softmax) so the script runs BEFORE a head is
    trained — the turnover column is head-independent anyway, and the
    fixed-W column with collapsed mass is a valid lower-fidelity stand-in
    flagged as such."""
    from rsuq.core.beliefs import cluster_prob_mass
    from rsuq.core.signals import credal_width
    from rsuq.align import locate_word_positions
    model.eval()
    kappa, sizes = frame.kappa, frame.sizes
    rows = []
    for tok, ctxA, ctxB in PROBES:
        row = {"token": tok}
        tops = {}
        skip = False
        for tag, ctx in (("A", ctxA), ("B", ctxB)):
            full = ctx + " " + tok
            try:
                loc = locate_word_positions(tokenizer, full, tok, which="first")
            except ValueError:
                skip = True; break
            ids = loc["input_ids"].to(DEVICE)
            pos = max(loc["score_index"] - 1, 0)           # predict-the-slot
            out = model(ids, output_hidden_states=(pool == "last4_mean"))
            logits = out.logits[0, pos]
            p = torch.softmax(logits, -1).cpu()
            if mass_head is None:
                m = cluster_prob_mass(p, kappa, frame.K)        # collapsed
            else:
                if pool == "last4_mean":
                    h = torch.stack(out.hidden_states[-4:], 0).mean(0)[0, pos]
                else:
                    h = out.hidden_states[-1][0, pos]
                m = mass_head.masses(h.to(DEVICE)).cpu()
            row[f"W_fixed_{tag}"] = float(credal_width(m, sizes))
            tops[tag] = set(logits.topk(topk).indices.cpu().tolist())
        if skip:
            continue
        a, b = tops["A"], tops["B"]
        row["competitor_jaccard"] = len(a & b)/len(a | b) if (a | b) else 1.0
        row["dW_fixed"] = abs(row["W_fixed_A"] - row["W_fixed_B"])
        row["W_ctx_A"] = None   # STUB — fill from ContextFrame (E-C pipeline)
        row["W_ctx_B"] = None
        row["dW_ctx"] = None
        rows.append(row)
    return rows


def summarise(rows):
    jac = np.array([r["competitor_jaccard"] for r in rows])
    dwf = np.array([r["dW_fixed"] for r in rows])
    out = {"n_probes": len(rows),
           "mean_competitor_jaccard": float(jac.mean()),
           "mean_dW_fixed": float(dwf.mean())}
    # The key Figure-1 correlation needs dW_ctx (stubbed). Once filled:
    #   corr(turnover, dW_ctx) should be strong; corr(turnover, dW_fixed) ~0.
    out["note"] = ("contextual-W column stubbed; fixed-W + turnover ready. "
                   "Fill dW_ctx from ContextFrame to complete the figure.")
    if any(r["dW_ctx"] is not None for r in rows):
        dwc = np.array([r["dW_ctx"] for r in rows])
        from scipy.stats import pearsonr
        out["corr_turnover_dWfixed"] = float(pearsonr(1-jac, dwf)[0])
        out["corr_turnover_dWctx"] = float(pearsonr(1-jac, dwc)[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--frame", default="results_EA_frame.pt",
                    help="reuse the frame built by run_EA_gate.py")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from rsuq.core.frame import FixedFrame

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if os.path.exists(args.frame):
        frame = FixedFrame.load(args.frame)
    else:
        frame = FixedFrame.from_model(model, K=args.k, pca_dim=args.pca)

    rows = fixed_W_and_turnover(model, tok, frame, mass_head=None)
    out = {"model": args.model, "rows": rows, "summary": summarise(rows)}
    print(json.dumps(out, indent=2))
    with open("results_EB_figure1.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nFixed-W + turnover computed. Contextual-W stubbed (needs E-C "
          "ContextFrame). Sanity: mean competitor Jaccard should be LOW "
          f"(got {out['summary']['mean_competitor_jaccard']:.3f}).")


if __name__ == "__main__":
    main()
