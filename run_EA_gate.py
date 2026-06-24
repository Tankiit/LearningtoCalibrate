#!/usr/bin/env python3
"""run_EA_gate.py — E-A: the foundation gate (run FIRST, on M2).

Two checks, both cheap, both falsifying:

  E-A.1  Confusion alignment (frame foundation). Do embedding-clustered focal
         sets align with the model's confusion structure? Top-k softmax
         competitors should land in the same cluster more than a size-matched
         random partition predicts. FAIL => the fixed frame is unsound and
         NOTHING downstream (Stage I/II, AAAI fixed baseline) is trustworthy.

  E-A.2  Sense-routing sanity. For a handful of classic polysemes, do the
         per-context hidden states actually separate into sensible senses,
         and do those senses route to DIFFERENT clusters? FAIL => the AAAI
         mechanism has nothing to act on; rethink sense construction before
         building E-C.

Usage (M2):
    pip install torch transformers datasets scikit-learn
    python run_EA_gate.py --model gpt2 --k 200 --max_positions 20000

Model-agnostic via AutoModelForCausalLM + get_input_embeddings().
"""
from __future__ import annotations
import argparse, json, sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rsuq"))
# from rsuq.core.frame import FixedFrame
# from rsuq.extract import chunk_texts, embedding_matrix

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")

# Classic polysemes for E-A.2 (token must be single-piece in the tokenizer;
# script checks and skips multi-piece ones).
POLYSEMES = ["bank", "spring", "bat", "bark", "plant", "match", "rock",
             "court", "club", "seal", "light", "pitch", "fair", "mine"]
# Disambiguating prefixes per token: (senseA_context, senseB_context)
CONTEXTS = {
    "bank":  ("I sat by the edge of the river",
              "I deposited the cheque at the"),
    "spring": ("The flowers bloom every warm",
               "The mattress coil is a metal"),
    "bat":   ("At dusk the cave released every",
              "He swung the heavy wooden"),
    "bark":  ("The old oak had rough grey",
              "The startled dog began to"),
    "plant": ("She watered the small green",
              "The factory was a huge industrial"),
    "match": ("He struck the wooden",
              "They won the football"),
    "rock":  ("The climber gripped the solid",
              "The band played loud"),
    "court": ("The king addressed his royal",
              "They played tennis on the"),
    "club":  ("He danced all night at the",
              "She swung the golf"),
    "seal":  ("On the ice rested a grey",
              "He pressed the wax"),
}


@torch.inference_mode()
def confusion_alignment(model, tokenizer, frame, blocks, ks=(5, 10),
                        max_positions=20000, n_boot=1000, seed=0):
    model.eval()
    kappa = frame.kappa
    same = {k: [] for k in ks}
    n = 0
    for i in range(0, blocks.shape[0]):
        if n >= max_positions:
            break
        ids = blocks[i:i+1].to(DEVICE)
        logits = model(ids).logits[0]                      # (T, V)
        for k in ks:
            top = logits.topk(k, -1).indices.cpu()         # (T, k)
            c = kappa[top]
            eq = (c.unsqueeze(-1) == c.unsqueeze(-2)).float()
            same[k].extend(((eq.sum((-1, -2)) - k) / (k*(k-1))).tolist())
        n += ids.shape[1]
    szs = frame.sizes.numpy(); V = float(len(kappa))
    base = float((szs*(szs-1)).sum() / (V*(V-1)))          # analytic baseline
    rng = np.random.default_rng(seed)
    out = {}
    for k in ks:
        a = np.array(same[k])
        ci = np.quantile([a[rng.integers(0, len(a), len(a))].mean()
                          for _ in range(n_boot)], [0.025, 0.975])
        out[k] = {"observed": float(a.mean()), "random_baseline": base,
                  "ratio": float(a.mean()/max(base, 1e-12)),
                  "ci": [float(ci[0]), float(ci[1])],
                  "pass": bool(ci[0] > base)}
    out["verdict"] = ("PASS" if all(out[k]["pass"] for k in ks)
                      else "FAIL: frame foundation unsound — stop, rethink "
                           "partition (PCA dim? K? clustering layer?)")
    return out


@torch.inference_mode()
def sense_routing_sanity(model, tokenizer, frame, pool="last4_mean"):
    """For each polyseme with two disambiguating contexts, pool the token's
    hidden state in each context and check (a) the two states are far apart
    (senses separate) and (b) their nearest fixed-frame cluster-prototype
    differs (routing would move the token). Prototype = mean wte of cluster
    members projected to hidden space is unavailable here, so we use the
    CHEAP proxy: does the token's top-competitor cluster differ by context?
    Full prototype routing is the E-C pipeline."""
    model.eval()
    from rsuq.align import locate_word_positions
    kappa = frame.kappa
    rows = []
    for tok, (ctxA, ctxB) in CONTEXTS.items():
        res = {"token": tok}
        ok = True
        for tag, ctx in (("A", ctxA), ("B", ctxB)):
            # Put the word IN the context, then score the position just BEFORE
            # it (the model's prediction of the word's slot). Robust to multi-
            # piece words via offset mapping; no leading-space guessing.
            full = ctx + " " + tok
            try:
                loc = locate_word_positions(tokenizer, full, tok, which="first")
            except ValueError:
                ok = False; break
            ids = loc["input_ids"].to(DEVICE)
            pos = max(loc["score_index"] - 1, 0)           # predict-the-slot
            logits = model(ids).logits[0, pos]
            top = logits.topk(10).indices.cpu()
            res[f"top_clusters_{tag}"] = sorted(set(kappa[top].tolist()))
        if not ok:
            continue
        a, b = set(res["top_clusters_A"]), set(res["top_clusters_B"])
        res["competitor_jaccard"] = len(a & b)/len(a | b) if (a|b) else 1.0
        rows.append(res)
    mean_j = float(np.mean([r["competitor_jaccard"] for r in rows])) if rows else 1.0
    return {"rows": rows, "mean_competitor_jaccard": mean_j,
            "verdict": ("PASS: contexts produce distinct competitor sets "
                        "(low Jaccard) — routing has signal to act on"
                        if mean_j < 0.5 else
                        "WEAK: competitor sets overlap; sense separation may "
                        "be too weak for routing to matter — inspect rows")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--pca", type=int, default=64)
    ap.add_argument("--max_positions", type=int, default=20000)
    ap.add_argument("--n_docs", type=int, default=2000)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from rsuq.core.frame import FixedFrame
    from rsuq.extract import chunk_texts

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    frame = FixedFrame.from_model(model, K=args.k, pca_dim=args.pca)
    frame.save("results_EA_frame.pt")

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if len(t.strip()) > 200][:args.n_docs]
    blocks = chunk_texts(tok, texts, block_size=256,
                         max_blocks=args.max_positions // 256 + 1)

    r1 = confusion_alignment(model, tok, frame, blocks,
                             max_positions=args.max_positions)
    r2 = sense_routing_sanity(model, tok, frame)
    out = {"model": args.model, "K": args.k, "EA1_confusion": r1,
           "EA2_sense_routing": r2}
    print(json.dumps(out, indent=2))
    with open("results_EA_gate.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n" + r1["verdict"])
    print(r2["verdict"])


if __name__ == "__main__":
    main()
