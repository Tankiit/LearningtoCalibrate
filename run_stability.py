#!/usr/bin/env python3
"""run_stability.py — Is the +0.0075 Stratum-P delta stable?

Varies frame+head seed together across a tau sweep. Caches the seed/tau-
INDEPENDENT work once (occurrences, QA instances, state cache).

Decision rule (primary = per-seed sign count, NOT the noisy tau-trend):
  STABLE : >=4/5 seeds give a positive Stratum-P delta at tau>=0.5
  FRAGILE: seeds straddle zero -> +0.0075 was a one-seed artifact.
"""
from __future__ import annotations
import argparse, json, math, sys, os
import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def parse_seeds(n_seeds: int, seed_list: list[int] | None) -> list[int]:
    if seed_list:
        return seed_list
    return list(range(n_seeds))


def stable_threshold(n: int, frac: float = 0.8) -> int:
    return max(1, math.ceil(frac * n))


def finite_or_none(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def p_label_counts(rows: list[dict]) -> dict:
    p_rows = [r for r in rows if r.get("in_P") == 1]
    return {
        "n_P_positions": len(p_rows),
        "n_P_error": int(sum(r.get("error", 0) for r in p_rows)),
        "n_P_correct": int(sum(1 - r.get("error", 0) for r in p_rows)),
    }


def safe_analyse(rows: list[dict]):
    from run_EC_ED import analyse

    try:
        return analyse(rows), None
    except ValueError as exc:
        # Degenerate bootstrap/AUROC cells happen when a stratum has one label.
        return None, f"analysis failed: {exc}"


def main():
    ap = argparse.ArgumentParser(
        description="Seed/tau stability sweep for the E-C/E-D Stratum-P delta.")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--dataset", default="halueval")
    ap.add_argument("--k", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed_list", type=int, nargs="+", default=None,
                    help="Explicit seeds; overrides --seeds.")
    ap.add_argument("--taus", type=float, nargs="+", default=[0.4, 0.5, 0.6, 0.7])
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--span_mode", default="first", choices=["first", "span"],
                    help="Match run_EC_ED: first answer token or all span tokens.")
    ap.add_argument("--n_docs", type=int, default=2000)
    ap.add_argument("--train_blocks", type=int, default=400)
    ap.add_argument("--n_candidates", type=int, default=2000)
    ap.add_argument("--max_occ", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", default="results_stability.json")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds, args.seed_list)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    from rsuq.core.frame import FixedFrame, ContextFrame
    from rsuq.core.sense_inventory import collect_occurrences, build_inventory
    from rsuq.core.beliefs import MassHead
    from rsuq.extract import chunk_texts, collect_states
    from rsuq.train import train_mass_head
    from rsuq.qa_data import load_qa_instances
    from run_EC_ED import score_answer_tokens

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.hidden_size

    # ---- cache the seed/tau-INDEPENDENT work ONCE ----
    print("caching shared work (corpus, occurrences, states, QA) ...")
    wt = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in wt["text"] if len(t.strip()) > 200][:args.n_docs]
    blocks = chunk_texts(tok, texts, block_size=256, max_blocks=args.train_blocks)
    if blocks.numel() == 0:
        raise ValueError("no WikiText training blocks were produced; increase --n_docs")
    flat = blocks.flatten()
    freq = torch.bincount(flat, minlength=model.config.vocab_size)
    n_candidates = min(args.n_candidates, int(freq.numel()))
    candidate_ids = set(torch.topk(freq, n_candidates).indices.tolist())
    occ = collect_occurrences(model, tok, blocks, candidate_ids, DEVICE,
                              max_per_token=args.max_occ)
    cache = collect_states(model, blocks, device=DEVICE)
    instances = load_qa_instances(args.dataset, args.n, seed=0)
    print(f"  device={DEVICE} blocks={blocks.shape[0]} candidates={len(candidate_ids)} "
          f"qa_instances={len(instances)} seeds={seeds}")

    # ---- loop seeds x taus (rebuild frame+inventory+head each seed) ----
    cells = []
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        torch.manual_seed(seed)
        frame = FixedFrame.from_model(model, K=args.k, seed=seed)
        head = MassHead(d_model=d, K=args.k)
        train_stats = train_mass_head(head, cache.h, cache.gold, frame,
                                      device=DEVICE, seed=seed,
                                      epochs=args.epochs)
        for tau in args.taus:
            inv = build_inventory(frame, occ, d, tau=tau, seed=seed)
            base_cell = {
                "tau": float(tau),
                "seed": int(seed),
                "n_P_tokens": int(inv["n_stratum_P"]),
                "train": train_stats,
            }
            if inv["n_stratum_P"] < 5:
                cells.append({**base_cell, "delta": None,
                              "note": "stratum P too small"})
                print(f"  seed={seed} tau={tau}: skipped "
                      f"nP_tokens={inv['n_stratum_P']}")
                continue
            ctx = ContextFrame(base=frame, poly_tokens=inv["poly_tokens"],
                               sense_centroids=inv["sense_centroids"],
                               sense_cluster=inv["sense_cluster"])
            rows = score_answer_tokens(model, tok, frame, ctx, head, instances,
                                       span_mode=args.span_mode)
            label_counts = p_label_counts(rows)
            res, note = safe_analyse(rows)
            if res is None:
                cells.append({**base_cell, **label_counts, "delta": None,
                              "note": note})
                print(f"  seed={seed} tau={tau}: skipped {note} "
                      f"nP_pos={label_counts['n_P_positions']}")
                continue
            P = res.get("P", {}).get("EC_auroc_delta_ctx_minus_fixed", {})
            delta = finite_or_none(P.get("delta"))
            cell = {
                **base_cell,
                **label_counts,
                "n_P_positions": res.get("P", {}).get("n"),
                "delta": delta, "ci": P.get("ci"),
                "sig": P.get("significant"),
                "screen_partial_fixed": res.get("P", {}).get("ED_screen_fixed", {}).get("partial_W_err_given_conf"),
                "screen_partial_ctx": res.get("P", {}).get("ED_screen_ctx", {}).get("partial_W_err_given_conf"),
                "mean_dW_routed": res["routing_diagnostics"]["mean_dW_when_routed"],
                "frac_P_routed": res["routing_diagnostics"]["frac_P_routed"],
            }
            cells.append(cell)
            delta_s = f"{delta:+.5f}" if delta is not None else "NA"
            print(f"  seed={seed} tau={tau}: delta={delta_s} "
                  f"sig={P.get('significant')} "
                  f"nP_tokens={inv['n_stratum_P']} "
                  f"nP_pos={cell['n_P_positions']}")

    # ---- aggregate ----
    from scipy.stats import spearmanr
    summary = {
        "meta": {
            "model": args.model,
            "dataset": args.dataset,
            "K": args.k,
            "seeds": seeds,
            "taus": args.taus,
            "n": args.n,
            "span_mode": args.span_mode,
            "n_docs": args.n_docs,
            "train_blocks": args.train_blocks,
            "n_candidates": n_candidates,
            "max_occ": args.max_occ,
            "epochs": args.epochs,
            "device": DEVICE,
        },
        "per_tau": {},
        "per_seed_main": {},
        "cells": cells,
    }
    tau_means = []
    for tau in args.taus:
        ds = [c["delta"] for c in cells if c["tau"] == tau and c["delta"] is not None]
        if not ds:
            continue
        ds = np.array(ds)
        n_pos = int((ds > 0).sum())
        summary["per_tau"][str(tau)] = {
            "mean_delta": float(ds.mean()), "std_delta": float(ds.std()),
            "n_seeds": len(ds), "n_positive": n_pos,
            "stable": bool(n_pos >= stable_threshold(len(ds)))}
        tau_means.append((tau, ds.mean()))
    if len(tau_means) >= 3:
        ts, ms = zip(*tau_means)
        rho = spearmanr(ts, ms)[0]
        summary["tau_monotonicity_spearman"] = finite_or_none(rho)

    main_taus = [t for t in args.taus if t >= 0.5]
    for seed in seeds:
        ds = [c["delta"] for c in cells
              if c["seed"] == seed and c["tau"] in main_taus
              and c["delta"] is not None]
        if not ds:
            summary["per_seed_main"][str(seed)] = {
                "mean_delta": None, "n_main_taus": 0, "positive": None}
            continue
        mean_delta = float(np.mean(ds))
        summary["per_seed_main"][str(seed)] = {
            "mean_delta": mean_delta,
            "n_main_taus": len(ds),
            "positive": bool(mean_delta > 0),
        }
    valid_seed_rows = [v for v in summary["per_seed_main"].values()
                       if v["positive"] is not None]
    n_pos_main = sum(int(v["positive"]) for v in valid_seed_rows)
    n_valid_main = len(valid_seed_rows)
    frac = n_pos_main / max(n_valid_main, 1)
    summary["frac_positive_main"] = frac
    summary["n_positive_main_seeds"] = n_pos_main
    summary["n_valid_main_seeds"] = n_valid_main
    summary["stable_threshold_main_seeds"] = stable_threshold(n_valid_main) if n_valid_main else None
    summary["verdict"] = (
        "STABLE: detection delta robust across seeds (>=80% positive seeds at "
        "tau>=0.5). Safe to write up; consider mass-routing fix for the "
        "deferral/screen story." if (
            n_valid_main and n_pos_main >= stable_threshold(n_valid_main)
        ) else
        "FRAGILE: delta straddles zero across seeds. +0.0075 was seed-"
        "dependent; the detection headline needs the mass-routing fix.")

    print("\n=== STABILITY SUMMARY ===")
    print(json.dumps(summary["per_tau"], indent=2))
    print("per-seed main tau means:", json.dumps(summary["per_seed_main"], indent=2))
    print("monotonicity (supporting only):", summary.get("tau_monotonicity_spearman"))
    print("\n" + summary["verdict"])
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
