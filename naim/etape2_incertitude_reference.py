"""Étape 2 (things2do.md §3) : intervalle joint pour D = rel_fwd − accord forward/reversed.

Définitions retrouvées le 2026-09-15 en recalculant exactement le ledger
cached_results/letter11_provenance_v1/summary.csv (le script qui l'a produit n'est dans aucune
branche git) :
- accord ρ : Spearman(V_pos forward, V_pos reversed), caches provenance-v1, 817 questions ;
- rel_fwd  : Spearman(V_pos, Valt_pos) sur alt_index (questions 0..499), cache d'origine
  logprobs.pt (prompt de confiance contre prompt reformulé) ;
- intervalles du ledger : un seul np.random.default_rng(20260908) partagé par les 6 cellules dans
  l'ordre du fichier, 2000 tirages, percentiles 2,5/97,5.

Bootstrap joint : les 817 questions sont rééchantillonnées (mêmes tirages pour les trois modèles).
À chaque tirage, ρ est recalculé sur le tirage et rel_fwd sur la partie du tirage qui tombe dans
alt_index, donc les questions partagées restent liées. Deux analyses sont rapportées :
- population_complete (principale) : ρ sur 817 questions, rel_fwd sur 500 ;
- sous_ensemble_commun_500 : ρ et rel_fwd sur les 500 questions communes (analyse étiquetée).
L'inférence est conditionnelle aux caches extraits : rien n'est ré-extrait ni réajusté.

Usage : ~/Documents/hf/bin/python naim/etape2_incertitude_reference.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from commun import (BOOT_SEED, MODELS, N_BOOT, RESULTS, ci, example_ids, input_hashes,
                    provenance, read_ledger, spearman, step1, write_json)

OUT = RESULTS / "etape2"
LEDGER_BOOT_SEED = 20260908
TOL = 5e-4


def load_cell(model: str) -> dict:
    fw, rv, lp = provenance(model, "forward"), provenance(model, "reversed"), step1(model, "logprobs")
    ids = example_ids(fw)
    if not (np.array_equal(ids, example_ids(rv)) and np.array_equal(ids, example_ids(lp))):
        raise ValueError(f"{model} : identifiants différents entre les caches")
    return {
        "ids": ids,
        "alt": np.asarray(lp["alt_index"], dtype=int),
        "fwd": np.asarray(fw["V_pos"], dtype=float),
        "rev": np.asarray(rv["V_pos"], dtype=float),
        "v_step1": np.asarray(lp["V_pos"], dtype=float),
        "valt": np.asarray(lp["Valt_pos"], dtype=float),
    }


def audit(cells: dict, ledger: list[dict]) -> list[dict]:
    """Vérifie les définitions, les populations et la reproduction des intervalles du ledger."""
    rows = {}
    for model, c in cells.items():
        ref = next(r for r in ledger if r["model"] == model and r["dataset"] == "truthfulqa")
        alt = c["alt"]
        rho, rel = spearman(c["fwd"], c["rev"]), spearman(c["v_step1"][alt], c["valt"])
        if abs(rho - float(ref["rho_forward_reversed"])) > TOL or abs(rel - float(ref["rel_fwd"])) > TOL:
            raise ValueError(f"{model} : définitions non retrouvées (rho={rho:.4f}, rel_fwd={rel:.4f})")
        rows[model] = {
            "model": model,
            "n_accord": len(c["fwd"]), "ledger_n": int(ref["n"]),
            "n_fiabilite": len(alt), "ledger_n_alt": int(ref["n_alt"]),
            "alt_index_egal_0_a_499": bool(np.array_equal(alt, np.arange(len(alt)))),
            "questions_communes": int(np.intersect1d(c["ids"], c["ids"][alt]).size),
            "rho_recalcule": round(rho, 6), "rho_ledger": float(ref["rho_forward_reversed"]),
            "rel_fwd_recalcule": round(rel, 6), "rel_fwd_ledger": float(ref["rel_fwd"]),
            "rel_fwd_si_V_provenance": round(spearman(c["fwd"][alt], c["valt"]), 6),
            "ecart_max_V_origine_vs_provenance": float(np.abs(c["v_step1"] - c["fwd"]).max()),
        }
    rng = np.random.default_rng(LEDGER_BOOT_SEED)
    for ref in ledger:
        n, draws = int(ref["n"]), int(ref["bootstrap_draws"])
        idx = rng.integers(0, n, (draws, n))  # consommé aussi pour pavlick_nli, comme dans le ledger
        if ref["dataset"] != "truthfulqa" or ref["model"] not in cells:
            continue
        c = cells[ref["model"]]
        lo, hi = ci([spearman(c["fwd"][i], c["rev"][i]) for i in idx])
        rows[ref["model"]].update(
            ic_rho_ledger=f"[{ref['bootstrap_lo']}, {ref['bootstrap_hi']}]",
            ic_rho_reproduit=f"[{lo:.4f}, {hi:.4f}]",
            ic_ledger_reproduit=bool(abs(lo - float(ref["bootstrap_lo"])) < TOL
                                     and abs(hi - float(ref["bootstrap_hi"])) < TOL))
    return list(rows.values())


def conclusion(lo: float, hi: float) -> str:
    if lo > 0:
        return "D > 0 : accord sous la référence (intervalle entièrement au-dessus de 0)"
    if hi < 0:
        return "D < 0 : accord au-dessus de la référence"
    return "non concluant : l'intervalle contient 0"


def joint_bootstrap(cells: dict, n_boot: int, seed: int) -> list[dict]:
    n = len(next(iter(cells.values()))["fwd"])
    idx = np.random.default_rng(seed).integers(0, n, (n_boot, n))
    rows = []
    for model, c in cells.items():
        alt = c["alt"]
        pos = np.full(n, -1)
        pos[alt] = np.arange(len(alt))
        v, valt = c["v_step1"][alt], c["valt"]
        draws = {"population_complete": [], "sous_ensemble_commun_500": []}
        for i in idx:
            j = i[pos[i] >= 0]
            rel = spearman(v[pos[j]], valt[pos[j]])
            draws["population_complete"].append((rel, spearman(c["fwd"][i], c["rev"][i])))
            draws["sous_ensemble_commun_500"].append((rel, spearman(c["fwd"][j], c["rev"][j])))
        rel_pt = spearman(v, valt)
        points = {"population_complete": (spearman(c["fwd"], c["rev"]), n),
                  "sous_ensemble_commun_500": (spearman(c["fwd"][alt], c["rev"][alt]), len(alt))}
        for analysis, pairs in draws.items():
            pairs = np.asarray(pairs)
            rho_pt, n_rho = points[analysis]
            d = pairs[:, 0] - pairs[:, 1]
            (rel_lo, rel_hi), (rho_lo, rho_hi), (d_lo, d_hi) = ci(pairs[:, 0]), ci(pairs[:, 1]), ci(d)
            rows.append({
                "model": model, "analyse": analysis,
                "n_questions_accord": n_rho, "n_questions_fiabilite": len(alt),
                "rel_fwd": rel_pt, "rel_fwd_ic_bas": rel_lo, "rel_fwd_ic_haut": rel_hi,
                "accord": rho_pt, "accord_ic_bas": rho_lo, "accord_ic_haut": rho_hi,
                "D": rel_pt - rho_pt, "D_ic_bas": d_lo, "D_ic_haut": d_hi,
                "part_tirages_D_negatif_ou_nul": float(np.mean(d <= 0)),
                "conclusion": conclusion(d_lo, d_hi),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Étape 2 : intervalle joint de D.")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=BOOT_SEED)
    args = parser.parse_args()

    cells = {m: load_cell(m) for m in MODELS}
    audit_rows = audit(cells, read_ledger())
    table = joint_bootstrap(cells, args.n_boot, args.seed)

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(OUT / "audit_sources.csv", index=False)
    pd.DataFrame(table).to_csv(OUT / "table_D.csv", index=False)
    write_json(OUT / "manifest.json", {
        "script": "naim/etape2_incertitude_reference.py",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "n_boot": args.n_boot, "seed": args.seed,
        "unite_bootstrap": "question (mêmes tirages pour les trois modèles)",
        "intervalles": "percentiles 2,5 / 97,5",
        "analyse_principale": "population_complete",
        "inference": "conditionnelle aux caches extraits ; aucun composant réajusté",
        "entrees_sha256": input_hashes(),
    })

    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(pd.DataFrame(audit_rows)[["model", "n_accord", "n_fiabilite", "rho_recalcule",
                                        "rel_fwd_recalcule", "ic_rho_reproduit", "ic_ledger_reproduit"]])
        print(pd.DataFrame(table)[["model", "analyse", "accord", "rel_fwd", "D", "D_ic_bas",
                                   "D_ic_haut", "conclusion"]].round(4))
    print(f"Résultats écrits dans {OUT}")


if __name__ == "__main__":
    main()
