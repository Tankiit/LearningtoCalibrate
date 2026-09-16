"""Étape 4, analyse des émissions collectées par etape4_emissions_modal.py.

Pour chaque modèle :
- validité par légende et raisons d'échec (une réponse invalide ne vaut jamais 0) ;
- changement numérique forward/reversed sur les candidats valides sous les deux légendes ;
- ordre des candidats avec les quatre réponses de chaque question : inversion stricte, égalité,
  même ordre ou paquet invalide, avec deux dénominateurs (paquets valides, toutes les questions) ;
- qualité prédictive par légende (AUROC, ordre correct) ;
- à part, le taux d'inversion du readout conditionnel sur les mêmes questions.
Intervalles : bootstrap par question, candidats et légendes gardés ensemble.
Le parser strict est principal ; --parser tolerant sert d'analyse de sensibilité.

Usage : ~/Documents/hf/bin/python naim/etape4_analyse_emissions.py --run shard
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from commun import BOOT_SEED, N_BOOT, RESULTS, as_array, ci, example_ids, provenance

CATEGORIES = ("inversion_stricte", "egalite", "meme_ordre", "invalide")


def load_rows(path: Path, parser_name: str) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    if parser_name == "tolerant":
        df["decoded_value"] = df["lenient_value"]
        df["valid"] = df["lenient_value"].notna()
    return df


def wide(df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par question, colonnes valeur_<candidat>_<légende> (NaN si invalide)."""
    df = df.assign(value=df["decoded_value"].where(df["valid"]))
    table = df.pivot_table(index="question_id", columns=["candidate_side", "legend"],
                           values="value", aggfunc="first", dropna=False)
    expected = pd.MultiIndex.from_product([("pos", "neg"), ("forward", "reversed")])
    table = table.reindex(columns=expected)
    table.columns = [f"valeur_{s}_{l}" for s, l in table.columns]
    return table


def categories(t: pd.DataFrame) -> np.ndarray:
    cols = ["valeur_pos_forward", "valeur_neg_forward", "valeur_pos_reversed", "valeur_neg_reversed"]
    v = t[cols].to_numpy(float)
    sf, sr = np.sign(v[:, 0] - v[:, 1]), np.sign(v[:, 2] - v[:, 3])
    cat = np.where((sf == 0) | (sr == 0), "egalite", np.where(sf != sr, "inversion_stricte", "meme_ordre"))
    return np.where(np.isnan(v).any(axis=1), "invalide", cat)


def rates(cat: np.ndarray) -> tuple[float, float]:
    valid = cat != "invalide"
    strict = cat == "inversion_stricte"
    return (float(strict[valid].mean()) if valid.any() else float("nan"), float(strict.mean()))


def analyse_model(model: str, df: pd.DataFrame, n_boot: int):
    t = wide(df)
    nq = len(t)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, nq, (n_boot, nq))

    validity = []
    for legend, g in df.groupby("legend"):
        validity.append({"model": model, "legende": legend, "reponses": len(g),
                         "valides": int(g["valid"].sum()), "taux_validite": float(g["valid"].mean())})
    reasons = (df[~df["valid"]].groupby(["legend", "failure_reason"], dropna=False).size()
               .rename("n").reset_index().assign(model=model))

    delta = np.abs(np.c_[t["valeur_pos_forward"] - t["valeur_pos_reversed"],
                         t["valeur_neg_forward"] - t["valeur_neg_reversed"]])

    def delta_stats(d):
        d = d[np.isfinite(d)]
        return (float(d.mean()), float((d > 0).mean())) if d.size else (float("nan"), float("nan"))

    mean_pt, share_pt = delta_stats(delta.ravel())
    boot_delta = np.array([delta_stats(delta[i].ravel()) for i in idx])
    numeric = {"model": model, "candidats_valides_deux_legendes": int(np.isfinite(delta).sum()),
               "candidats_total": int(delta.size), "ecart_absolu_moyen": mean_pt,
               "ecart_ic_bas": ci(boot_delta[:, 0])[0], "ecart_ic_haut": ci(boot_delta[:, 0])[1],
               "part_valeur_changee": share_pt,
               "part_ic_bas": ci(boot_delta[:, 1])[0], "part_ic_haut": ci(boot_delta[:, 1])[1]}

    cat = categories(t)
    boot_rates = np.array([rates(cat[i]) for i in idx])
    valid_pt, all_pt = rates(cat)
    fw, rv = provenance(model, "forward"), provenance(model, "reversed")
    pos_in_cache = pd.Series(np.arange(len(example_ids(fw))), index=example_ids(fw))[t.index].to_numpy()
    readout_flip = ((as_array(fw["V_pos"]) > as_array(fw["V_neg"]))
                    != (as_array(rv["V_pos"]) > as_array(rv["V_neg"])))[pos_in_cache]
    order = {"model": model, "questions": nq,
             **{f"n_{c}": int((cat == c).sum()) for c in CATEGORIES},
             "inversion_sur_paquets_valides": valid_pt,
             "ic_bas_valides": ci(boot_rates[:, 0])[0], "ic_haut_valides": ci(boot_rates[:, 0])[1],
             "inversion_sur_toutes_questions": all_pt,
             "ic_bas_toutes": ci(boot_rates[:, 1])[0], "ic_haut_toutes": ci(boot_rates[:, 1])[1],
             "readout_conditionnel_inversion_memes_questions": float(readout_flip.mean())}

    quality = []
    for legend in ("forward", "reversed"):
        g = df[(df["legend"] == legend) & df["valid"]]
        auc = (float(roc_auc_score(g["correct"], g["decoded_value"]))
               if g["correct"].nunique() == 2 else float("nan"))
        vp, vn = t[f"valeur_pos_{legend}"].to_numpy(float), t[f"valeur_neg_{legend}"].to_numpy(float)
        both = np.isfinite(vp) & np.isfinite(vn)
        acc = float(np.mean((vp[both] > vn[both]) + 0.5 * (vp[both] == vn[both]))) if both.any() else float("nan")
        quality.append({"model": model, "legende": legend, "reponses_valides": len(g), "auroc": auc,
                        "questions_deux_candidats_valides": int(both.sum()), "ordre_correct": acc})

    joined = t.assign(model=model, categorie=cat,
                      readout_V_pos_forward=as_array(fw["V_pos"])[pos_in_cache],
                      readout_V_neg_forward=as_array(fw["V_neg"])[pos_in_cache],
                      readout_V_pos_reversed=as_array(rv["V_pos"])[pos_in_cache],
                      readout_V_neg_reversed=as_array(rv["V_neg"])[pos_in_cache]).reset_index()
    return validity, reasons, numeric, order, quality, joined


def main() -> None:
    parser = argparse.ArgumentParser(description="Étape 4 : analyse des émissions.")
    parser.add_argument("--run", default="shard", help="shard ou complet")
    parser.add_argument("--emissions", type=Path, help="fichier emissions.jsonl (sinon results/etape4/<run>)")
    parser.add_argument("--parser", choices=("strict", "tolerant"), default="strict")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    path = args.emissions or RESULTS / "etape4" / args.run / "emissions.jsonl"
    out = path.parent / f"analyse_{args.parser}"
    out.mkdir(parents=True, exist_ok=True)
    df = load_rows(path, args.parser)
    if df["protocol_sha256"].nunique() != 1:
        raise ValueError("plusieurs protocoles mélangés dans le même fichier")

    validity, reasons, numeric, order, quality, joined = [], [], [], [], [], []
    for model, g in df.groupby("model_key"):
        v, r, n, o, q, j = analyse_model(model, g, args.n_boot)
        validity += v; reasons.append(r); numeric.append(n); order.append(o); quality += q; joined.append(j)

    tables = {"validite": pd.DataFrame(validity), "raisons_echec": pd.concat(reasons),
              "changement_numerique": pd.DataFrame(numeric), "ordre_candidats": pd.DataFrame(order),
              "qualite_par_legende": pd.DataFrame(quality), "table_jointe": pd.concat(joined)}
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    (out / "manifest.json").write_text(json.dumps({
        "emissions": str(path), "parser": args.parser, "n_boot": args.n_boot, "seed": BOOT_SEED,
        "protocol_sha256": df["protocol_sha256"].iloc[0], "reponses": len(df),
    }, indent=2, ensure_ascii=False) + "\n")

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        for name in ("validite", "raisons_echec", "changement_numerique", "ordre_candidats", "qualite_par_legende"):
            print(f"\n== {name}\n{tables[name].round(4).to_string(index=False)}")
    print(f"\nRésultats écrits dans {out}")


if __name__ == "__main__":
    main()
