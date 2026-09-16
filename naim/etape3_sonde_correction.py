"""Étape 3 (things2do.md §2) : sonde linéaire de correction par candidat.

Lignes (question_id, candidat, h(q,a), correct) :
- h(q,a) : hidden_states.pt d'origine (h_pos / h_neg), passe « Question: q / Answer: a »,
  premier token de la réponse, moyenne des 8 dernières couches (couche et pooling figés) ;
- correct : 1 pour la réponse correcte mc1 de TruthfulQA, 0 pour la première réponse incorrecte
  (labels curés, indépendants de la vraisemblance du modèle).
Le lecteur note chaque candidat séparément : il ne reçoit jamais la paire ni la position.

Splits : les trois splits disjoints par question (seeds 0–2) déjà utilisés, relus depuis
cached_results/state_readout/cross_fitted_predictions.csv (326 questions test, 491 train).
Les deux candidats d'une question restent du même côté. Standardisation et C choisis sur
l'entraînement seulement (validation croisée groupée par question, AUROC).

Comparaisons sur exactement les mêmes candidats test : confiance décodée forward, reversed, et
une baseline de longueur (log nombre de tokens, même lecteur). Métriques : AUROC candidat et
ordre intra-question (égalité = 1/2), intervalles bootstrap par question pour sonde − rapport.
Les seeds sont rapportés séparément. Analyse secondaire, après ajustement : questions où le
rapport classe l'incorrect au moins aussi haut sous les deux légendes.

Contrôle « question seule » : il donne le même score aux deux candidats, donc 1/2 d'ordre par
construction ; ce n'est pas un résultat et il n'est pas calculé.

Usage : ~/Documents/hf/bin/python naim/etape3_sonde_correction.py
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from commun import (BOOT_SEED, MODELS, N_BOOT, RESULTS, SPLITS_CSV, as_array, ci, example_ids,
                    input_hashes, provenance, step1, write_json)

OUT = RESULTS / "etape3"
EMISSIONS = RESULTS / "etape4" / "complet" / "emissions.jsonl"
SEEDS = (0, 1, 2)
C_GRID = [float(c) for c in np.logspace(-4, 0, 5)]
INNER_FOLDS = 5
METHODS = ("sonde_etat", "longueur", "rapport_forward", "rapport_reversed")
COMPARISONS = ("rapport_forward", "rapport_reversed", "longueur")


def load_cell(model: str) -> dict:
    hs, lp = step1(model, "hidden_states"), step1(model, "logprobs")
    fw, rv = provenance(model, "forward"), provenance(model, "reversed")
    ids = example_ids(hs)
    for other in (lp, fw, rv):
        if not np.array_equal(ids, example_ids(other)):
            raise ValueError(f"{model} : identifiants différents entre les caches")
    return {
        "ids": ids,
        "h_pos": as_array(hs["h_pos"], np.float32), "h_neg": as_array(hs["h_neg"], np.float32),
        "len_pos": np.log1p(as_array(lp["ntok_pos"]))[:, None],
        "len_neg": np.log1p(as_array(lp["ntok_neg"]))[:, None],
        "fwd_pos": as_array(fw["V_pos"]), "fwd_neg": as_array(fw["V_neg"]),
        "rev_pos": as_array(rv["V_pos"]), "rev_neg": as_array(rv["V_neg"]),
        "meta": {k: hs["meta"].get(k) for k in ("model_id", "layer", "n_probe_layers", "pool_method")},
    }


def load_splits(n: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    df = pd.read_csv(SPLITS_CSV)
    df = df[(df.dataset == "truthfulqa") & (df.format == "letter11")]
    splits = {}
    for seed in SEEDS:
        test = np.sort(df.loc[df.seed == seed, "item_index"].unique())
        splits[seed] = (np.setdiff1d(np.arange(n), test), test)
    return splits


def candidate_rows(pos: np.ndarray, neg: np.ndarray, questions: np.ndarray):
    X = np.concatenate([pos[questions], neg[questions]])
    y = np.r_[np.ones(len(questions)), np.zeros(len(questions))]
    return X, y, np.r_[questions, questions]


def fit_reader(X, y, groups):
    search = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
        {"logisticregression__C": C_GRID}, cv=GroupKFold(INNER_FOLDS), scoring="roc_auc", n_jobs=-1)
    search.fit(X, y, groups=groups)
    return search.best_estimator_, search.best_params_["logisticregression__C"]


def auroc(sp, sn) -> float:
    return float(roc_auc_score(np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn]))


def ordering(sp, sn) -> float:
    return float(np.mean((sp > sn) + 0.5 * (sp == sn)))


def load_emissions(model: str, ids: np.ndarray):
    """Confiances réellement écrites (parser strict), alignées sur l'ordre des caches ; NaN si invalide."""
    if not EMISSIONS.exists():
        return None
    df = pd.read_json(EMISSIONS, lines=True)
    df = df[df.model_key == model]
    if df.empty:
        return None
    place = {q: i for i, q in enumerate(ids)}
    out = {}
    for legend in ("forward", "reversed"):
        sides = {}
        for side in ("pos", "neg"):
            values = np.full(len(ids), np.nan)
            g = df[(df.legend == legend) & (df.candidate_side == side) & df.valid]
            values[[place[q] for q in g.question_id]] = g.decoded_value.to_numpy(dtype=float)
            sides[side] = values
        out[legend] = sides
    return out


def emission_rows(model, seed, test, probe_scores, emissions, n_boot):
    """things2do §2 ligne 79 : comparaison séparée sonde / confiance écrite, masque de validité inclus.

    On garde les questions dont les DEUX réponses ont une lettre lisible sous cette légende, pour que
    la sonde et l'émission soient jugées sur exactement les mêmes candidats. Une réponse invalide est
    exclue, jamais comptée comme une confiance de 0.
    """
    sp, sn = probe_scores
    rows = []
    for legend in ("forward", "reversed"):
        vp, vn = emissions[legend]["pos"][test], emissions[legend]["neg"][test]
        keep = np.isfinite(vp) & np.isfinite(vn)
        n = int(keep.sum())
        row = {"model": model, "seed": seed, "legende": legend, "questions_test": len(test),
               "questions_emission_valide": n, "couverture": float(keep.mean())}
        if n >= 10:
            e_p, e_n, p_p, p_n = vp[keep], vn[keep], sp[keep], sn[keep]
            idx = np.random.default_rng(BOOT_SEED + 200 + seed).integers(0, n, (n_boot, n))
            boot = np.array([[auroc(e_p[i], e_n[i]), ordering(e_p[i], e_n[i]),
                              auroc(p_p[i], p_n[i]), ordering(p_p[i], p_n[i])] for i in idx])
            a_lo, a_hi = ci(boot[:, 2] - boot[:, 0])
            o_lo, o_hi = ci(boot[:, 3] - boot[:, 1])
            row.update(auroc_emission=auroc(e_p, e_n), ordre_emission=ordering(e_p, e_n),
                       auroc_sonde_meme_sous_ensemble=auroc(p_p, p_n),
                       ordre_sonde_meme_sous_ensemble=ordering(p_p, p_n),
                       diff_auroc=auroc(p_p, p_n) - auroc(e_p, e_n),
                       diff_auroc_ic_bas=a_lo, diff_auroc_ic_haut=a_hi,
                       diff_ordre=ordering(p_p, p_n) - ordering(e_p, e_n),
                       diff_ordre_ic_bas=o_lo, diff_ordre_ic_haut=o_hi)
        rows.append(row)
    return rows


def run_seed(model, c, seed, train, test, n_boot):
    X, y, g = candidate_rows(c["h_pos"], c["h_neg"], train)
    probe, c_probe = fit_reader(X, y, g)
    L, yl, gl = candidate_rows(c["len_pos"], c["len_neg"], train)
    length, c_len = fit_reader(L, yl, gl)
    scores = {
        "sonde_etat": (probe.decision_function(c["h_pos"][test]), probe.decision_function(c["h_neg"][test])),
        "longueur": (length.decision_function(c["len_pos"][test]), length.decision_function(c["len_neg"][test])),
        "rapport_forward": (c["fwd_pos"][test], c["fwd_neg"][test]),
        "rapport_reversed": (c["rev_pos"][test], c["rev_neg"][test]),
    }
    fit = {"model": model, "seed": seed, "questions_train": len(train), "questions_test": len(test),
           "candidats_train": 2 * len(train), "candidats_test": 2 * len(test),
           "C_sonde": c_probe, "C_longueur": c_len}

    rng = np.random.default_rng(BOOT_SEED + seed)
    idx = rng.integers(0, len(test), (n_boot, len(test)))
    boot = {m: np.array([[auroc(sp[i], sn[i]), ordering(sp[i], sn[i])] for i in idx])
            for m, (sp, sn) in scores.items()}
    point = {m: np.array([auroc(sp, sn), ordering(sp, sn)]) for m, (sp, sn) in scores.items()}

    metrics, diffs = [], []
    for m in METHODS:
        (a_lo, a_hi), (o_lo, o_hi) = ci(boot[m][:, 0]), ci(boot[m][:, 1])
        metrics.append({"model": model, "seed": seed, "methode": m, "questions_test": len(test),
                        "auroc": point[m][0], "auroc_ic_bas": a_lo, "auroc_ic_haut": a_hi,
                        "ordre": point[m][1], "ordre_ic_bas": o_lo, "ordre_ic_haut": o_hi})
    for other in COMPARISONS:
        for j, metric in enumerate(("auroc", "ordre")):
            lo, hi = ci(boot["sonde_etat"][:, j] - boot[other][:, j])
            diffs.append({"model": model, "seed": seed, "comparaison": f"sonde_etat - {other}",
                          "metrique": metric, "difference": point["sonde_etat"][j] - point[other][j],
                          "ic_bas": lo, "ic_haut": hi})

    failed = (c["fwd_pos"][test] <= c["fwd_neg"][test]) & (c["rev_pos"][test] <= c["rev_neg"][test])
    k = np.flatnonzero(failed)
    subset = []
    for m in ("sonde_etat", "longueur"):
        sp, sn = scores[m]
        row = {"model": model, "seed": seed, "methode": m, "questions_sous_ensemble": len(k),
               "ordre": float("nan"), "ordre_ic_bas": float("nan"), "ordre_ic_haut": float("nan")}
        if len(k):
            sub_idx = np.random.default_rng(BOOT_SEED + 100 + seed).integers(0, len(k), (n_boot, len(k)))
            lo, hi = ci([ordering(sp[k][i], sn[k][i]) for i in sub_idx])
            row.update(ordre=ordering(sp[k], sn[k]), ordre_ic_bas=lo, ordre_ic_haut=hi)
        subset.append(row)

    preds = []
    for t, q in enumerate(test):
        for side, label in (("pos", 1), ("neg", 0)):
            s = 0 if side == "pos" else 1
            preds.append({"model": model, "seed": seed, "item_index": int(q), "question_id": c["ids"][q],
                          "candidat": side, "correct": label,
                          **{m: float(scores[m][s][t]) for m in METHODS},
                          "echec_rapport_deux_legendes": bool(failed[t])})
    return fit, metrics, diffs, subset, preds, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Étape 3 : sonde de correction par candidat.")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args()

    fits, metrics, diffs, subsets, preds, emis_rows, metas = [], [], [], [], [], [], {}
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        c = load_cell(model)
        emissions = load_emissions(model, c["ids"])
        metas[model] = c["meta"]
        for seed, (train, test) in load_splits(len(c["ids"])).items():
            print(f"{model} seed {seed} : {len(train)} questions train, {len(test)} test")
            out = run_seed(model, c, seed, train, test, args.n_boot)
            fits.append(out[0]); metrics += out[1]; diffs += out[2]; subsets += out[3]; preds += out[4]
            if emissions is not None:
                emis_rows += emission_rows(model, seed, test, out[5]["sonde_etat"], emissions, args.n_boot)

    OUT.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(metrics)
    pd.DataFrame(fits).to_csv(OUT / "ajustements.csv", index=False)
    metrics_df.to_csv(OUT / "metriques_par_seed.csv", index=False)
    pd.DataFrame(diffs).to_csv(OUT / "differences_par_seed.csv", index=False)
    pd.DataFrame(subsets).to_csv(OUT / "sous_ensemble_echec_rapport.csv", index=False)
    if emis_rows:
        pd.DataFrame(emis_rows).to_csv(OUT / "emission_vs_sonde.csv", index=False)
    pd.DataFrame(preds).to_csv(OUT / "predictions_test.csv", index=False)
    spread = (metrics_df.groupby(["model", "methode"])[["auroc", "ordre"]]
              .agg(["mean", "min", "max"]).round(4))
    spread.to_csv(OUT / "variation_entre_seeds.csv")
    write_json(OUT / "manifest.json", {
        "script": "naim/etape3_sonde_correction.py",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "cible": "correction curée du candidat (mc1 TruthfulQA), pas la préférence teacher-forced",
        "entree_lecteur": "h(q,a) d'un seul candidat ; ni paire ordonnée ni position",
        "etats": metas,
        "splits": "cached_results/state_readout/cross_fitted_predictions.csv (truthfulqa, letter11, seeds 0-2)",
        "lecteur": {"modele": "StandardScaler + LogisticRegression L2", "grille_C": C_GRID,
                    "selection": f"GroupKFold({INNER_FOLDS}) par question sur le train, AUROC"},
        "baseline_longueur": "log1p(nombre de tokens du candidat), même lecteur",
        "population": "817 paires = population du rapport ; test = 326 questions par seed",
        "bootstrap": {"unite": "question test (deux candidats ensemble)", "n": args.n_boot,
                      "seed": f"{BOOT_SEED} + seed"},
        "controle_question_seule": "ordre = 1/2 par construction, non calculé",
        "sous_ensemble_secondaire": "V_pos <= V_neg sous forward ET reversed ; évalué après ajustement",
        "emissions": {
            "source": str(EMISSIONS), "parser": "strict",
            "regle": "questions dont les deux réponses ont une lettre lisible sous la légende ; "
                     "une réponse invalide est exclue, jamais comptée comme confiance 0",
            "comparaison": "sonde recalculée sur ce même sous-ensemble, différence bootstrapée par question",
        },
        "entrees_sha256": input_hashes(),
    })

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(spread)
        print(pd.DataFrame(diffs).round(4))
        print(pd.DataFrame(subsets).round(4))
        if emis_rows:
            print(pd.DataFrame(emis_rows).round(4))
    print(f"Résultats écrits dans {OUT}")


if __name__ == "__main__":
    main()
