"""alphabet_flip_analysis.py — decomposing the Llama/TruthfulQA sign flip.

OBSERVED: report-channel dAURC on llama3_8b/truthfulqa is -0.038 +/- 0.023 under
the letter alphabet and +0.079 +/- 0.023 under the numeric one, same items.
Llama groups digit runs, so its numeric readout is uncontaminated and both
alphabets are validly readable there. The flip is +0.117, which is the size of
the ANSWER effect on the state channel (TruthfulQA +0.104 to +0.154, PopQA
+0.071 to +0.092). A formatting choice moves the report score as much as the
answer moves the state score.

PRECONDITION, ANSWERED. ``cache_inventory`` is the check. Numeric Vdist caches
(``fine_conf.pt``, full 101-level distributions) exist for llama3_8b/truthfulqa
and llama3_8b/pavlick_nli and for no other cell. So (1), (3), (4), (5) and (6)
are free on both llama cells; (2) is free for those two and needs a fresh
elicitation pass for the four mistral/qwen cells.

NAMING CORRECTION. The cached numeric run is ``digits101-v1``: levels 0..100 in
steps of 1, no legend, prompt "Answer with a number from 0 to 100". It is not a
``numeric11``. The contrast that produced the flip therefore moves three things
at once:

    glyph class     letter A..L            vs digit strings
    legend          in-context A=0% ...    vs none, the token IS the value
    resolution      K = 11                 vs K = 101

Any of the three could carry +0.117. (4) separates resolution from the other two
by re-reading the SAME numeric distributions restricted to the 11 letter-grid
levels; nothing available offline separates glyph from legend, which is what
NEXT_EXPERIMENT is for.

FOLDS. Checks that never touch the outcome (1, 2, 4-descriptive) are computed on
all items -- there is no fold to spend. Checks that score against y (3, 5, 6) are
reported on both folds with s_q always fitted out-of-fold: fold="test" fits s_q
on train and reproduces the published number, fold="train" is its mirror. The
test-fold rows re-describe an already-observed flip; they open no new confirmatory
count.

DISCIPLINE: this is a decomposition of an observed flip, so it is exploratory by
construction. Nothing here enters a confirmatory count.

Examples
--------
    python verbalized_conformal_confidence/alphabet_flip_analysis.py
    python verbalized_conformal_confidence/alphabet_flip_analysis.py \
        --checks cache_inventory which_side_flips paired_seed_test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
for _p in (str(ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiments.ladder import _split  # noqa: E402
from scripts.step_token_contribution import (  # noqa: E402
    aurc,
    build_targets,
    fit_predict,
    load_cell,
)
from utils.paths import logprobs_path  # noqa: E402

from report_channel_robustness import (  # noqa: E402
    _adequacy,
    _entropy_norm,
    _normalise_rows,
)


LETTER = "letter11"
NUMERIC = "digits101"
ALPHABETS = (LETTER, NUMERIC)
CELLS = [(m, d) for m in ("llama3_8b", "mistral_7b", "qwen2_5_7b")
         for d in ("truthfulqa", "pavlick_nli")]
FLIP_CELL = ("llama3_8b", "truthfulqa")
READABLE_CELLS = [("llama3_8b", "truthfulqa"), ("llama3_8b", "pavlick_nli")]
SEEDS = (0, 1, 2)
FOLDS = ("test", "train")
N_BOOT = 2000
OUT = _HERE / "cached_results" / "alphabet_flip"

# The letter alphabet's support, as a fraction. ABCDEFGHJKL = 0..100 step 10.
LETTER_GRID = np.arange(0, 101, 10)


def _cell_name(cell) -> str:
    return f"{cell[0]}/{cell[1]}"


def numeric_path(cell) -> Path:
    return logprobs_path(*cell).with_name("fine_conf.pt")


def load_alphabets(cell) -> tuple[dict, np.ndarray, dict[str, dict]]:
    """Return (state cache, target, {alphabet: confidence blob}) for one cell."""
    cell = tuple(cell)
    d, err = load_cell(*cell)
    if err:
        raise FileNotFoundError(err["error"])
    letter = torch.load(logprobs_path(*cell), map_location="cpu", weights_only=False)
    path = numeric_path(cell)
    if not path.exists():
        raise FileNotFoundError(
            f"no numeric-alphabet cache for {_cell_name(cell)} at {path}; this "
            f"cell needs a re-elicitation pass (see cache_inventory)")
    numeric = torch.load(path, map_location="cpu", weights_only=False)
    if not np.array_equal(np.asarray(letter["example_ids"]),
                          np.asarray(numeric["example_ids"])):
        raise ValueError(f"item IDs do not align for {_cell_name(cell)}")
    y = build_targets(d)["mean"][0]
    return d, y, {LETTER: letter, NUMERIC: numeric}


def side_scores(blob: dict) -> dict[str, np.ndarray]:
    """The four report scores that (3) separates. ``minus_V_neg`` exists because
    AURC is direction-sensitive and V- orders confidence the other way round."""
    vp = np.asarray(blob["V_pos"], dtype=np.float64)
    vn = np.asarray(blob["V_neg"], dtype=np.float64)
    return {"V_pos": vp, "V_neg": vn, "minus_V_neg": -vn, "gap": vp - vn}


def _fold_scores(d: dict, y: np.ndarray, seed: int, fold: str):
    """Evaluation index and an s_q always fitted on the OTHER fold."""
    tr, te = _split(len(y), seed)
    fit, ev = (tr, te) if fold == "test" else (te, tr)
    sq = fit_predict(d["h_q"][fit], y[fit], d["h_q"][ev], seed)
    return np.flatnonzero(ev), sq


def _boot_index(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def _ci(samples: np.ndarray) -> tuple[float, float]:
    s = np.asarray(samples, dtype=float)
    s = s[np.isfinite(s)]
    if not len(s):
        return float("nan"), float("nan")
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import rankdata
    return _pearson(rankdata(a), rankdata(b))


# ───────────────────────────────────────────────────── precondition


def cache_inventory(cells: Sequence = CELLS) -> pd.DataFrame:
    """(0) PRECONDITION. Which cells hold a numeric Vdist, not just its scalars.

    Only cells with a full ``Vdist`` under BOTH alphabets can be re-read; a cell
    for which only V_pos/V_neg survived cannot answer (2) or (4).
    """
    rows = []
    for cell in cells:
        cell = tuple(cell)
        row = {"model": cell[0], "dataset": cell[1],
               "letter_cache": False, "numeric_cache": False,
               "letter_K": np.nan, "numeric_K": np.nan, "n_items": np.nan,
               "numeric_has_Vdist": False, "ids_aligned": False,
               "has_alt_form": False, "conf_scheme": "",
               "status": "requires_re_elicitation"}
        lp = logprobs_path(*cell)
        if lp.exists():
            letter = torch.load(lp, map_location="cpu", weights_only=False)
            row["letter_cache"] = "Vdist_pos" in letter
            row["letter_K"] = int(len(letter.get("conf_values", [])))
            row["n_items"] = int(len(letter["example_ids"]))
        path = numeric_path(cell)
        if path.exists():
            numeric = torch.load(path, map_location="cpu", weights_only=False)
            row["numeric_cache"] = True
            row["numeric_has_Vdist"] = "Vdist_pos" in numeric
            row["numeric_K"] = int(len(numeric.get("conf_values", [])))
            row["has_alt_form"] = "Valt_pos" in numeric
            meta = numeric.get("meta", {})
            meta = meta.item() if hasattr(meta, "item") else meta
            row["conf_scheme"] = str(meta.get("conf_scheme", ""))
            if lp.exists():
                row["ids_aligned"] = bool(np.array_equal(
                    np.asarray(letter["example_ids"]),
                    np.asarray(numeric["example_ids"])))
            row["status"] = ("readable_both_alphabets"
                             if row["numeric_has_Vdist"] and row["ids_aligned"]
                             else "numeric_scalars_only")
        rows.append(row)
    return pd.DataFrame(rows)


def shuffled_legend_status() -> dict:
    """The control that has been open since August, and why it is still open.

    Permuting the legend (A=70%, B=10%, ...) while holding the letters fixed
    separates "V tracks the glyph" from "V tracks the value the glyph denotes".
    It cannot be computed from any cached artifact: the legend lives inside the
    prompt, so a permutation is a different forward pass. Recorded here so the
    manifest does not silently omit it.
    """
    return {
        "available_offline": False,
        "reason": ("the legend is part of CONF_PROMPT, so a permuted legend is a "
                   "new elicitation; no cached blob holds it"),
        "requires": ("a CONF_LEVELS permutation in extraction/modal_app.py, then "
                     "one confidence pass over the same items"),
        "decides": ("if V is unchanged under permutation, V tracks the letter "
                    "and not the value it denotes"),
    }


# ─────────────────────────────────────────────── (1) answer invariance


def answer_invariance_by_alphabet(cell=FLIP_CELL, n_boot: int = N_BOOT,
                                  boot_seed: int = 0) -> pd.DataFrame:
    """(1) Does the legend indirection make the report answer-invariant?

    rho(V_pos, V_neg) under each alphabet on the same items, with an item
    bootstrap CI on the DIFFERENCE between the two correlations (same resampled
    items for both alphabets, so the pairing is preserved).

    HYPOTHESIS: under letters the two sides are tightly coupled -- the report
    barely distinguishes which candidate it was shown -- and under numeric they
    decouple. That is the same signature as llama/pavlick under letters
    (rho = .867 against alternate-form reliability of .147/.171).

    IF CONFIRMED: the mechanism is the legend. A letter alphabet requires
    resolving an in-context mapping (A -> 0%, ... , L -> 100%) that a numeric
    alphabet does not, and the cost of that indirection is the answer signal.
    That is a statement about elicitation design, reportable independently of
    which alphabet is "right".

    IF NOT: the coupling is unchanged and the flip comes from somewhere else.
    Go to (4) -- resolution -- before reaching for a semantic explanation.

    The comparison that makes rho(V+, V-) interpretable is alternate-form
    reliability on the SAME items: coupling only means answer-invariance if the
    instrument is reliable enough for the two sides to have been able to differ.
    Both alphabets carry a Valt pass, so both ratios are computed here.

    The shuffled-legend control is not computable offline; see
    ``shuffled_legend_status``.
    """
    cell = tuple(cell)
    _, y, blobs = load_alphabets(cell)
    n = len(y)
    boot = _boot_index(n, n_boot, boot_seed)

    rows, coupling_boot = [], {}
    for alphabet, blob in blobs.items():
        s = side_scores(blob)
        vp, vn = s["V_pos"], s["V_neg"]
        coupling_boot[alphabet] = np.array(
            [_pearson(vp[ix], vn[ix]) for ix in boot])
        lo, hi = _ci(coupling_boot[alphabet])
        row = {"model": cell[0], "dataset": cell[1], "alphabet": alphabet,
               "quantity": "rho(V_pos, V_neg)", "n": n,
               "pearson": _pearson(vp, vn), "spearman": _spearman(vp, vn),
               "ci_lo": lo, "ci_hi": hi}
        # Alternate-form reliability on the same items, per side.
        if "Valt_pos" in blob:
            j = np.asarray(blob["alt_index"], dtype=int)
            for side, v, valt in (("pos", vp, blob["Valt_pos"]),
                                  ("neg", vn, blob["Valt_neg"])):
                valt = np.asarray(valt, dtype=np.float64)
                row[f"alt_rho_{side}"] = _pearson(v[j], valt)
                row[f"alt_abs_dV_{side}"] = float(np.abs(v[j] - valt).mean())
            row["n_alt"] = int(len(j))
            # >1 means the two sides agree with each other more than either
            # agrees with its own reworded self: the report is not resolving
            # which candidate it was shown.
            row["coupling_over_reliability"] = (
                row["pearson"] / row["alt_rho_pos"]
                if row.get("alt_rho_pos") else float("nan"))
        rows.append(row)

    diff = coupling_boot[NUMERIC] - coupling_boot[LETTER]
    lo, hi = _ci(diff)
    point = (_pearson(*[side_scores(blobs[NUMERIC])[k] for k in ("V_pos", "V_neg")])
             - _pearson(*[side_scores(blobs[LETTER])[k] for k in ("V_pos", "V_neg")]))
    rows.append({
        "model": cell[0], "dataset": cell[1], "alphabet": "numeric - letter",
        "quantity": "d rho(V_pos, V_neg)", "n": n, "pearson": point,
        "spearman": float("nan"), "ci_lo": lo, "ci_hi": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "decoupling_confirmed": bool(hi < 0),
    })
    return pd.DataFrame(rows)


# ────────────────────────────────────────────── (2) adequacy under numeric


def adequacy_under_numeric(cells: Sequence = CELLS) -> pd.DataFrame:
    """(2) Does the numeric alphabet rescue the excluded cell?

    llama/pavlick fails the adequacy gate under letters: 1.07 effective bins,
    98.8% of mass in one, 2.4% of item pairs separated. Run the SAME fixed gate
    -- 5pp bins anchored at zero, exponentiated bin entropy, threshold 2 -- on
    the numeric gap for every cell.

    WHY THIS MATTERS MORE THAN IT LOOKS: if llama/pavlick passes under numeric,
    the cell was excluded for a property of the elicitation format rather than of
    the model. The exclusion stays correct as applied (the gate is computed on
    the score actually used), but Sec. 5's framing changes: the sentence becomes
    "under the letter alphabet this cell yields no usable ordering", not "this
    model's report is degenerate on this dataset". Those are different claims and
    only the first is supported.

    The gate is imported from report_channel_robustness, not re-derived.
    """
    rows = []
    for cell in cells:
        cell = tuple(cell)
        try:
            _, y, blobs = load_alphabets(cell)
        except FileNotFoundError as exc:
            rows.append({"model": cell[0], "dataset": cell[1],
                         "alphabet": NUMERIC, "K": np.nan, "n": np.nan,
                         "effective_bins": np.nan, "max_bin_mass": np.nan,
                         "measurable": False, "pair_separation": np.nan,
                         "status": "requires_re_elicitation", "note": str(exc)})
            continue
        for alphabet, blob in blobs.items():
            gap = side_scores(blob)["gap"]
            gate = _adequacy(gap)
            rows.append({
                "model": cell[0], "dataset": cell[1], "alphabet": alphabet,
                "K": int(len(blob["conf_values"])), "n": int(len(gap)),
                "pair_separation": _pair_separation(gap),
                "status": "readable", "note": "", **gate,
            })
    frame = pd.DataFrame(rows)
    if {"model", "dataset"}.issubset(frame.columns):
        wide = frame.pivot_table(index=["model", "dataset"], columns="alphabet",
                                 values="measurable", aggfunc="first")
        if LETTER in wide and NUMERIC in wide:
            let = wide[LETTER].to_numpy(dtype=object) == True  # noqa: E712
            num = wide[NUMERIC].to_numpy(dtype=object) == True  # noqa: E712
            frame.attrs["rescued_cells"] = [
                "/".join(c) for c in wide.index[~let & num]]
    return frame


def _pair_separation(score: np.ndarray, width: float = 0.05) -> float:
    """Fraction of item pairs the score separates AT THE GATE'S RESOLUTION.

    Binned at the same 5pp grid the adequacy gate uses, not at machine epsilon:
    a soft V is almost never exactly tied, so an exact-tie count would report
    ~1.00 for a score with 98.8% of its mass in one bin. This is the definition
    that reproduces the 2.4% quoted for llama/pavlick under letters.
    """
    s = np.asarray(score, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < 2:
        return float("nan")
    _, counts = np.unique(np.floor(s / width + 0.5) * width, return_counts=True)
    total = len(s) * (len(s) - 1) / 2
    tied = float(np.sum(counts * (counts - 1) / 2))
    return float((total - tied) / total)


# ───────────────────────────────────────────────── scoring machinery


def _sq_cache(d: dict, y: np.ndarray, folds=FOLDS, seeds=SEEDS) -> dict:
    """s_q for every (fold, seed), always fitted on the complementary fold."""
    return {(fold, seed): _fold_scores(d, y, seed, fold)
            for fold in folds for seed in seeds}


def _aurc_rows(cell, y, named_scores, cache) -> list[dict]:
    """named_scores: iterable of (alphabet, readout, score). s_q is re-evaluated
    on each score's own finite mask so the two AURCs always share their items."""
    rows = []
    for (fold, seed), (ix, sq) in cache.items():
        risk = 1.0 - y[ix]
        for alphabet, readout, score in named_scores:
            s = np.asarray(score, dtype=float)[ix]
            keep = np.isfinite(s)
            if not keep.any():
                continue
            a_q = aurc(sq[keep], risk[keep])
            a_r = aurc(s[keep], risk[keep])
            rows.append({
                "model": cell[0], "dataset": cell[1], "fold": fold, "seed": seed,
                "alphabet": alphabet, "readout": readout,
                "n_eval": int(keep.sum()), "n_dropped": int((~keep).sum()),
                "AURC_sq": a_q, "AURC_report": a_r, "delta": a_q - a_r,
            })
    return rows


# ────────────────────────────────────────────────── (3) which side flips


def which_side_flips(cells: Sequence = READABLE_CELLS) -> pd.DataFrame:
    """(3) Is the flip in V+, in V-, or only in their difference?

    dAURC against s_q for four scores x two alphabets:
        V_pos alone, V_neg alone, -V_neg, V_pos - V_neg.

    ``minus_V_neg`` is carried because AURC ranks by descending score and V-
    orders confidence the other way; reporting V- alone without it invites a
    sign misreading rather than a finding.

    READING:
      - V_pos flips too -> the alphabet effect is about verbalized confidence
        generally. Larger claim, and it reaches the deployable object, so it must
        appear in Sec. 1 rather than in a robustness paragraph.
      - only the difference flips -> the effect is specific to the paired
        construction. Narrower and more awkward: it would mean the format
        interacts with the contrast rather than with the report, and the paired
        design would need defending on exactly the axis it was introduced to fix.
      - V_neg alone carries it -> the dispreferred side is where the format bites.
        Check this against (1): a V_neg that only responds under numeric is what
        decoupling looks like from the other direction.
    """
    rows = []
    for cell in cells:
        cell = tuple(cell)
        d, y, blobs = load_alphabets(cell)
        cache = _sq_cache(d, y)
        named = [(alphabet, readout, score)
                 for alphabet, blob in blobs.items()
                 for readout, score in side_scores(blob).items()]
        rows += _aurc_rows(cell, y, named, cache)
    return pd.DataFrame(rows)


def flip_by_readout(frame: pd.DataFrame) -> pd.DataFrame:
    """A_letter - A_numeric per readout: the flip itself, with s_q cancelled."""
    keys = ["model", "dataset", "fold", "seed", "readout"]
    wide = frame.pivot_table(index=keys, columns="alphabet",
                             values="AURC_report").reset_index()
    wide["flip"] = wide[LETTER] - wide[NUMERIC]
    out = (wide.groupby(["model", "dataset", "fold", "readout"], as_index=False)
                .agg(flip_mean=("flip", "mean"), flip_sd=("flip", "std"),
                     AURC_letter=(LETTER, "mean"), AURC_numeric=(NUMERIC, "mean")))
    return out.sort_values(["model", "dataset", "fold", "flip_mean"],
                           ascending=[True, True, True, False])


# ──────────────────────────────────────── (4) resolution vs content


def restrict_to_letter_grid(blob: dict) -> dict[str, np.ndarray]:
    """Re-read the numeric distribution on the letter alphabet's own support.

    The letter readout is a restrict-and-renormalize over 11 levels. Applying
    exactly that restriction to the 101-level distribution holds resolution
    fixed at K = 11 while keeping the numeric elicitation, so whatever survives
    is not a resolution effect. Rows with no mass on the grid are returned as
    NaN and counted rather than renormalized from nothing.
    """
    values = np.asarray(blob["conf_values"], dtype=float)
    keep = np.isin(values.astype(int), LETTER_GRID)
    grid = values[keep] / 100.0
    out = {}
    for side in ("pos", "neg"):
        p = np.clip(np.asarray(blob[f"Vdist_{side}"], dtype=np.float64), 0.0, None)[:, keep]
        total = p.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            q = np.where(total > 0, p / np.where(total > 0, total, 1.0), np.nan)
        out[f"V_{side}"] = q @ grid
        out[f"grid_mass_{side}"] = total.ravel()
    out["n_zero_grid_mass"] = int(np.sum(~np.isfinite(out["V_pos"])
                                         | ~np.isfinite(out["V_neg"])))
    return out


def resolution_vs_content(cells: Sequence = READABLE_CELLS) -> pd.DataFrame:
    """(4) Is the flip a measurement-resolution effect or a different judgement?

    Per alphabet, on the gap: effective bins, across-item SD, distinct values,
    and H_norm of the per-item Vdist. Then the control the K = 11 vs K = 101
    confound demands -- the numeric distribution re-read on the letter alphabet's
    own 11-level support, and (secondarily) the numeric scalar rounded onto that
    grid -- scored through the same dAURC machinery.

    TWO OUTCOMES, very different consequences:
      - numeric has materially MORE effective bins / higher across-item SD, AND
        the advantage disappears once the numeric read is restricted to 11
        levels -> the letter readout was compressing the score toward the
        midpoint and the letter result is a resolution failure. This is the
        benign reading and it makes the single-digit control the obvious fix.
      - similar support under both, or the advantage SURVIVES restriction to 11
        levels -> the two alphabets elicit different judgements on the same
        items. That is a much stranger result, it is not fixed by a better
        alphabet, and it would need its own experiment rather than a footnote.
        Do not report this outcome quickly.
    """
    rows = []
    for cell in cells:
        cell = tuple(cell)
        d, y, blobs = load_alphabets(cell)
        cache = _sq_cache(d, y)

        restricted = restrict_to_letter_grid(blobs[NUMERIC])
        num = side_scores(blobs[NUMERIC])
        variants = {
            LETTER: side_scores(blobs[LETTER])["gap"],
            NUMERIC: num["gap"],
            "digits101_on_letter_support": restricted["V_pos"] - restricted["V_neg"],
            "digits101_rounded_to_grid": (np.round(num["V_pos"] * 10) / 10
                                          - np.round(num["V_neg"] * 10) / 10),
        }
        deltas = pd.DataFrame(_aurc_rows(
            cell, y, [(name, "gap", s) for name, s in variants.items()], cache))
        mean_delta = (deltas.groupby(["alphabet", "fold"])["delta"]
                            .agg(["mean", "std"]).unstack())

        for name, gap in variants.items():
            g = np.asarray(gap, dtype=float)
            finite = g[np.isfinite(g)]
            row = {
                "model": cell[0], "dataset": cell[1], "variant": name,
                "n_finite": int(len(finite)),
                "sd_across_items": float(finite.std(ddof=1)),
                "iqr": float(np.subtract(*np.percentile(finite, [75, 25]))),
                "distinct_values": int(len(np.unique(np.round(finite, 9)))),
                **_adequacy(g),
                "pair_separation": _pair_separation(g),
            }
            for fold in FOLDS:
                row[f"delta_{fold}"] = float(mean_delta[("mean", fold)][name])
                row[f"delta_sd_{fold}"] = float(mean_delta[("std", fold)][name])
            if name in blobs:
                blob = blobs[name]
                for side in ("pos", "neg"):
                    h = _entropy_norm(blob[f"Vdist_{side}"])
                    row[f"H_norm_{side}"] = float(h.mean())
                    row[f"eff_support_{side}"] = float(np.exp(
                        h * np.log(blob[f"Vdist_{side}"].shape[1])).mean())
                    row[f"Vmass_{side}"] = float(np.asarray(blob[f"Vmass_{side}"]).mean())
                row["K"] = int(len(blob["conf_values"]))
            elif name == "digits101_on_letter_support":
                row["K"] = len(LETTER_GRID)
                row["n_zero_grid_mass"] = restricted["n_zero_grid_mass"]
                row["grid_mass_pos"] = float(np.mean(restricted["grid_mass_pos"]))
                row["grid_mass_neg"] = float(np.mean(restricted["grid_mass_neg"]))
            else:
                row["K"] = len(LETTER_GRID)
            rows.append(row)
    return pd.DataFrame(rows)


def resolution_verdict(frame: pd.DataFrame, margin: float = 0.02) -> dict:
    """Decide (4) from its own table, on the fold that reproduces the headline."""
    out = {}
    for (model, dataset), sub in frame.groupby(["model", "dataset"]):
        s = sub.set_index("variant")
        if not {LETTER, NUMERIC, "digits101_on_letter_support"}.issubset(s.index):
            continue
        full = s.loc[NUMERIC, "delta_test"] - s.loc[LETTER, "delta_test"]
        held = s.loc["digits101_on_letter_support", "delta_test"] - s.loc[LETTER, "delta_test"]
        retained = held / full if full else float("nan")
        out[f"{model}/{dataset}"] = {
            "flip_full": float(full),
            "flip_at_matched_resolution": float(held),
            "fraction_retained": float(retained),
            "effective_bins_letter": float(s.loc[LETTER, "effective_bins"]),
            "effective_bins_numeric": float(s.loc[NUMERIC, "effective_bins"]),
            "verdict": ("resolution_failure" if abs(held) < margin else
                        "different_judgement" if abs(retained) >= 0.5 else
                        "mixed_resolution_and_content"),
        }
    return out


# ─────────────────────────────────────────── (5) cross-alphabet agreement


def _aurc_influence(score: np.ndarray, risk: np.ndarray) -> np.ndarray:
    """Leave-one-out change in AURC. A ranking device for locating the flip,
    not a decomposition: these do not sum to the AURC difference."""
    n = len(score)
    base = aurc(score, risk)
    out = np.empty(n)
    keep = np.ones(n, bool)
    for i in range(n):
        keep[i] = False
        out[i] = base - aurc(score[keep], risk[keep])
        keep[i] = True
    return out


def _alt_form_reliability(blob: dict) -> dict[str, float]:
    """rho(V, V_reworded) per readout on the alt-form subset, or {} if absent.

    This is the ceiling on how well any two readings of the same construct can
    agree, and it is what makes a low cross-alphabet correlation interpretable.
    """
    if "Valt_pos" not in blob:
        return {}
    j = np.asarray(blob["alt_index"], dtype=int)
    s = side_scores(blob)
    ap = np.asarray(blob["Valt_pos"], dtype=np.float64)
    an = np.asarray(blob["Valt_neg"], dtype=np.float64)
    return {"V_pos": _pearson(s["V_pos"][j], ap),
            "V_neg": _pearson(s["V_neg"][j], an),
            "gap": _pearson(s["gap"][j], ap - an)}


def cross_alphabet_agreement(cells: Sequence = READABLE_CELLS,
                             n_boot: int = N_BOOT, boot_seed: int = 0,
                             n_null: int = 200) -> pd.DataFrame:
    """(5) Are the two readings the same measurement at all?

    rho(V_letter, V_numeric) per side on the same items, with the scatter data
    written out. Then locate where the AURC difference comes from: rank the test
    items by their leave-one-out contribution to the risk-coverage difference and
    check whether it is spread or carried by a tail. AURC is rank-based and a
    small number of items moving across the ranking can produce a large change.

    Top-k removal is compared against removing k items at random, because
    deleting the k most influential items shrinks any difference mechanically;
    only the excess over the random null is evidence of a tail.

    IF A TAIL CARRIES IT: show the items rather than the summary. A flip driven
    by twenty items is a different fact from a flip driven by the whole
    distribution, and only the second supports a general statement about
    elicitation format.
    """
    rows, per_item = [], []
    for cell in cells:
        cell = tuple(cell)
        d, y, blobs = load_alphabets(cell)
        ids = np.asarray(blobs[LETTER]["example_ids"])
        L, N = side_scores(blobs[LETTER]), side_scores(blobs[NUMERIC])
        boot = _boot_index(len(y), n_boot, boot_seed)

        rel = {alph: _alt_form_reliability(blob)
               for alph, blob in ((LETTER, blobs[LETTER]), (NUMERIC, blobs[NUMERIC]))}
        for readout in ("V_pos", "V_neg", "gap"):
            a, b = L[readout], N[readout]
            draws = np.array([_pearson(a[ix], b[ix]) for ix in boot])
            lo, hi = _ci(draws)
            r_l, r_n = rel[LETTER].get(readout), rel[NUMERIC].get(readout)
            r_ln = _pearson(a, b)
            # Disattenuation. Two unreliable instruments cannot correlate above
            # sqrt(rel_1 * rel_2) even when they measure the same thing, so a low
            # raw agreement is only evidence of a DIFFERENT judgement once the
            # ceiling is divided out. Values above 1 mean the correction has run
            # out of headroom and should be read as "at the ceiling", not as
            # agreement above unity.
            ceiling = (np.sqrt(r_l * r_n)
                       if r_l is not None and r_n is not None
                       and r_l > 0 and r_n > 0 else float("nan"))
            rows.append({
                "model": cell[0], "dataset": cell[1], "quantity": "agreement",
                "readout": readout, "fold": "all", "n": len(y),
                "pearson": r_ln, "spearman": _spearman(a, b),
                "ci_lo": lo, "ci_hi": hi,
                "mean_abs_diff": float(np.abs(a - b).mean()),
                "reliability_letter": r_l, "reliability_numeric": r_n,
                "agreement_ceiling": float(ceiling),
                "disattenuated": float(r_ln / ceiling) if ceiling == ceiling else float("nan"),
            })

        rng = np.random.default_rng(boot_seed)
        for seed in SEEDS:
            ix, _ = _fold_scores(d, y, seed, "test")
            risk = 1.0 - y[ix]
            sl, sn = L["gap"][ix], N["gap"][ix]
            flip = aurc(sl, risk) - aurc(sn, risk)
            infl = _aurc_influence(sl, risk) - _aurc_influence(sn, risk)
            order = np.argsort(-np.abs(infl))
            total_abs = float(np.abs(infl).sum())
            n_te = len(ix)
            per_item += [{
                "model": cell[0], "dataset": cell[1], "seed": seed,
                "example_id": ids[ix][i], "y": int(y[ix][i]),
                "V_pos_letter": L["V_pos"][ix][i], "V_pos_numeric": N["V_pos"][ix][i],
                "V_neg_letter": L["V_neg"][ix][i], "V_neg_numeric": N["V_neg"][ix][i],
                "gap_letter": sl[i], "gap_numeric": sn[i],
                "flip_influence": infl[i],
                "rank": int(np.flatnonzero(order == i)[0]),
            } for i in range(n_te)]

            row = {"model": cell[0], "dataset": cell[1], "quantity": "tail",
                   "readout": "gap", "fold": "test", "seed": seed, "n": n_te,
                   "flip_all_items": flip,
                   "top5pct_share_of_abs_influence": float(
                       np.abs(infl[order[:max(1, n_te // 20)]]).sum() / total_abs)
                   if total_abs > 0 else float("nan")}
            for k in (5, 10, 20):
                keep = np.ones(n_te, bool)
                keep[order[:k]] = False
                row[f"flip_drop_top{k}"] = aurc(sl[keep], risk[keep]) - aurc(sn[keep], risk[keep])
                null = [
                    (lambda m: aurc(sl[m], risk[m]) - aurc(sn[m], risk[m]))(
                        np.isin(np.arange(n_te), rng.choice(n_te, n_te - k, replace=False)))
                    for _ in range(n_null)]
                row[f"flip_drop_random{k}"] = float(np.mean(null))
                row[f"excess_top{k}"] = row[f"flip_drop_random{k}"] - row[f"flip_drop_top{k}"]
            rows.append(row)

    frame = pd.DataFrame(rows)
    frame.attrs["per_item"] = pd.DataFrame(per_item)
    return frame


def scatter_figure(cells: Sequence = READABLE_CELLS, out_dir: Path = OUT) -> list[Path]:
    """One scatter per cell: letter reading against numeric reading, three panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for cell in cells:
        cell = tuple(cell)
        _, y, blobs = load_alphabets(cell)
        L, N = side_scores(blobs[LETTER]), side_scores(blobs[NUMERIC])
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), constrained_layout=True)
        for ax, readout in zip(axes, ("V_pos", "V_neg", "gap")):
            a, b = L[readout], N[readout]
            ax.scatter(a, b, s=7, alpha=0.25, edgecolors="none", color="#1f4e79")
            lim = [min(a.min(), b.min()), max(a.max(), b.max())]
            ax.plot(lim, lim, lw=0.8, color="0.5", ls="--")
            ax.set_title(f"{readout}   r={_pearson(a, b):.3f}  "
                         rf"$\rho$={_spearman(a, b):.3f}", fontsize=9)
            ax.set_xlabel(f"{LETTER}")
            ax.set_ylabel(f"{NUMERIC}")
        fig.suptitle(f"{_cell_name(cell)} — same items, two alphabets", fontsize=10)
        path = Path(out_dir) / f"scatter_{cell[0]}_{cell[1]}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


# ──────────────────────────────────────────────── (6) paired seed test


def paired_seed_test(cells: Sequence = READABLE_CELLS, n_boot: int = N_BOOT,
                     boot_seed: int = 0) -> pd.DataFrame:
    """(6) Test the flip paired across seeds, not by comparing two intervals.

    -0.038 +/- 0.023 and +0.079 +/- 0.023 share items and seeds, so the
    difference has much less variance than the two SDs suggest independently.
    Compute the per-seed difference and its interval directly.

    The algebra is worth stating in the paper: the contrast is

        (A_q - A_numeric) - (A_q - A_letter) = A_letter - A_numeric,

    so s_q cancels exactly and the flip does not depend on the baseline at all.
    Nearly all of the +/- 0.023 on each alphabet is s_q's seed-to-seed variance,
    which is common to both arms.

    Report it that way in the paper too. Two overlapping-looking intervals
    invite a reviewer to conclude the flip is not resolved when the paired test
    may show it clearly is.
    """
    rows = []
    for cell in cells:
        cell = tuple(cell)
        d, y, blobs = load_alphabets(cell)
        L, N = side_scores(blobs[LETTER])["gap"], side_scores(blobs[NUMERIC])["gap"]
        for fold in FOLDS:
            per_seed, boot_draws = [], []
            for seed in SEEDS:
                ix, sq = _fold_scores(d, y, seed, fold)
                risk = 1.0 - y[ix]
                a_q, a_l, a_n = aurc(sq, risk), aurc(L[ix], risk), aurc(N[ix], risk)
                d_l, d_n = a_q - a_l, a_q - a_n
                boot = _boot_index(len(ix), n_boot, boot_seed + seed)
                boot_draws.append(np.array([
                    aurc(L[ix][b], risk[b]) - aurc(N[ix][b], risk[b]) for b in boot]))
                per_seed.append({
                    "seed": seed, "n": len(ix), "AURC_sq": a_q,
                    "AURC_letter": a_l, "AURC_numeric": a_n,
                    "delta_letter": d_l, "delta_numeric": d_n,
                    "paired_flip": d_n - d_l, "identity_check": a_l - a_n,
                })
            ps = pd.DataFrame(per_seed)
            pooled = np.mean(np.vstack(boot_draws), axis=0)
            lo, hi = _ci(pooled)
            for r in per_seed:
                rows.append({"model": cell[0], "dataset": cell[1], "fold": fold,
                             "scope": "per_seed", **r})
            rows.append({
                "model": cell[0], "dataset": cell[1], "fold": fold,
                "scope": "summary", "seed": -1, "n": int(ps["n"].mean()),
                "AURC_sq": ps["AURC_sq"].mean(),
                "AURC_letter": ps["AURC_letter"].mean(),
                "AURC_numeric": ps["AURC_numeric"].mean(),
                "delta_letter": ps["delta_letter"].mean(),
                "delta_numeric": ps["delta_numeric"].mean(),
                "paired_flip": ps["paired_flip"].mean(),
                "identity_check": ps["identity_check"].mean(),
                "sd_delta_letter": ps["delta_letter"].std(ddof=1),
                "sd_delta_numeric": ps["delta_numeric"].std(ddof=1),
                "sd_paired_flip": ps["paired_flip"].std(ddof=1),
                "sd_ratio_paired_over_worst_arm": float(
                    ps["paired_flip"].std(ddof=1)
                    / max(ps["delta_letter"].std(ddof=1),
                          ps["delta_numeric"].std(ddof=1), 1e-12)),
                "variance_removed_by_pairing": float(
                    1.0 - ps["paired_flip"].std(ddof=1)
                    / max(ps["delta_letter"].std(ddof=1),
                          ps["delta_numeric"].std(ddof=1), 1e-12)),
                "item_boot_ci_lo": lo, "item_boot_ci_hi": hi,
                "excludes_zero": bool(lo > 0 or hi < 0),
            })
    return pd.DataFrame(rows)


NEXT_EXPERIMENT = """
SINGLE-DIGIT ALPHABET (digit10-direct), levels 0-9 mapped to 0-90%.

Single digits are single tokens under both grouping and splitting tokenizers, so
this alphabet is clean on all three models. It is the only way to separate the
alphabet effect from the tokenization effect: "numeric vs letter" is currently
entangled with "contaminated vs clean", and exactly one cell is interpretable.

WHAT IT HOLDS, AND HOW LOOSELY. K = 10 against the letter arm's 11, so
resolution is held APPROXIMATELY, not matched by design. That the remaining
one-level mismatch does not carry the effect rests on (4) -- on Llama,
restricting the 101-level distribution to the 11-level letter support retained
99% of the flip -- which is an argument plus a one-level gap, not a matched
control. Write it that way. K was never matched, and nobody should later be able
to read this line as saying it was.

WHAT STAYS ENTANGLED. digit10-direct against letter11 still varies glyph and
mapping together, exactly as digits101 did: a digit carries its value directly,
a letter requires resolving an in-context legend. So the run establishes whether
the effect EXISTS beyond Llama. It does not say what the effect is made of.
Characterisation needs the permuted-legend arm (see shuffled_legend_status),
which holds glyph fixed and moves mapping alone. A reviewer who reads
"characterised" will ask which factor is responsible; without the permuted arm
there is no answer.

WHAT IT MIGHT RETURN. Not necessarily six cells of one effect. An effect on
Llama with nulls on Mistral and Qwen is a live outcome, and it yields a NARROWER
claim -- one model, format-sensitive -- rather than a broader one. Still worth
the pass, and still the run to do, but the decision rule mapping each outcome
onto a claim must be fixed BEFORE the look, in the house style of
paper/PREREGISTRATION_ADDITIVITY.md, whose second rule is the relevant shape:
partial replication reports the narrower finding rather than nothing. No such
prereg exists for this arm yet; it is a prerequisite, not a formality.

Cost: one elicitation pass. TruthfulQA alone is 817 items x 3 models x 2 sides.

What it converts: an anomaly in one cell into an ESTABLISHED effect across six.
Without it the paper says the report result flipped on Llama under a different
alphabet; with it the paper says whether elicitation format bears on the report
channel beyond one model. Protect this over the remaining appendix analyses.
"""


CHECKS = {
    "cache_inventory": cache_inventory,
    "answer_invariance_by_alphabet": answer_invariance_by_alphabet,
    "adequacy_under_numeric": adequacy_under_numeric,
    "which_side_flips": which_side_flips,
    "resolution_vs_content": resolution_vs_content,
    "cross_alphabet_agreement": cross_alphabet_agreement,
    "paired_seed_test": paired_seed_test,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checks", nargs="+", choices=tuple(CHECKS),
                        default=tuple(CHECKS))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "flip_cell": _cell_name(FLIP_CELL),
        "alphabets": {LETTER: "ABCDEFGHJKL = 0..100 step 10, in-context legend",
                      NUMERIC: "digits101-v1, 0..100 step 1, no legend"},
        "confounds": ["glyph class", "legend indirection", "resolution K=11 vs K=101"],
        "seeds": list(SEEDS), "folds": list(FOLDS),
        "s_q_protocol": "always fitted on the fold complementary to the one scored",
        "discipline": "exploratory decomposition of an observed flip; no confirmatory count",
        "shuffled_legend_control": shuffled_legend_status(),
        "next_experiment": NEXT_EXPERIMENT.strip(),
        "checks": {},
    }
    frames = {}
    for name in args.checks:
        frame = CHECKS[name]()
        frames[name] = frame
        frame.to_csv(args.out / f"{name}.csv", index=False)
        entry = {"rows": len(frame), "csv": f"{name}.csv"}
        if "per_item" in frame.attrs:
            frame.attrs["per_item"].to_csv(args.out / f"{name}_per_item.csv", index=False)
            entry["per_item_csv"] = f"{name}_per_item.csv"
        if name == "adequacy_under_numeric":
            entry["rescued_cells"] = frame.attrs.get("rescued_cells", [])
            entry["blocked_cells"] = sorted(
                {f"{m}/{d}" for m, d, s in
                 frame[["model", "dataset", "status"]].itertuples(index=False)
                 if s == "requires_re_elicitation"})
        if name == "resolution_vs_content":
            entry["verdict"] = resolution_verdict(frame)
        if name == "which_side_flips":
            by_readout = flip_by_readout(frame)
            by_readout.to_csv(args.out / "which_side_flips_by_readout.csv", index=False)
            entry["by_readout_csv"] = "which_side_flips_by_readout.csv"
        manifest["checks"][name] = entry
        print(f"\n=== {name} ===\n{frame.to_string(index=False)}")
        if name == "which_side_flips":
            print(f"\n--- A_letter - A_numeric by readout ---\n"
                  f"{by_readout.to_string(index=False)}")

    if not args.no_figures and "cross_alphabet_agreement" in args.checks:
        manifest["figures"] = [p.name for p in scatter_figure(out_dir=args.out)]

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(f"\nWrote alphabet-flip outputs to {args.out}")


if __name__ == "__main__":
    main()
