"""Legend-reversal figures (Fig 1a rank tracks, Fig 4 quintile transitions).

Run from the repo root:  python -m experiments.fig_legend

Data (read-only): the frozen p2_run1 extraction used for the original Fig 1a / Fig 4,
  ../verbalized_conformal_confidence/LearningtoCalibrate/cached_results/p2_run1/run1_20260919_464acbb/
    {model}/main/fwd.pt  forward mapping, canonical confidence prompt    (F0)
    {model}/main/rev.pt  reversed mapping, canonical confidence prompt   (R0)
    {model}/alt/fwd.pt   forward mapping, reworded confidence prompt     (F1, "re-asked")
Score: V_pos (decoded expected confidence for the supplied correct candidate).
Outputs go to paper/figs/.
"""
from __future__ import annotations

import hashlib
import json
import os
import textwrap
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from scipy.stats import rankdata, spearmanr  # noqa: E402

VCC = REPO.parent / "verbalized_conformal_confidence"
RUN = VCC / "LearningtoCalibrate/cached_results/p2_run1/run1_20260919_464acbb"
PROV = VCC / "letter11_provenance_v1"
OUT = REPO / "paper/figs"
MODELS = [("llama3_8b", "Llama-3.1-8B"), ("mistral_7b", "Mistral-7B"), ("qwen2_5_7b", "Qwen2.5-7B")]
FIG1A_MODEL = "llama3_8b"  # model of the original Fig 1a (lear_rank_panel_20260921)
OLD_ITEM_ID = "e352a2310e2523f3"  # "Bill ... Lear" item of the original Fig 1a
N_GROUPS = 5

# Old-figure check numbers (target values from the brief)
TARGET_STAY = {"llama3_8b": (24.0, 28.4), "mistral_7b": (20.2, 31.8), "qwen2_5_7b": (24.4, 33.7)}
TARGET_OLD_ITEM = dict(move_rev=165, med_rev=203, move_reask=4, med_reask=164)

# Okabe-Ito colours
C_FWD, C_REASK, C_REV = "#000000", "#0072B2", "#D55E00"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
    "figure.facecolor": "white", "savefig.facecolor": "white",
})


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(model: str, form: str, legend: str) -> dict:
    return torch.load(RUN / model / form / f"{legend}.pt", weights_only=False, map_location="cpu")


def conf_rank(v: np.ndarray) -> np.ndarray:
    """Rank among all items, 1 = most confident, ties -> average rank."""
    return rankdata(-np.asarray(v, dtype=np.float64), method="average")


def groups(rank: np.ndarray, n: int) -> np.ndarray:
    """0-based equal-count groups; group 0 (= 'group 1') holds the most confident items.

    Binned exactly as the old Fig 4 (make_rank_quintiles.py): ascending rank r_up = n + 1 - rank
    (1 = least confident), quintile q = floor((r_up-1)*5/n) clipped at 4, then relabelled g = 4 - q.
    This reproduces the old matrices exactly (sizes from most to least confident: 163,163,164,163,164).
    Binning the descending rank directly moves a few boundary items and changes stay shares by ~0.1 pp.
    """
    r_up = n + 1 - np.asarray(rank)
    q = np.minimum(np.floor((r_up - 1) * N_GROUPS / n), N_GROUPS - 1).astype(int)
    return (N_GROUPS - 1) - q


def transition(gf: np.ndarray, go: np.ndarray):
    counts = np.zeros((N_GROUPS, N_GROUPS), dtype=int)
    np.add.at(counts, (gf, go), 1)
    M = counts / counts.sum(1, keepdims=True)
    return counts, M


# --------------------------------------------------------------------------- data
def build_data():
    questions = [json.loads(l) for l in (RUN / "questions.jsonl").read_text().splitlines() if l.strip()]
    qids = [q["question_id"] for q in questions]
    data, sources = {}, {}
    for model, _ in MODELS:
        arms = {"F0": load(model, "main", "fwd"), "R0": load(model, "main", "rev"), "F1": load(model, "alt", "fwd")}
        for arm, d in arms.items():
            assert list(d["example_ids"]) == qids, (model, arm)
            assert np.isfinite(d["V_pos"]).all()
        for rel in ("main/fwd.pt", "main/rev.pt", "alt/fwd.pt"):
            sources[f"{model}/{rel}"] = sha(RUN / model / rel)
        V = {arm: np.asarray(d["V_pos"], dtype=np.float64) for arm, d in arms.items()}
        R = {arm: conf_rank(v) for arm, v in V.items()}
        # cross-check against the independent letter11_provenance_v1 extraction (no alt arm there)
        xcheck = {}
        for arm, fn in (("F0", "forward"), ("R0", "reversed")):
            p = PROV / model / "truthfulqa" / f"{fn}.pt"
            if p.exists():
                pv = torch.load(p, weights_only=False, map_location="cpu")
                same_ids = list(pv["example_ids"]) == qids
                xcheck[arm] = dict(file=str(p.relative_to(VCC.parent)), same_ids=same_ids,
                                   max_abs_diff_V_pos=float(np.abs(np.asarray(pv["V_pos"]) - V[arm]).max()) if same_ids else None,
                                   spearman_with_p2=float(spearmanr(np.asarray(pv["V_pos"]), V[arm]).statistic) if same_ids else None)
        data[model] = dict(V=V, R=R, xcheck=xcheck, legend_meta={
            arm: dict(form=arms[arm]["meta"]["form"], legend=arms[arm]["meta"]["legend"],
                      prompt_template=arms[arm]["meta"]["prompt_template"]) for arm in arms})
    return questions, data, sources


def movement_stats(R):
    mv_rev = np.abs(R["R0"] - R["F0"])
    mv_reask = np.abs(R["F1"] - R["F0"])
    return mv_rev, mv_reask, float(np.median(mv_rev)), float(np.median(mv_reask))


# --------------------------------------------------------------------------- fig 1a
def shorten(q: str, head: int = 52, tail: int = 34) -> str:
    q = " ".join(q.split())
    if len(q) <= head + tail + 3:
        return q
    h = q[:head].rsplit(" ", 1)[0].rstrip(",;:")
    t = q[-tail:].split(" ", 1)[1] if " " in q[-tail:] else q[-tail:]
    return f"{h} … {t}"


def clean_candidate(a: str, maxlen: int = 40) -> str:
    a = " ".join(a.split()).rstrip(".")
    return a if len(a) <= maxlen else a[: maxlen].rsplit(" ", 1)[0] + "…"


def draw_fig1a(path_stem: Path, q: dict, idx: int, R, med_rev, med_reask, n, model_label):
    for head, tail in ((52, 34), (46, 30), (40, 28), (34, 26), (28, 24), (24, 20)):
        try:
            return _draw_fig1a(path_stem, q, idx, R, med_rev, med_reask, n, model_label, head, tail)
        except OverflowError:
            plt.close("all")
    raise RuntimeError("prompt strip overflow")


def _draw_fig1a(path_stem, q, idx, R, med_rev, med_reask, n, model_label, head, tail):
    rf, rr, ra = float(R["F0"][idx]), float(R["R0"][idx]), float(R["F1"][idx])
    fig = plt.figure(figsize=(6.75, 2.4))
    # ---- prompt strip
    strip = fig.add_axes([0.012, 0.80, 0.976, 0.17])
    strip.set_axis_off()
    strip.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.08",
                                   transform=strip.transAxes, fc="#F2F2F2", ec="#BDBDBD", lw=0.5))
    qtxt = f"“{shorten(q['question'], head, tail)}”"
    strip.text(0.012, 0.5, "Q:", fontsize=7.5, weight="bold", va="center", transform=strip.transAxes)
    strip.text(0.040, 0.5, qtxt, fontsize=7.5, va="center", transform=strip.transAxes, style="italic")
    strip.text(0.988, 0.5, "", transform=strip.transAxes)
    fig.canvas.draw()
    bb = strip.texts[1].get_window_extent().transformed(strip.transAxes.inverted())
    x = bb.x1 + 0.014
    strip.text(x, 0.5, "Candidate:", fontsize=7.5, weight="bold", va="center", transform=strip.transAxes)
    fig.canvas.draw()
    x = strip.texts[-1].get_window_extent().transformed(strip.transAxes.inverted()).x1 + 0.006
    strip.text(x, 0.5, clean_candidate(q["pos"]), fontsize=7.5, va="center", transform=strip.transAxes)
    strip.text(0.988, 0.5, "Scale: A = 0% … L = 100%   vs   A = 100% … L = 0%",
               fontsize=7.5, va="center", ha="right", transform=strip.transAxes)
    fig.canvas.draw()
    # guard: candidate must not collide with the scale text
    t_cand, t_scale = strip.texts[-2], strip.texts[-1]
    if t_cand.get_window_extent().x1 > t_scale.get_window_extent().x0 - 6:
        raise OverflowError

    # ---- tracks
    ax = fig.add_axes([0.235, 0.20, 0.645, 0.56])
    tracks = [(2, "Forward", None, rf, C_FWD, None, None),
              (1, "Reworded prompt", "Same legend as forward", ra, C_REASK, med_reask, abs(ra - rf)),
              (0, "Reversed mapping", None, rr, C_REV, med_rev, abs(rr - rf))]
    for y, lab, gloss, r, col, med, mv in tracks:
        ax.plot([1, n], [y, y], color="#D0D0D0", lw=0.6, zorder=0)
        if med is not None:
            lo, hi = max(1, rf - med), min(n, rf + med)
            ax.add_patch(Rectangle((lo, y - 0.17), hi - lo, 0.34, fc="#D9D9D9", ec="none", zorder=1))
        ax.scatter([r], [y], s=34, color=col, zorder=4, edgecolor="white", linewidth=0.6,
                   marker="o" if med is None else "D")
        ax.text(r, y + 0.24, f"{r:.0f}", ha="center", va="bottom", fontsize=7, color=col, zorder=5,
                bbox=dict(fc="white", ec="none", pad=0.6))
        # left labels
        if gloss is None:
            ax.text(-0.015, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.5)
        else:
            ax.text(-0.015, y + 0.11, lab, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.5)
            ax.text(-0.015, y - 0.20, gloss, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=7, color="#555555")
        # right annotations
        if med is None:
            ax.text(1.015, y, "reference", transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=7, color="#555555")
        else:
            ax.text(1.015, y + 0.11, f"moved {mv:.0f}", transform=ax.get_yaxis_transform(), ha="left",
                    va="center", fontsize=7.5, color=col)
            ax.text(1.015, y - 0.20, f"median {med:.0f}", transform=ax.get_yaxis_transform(), ha="left",
                    va="center", fontsize=7, color="#555555")
    ax.axvline(rf, color=C_FWD, lw=0.7, ls=(0, (3, 2)), zorder=2)
    ax.set_xlim(-12, n + 12)
    ax.set_ylim(-0.45, 2.55)
    ax.set_yticks([])
    ax.set_xticks([1, 200, 400, 600, n])
    ax.tick_params(axis="x", length=2.5, pad=1.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_bounds(1, n)
    ax.spines["bottom"].set_color("#777777")
    ax.set_xlabel(f"Rank among {n} questions ({model_label}; 1 = most confident)", labelpad=2)
    # small band key
    fig.text(0.235, 0.015, "Grey band: forward rank ± population median |rank move| for that comparison.",
             fontsize=7, color="#555555", va="bottom")
    fig.savefig(path_stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
    return dict(rank_forward=rf, rank_reasked=ra, rank_reversed=rr,
                move_reasked=abs(ra - rf), move_reversed=abs(rr - rf))


# --------------------------------------------------------------------------- fig 4
def draw_fig4(path_stem: Path, mats: dict, stays: dict, vmax: float):
    W, H = 6.75, 2.6
    fig = plt.figure(figsize=(W, H))
    cmap = plt.get_cmap("YlGnBu")
    norm = Normalize(0, vmax)
    s, x0, pitch, yb = 1.12, 0.50, 1.93, 0.74  # heatmap side, first left edge, pitch, bottom (inches)

    def ax_in(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    for j, (model, label) in enumerate(MODELS):
        ax = ax_in(x0 + j * pitch, yb, s, s)
        M = mats[model]
        # Orientation follows the old figure: rows = forward group on y, columns = reversed group on x,
        # high-confidence corner at TOP-RIGHT (diagonal runs bottom-left -> top-right).
        # Internal index 0 = group 1 = most confident, so display index = 4 - g on both axes.
        D = M[::-1, ::-1]
        ax.imshow(D, cmap=cmap, norm=norm, origin="lower", interpolation="nearest", aspect="equal")
        for a in range(N_GROUPS):
            for b in range(N_GROUPS):
                v = D[a, b]
                ax.text(b, a, f"{100 * v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 0.6 * vmax else "#1A1A1A")
        for k in range(N_GROUPS):  # outline diagonal (the "stay" cells)
            ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, ec="#222222", lw=0.7))
        labels = [str(N_GROUPS - i) for i in range(N_GROUPS)]  # display 0..4 -> group 5..1
        ax.set_xticks(range(N_GROUPS), labels=labels)
        ax.set_yticks(range(N_GROUPS), labels=labels)
        ax.tick_params(length=0, pad=1.5)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("Group under reversal", labelpad=1.5)
        if j == 0:
            ax.set_ylabel("Forward group", labelpad=1.5)
        st_rev, st_re = stays[model]
        cx = (x0 + j * pitch + s / 2) / W
        fig.text(cx, (yb + s + 0.42) / H, label, ha="center", va="bottom", fontsize=8, weight="bold")
        fig.text(cx, (yb + s + 0.22) / H, f"reversal stay = {st_rev:.1f}%", ha="center", va="bottom",
                 fontsize=7.5, color=C_REV, weight="bold")
        fig.text(cx, (yb + s + 0.05) / H, f"reworded-prompt stay = {st_re:.1f}%", ha="center", va="bottom",
                 fontsize=7.5, color=C_REASK)
    cax = ax_in(x0 + 2 * pitch + s + 0.22, yb, 0.08, s)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, format=PercentFormatter(1, decimals=0))
    cb.set_ticks(np.arange(0, vmax + 1e-9, 0.1))
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.5, labelsize=7, pad=1.5)
    cb.set_label("Row share", fontsize=7, labelpad=7)
    fig.text(x0 / W, (H - 0.04) / H, "Group 1 = most confident fifth (163\u2013164 questions); rows sum to 100%; "
             "full scrambling would give 20% in every cell and stay = 20%.",
             fontsize=7, color="#444444", va="top")
    # frozen-state strip (deliberately not a heatmap)
    sx = ax_in(x0, 0.05, 2 * pitch + s + 0.62, 0.22)
    sx.set_axis_off()
    sx.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.3",
                                transform=sx.transAxes, fc="white", ec="#888888", lw=0.6, ls=(0, (3, 2)),
                                mutation_aspect=0.04))
    sx.text(0.012, 0.5, "Frozen state:", transform=sx.transAxes, va="center", ha="left", fontsize=7.5, weight="bold")
    fig.canvas.draw()
    x1 = sx.texts[0].get_window_extent().transformed(sx.transAxes.inverted()).x1 + 0.008
    sx.text(x1, 0.5, "cells fixed before elicitation: stay = 100% by construction (definition, not a result)",
            transform=sx.transAxes, va="center", ha="left", fontsize=7.5, style="italic", color="#333333")
    fig.savefig(path_stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(path_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)

# --------------------------------------------------------------------------- main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    questions, data, sources = build_data()
    n = len(questions)
    assert n == 817
    out = dict(n_questions=n, score="V_pos (decoded expected confidence for the correct candidate)",
               rank_convention="scipy rankdata(-V_pos, method='average'); 1 = most confident",
               group_convention="group = floor((rank-1)*5/817)+1, group 1 = most confident",
               source_run=str(RUN.relative_to(VCC.parent)), source_sha256=sources, models={})

    mats, stays = {}, {}
    for model, label in MODELS:
        R = data[model]["R"]
        gF, gR, gA = groups(R["F0"], n), groups(R["R0"], n), groups(R["F1"], n)
        cR, MR = transition(gF, gR)
        cA, MA = transition(gF, gA)
        stay_rev, stay_re = 100 * np.trace(MR) / N_GROUPS, 100 * np.trace(MA) / N_GROUPS
        mv_rev, mv_re, med_rev, med_re = movement_stats(R)
        mats[model], stays[model] = MR, (stay_rev, stay_re)
        out["models"][model] = dict(
            label=label,
            group_sizes_forward=np.bincount(gF, minlength=5).tolist(),
            reversal=dict(counts=cR.tolist(), row_shares=MR.tolist(), diagonal=np.diag(MR).tolist(),
                          stay_share_pct=stay_rev, stay_share_question_weighted_pct=100 * np.trace(cR) / n),
            reasking=dict(counts=cA.tolist(), row_shares=MA.tolist(), diagonal=np.diag(MA).tolist(),
                          stay_share_pct=stay_re, stay_share_question_weighted_pct=100 * np.trace(cA) / n),
            median_abs_rank_move=dict(reversal=med_rev, reasking=med_re),
            spearman=dict(F0_R0=float(spearmanr(data[model]["V"]["F0"], data[model]["V"]["R0"]).statistic),
                          F0_F1=float(spearmanr(data[model]["V"]["F0"], data[model]["V"]["F1"]).statistic)),
            reproduction_check=dict(target_stay_reversal_pct=TARGET_STAY[model][0],
                                    target_stay_reasking_pct=TARGET_STAY[model][1],
                                    match=bool(round(stay_rev, 1) == TARGET_STAY[model][0]
                                               and round(stay_re, 1) == TARGET_STAY[model][1])),
            crosscheck_letter11_provenance_v1=data[model]["xcheck"],
            prompts=data[model]["legend_meta"])
        print(f"{label:13s} stay rev {stay_rev:.2f}%  reask {stay_re:.2f}%  med move rev {med_rev:.0f} reask {med_re:.0f}")

    allmax = max(m.max() for m in mats.values())
    vmax = float(np.ceil(allmax * 10) / 10)
    draw_fig4(OUT / "fig4_quintile_reversal", mats, stays, vmax)
    out["fig4"] = dict(vmax=vmax, max_cell=float(allmax), colormap="YlGnBu (ColorBrewer sequential)",
                       orientation="rows = forward group (y), columns = group under reversal (x); "
                                   "group 1 (most confident) at top / right, as in the old figure "
                                   "(old: quintile 5 = highest at top/right, origin lower)")

    # ---- Fig 1a
    R = data[FIG1A_MODEL]["R"]
    mv_rev, mv_re, med_rev, med_re = movement_stats(R)
    score = np.abs(mv_rev - med_rev) / med_rev + np.abs(mv_re - med_re) / med_re
    best = float(score.min())
    chosen = int(np.flatnonzero(np.isclose(score, best, rtol=0, atol=1e-12))[0])  # lowest index on ties
    order = np.argsort(score, kind="stable")[:5]
    old_idx = [q["question_id"] for q in questions].index(OLD_ITEM_ID)
    label = dict(MODELS)[FIG1A_MODEL]
    res_new = draw_fig1a(OUT / "fig1a", questions[chosen], chosen, R, med_rev, med_re, n, label)
    res_old = draw_fig1a(OUT / "fig1a_old_item", questions[old_idx], old_idx, R, med_rev, med_re, n, label)
    pct = lambda mv, x: float(100 * ((mv < x).sum() + 0.5 * (mv == x).sum()) / n)  # noqa: E731
    out["fig1a"] = dict(
        model=FIG1A_MODEL, median_abs_rank_move=dict(reversal=med_rev, reasking=med_re),
        band="symmetric around forward rank, half-width = population median |rank move|, clipped to [1, 817]",
        selection_rule="argmin_i |move_rev_i - med_rev|/med_rev + |move_reask_i - med_reask|/med_reask; "
                       "ties -> lowest question index (questions.jsonl order, 0-based)",
        chosen=dict(index=chosen, question_id=questions[chosen]["question_id"], question=questions[chosen]["question"],
                    candidate=questions[chosen]["pos"], score=best,
                    percentile_move_reversal=pct(mv_rev, res_new["move_reversed"]),
                    percentile_move_reasking=pct(mv_re, res_new["move_reasked"]), **res_new),
        runners_up=[dict(index=int(i), question_id=questions[i]["question_id"], score=float(score[i]),
                         move_reversed=float(mv_rev[i]), move_reasked=float(mv_re[i])) for i in order],
        old_item=dict(index=old_idx, question_id=OLD_ITEM_ID, question=questions[old_idx]["question"],
                      candidate=questions[old_idx]["pos"], score=float(score[old_idx]),
                      percentile_move_reversal=pct(mv_rev, res_old["move_reversed"]),
                      percentile_move_reasking=pct(mv_re, res_old["move_reasked"]), **res_old,
                      reproduction_check=dict(**TARGET_OLD_ITEM, match=bool(
                          res_old["move_reversed"] == 165 and res_old["move_reasked"] == 4
                          and med_rev == 203 and med_re == 164))),
        population_iqr_abs_move=dict(reversal=np.percentile(mv_rev, [25, 75]).tolist(),
                                     reasking=np.percentile(mv_re, [25, 75]).tolist()))
    (OUT / "fig_legend_numbers.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out["fig1a"][k] for k in ("chosen", "old_item")}, indent=1))


if __name__ == "__main__":
    main()
