"""Build the method figures from tokenizer metadata and cached readouts.

Figure 2 shows which intended numeric values share a first-token identity.
Figure 3 shows the paired Llama/TruthfulQA confidence readouts.  The script
fails closed when a cache schema or item-id alignment is not what it expects.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    # These are the IDs used by extraction/modal_app.py.
    "llama3_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen2_5_7b": "Qwen/Qwen2.5-7B-Instruct",
}
VALUE_FMT = " {v}"
COL_SPLIT, COL_CLEAN = "#A32D2D", "#5F5E5A"


def first_token_map(tok, values, fmt=VALUE_FMT) -> dict[int, str]:
    """Map values to the first token after their shared separator prefix."""
    encoded = {
        value: tok(fmt.format(v=value), add_special_tokens=False)["input_ids"]
        for value in values
    }
    if any(not ids for ids in encoded.values()):
        raise RuntimeError("tokenizer returned an empty value encoding")
    # A leading-space marker is a shared separator, not the emitted value
    # token.  This is the same longest-common-prefix operation used by the
    # extractor's guarded confidence-token audit.
    common = 0
    while all(len(ids) > common for ids in encoded.values()):
        tokens = {ids[common] for ids in encoded.values()}
        if len(tokens) != 1:
            break
        common += 1
    out = {}
    for value, ids in encoded.items():
        tail = ids[common:]
        if not tail:
            raise RuntimeError(f"no value token remains for {value}")
        out[value] = tok.convert_ids_to_tokens(tail[0])
    return out


def collisions(fmap: dict[int, str]) -> dict[str, list[int]]:
    inverse = defaultdict(list)
    for value, token in fmap.items():
        inverse[token].append(value)
    return {token: values for token, values in inverse.items()}


def draw_collision_strip(ax, fmap, title, ymax=100, neutral_text=None,
                         highlight=None, neutral_xy=(0.99, 0.06),
                         neutral_ha="right"):
    """Show one equal-width cell per intended value.

    Red cells are values whose first token is shared by another value.  The
    x-axis remains the intended numeric value, so collision density cannot be
    mistaken for a change in scale or bin spacing.
    """
    inverse = collisions(fmap)
    for value, token in fmap.items():
        shared = len(inverse[token]) > 1
        edge = "#1D1D1D" if value == highlight else "none"
        ax.add_patch(Rectangle(
            (value - 0.5, 0.25), 1.0, 0.5,
            facecolor=COL_SPLIT if shared else COL_CLEAN,
            edgecolor=edge, linewidth=1.2 if value == highlight else 0.25,
            alpha=0.85,
        ))
    # Token labels are useful for the ten-bin split; on the 101-bin row the
    # intended-value ticks carry the readable overview and avoid label noise.
    if len(inverse) <= 20:
        for token, values in inverse.items():
            ax.text(np.mean(values), 0.51, token, fontsize=6,
                    ha="center", va="center", color="white")
    step = 10 if ymax > 20 else 1
    ticks = np.arange(0, ymax + 1, step)
    if ymax > 20:
        for tick in ticks[1:-1]:
            ax.axvline(tick - 0.5, color="white", lw=0.35, alpha=0.7, zorder=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(v)) for v in ticks], fontsize=6)
    ax.set_yticks([])
    ax.set_xlim(-0.5, ymax + 0.5)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=8, loc="left")
    if neutral_text:
        ax.text(*neutral_xy, neutral_text, ha=neutral_ha, va="bottom",
                fontsize=6, transform=ax.transAxes)
    if highlight is not None:
        ax.annotate(str(highlight), (highlight, 0.8), xytext=(highlight - 4, 0.98),
                    fontsize=6, ha="center", color=COL_SPLIT,
                    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": COL_SPLIT})
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def _neutral_mass(model_key, scheme):
    path = Path(__file__).parent / "neutral_prior" / f"{model_key}.json"
    if not path.exists():
        return None
    data = __import__("json").loads(path.read_text())
    scheme_key = "digit10-direct" if scheme == "digit10" else scheme
    for row in data.get("rows", []):
        if row.get("conf_scheme") == scheme_key:
            return row.get("candidate_mass")
    return None


def _tokenizer_source(model_key, hf_id):
    """Prefer the exact local snapshot used by this workspace when present."""
    cache_names = {
        "llama3_8b": "models--meta-llama--Llama-3.1-8B-Instruct",
        "mistral_7b": "models--mistralai--Mistral-7B-Instruct-v0.3",
        "qwen2_5_7b": "models--Qwen--Qwen2.5-7B-Instruct",
    }
    cache_root = Path.home() / ".cache" / "huggingface" / "hub" / cache_names[model_key]
    refs = cache_root / "refs" / "main"
    if refs.exists():
        snapshot = cache_root / "snapshots" / refs.read_text().strip()
        if (snapshot / "tokenizer_config.json").exists():
            return str(snapshot)
    return hf_id


def load_vdist_item(cache_path: Path, item_id: str, side="pos"):
    """Load one real Vdist row, preserving the cache's item identity."""
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    ids = np.asarray(data["example_ids"], dtype=object)
    matches = np.flatnonzero([str(x) == str(item_id) for x in ids])
    if len(matches) != 1:
        raise KeyError(f"{cache_path}: expected one item_id={item_id!r}")
    values = np.asarray(data["conf_values"], dtype=float) / 100.0
    dist = np.asarray(data[f"Vdist_{side}"][matches[0]], dtype=float)
    if dist.shape != values.shape:
        raise ValueError(f"{cache_path}: incompatible Vdist/conf_values shapes")
    return dist, values


def load_V_item(cache_path: Path, item_id: str):
    """Return one item's V+ and V- in denoted-value space."""
    pos, values = load_vdist_item(cache_path, item_id, "pos")
    neg, values_neg = load_vdist_item(cache_path, item_id, "neg")
    if not np.array_equal(values, values_neg):
        raise ValueError(f"{cache_path}: positive/negative levels differ")
    return float(pos @ values), float(neg @ values)


def _draw_token_failure(ax, cell_dir: Path, item_id: str):
    """Panel A: one expressed 101-level distribution and split-token reread."""
    tok = AutoTokenizer.from_pretrained(
        _tokenizer_source("mistral_7b", MODELS["mistral_7b"]),
        local_files_only=True,
    )
    fmap = first_token_map(tok, range(101))
    inverse = collisions(fmap)
    bins = sorted(inverse, key=lambda token: min(inverse[token]))
    vdist, values = load_vdist_item(cell_dir / "fine_conf.pt", item_id, "pos")
    reread = np.asarray([vdist[inverse[token]].sum() for token in bins])
    ax.bar(values, vdist, width=0.0075, color="0.55", linewidth=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(0.42, float(vdist.max()) * 1.25))
    ax.set_xlabel("intended value", fontsize=7)
    ax.set_ylabel("model mass", fontsize=7)
    ax.set_title("A. Token failure: one distribution, two readings", fontsize=8,
                 loc="left", pad=5)
    # Literal tokenization schematic: all three strings begin with the same
    # first token under the splitting tokenizer.  It occupies the otherwise
    # empty upper-left area and makes the collision readable without a legend.
    ax.text(0.03, 0.93, '1    15    100  →  “1”',
            transform=ax.transAxes, fontsize=6.5, family="monospace",
            color="#A32D2D", va="top")
    ax.tick_params(labelsize=6)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels(["0", "20", "40", "60", "80", "100"], fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="0.88", lw=0.4)
    ax.annotate(f"P(100) = {vdist[100]:.3f}", (1.0, vdist[100]),
                xytext=(0.60, ax.get_ylim()[1] * 0.83), fontsize=6,
                arrowprops={"arrowstyle": "->", "lw": 0.5, "color": "#A32D2D"},
                color="#A32D2D")
    inset = ax.inset_axes([0.57, 0.43, 0.40, 0.50])
    colors = ["#A32D2D" if token == fmap[100] else "0.55" for token in bins]
    inset.bar(np.arange(len(bins)), reread, color=colors, width=0.78)
    inset.set_xticks(np.arange(len(bins)))
    inset.set_xticklabels(bins, fontsize=5, rotation=90)
    inset.set_yticks([])
    inset.set_title("received after first-token read", fontsize=6, pad=2)
    inset.text(0.98, 0.90, f"100 → {fmap[100]}", transform=inset.transAxes,
               ha="right", va="top", fontsize=6, color="#A32D2D")
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.spines["left"].set_visible(False)
    inset.spines["bottom"].set_color("0.6")
    return float(vdist[100]), str(fmap[100])


def _draw_elicitation_failure(ax, cell_dir: Path, item_id: str):
    """Panel B: paired V+ and V- for one item under two legends."""
    rows = [("letter11-v1", cell_dir / "logprobs.pt", 1.0),
            ("letter11-permuted", Path(__file__).parent /
             "letter11_permuted_remote" / "llama3_8b_truthfulqa.pt", 0.0)]
    for label, path, y in rows:
        vp, vn = load_V_item(path, item_id)
        ax.hlines(y, 0, 1, color="0.82", lw=0.7)
        ax.plot(vp, y, "o", color="#534AB7", ms=5, zorder=3)
        ax.plot(vn, y, "s", color="#AFA9EC", ms=5, zorder=3)
        ax.text(-0.02, y, label, ha="right", va="center", fontsize=6)
        ax.text(vp, y + 0.14, f"V⁺ {vp:.2f}", ha="center", fontsize=5.5,
                color="#534AB7")
        ax.text(vn, y - 0.16, f"V⁻ {vn:.2f}", ha="center", fontsize=5.5,
                color="#534AB7")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.45, 1.45)
    ax.set_yticks([])
    ax.set_xlabel("V (denoted value)", fontsize=7)
    ax.set_title("B. Elicitation failure: same tokens, two legends", fontsize=8,
                 loc="left", pad=5)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels(["0", ".2", ".4", ".6", ".8", "1"], fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color="none",
                              markerfacecolor="#534AB7", markersize=5, label="V⁺"),
                       Line2D([0], [0], marker="s", color="none",
                              markerfacecolor="#AFA9EC", markersize=5, label="V⁻")],
              fontsize=6, frameon=False, loc="upper right")


def fig2(out: Path, item_id: str, cell_dir: Path):
    """Two real-cache failures, using one fixed Llama/TruthfulQA item."""
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(6.75, 2.9), gridspec_kw={"width_ratios": [1, 1]})
    p100, first_token = _draw_token_failure(ax_a, cell_dir, item_id)
    _draw_elicitation_failure(ax_b, cell_dir, item_id)
    fig.text(0.5, 0.015,
             "Llama 3 8B / TruthfulQA: no token failure, still shows B",
             ha="center", fontsize=7)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.86, bottom=0.23,
                        wspace=0.42)
    fig.savefig(out / "fig2_two_failures.pdf", bbox_inches="tight")
    plt.close(fig)
    (out / "fig2_two_failures_manifest.txt").write_text(
        f"item_id={item_id}\n"
        "selection=closest to median absolute delta V+ among fixed filtered set\n"
        f"numeric_P100={p100:.6f}\nfirst_split_token={first_token}\n"
        "panel_A_source=llama3_8b/truthfulqa/fine_conf.pt\n"
        "panel_B_source=canonical letter11-v1 plus letter11-permuted\n"
    )


def load_V(cache_path: Path, scheme: str):
    """Load V+ and V- from a cache and derive them from its Vdist rows."""
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    for key in ("Vdist_pos", "Vdist_neg", "conf_values", "example_ids"):
        if key not in data:
            raise KeyError(f"{cache_path}: missing {key}")
    pos = np.asarray(data["Vdist_pos"], dtype=float)
    neg = np.asarray(data["Vdist_neg"], dtype=float)
    levels = np.asarray(data["conf_values"], dtype=float).reshape(-1) / 100.0
    if pos.ndim != 2 or neg.ndim != 2 or pos.shape != neg.shape:
        raise ValueError(f"{cache_path}: incompatible Vdist shapes")
    if pos.shape[1] != len(levels):
        raise ValueError(f"{cache_path}: Vdist width does not match conf_values")
    if scheme == "letter11" and len(levels) != 11:
        raise ValueError(f"letter cache has {len(levels)} levels, expected 11")
    if scheme == "digits101" and len(levels) != 101:
        raise ValueError(f"numeric cache has {len(levels)} levels, expected 101")
    ids = np.asarray(data["example_ids"], dtype=object)
    if len(ids) != len(pos) or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{cache_path}: item IDs are missing or non-unique")
    return pos @ levels, neg @ levels, ids


def _align(ids_a, ids_b):
    index_b = {item_id: i for i, item_id in enumerate(ids_b)}
    if set(ids_a) != set(ids_b):
        raise ValueError("letter and numeric caches do not contain the same item IDs")
    return np.array([index_b[item_id] for item_id in ids_a], dtype=int)


def _rank(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def spearman(x, y):
    return float(np.corrcoef(_rank(x), _rank(y))[0, 1])


def panel(ax, x, y, title, xlabel, ylabel, color, band=None, stats_loc="br"):
    ax.scatter(x, y, s=4, alpha=0.35, color=color, linewidths=0)
    ax.plot([0, 1], [0, 1], lw=0.5, color="0.6")
    if band is not None:
        ax.axvspan(*band, color="#534AB7", alpha=0.08, lw=0)
        ax.text(np.mean(band), 0.04, "letter V⁺ 5–95%", fontsize=6,
                ha="center", color="#534AB7")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=8, loc="left", pad=4)
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6)
    pearson = float(np.corrcoef(x, y)[0, 1])
    if stats_loc == "tl":
        xy, ha, va = (0.03, 0.93), "left", "top"
    else:
        xy, ha, va = (0.97, 0.05), "right", "bottom"
    ax.text(*xy, f"ρ_S {spearman(x, y):.3f}\nρ_P {pearson:.3f}",
            fontsize=6, ha=ha, va=va, transform=ax.transAxes)


def fig3(out: Path, cell_dir: Path):
    letter_pos, letter_neg, letter_ids = load_V(
        cell_dir / "logprobs.pt", "letter11")
    digit_pos, digit_neg, digit_ids = load_V(
        cell_dir / "fine_conf.pt", "digits101")
    digit_order = _align(letter_ids, digit_ids)
    digit_pos, digit_neg = digit_pos[digit_order], digit_neg[digit_order]
    p05, p95 = np.percentile(letter_pos, [5, 95])
    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.6))
    panel(axes[0], letter_pos, letter_neg, "letter11", "V⁺", "V⁻", "#534AB7")
    panel(axes[1], digit_pos, digit_neg, "numeric101", "V⁺", "V⁻", "#0F6E56",
          stats_loc="tl")
    panel(axes[2], letter_pos, digit_pos, "same items, two formats",
          "V⁺ letter", "V⁺ numeric", "#993C1D", band=(p05, p95))
    fig.subplots_adjust(top=0.86, bottom=0.20, left=0.08, right=0.98,
                        wspace=0.35)
    fig.savefig(out / "format_panels.pdf")
    plt.close(fig)
    print("format Spearman:", spearman(letter_pos, digit_pos))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("figs"))
    parser.add_argument("--cell", type=Path,
                        default=ROOT / "outputs" / "step1_extract" /
                        "llama3_8b" / "truthfulqa")
    parser.add_argument("--fig2-item", default="d28dd71f5793f297")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig2(args.out, args.fig2_item, args.cell)
    fig3(args.out, args.cell)
