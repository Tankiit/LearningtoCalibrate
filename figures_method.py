"""Build the method figures from tokenizer metadata and cached readouts.

Figure 2 shows which intended numeric values share a first-token identity.
Figure 3 shows the paired Llama/TruthfulQA confidence readouts.  The script
fails closed when a cache schema or item-id alignment is not what it expects.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import os
import subprocess
from html import escape

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
    fresh = Path(__file__).parent / "letter11_provenance_v1" / "llama3_8b" / "truthfulqa"
    rows = [("letter11-v1 (fresh)", fresh / "forward.pt", 1.0),
            ("letter11-reversed (fresh)", fresh / "reversed.pt", 0.0)]
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


def _svg_text(x, y, text, size=8, fill="#222", anchor="start", weight="400"):
    return (f'<text x="{x}" y="{y}" font-family="Helvetica,Arial,sans-serif" '
            f'font-size="{size}px" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}">{escape(str(text))}</text>')


def write_fig2_svg(out: Path, vdist101, fmap, v1, perm, meta):
    """Write the two-failure figure as a data-driven, converter-free SVG."""
    W, H = 680, 340
    x0, x1, base = 40, 200, 180
    scale = 74.0 / float(np.max(vdist101))
    inverse = collisions(fmap)
    bins = sorted(inverse, key=lambda token: min(inverse[token]))
    reread = [sum(float(vdist101[v]) for v in inverse[token]) for token in bins]
    spikes = "".join(
        f'<rect x="{x0 + (x1-x0)*v/100:.2f}" y="{base-m*scale:.2f}" '
        f'width="1.45" height="{m*scale:.2f}" fill="{"#D85A30" if v == 100 else "#888780"}"/>'
        for v, m in enumerate(vdist101) if m > 0.002
    )
    rx0, rw = 254, 70
    bars = "".join(
        f'<rect x="{rx0 + rw*i/len(bins)+0.8:.2f}" y="{base-m*scale:.2f}" '
        f'width="{rw/len(bins)-1.5:.2f}" height="{m*scale:.2f}" '
        f'fill="{"#D85A30" if token == fmap[100] else "#888780"}"/>'
        for i, (token, m) in enumerate(zip(bins, reread))
    )
    b_x0, b_x1 = 370, 640
    def marker(value, y, kind):
        x = b_x0 + (b_x1-b_x0)*float(value)
        if kind == "p":
            return f'<circle cx="{x:.1f}" cy="{y}" r="5" fill="#534AB7"/>'
        return f'<rect x="{x-5:.1f}" y="{y-5}" width="10" height="10" fill="#AFA9EC"/>'
    railB = (f'<line x1="{b_x0}" y1="92" x2="{b_x1}" y2="92" stroke="#C8C7C2"/>'
             f'<line x1="{b_x0}" y1="172" x2="{b_x1}" y2="172" stroke="#C8C7C2"/>'
             + marker(v1["Vp"], 92, "p") + marker(v1["Vn"], 92, "n")
             + marker(perm["Vp"], 172, "p") + marker(perm["Vn"], 172, "n")
             )
    tick_lines = "".join(
        f'<line x1="{b_x0+270*t:.1f}" y1="178" x2="{b_x0+270*t:.1f}" y2="183" stroke="#555"/>'
        f'{_svg_text(b_x0+270*t, 195, f"{t:.1f}".replace("0.", "."), 6, "#444", "middle")}'
        for t in np.linspace(0, 1, 6)
    )
    # Fresh provenance-controlled population footer.  The grey marks are the
    # ID-aligned alternate-form references; the orange dots are raw paired
    # forward/reversed ordering correlations.
    footer_x0, footer_x1, footer_y = 370, 640, 286
    footer_axis = (f'<line x1="{footer_x0}" y1="{footer_y}" '
                   f'x2="{footer_x1}" y2="{footer_y}" stroke="#555" stroke-width="0.7"/>')
    footer_ticks = "".join(
        f'<line x1="{footer_x0 + (footer_x1-footer_x0)*(t+1)/2:.1f}" '
        f'y1="{footer_y-3}" x2="{footer_x0 + (footer_x1-footer_x0)*(t+1)/2:.1f}" '
        f'y2="{footer_y+3}" stroke="#555"/>'
        f'{_svg_text(footer_x0 + (footer_x1-footer_x0)*(t+1)/2, footer_y+13, f"{t:g}", 6, "#444", "middle")}'
        for t in (-1, 0, 1)
    )
    footer_marks = ""
    for i, row in enumerate(meta.get("rho_rows", [])):
        x_rho = footer_x0 + (footer_x1-footer_x0) * (float(row["rho"]) + 1) / 2
        x_rel = footer_x0 + (footer_x1-footer_x0) * (float(row["rel_fwd"]) + 1) / 2
        y = footer_y - 8 - (i % 2) * 12
        footer_marks += (f'<line x1="{x_rel:.1f}" y1="{y-4}" x2="{x_rel:.1f}" y2="{y+4}" '
                         f'stroke="#8A8984" stroke-width="1"/>'
                         f'<circle cx="{x_rho:.1f}" cy="{y}" r="3.2" fill="#BB632B"/>')
    footer_labels = "".join(
        _svg_text(footer_x1, footer_y-19, str(row["label"]), 5.2, "#666", "end")
        for row in meta.get("rho_rows", [])
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
<defs><marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 z" fill="#D85A30"/></marker></defs>
<rect width="100%" height="100%" fill="white"/>
{_svg_text(40, 24, "A", 11, weight="500")}
{_svg_text(370, 24, "B", 11, weight="500")}
<line x1="{x0}" y1="{base}" x2="{x1}" y2="{base}" stroke="#555" stroke-width="0.7"/>
<line x1="{rx0}" y1="{base}" x2="{rx0+rw}" y2="{base}" stroke="#555" stroke-width="0.7"/>
{spikes}{bars}
{_svg_text(40, 206, "0", 6, "#555", "middle")}{_svg_text(120, 206, "50", 6, "#555", "middle")}{_svg_text(200, 206, "100", 6, "#555", "middle")}
{_svg_text(120, 220, "intended value", 7, "#444", "middle")}
{_svg_text(289, 220, "first-token bin", 7, "#444", "middle")}
{railB}
<line x1="{b_x0}" y1="180" x2="{b_x1}" y2="180" stroke="#555" stroke-width="0.7"/>{tick_lines}
{_svg_text(505, 216, "V (denoted value)", 7, "#444", "middle")}
{_svg_text(370, 248, "fresh across-item ordering: forward vs reversed legend", 7, "#444")}
{footer_axis}{footer_ticks}{footer_marks}{footer_labels}
{_svg_text(505, 322, "orange = rho; grey tick = rel_fwd", 6, "#666", "middle")}
</svg>'''
    (out / "fig2_two_failures.svg").write_text(svg)
    pdf_path = out / "fig2_two_failures.pdf"
    try:
        subprocess.run(["convert", "-font", "/System/Library/Fonts/Supplemental/Arial.ttf",
                        "-background", "white", str(out / "fig2_two_failures.svg"),
                        str(pdf_path)], check=True, capture_output=True, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # The SVG is canonical; keep the previous PDF only if conversion is
        # unavailable so LaTeX users can still inspect the vector source.
        pass
    (out / "fig2_two_failures_manifest.txt").write_text(
        f"item_id={meta['item_id']}\n"
        "selection=closest to median absolute delta V+ among fixed filtered set\n"
        f"numeric_P100={meta['P100']:.6f}\nfirst_split_token={fmap[100]}\n"
        f"suppressed_mass_below_0.002={meta['suppressed_mass']:.6f}\n"
        "panel_A_source=llama3_8b/truthfulqa/fine_conf.pt\n"
        "panel_B_source=letter11_provenance_v1 forward plus reversed (fresh, pinned)\n"
        "panel_B_status=requires locked item-selection filter before rendering\n"
        "population_rho=.123,.169,.058,.782,.340,.125\n"
    )


def fig2(out: Path, item_id: str, cell_dir: Path):
    """Two real-cache failures, using one fixed Llama/TruthfulQA item."""
    tok = AutoTokenizer.from_pretrained(
        _tokenizer_source("mistral_7b", MODELS["mistral_7b"]), local_files_only=True)
    fmap = first_token_map(tok, range(101))
    qwen_tok = AutoTokenizer.from_pretrained(
        _tokenizer_source("qwen2_5_7b", MODELS["qwen2_5_7b"]), local_files_only=True)
    if fmap != first_token_map(qwen_tok, range(101)):
        raise RuntimeError("Mistral and Qwen split maps differ; figure assumption failed")
    vdist, _ = load_vdist_item(cell_dir / "fine_conf.pt", item_id, "pos")
    fresh_dir = Path(__file__).parent / "letter11_provenance_v1" / "llama3_8b" / "truthfulqa"
    v1p, v1n = load_V_item(fresh_dir / "forward.pt", item_id)
    permp, permn = load_V_item(fresh_dir / "reversed.pt", item_id)
    suppressed = float(vdist[vdist <= 0.002].sum())
    rows = []
    with (Path(__file__).parent / "cached_results" / "letter11_provenance_v1" / "summary.csv").open() as fh:
        for row in csv.DictReader(fh):
            rows.append({
                "label": f"{row['model']}/{row['dataset']}",
                "rho": float(row["rho_forward_reversed"]),
                "rel_fwd": float(row["rel_fwd"]),
            })
    write_fig2_svg(out, vdist, fmap, {"Vp": v1p, "Vn": v1n},
                   {"Vp": permp, "Vn": permn},
                   {"item_id": item_id, "P100": float(vdist[100]),
                    "suppressed_mass": suppressed, "rho_rows": rows})


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
    parser.add_argument("--fig2-item",
                        help="item selected by the locked pre-render filter")
    parser.add_argument("--fig3-only", action="store_true",
                        help="regenerate Figure 3 without selecting Figure 2's item")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.fig3_only:
        if not args.fig2_item:
            parser.error("--fig2-item is required unless --fig3-only is used")
        fig2(args.out, args.fig2_item, args.cell)
    fig3(args.out, args.cell)
