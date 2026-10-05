"""Regenerate the two Label-Free-Transfer tables in paper/main.tex from the run outputs.

  python paper/gen_transport_tables.py

Reads runs/p1_e8/e8_result.json (TruthfulQA frozen models, TriviaQA reporters) and
runs/p1_e9/e9c_e8_*.json (PopQA-Elicit). Replaces the tables whose header starts with
"A $\\to$ B & same procedure" (summary) and "A $\\to$ B & map & coverage" (full) in place.
Lives in paper/, outside the hashed source folders.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "paper" / "main.tex"
NAME = {"llama3_8b": "Llama", "mistral_7b": "Mistral", "qwen2_5_7b": "Qwen"}
SHIFT = {"rev": "reversed", "alt": "reworded"}
ARMS = [("A->A", "same"), ("naive", "naive"), ("ot_quantile", "quantile"), ("paired_iso", "paired iso."),
        ("paired_iso_group", "paired iso., groups"), ("recal_B_k30", "labels 15+15"), ("recal_B_k120", "labels 60+60")]


def load():
    e8 = json.loads((ROOT / "runs/p1_e8/e8_result.json").read_text())
    e9 = json.loads((ROOT / "runs/p1_e9/e9c_e8_llama3_8b_mistral_7b_qwen2_5_7b.json").read_text())
    return e8, e9


def cs(a, k):
    return f"{a[k]['coverage']:.3f} ({a[k]['set_size']:.2f})"


def summary(e8):
    rows = []
    for m in NAME:
        for B in ("rev", "alt"):
            a = e8["frozen"][m][B]
            rows.append(f"{NAME[m]}, fwd$\\to${SHIFT[B]} & {cs(a, 'A->A')} & {a['naive']['coverage']:.3f} & {cs(a, 'ot_quantile')} & "
                        f"{cs(a, 'paired_iso')} & {cs(a, 'recal_B_k30')} & {cs(a, 'recal_B_k120')}\\\\")
    for arm, lab in (("q", "TriviaQA question-only, normal$\\to$reversed"), ("qa", "TriviaQA answer-aware, normal$\\to$reversed")):
        a = e8["tinker"][arm]
        rows.append(f"{lab} & {cs(a, 'A->A')} & {a['naive']['coverage']:.3f} & {cs(a, 'ot_quantile')} & {cs(a, 'paired_iso')} & "
                    f"{cs(a, 'recal_B_k30')} & {cs(a, 'recal_B_k120')}\\\\")
    return ("\\begin{center}\\small\n\\begin{tabular}{lcccccc}\n\\toprule\n"
            "A $\\to$ B & same procedure & naive & quantile map & paired isotonic & labels $15{+}15$ & labels $60{+}60$\\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{center}")


def full(e8, e9):
    blocks = []
    for data, R in (("TruthfulQA", e8["frozen"]), ("PopQA", e9)):
        for m in NAME:
            for B in ("rev", "alt"):
                a = R[m][B]; base = a["A->A"]["cell_gap_min"]; rows = []
                for k, lab in ARMS:
                    v = a[k]
                    rows.append(f"{data} {NAME[m]} fwd$\\to${SHIFT[B]} & {lab} & {v['coverage']:.3f} & "
                                f"[{v['coverage_q10_q90'][0]:.2f}, {v['coverage_q10_q90'][1]:.2f}] & {v['set_size']:.2f} & "
                                f"{v['brier']:.4f} & {v['crossing_mass']:.3f} & {v['label_loss_mass']:.3f} & "
                                f"{v['cell_gap_min'] - base:+.3f}\\\\")
                blocks.append("\n".join(rows))
    for arm, lab in (("q", "TriviaQA question-only"), ("qa", "TriviaQA answer-aware")):
        a = e8["tinker"][arm]; rows = []
        for k, l in ARMS:
            if k not in a or k == "paired_iso_group":
                continue
            v = a[k]; ci = v.get("coverage_ci95")
            rows.append(f"{lab} & {l} & {v['coverage']:.3f} & {'[%.2f, %.2f]' % tuple(ci) if ci else '--'} & {v['set_size']:.2f} & "
                        f"{v['brier']:.4f} & {v['crossing_mass']:.3f} & {v['label_loss_mass']:.3f} & --\\\\")
        blocks.append("\n".join(rows))
    return ("\\begin{center}\\small\n\\begin{tabular}{llccccccc}\n\\toprule\n"
            "A $\\to$ B & map & coverage & 10--90\\% (frozen) / 95\\% CI (TriviaQA) & set size & Brier & crossing mass & "
            "label-loss mass & worst group, net of A$\\to$A\\\\\n\\midrule\n" + "\n\\midrule\n".join(blocks) +
            "\n\\bottomrule\n\\end{tabular}\n\\end{center}")


def replace(s, header_start, new):
    pat = re.compile(r"\\begin\{center\}\\small\n\\begin\{tabular\}\{[^}]*\}\n\\toprule\n" + re.escape(header_start) + r".*?\\end\{center\}", re.S)
    hits = pat.findall(s)
    assert len(hits) == 1, (header_start, len(hits))
    return pat.sub(lambda _: new, s)


def main():
    e8, e9 = load()
    s = TEX.read_text()
    s = replace(s, "A $\\to$ B & same procedure", summary(e8))
    s = replace(s, "A $\\to$ B & map & coverage", full(e8, e9))
    TEX.write_text(s)
    print("tables regenerated")


if __name__ == "__main__":
    main()
