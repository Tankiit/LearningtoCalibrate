"""Compile every report and decision into one portable markdown file.

  python reports/compile_handoff.py      -> HANDOFF_AISTATS_P1.md (in mondrian/)

The repo's .gitignore keeps markdown out of git except an explicit whitelist; this one compiled
file is the single exception, so the reports and decisions travel with the repository.
Rerun after any report changes.
"""
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "HANDOFF_AISTATS_P1.md"
VAULT_DECISIONS = ROOT.parent / "verbalized_conformal_confidence/vault/wiki/decisions.md"

SECTIONS = [
    ("Registered abstract (binding scope)", ["reports/registered_abstract_2026-09-29.md"]),
    ("Status page", ["reports/AISTATS_P1_STATUS_2026-10-03.md"]),
    ("Experiment lock", ["reports/LOCK_2026-10-04.md"]),
    ("Editorial decisions (simulated review rounds 1-4)", [
        "reports/review/editorial_decision.md", "reports/rereview/editorial_decision_round2.md",
        "reports/review_round3/editorial_decision_round3.md", "reports/review_round4/editorial_decision_round4.md"]),
    ("Dated protocol notes", [
        "reports/estimand_note_2026-10-02.md", "reports/estimand_note_E123_2026-10-02.md",
        "reports/estimand_note_E5_2026-10-02.md", "reports/estimand_note_E8_2026-10-02.md",
        "reports/estimand_note_E9_popqa_2026-10-03.md", "reports/calibration_toolbox_note_2026-10-03.md",
        "reports/transport_controls_note_2026-10-03.md", "reports/postlock_analyses_note_2026-10-05.md"]),
    ("Theory, novelty and audits", [
        "reports/theory.md", "reports/lemma_diameter.md", "reports/novelty_check.md", "reports/transport_prior_work.md",
        "reports/related_work_bib_report.md", "reports/claim_audit_2026-10-02.md", "reports/implementation_audit.md",
        "reports/arr_overlap.md", "reports/archive_README.md"]),
    ("Reviewer seat reports", [f"reports/{d}/{s}.md" for d in ("review", "rereview", "review_round3", "review_round4")
                               for s in ("eic", "methodology", "domain", "perspective", "devils_advocate")]),
    ("All results (generated from run outputs)", ["reports/README_RESULTS.md"]),
]

HANDOFF = """## 0. Handoff: where things stand and how to resume

**Paper.** *How Finely to Group? Learned Partitions for Calibrating LLM Confidence across Elicitation Procedures*,
AISTATS 2027. The full paper is due **6 Oct 2026, 23:59 AoE**.

### State
- **Source of truth:** `paper/main.tex` (the draft).
  - Every body element carries a `% WHY:` comment saying why it is there and where it comes from.
  - Every result carries a `\\todo` naming its command and output file.
- **Submission:** `paper/aistats2027_submission.tex`, built from the draft, with WHY comments and todos stripped.
  - PDF: `paper/aistats2027_submission.pdf`.
  - The body ends on page 8.
- **Experiments are locked** at source `32d32a34ede1` (`runs/run_all_manifest.json`, 32/32 ok). Do not edit `mondrian/`, `models/`, `data/`, `experiments/` or `tests/`.
- **Post-lock analyses:** `analyses/`, with outputs in `runs_postlock/`. These are partition efficiency with a fixed score and correctness-probe baselines.
- **Figures:**
  - Figure 1 is the TikZ schematic `paper/figs/aistats_main_fig.tex`.
  - Figure 2 is `paper/figs/fig_overview` (`python paper/make_figs.py`).
  - The appendix transport tables come from `python paper/gen_transport_tables.py`.
- **Reviews:** four simulated rounds (sections 5 and 8 below). Round 4 is Minor Revision, and all its required items are applied.

### Resume
```sh
cd mondrian
python -m experiments.run_all --popqa --check   # lock intact: expect 0 job(s) not current
python paper/build_submission.py                 # build the AISTATS .tex from main.tex
cd paper && tectonic aistats2027_submission.tex  # compile the PDF; also: tectonic main.tex (draft, todos visible)
python reports/compile_results.py                # regenerate reports/README_RESULTS.md
python reports/compile_handoff.py                # regenerate this file
```

### Open items for the author
- **AI-use statement.** Write it accurately. It is a desk-reject item, placed immediately before the references.
- **Bibliography corrections.** These are listed in `reports/related_work_bib.bib` (CORRECTIONS). Confirm the venue and year of `moreno2025correctness`.
- **Title.** Confirm it matches the registered title word for word.
- **Abstract.** It is now the author's rewrite (174 words). If OpenReview can be updated, use the same text there. Otherwise consider the registered wording with only the findings swapped in.
- **Repository visibility.** The repository is public and review is double-blind. Check the AISTATS preprint and anonymity policy.
- **Optional, not done:**
  - a PopQA relation-category control;
  - more than 16 groups;
  - a learned non-monotone transport map (stated as future work).

### Git
- **Branch:** `aistats-2027-clean`, pushed to `origin/aistats-2027`.
- **Not pushed:** the old local branch `aistats-2027`, which carries large `outputs/*.pt` history.
- **Excluded from git on purpose:**
  - `.claude/`, the local copy of the reviewer skill;
  - all markdown except whitelisted files. This compiled file is the one exception.

### Elsewhere
- **Vault:** `verbalized_conformal_confidence/vault/`. `Home.md` has the AISTATS section; `wiki/decisions.md` is the dated log, whose AISTATS entries are copied into section 1 below.
- **External code:** cloned into `../third_party/` and not committed:
  - calibration-toolbox @ 1c6f60b;
  - correctness-model-internals @ aa7270c.
"""


def demote(text, by=2):
    out, fence = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
        if not fence and re.match(r"^#{1,6} ", line):
            hashes = len(line) - len(line.lstrip("#"))
            line = "#" * min(6, hashes + by) + line[hashes:]
        out.append(line)
    return "\n".join(out)


def anchor(title):
    return re.sub(r"[^a-z0-9 -]", "", title.lower()).strip().replace(" ", "-")


def main():
    commit = subprocess.run(["git", "log", "-1", "--format=%h %s"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    parts = ["# AISTATS 2027 P1: reports and decisions (compiled)", "",
             f"Compiled {time.strftime('%Y-%m-%d %H:%M')} by `python reports/compile_handoff.py`; last commit: `{commit}`.",
             "This single file is the git-tracked exception to the repo's markdown rule, so it can be carried to another machine.",
             "", "**Contents**", "", "0. [Handoff](#0-handoff-where-things-stand-and-how-to-resume)"]
    titles = ["Decisions log (dated, from the vault)"] + [t for t, _ in SECTIONS]
    for k, t in enumerate(titles, 1):
        parts.append(f"{k}. [{t}](#{k}-{anchor(t)})")
    parts += ["", HANDOFF]
    # decisions log: AISTATS entries from the vault
    parts.append("## 1. Decisions log (dated, from the vault)\n")
    if VAULT_DECISIONS.exists():
        d = VAULT_DECISIONS.read_text()
        start = d.find("- [x] **1–2 Oct AISTATS P1")
        parts.append(demote(d[start:] if start >= 0 else d).replace("](../mondrian/", "]("))
    else:
        parts.append("_Vault decisions file not found on this machine._")
    for k, (title, files) in enumerate(SECTIONS, 2):
        parts.append(f"\n## {k}. {title}\n")
        for rel in files:
            p = ROOT / rel
            parts.append(f"\n### `{rel}`\n")
            parts.append(demote(p.read_text(), by=3) if p.exists() else "_missing_")
    OUT.write_text("\n".join(parts) + "\n")
    print("wrote", OUT, f"{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
