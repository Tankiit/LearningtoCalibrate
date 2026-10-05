"""Build the AISTATS 2027 submission file from the working draft.

    python paper/build_submission.py      (from the repo root)

Reads paper/main.tex (working draft with status tags) and the AISTATS template preamble, writes
paper/aistats2027_submission.tex. The script strips status tags and notes, drops internal
sections, converts tables and wide figures to two-column floats, and appends the AI-use
statement, the checklist and the single-column supplement. Rerun it after editing main.tex.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC, OUT = ROOT / "main.tex", ROOT / "aistats2027_submission.tex"

TABLE_CAPTIONS = [   # (substring of the table's header row, caption)
    ("Stage & Known", "Stages of information in an LLM pipeline, with example stratifiers."),
    ("same elicitation & legend reversed", "Coverage on three frozen models when the elicitation changes between "
     "calibration and deployment, on TruthfulQA (817 questions) and \\expname{PopQA-Elicit} (7{,}189 questions); 500 random splits, $M=4$."),
    ("labels & model & G$_4-$R", "\\expname{PopQA-Elicit}: Brier differences against procedure-specific recalibration "
     "($\\times10^3$; below zero: better), averaged over the three procedures, with Nadeau--Bengio-corrected 95\\% "
     "intervals over 100 random splits."),
    ("data (calibration points) & partition", "Post-hoc partition-efficiency check: one correctness model per split is "
     "held fixed (R or D) and only the partition changes; mean set size, fallback mass (test mass in cells below the floor) "
     "and worst-cell coverage, ranges over the three models, 100 splits."),
    ("data & model & P-full", "Post-hoc correctness-probe baselines: Brier difference against recalibrating the report "
     "($\\times10^3$; below zero: better) at the largest budget, with Nadeau--Bengio-corrected 95\\% intervals for P-full."),
    ("forecaster & model & tuning", "Forecasters and hyperparameters (\\expname{Granularity})."),
    ("experiment & identifier in notes", "Experiment names and the identifiers used in the dated notes and run directories."),
    ("A $\\to$ B & map & coverage", "Full \\expname{Label-Free-Transfer} results on TruthfulQA (500 splits), \\expname{PopQA-Elicit} (100 splits) and TriviaQA "
     "(one split): coverage at $\\alpha=0.1$, its spread, mean set size, Brier score of the mapped forecaster, crossing mass "
     "(two-sided bound on the coverage change), label-loss mass (one-sided bound; Proposition~\\ref{prop:transport}(a)), and the "
     "worst of the four frozen groups' coverage gaps net of the same statistic for A$\\to$A (which is below zero "
     "without any shift). Forecaster and thresholds use disjoint halves of the labelled fold."),
    ("A $\\to$ B & same procedure & naive & quantile", "\\expname{Label-Free-Transfer}: coverage ($\\alpha=0.1$; mean set size in parentheses) when a "
     "calibration fitted under procedure A is carried to B by a label-free map, against recalibrating under B with "
     "$k$ labels (half to fit, half to set the threshold)."),
    ("$p$ & 0.24", "X4: coverage on the full test stream by stratifier and answerability rate $p$ (50 seeds; worst "
     "bin is the seed mean of the worst bin)."),
    ("arm & coverage & cell MSCE", "X2: Communities and Crime, exemplar cells against one global cell and state "
     "strata (8 seeds)."),
    ("signal & stage & Brier", "E1--E3: binned recalibration on real data (pre-set split)."),
    ("conformal arm", "E1--E3 and E2: conformal coverage by stratifier stage ($\\alpha=0.1$, $M=8$)."),
    ("analysis & result & key numbers", "Sensitivity analyses with re-randomised calibration/test splits (post hoc)."),
    ("Brier difference vs.\\ R", "Pre-set E5 analysis (superseded): Brier difference against R with naive "
     "intervals, fixed penalty. Bold: worse than R."),
]


def strip_macro(s, name):
    """Remove \\name{...} with balanced braces (and a following space)."""
    out, i, pat = [], 0, "\\" + name + "{"
    while True:
        j = s.find(pat, i)
        if j < 0:
            out.append(s[i:]); break
        out.append(s[i:j]); k, depth = j + len(pat), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(s[k], 0); k += 1
        while k < len(s) and s[k] == " ":
            k += 1
        i = k
    return "".join(out)


def section(s, start, end):
    a = s.index(start); b = s.index(end, a) if end else len(s)
    return s[a:b]


def clean(body):
    # internal justification comments (% WHY: ...) stay in main.tex only
    body = re.sub(r"(?m)^% WHY:.*\n", "", body)
    for m in ("NOTE", "GATED", "HOLD"):
        body = strip_macro(body, m)
    body = re.sub(r"\\todo\[[^\]]*\]\{", r"\\todo{", body)   # drop options, then strip with braces
    body = strip_macro(body, "todo")
    body = body.replace("\\ifdraft\\listoftodos\\fi\n", "")
    body = re.sub(r"\\(PROPOSED|DECIDED|REVIEW)(\\ |\{\}| )?", "", body)
    # internal paragraphs
    body = re.sub(r"\\paragraph\{One-sentence story[^\n]*\n(?:.+\n)+?\n", "", body)
    body = re.sub(r"\\paragraph\{X5: LLM cache\.\}(?:.+\n)+?\n", "", body)
    body = body.replace(r"\section{Introduction: the retrieval problem}", r"\section{Introduction}")
    # residual-cell proposition with no statement -> remark
    body = re.sub(r"\\begin\{proposition\}\[Residual cell; Claim D\]\n\\label\{prop:D\}\n(?:.*\n)*?\\end\{proposition\}",
                  r"\\begin{remark}[Residual cell]\n\\label{prop:D}\nEvery query is assigned: cell 0 collects "
                  r"queries outside every exemplar radius fitted on $\\Dfit$. Distribution-free control of the "
                  r"assignment rate follows from conformal outlier tests \\citep{bates2023testing,guan2022bcops}; "
                  r"we claim nothing inside cell 0.\n\\end{remark}", body)
    # theorem titles: drop internal claim labels
    body = re.sub(r"(\\begin\{(?:theorem|proposition|corollary|conjecture|example|lemma)\}\[[^\]]*?);\s*Claim [^\]]*\]",
                  r"\1]", body)
    # pointers to local files
    body = re.sub(r"\s*Runs: \\texttt\{runs[^.]*\.", "", body)
    body = re.sub(r"\(Criteria: \\texttt\{reports/[^}]*\}, fixed before the run\.\)",
                  "(Criteria fixed in a dated protocol note before the run; supplementary material.)", body)
    body = re.sub(r"\(\\texttt\{reports/[^}]*\}\)", "(supplementary material)", body)
    body = re.sub(r"\\texttt\{reports/[^}]*\}", "a dated protocol note (supplementary material)", body)
    body = re.sub(r"\(\\texttt\{runs/[^}]*\};\s*", "(", body)
    body = re.sub(r"\\texttt\{runs/[^}]*\}", "", body)
    body = re.sub(r"Addendum~[A-D]", "a dated addendum", body)
    body = body.replace("Addendum A of the E1--E3 note", "an addendum to the E1--E3 note")
    body = re.sub(r"\\paragraph\{([^}]*?) \(Claim [A-G]\)\.\}", r"\\paragraph{\1.}", body)
    body = body.replace("Full proof:\n, to be moved to Appendix~\\ref{app:lemma}.", "Full proof in Appendix~\\ref{app:thmB}.")
    body = re.sub(r"Full proof:\s*\S*\s*to be moved to\s*Appendix~\\ref\{app:lemma\}\.",
                  r"Full proof in Appendix~\\ref{app:thmB}.", body)
    # tables -> table* with captions
    def table(m):
        tab = m.group(1)
        header = tab.split("\\toprule", 1)[1].strip().split("\n", 1)[0] if "\\toprule" in tab else ""
        cap = next((c for key, c in TABLE_CAPTIONS if key in header), None)
        if cap is None:
            raise ValueError("No caption for table with header: " + header[:80])
        return ("\\begin{table*}[t]\n\\caption{" + cap + "}\n\\centering\\small\n"
                + "\\begin{adjustbox}{max width=\\textwidth}\n" + tab.replace("\\small", "").strip()
                + "\n\\end{adjustbox}\n\\end{table*}")
    body = re.sub(r"\\begin\{center\}\\small\n?(.*?)\\end\{center\}|\\begin\{center\}\n(\\begin\{tabular\}.*?)\\end\{center\}",
                  lambda m: table(type("M", (), {"group": lambda self, i: m.group(1) or m.group(2)})()),
                  body, flags=re.S)
    # wide figures
    body = body.replace("\\begin{figure}[t]", "\\begin{figure*}[t]").replace("\\end{figure}", "\\end{figure*}")
    body = body.replace("\\includegraphics[width=\\linewidth]", "\\includegraphics[width=\\textwidth]")
    body = re.sub(r"(\S) +([.:;,])(?=\s)", r"\1\2", body)   # space left where a tag was stripped
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body


THM_B_PROOF = r"""
\section{Proof of Theorem~\ref{thm:B}}
\label{app:thmB}
Work on the event $\Omega_n$ of Lemma~\ref{lem:diam}; off it $R_{\mathrm{pt}}\le1$, which gives the last
term. \emph{Quantisation.} For $x$ in cell $j$, $|h(x)-C_j|\le\sup_{x'\in V_j}\sup_s|F_x(s)-F_{x'}(s)|\le
L\,\diam_j^{\beta}\le L K^{\beta}M^{-\beta/d}$, uniformly in the realised threshold. \emph{Cell masses.}
Exemplars are at least $r_M$ apart (Step~1 of Lemma~\ref{lem:diam}), so each cell contains
$B(e_j,r_M/2)\cap S$; with the density bounds and $M\le m'(n_{\mathrm{fit}})$ this gives $p_j\ge b/M$
with $b=c\kappa/(4^d\bar c)$. \emph{Variance.} Conditional on $\Dfit$, Corollary~\ref{prop:B0} gives
$\mathrm{Var}(C_j)\le t(1-t)/(n_j+2)+1/((n_j+1)(n_j+2))$ above the floor; with
$n_j\sim\mathrm{Bin}(n_{\mathrm{cal}},p_j)$ and $\E[1/(n_j+1)]\le1/((n_{\mathrm{cal}}+1)p_j)$, summing
$p_j\cdot$ over cells gives $t(1-t)M/(n_{\mathrm{cal}}+1)$ without any balance assumption.
\emph{Rounding.} $(k_j/(n_j+1)-t)^2\le(n_j+1)^{-2}$ and
$\E[1/((n_j+1)(n_j+2))]\le1/((n_{\mathrm{cal}}+1)(n_{\mathrm{cal}}+2)p_j^2)$ with $p_j\ge b/M$ give the
third term. \emph{Floor.} By a Chernoff bound $\P(n_j<n_{\mathrm{cal}}p_j/2)\le e^{-n_{\mathrm{cal}}p_j/8}$;
when $n_{\min}\le n_{\mathrm{cal}}p_j/2$ a below-floor cell has probability at most
$e^{-b n_{\mathrm{cal}}/(8M)}$ and costs at most $\alpha^2$ per unit mass. \hfill$\square$

"""

AI_STATEMENT = r"""
\subsection*{AI use statement}
In this work, we used generative AI tools to write and test analysis code, to run and summarise
experiments, to search and summarise related literature, to draft and revise text, and to draft
proofs. Criteria for most experiments were fixed in dated notes before they ran (exceptions are
listed in Appendix~\ref{app:protocol}), and every result reported was produced by code that we
reviewed. All AI-assisted proofs were checked by the
authors, and all citations were verified against their sources. We take responsibility for the
final content of this work, including text, claims, or artifacts produced with the aid of
generative AI.
"""

CHECKLIST = r"""
\section*{Checklist}
\begin{enumerate}
  \item For all models and algorithms presented, check if you include:
  \begin{enumerate}
    \item A clear description of the mathematical setting, assumptions, algorithm, and/or model. Yes (Sections~\ref{sec:setup}--\ref{sec:decomp}).
    \item An analysis of the properties and complexity (time, space, sample size) of any algorithm. Yes: sample-size dependence in Theorem~\ref{thm:B}; farthest-first traversal costs $O(Mn_{\mathrm{fit}})$ distance evaluations.
    \item (Optional) Anonymized source code, with specification of all dependencies, including external libraries. Yes (supplementary material).
  \end{enumerate}
  \item For any theoretical claim, check if you include:
  \begin{enumerate}
    \item Statements of the full set of assumptions of all theoretical results. Yes.
    \item Complete proofs of all theoretical results. Yes (main text and Appendices~\ref{app:lemma}--\ref{app:thmB}); Conjecture~\ref{conj:B} is stated as a conjecture.
    \item Clear explanations of any assumptions. Yes.
  \end{enumerate}
  \item For all figures and tables that present empirical results, check if you include:
  \begin{enumerate}
    \item The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL). Yes (supplementary material).
    \item All the training details (e.g., data splits, hyperparameters, how they were chosen). Yes (Section~\ref{sec:experiments} and the dated protocol notes).
    \item A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times). Yes.
    \item A description of the computing infrastructure used. Yes: all analyses run on a single laptop CPU from cached model outputs.
  \end{enumerate}
  \item If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include:
  \begin{enumerate}
    \item Citations of the creator If your work uses existing assets. Yes.
    \item The license information of the assets, if applicable. Yes (Communities and Crime: CC BY 4.0; model licences as published).
    \item New assets either in the supplemental material or as a URL, if applicable. Not Applicable.
    \item Information about consent from data providers/curators. Not Applicable.
    \item Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. Not Applicable.
  \end{enumerate}
  \item If you used crowdsourcing or conducted research with human subjects, check if you include:
  \begin{enumerate}
    \item The full text of instructions given to participants and screenshots. Not Applicable.
    \item Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. Not Applicable.
    \item The estimated hourly wage paid to participants and the total amount spent on participant compensation. Not Applicable.
  \end{enumerate}
\end{enumerate}
"""


def main():
    src = SRC.read_text()
    title = re.search(r"\\title\{(.*?)\}\n", src, re.S).group(1).replace("\\\\", " ")
    abstract = clean(section(src, r"\begin{abstract}", r"\end{abstract}"))
    abstract = re.sub(r"^\\begin\{abstract\}\n(%.*\n)*", "", abstract).strip()
    body = clean(section(src, r"\section{Introduction", r"\appendix"))
    app = clean(section(src, r"\appendix", r"\ifdraft" + "\n" + r"\section{Status ledger"))
    app = app.replace(r"\appendix", "").strip()
    # mathematical preamble from the draft (macros, theorem environments)
    pre = section(src, r"\newtheorem{theorem}", r"% Title keeps")
    pre += section(src, r"\newcommand{\Dfit}", r"% Title keeps") if r"\newcommand{\Dfit}" not in pre else ""
    preamble = (r"""\documentclass[twoside]{article}
\usepackage{aistats2027}
\usepackage[round]{natbib}
\renewcommand{\bibname}{References}
\renewcommand{\bibsection}{\subsubsection*{\bibname}}
\usepackage{amsmath,amssymb,amsthm,mathtools}
\usepackage{booktabs,graphicx,xcolor,adjustbox}
\usepackage{tikz}
\usetikzlibrary{positioning,arrows.meta,fit,calc}
\usepackage[hidelinks]{hyperref}
""" + pre)
    doc = (preamble + "\n\\begin{document}\n\n\\twocolumn[\n\n\\aistatstitle{" + title + "}\n\n"
           "\\aistatsauthor{ Anonymous Author(s) }\n\n\\aistatsaddress{ Anonymous Institution(s) } ]\n\n"
           "\\begin{abstract}\n" + abstract + "\n\\end{abstract}\n\n" + body.strip() + "\n\n"
           + AI_STATEMENT + "\n\\bibliographystyle{apalike}\n\\bibliography{refs}\n\n" + CHECKLIST
           + "\n\\clearpage\n\\appendix\n\\thispagestyle{empty}\n\\onecolumn\n\\aistatstitle{" + title
           + ":\\\\Supplementary Materials}\n\n" + app + "\n\\end{document}\n")
    OUT.write_text(doc)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
