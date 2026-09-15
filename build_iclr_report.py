"""Build the complete local ICLR evidence ledger without changing any result cache."""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
U='tinker_uq/runs/'
F=U+'scale_family_v1/'
sources={};parts=[]


def track(path):
    p=ROOT/path
    sources.setdefault(path,{}).update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    return p
def rows(path):
    with track(path).open(newline='') as f:return list(csv.DictReader(f))
def obj(path):return json.loads(track(path).read_text())
def esc(x):
    return str(x if x not in (None,'') else '—').replace('|','\\|').replace('\r','').replace('\n','<br>')
def table(data,columns=None):
    if not data:return '_No saved rows._\n'
    columns=columns or list(data[0])
    return '\n'.join(['| '+' | '.join(map(esc,columns))+' |','| '+' | '.join(['---']*len(columns))+' |']+['| '+' | '.join(esc(r.get(k)) for k in columns)+' |' for r in data])+'\n'
def add(s):parts.append(s.strip())
def source_table(path,columns=None,select=None):
    rr=rows(path)
    if select:rr=[r for r in rr if select(r)]
    add(f'Source: [{path}]({path}).\n\n'+table(rr,columns))
def category(path):
    if 'direction_cosines' in path:return 'WITHDRAWN / provenance review: target-selected geometry; no live inference'
    if '/legend_stability/' in path or '/letter11_permuted/' in path:return 'LEGACY audit-only: provenance failure; not current reversal/correctness evidence'
    if 'gepa_emission_coarse_v1' in path:return 'SUPERSEDED plumbing pilot; specialized-reflector failure; not main GEPA comparison'
    if 'optional_percent_v1' in path:return 'SEPARATE retrospective percentage-units-only parser; not lenient_v1'
    if 'smoke_' in path or 'pilot_qwen' in path:return 'SMOKE/PILOT; not independent full replication'
    if 'verbalized_confidence/' in path:return 'UPSTREAM descriptive input/readout; not new local experiment'
    if '/alphabet_flip/' in path:return 'SUPPORTING protocol diagnostic; included numeral arms are retired from the live ICLR claim'
    if 'cross_fitted_predictions' in path or 'per_item' in path:return 'ROW-LEVEL records; keep questions/candidates grouped; not independent result cells'
    if 'analysis_coarse' in path or 'analysis_selected_methods' in path:return 'SUBSET/earlier view; overlaps final full analysis; not another replication'
    if 'trained_emission' in path or 'emitted_validation' in path:return 'ACTUAL emissions; strict primary / lenient secondary; retain invalidity and sample labels'
    if 'frame_b_report_sets' in path:return 'SET DIAGNOSTIC; paired ordering uses conditional means, threshold table uses actual emissions'
    if 'fresh_candidate_correctness' in path:return 'FRESH CONDITIONAL candidate-correctness readout; curated pair population'
    if 'letter11_provenance_v1' in path or 'letter11_mapping_control' in path:return 'FRESH original-cell protocol/control evidence; conditional and generation readouts separate'
    if 'qa_average_verified' in path:return 'AUDITED binary P(True), not emitted numerical confidence'
    if '/students/' in path or '/scale_family_v1/analysis/' in path:return 'AUXILIARY conditional distributions/decoded expectations; exploratory pilot'
    if 'dinco_' in path or 'gepa_' in path:return 'DEVELOPMENT estimator/prompt pilot; resource and validity caveats apply'
    if path.startswith('cached_results/') or path.startswith('token_mode_mean/'):
        return 'FROZEN-model diagnostic; default target is mean-logprob preference unless source specifies otherwise'
    return 'Saved result: use source-specific target, split, readout and uncertainty; not independently re-audited here'


def main():
    add('''# ICLR report — complete local evidence and requirements

As of **15 September 2026**. This is the single-file working ledger for the current two-failure ICLR narrative and its separate follow-up. It includes every saved CSV result cell found in the result roots, including seed-level, overlapping, legacy and row-level tables, with explicit status labels. Exact source strings are preserved in the appendices. Raw JSONL predictions, tensors and per-token arrays are linked rather than copied into prose. This is a local inventory, not a claim to have searched unsynced remote jobs or other workspaces.

The current ICLR structure stays unchanged. New ConfTuner/fine-grid training is complete and belongs to the follow-up. The earlier “fine training pending” statements in September 14 compilations and `RECENT_BASELINES.md` are stale; this report supersedes those status statements, not their historical numbers.

**Immediate editorial issue:** the original reversal and paired-ordering cells below use conditional decoded expectations. They do not by themselves establish Failure 2 for an actually emitted number. Keep §5's readout distinction explicit. The mapping control supports supplied-value encoding under the same conditional next-symbol mechanism, with an additional unconstrained-first-token control; it does not prove natural confidence judgments always follow the legend.

## Navigation

- [Requirements and decisions](#requirements-and-decisions)
- [Readouts, targets, pairs and signs](#readouts-targets-pairs-and-signs)
- [Original frozen-state and report cells](#original-frozen-state-and-report-cells)
- [Fresh original-cell reversal and mapping controls](#fresh-original-cell-reversal-and-mapping-controls)
- [Constructive Frame B: paired ordering and sets](#constructive-frame-b-paired-ordering-and-sets)
- [Actual emitted confidence and parser contracts](#actual-emitted-confidence-and-parser-contracts)
- [Completed matched training](#completed-matched-training)
- [Averaging, family selection, DINCO and GEPA](#averaging-family-selection-dinco-and-gepa)
- [Historical binary q/qa and conformal results](#historical-binary-qqa-and-conformal-results)
- [Audits, exclusions and manuscript readiness](#audits-exclusions-and-manuscript-readiness)
- [Complete CSV appendix](#complete-csv-appendix)
- [JSON result and protocol appendix](#json-result-and-protocol-appendix)
- [Prose-only audit appendix](#prose-only-audit-appendix)
''')
    add('''## Editorial choices for the current abstract

**Hole A:** end the sentence at “neither.” Do not restore the retired numeral-fails-first/letter-fails-last sentence. This choice makes no additional clause-level claim; it does not independently validate whatever precedes “neither,” which still needs the readout qualification.

**Hole B:** keep the set perspective in §7 only—one sentence, one location, not an abstract closing claim that a set passes both checks. Proposed sentence: “Retaining the reports as a set makes explicit which answer-related conclusions survive the evaluated legends.”

The longer Frame B set analysis is supporting/follow-up material, not permission to duplicate that sentence or claim a superior set predictor. Numeral results below are retained for audit/history, not reinstated as live ICLR abstract evidence.
''')
    add('''## Requirements and decisions

| Priority / scope | Item | Current status | What is still required / decision |
|---|---|---|---|
| ICLR, before the user's 25th deadline | Preserve two failures and correct abstract/readout wording | Original conditional reversal available; numeral result retired from the live claim | If the abstract insists on actual stated numbers for original cells, collect natural-confidence emissions; otherwise explicitly narrow that clause to its measured readout. Conditional expectations cannot silently stand in for emissions. |
| ICLR | Original letter11 reversed mapping control | COMPLETE: 383/384 reversed contextual encodings correct; all six model–dataset cells below | Insert conditional argmax, first-token, alphabet-mass and whole-response distinctions; no inference that every natural judgment follows the legend. |
| ICLR / Failure 1 premise | “The state knows, the number doesn't” on wrong-under-both pairs | NOT ESTABLISHED by joining the currently reported summaries | Fit/validate a candidate-level correctness state probe with question-disjoint splits and no candidate-position leakage, then evaluate the exact paired questions and persistent-wrong subset. Existing frozen-state target is mean-logprob preference. |
| ICLR / Frame B | Which candidate ordering survives the convention? | COMPLETE for original TruthfulQA conditional expectations; 3 × 817 pairs | Label as auxiliary/estimator-side. Correct, incorrect, ties and sign changes are separate. This is not a new-method comparator or actual-emission result. |
| ICLR | Paired natural-confidence actual emissions on original TruthfulQA candidates | MISSING | Same original models, candidate pair, grid, prompts, decoding policy and both legends; preserve raw responses and invalidity. Known-value controls do not fill this gap. |
| ICLR Figure 1 | Locked real-item selection and single-item forward/reverse values | PENDING / old illustration withdrawn | Rerun the locked train-fold selection rule from `main_fig.md`; do not reuse the illustrative Tusk/Trump item or fill in invented values. |
| ICLR | Actual manuscript insertion | Insertion-ready Markdown exists; no local manuscript .tex source found | Apply the control and the single §7 set sentence to the real draft; keep the longer Frame B note as supporting/follow-up material and check abstract, caption and §5 consistency. This compilation does not claim manuscript edits or a fresh LaTeX build. |
| Follow-up | ConfTuner normal-only versus both-legend, coarse and fine | COMPLETE matched pilot | Replicate independently before confirmatory claims; do not retune on inspected 128-question test results. |
| Follow-up | Fine mean-Brier, consistency, distillation, distillation+labels | COMPLETE matched pilot | Distinguish one-report quality, ensemble quality, imitation and held-out sensitivity. No objective selected from these test results. |
| Follow-up | Six-bank support | COMPLETE emissions/distributions; additional fine mappings evaluated | Keep as support: normal/reversal agreement does not guarantee bank-wide agreement. Six coarse mappings exhaustive; fine bank sampled. |
| Follow-up | Stronger set predictor | NOT ESTABLISHED | Compare on the strongest new control's same reports against mixture features and mean+range/variance; fit selectors on development questions only. The earlier richer-family paired intervals cross zero. |
| Follow-up | p-box interpretation | COMPLETE auxiliary per-question CDF envelopes | Keep underlying paired distributions. No true-correctness bounds, DS belief claim or conformal coverage follows from report envelopes. |
| Follow-up | Direct grid resolution | COMPLETE 1,000-question audit | Added fine values used on only 60 questions; do not treat similar Brier as identical emissions or a strong resolution stress test. |
| Follow-up | strict_v1 / lenient_v1 | COMPLETE, separate contracts | Strict remains primary; old lenient rescoring is retrospective. New generation contracts were declared before collection. Do not relabel GEPA as lenient-optimized. |
| Follow-up | DINCO | COMPLETE 32-question development pilot | Full exact-cost comparison; frozen auxiliary candidates/SC; keep raw, NVC, SC and mixture separate. Interrupted requests make saved cost a lower bound. |
| Follow-up | DSPy–GEPA | COMPLETE coarse development pilot | Equal realized budgets, larger development search, fine grid and withheld mappings; 68 vs 64 metric calls in current arms are not exact resource parity. |
| Training / answer-sensitivity claim | ADVICE and ADVICE+ConfTuner | NOT RUN | Obtain graded correct/incorrect training candidates; match that supervision across controls. Single-answer TriviaQA labels are not the required paired data. |
| Broader method claims | SteerConf / ORCE / DirEAG | Rule-only or literature/protocol stage, not reproduced results | SteerConf needs steered reports; ORCE extra graded samples and verified runner; verify DirEAG before specifying reproduction. No superiority claim over these methods. |
| Conformal extension | Useful family bounds with coverage | NOT ESTABLISHED | Define target, assumptions, calibration split and method first; do not treat sensitivity envelopes as calibrated intervals. |
''')
    add('''## Readouts, targets, pairs and signs

**One pair** means one question with one curated correct candidate and one curated incorrect candidate. The original paired TruthfulQA analysis has 817 questions per model, not 1,634 independent question draws. Within legend g, `d_g = c_g(q,a+) − c_g(q,a−)`: positive favors the correct candidate, negative favors the incorrect candidate. Compare signs across the same two legends. Never compare forward/correct against reversed/incorrect.

| Readout | Meaning | Valid use | Do not call it |
|---|---|---|---|
| Actual emitted value | Parse the complete raw generation under a declared grammar | Main numerical emission Brier, threshold flips, invalidity | Conditional mean or permitted-token argmax |
| Original next-symbol conditional mean | Normalize candidate-letter probabilities, decode numerical expectation | Original auxiliary reversal, candidate contrasts, CDF family | Actually stated/generated confidence |
| Conditional argmax | Highest-probability permitted symbol | Original mapping-control readout | Unconstrained generation compliance |
| Unconstrained first token | First token from free generation | Additional mapping-control evidence | Whole-response validity or natural confidence judgment |
| New letter-plus-newline distribution | Normalize full allowed completion-sequence likelihoods | Training losses, teacher targets, auxiliary decoded means | Original next-token distribution; these have different mass semantics |
| Historical binary P(True) | Decoded probability of correctness under binary codebook | Supporting q/qa, legend and averaging evidence | A generated confidence percentage |
| Final estimator score | Averaging, normalization, DINCO or selector output | Estimator-side quality, with its call budget | Repaired individual verbal report |

| Target / population | Source | Interpretation |
|---|---|---|
| Frozen legacy state/report target | `1[logprob_pos_mean > logprob_neg_mean]` | Teacher-forced candidate preference; not independently graded answer correctness. |
| Original TruthfulQA paired candidates | Curated MC1 correct/incorrect candidate labels | Tests relative assessment of this selected pair; not natural answer-generation accuracy. |
| Pavlick NLI candidates | Adapter sign of mean graded NLI judgment | Binarized entailment/non-entailment instrument, not an unqualified factual correctness target. |
| TriviaQA fixed generated answers | Imported `correct_revisited` | Generated-answer correctness supervision, not new human confidence labels. |

Question-bootstrap intervals keep all legends and candidates together. Old seed ± values are SD over seeds 0–2, not confidence intervals. Saved intervals condition on fitted models and mappings. Different training seeds, datasets, candidate populations, readouts, validity masks and budgets are not pooled. Blank source fields remain missing, never zero.

`strict_v1`: requires the percent suffix for direct reports and retains the historical numerical tolerance. `lenient_v1`: one bare [0,1] number means probability; bare (1,100] or suffixed values use percentage units; exact Decimal grid membership; no off-grid rounding, multiple numbers, explanatory text, clipping or semantic retry. `optional_percent_v1` is an older, distinct percentage-units-only diagnostic. Invalid outputs receive Brier penalty 1; valid-only AUROC/AURC and stability retain explicit denominators.
''')
    add('## Original frozen-state and report cells\n\nAll six mean-pooled state comparisons improve over the question baseline on their **preference target**. This does not establish candidate-correctness state knowledge on the newer persistent-wrong subset. The saved `I_answer_given_q` column is a source estimate, not an identified information fraction. Fixed-order paired features are `[h_pos,h_neg,h_pos−h_neg]`; target-selected direction geometry is withdrawn.')
    source_table('token_mode_mean/summary_all_datasets.csv')
    add('Standard letter-report comparison: conditional expectation gap; one of five measurable cells improves. Llama/Pavlick fails the resolution gate and is not a measured negative. A decoded mean is continuous even on an eleven-value grid, so effective-bin counts are not restricted to eleven.')
    source_table('cached_results/report_channel_per_cell.csv')
    add('All alternative readout, alphabet, compliance, target-definition and seed-level cells appear in Appendix A. Numeric controls do change some conclusions; a universal readout-independent null is unsupported. Clean matched numeric caches are not available for every model/interface. Tokenizer collision diagnostics do not supply missing generated-number results.')
    add('## Fresh original-cell reversal and mapping controls\n\nThe fresh provenance-controlled reversal has five rule failures and one indeterminate cell. Do not reuse the old “near-zero correlation in six of six” claim. The decision rule compares a paired item-bootstrap interval to a saved forward reliability point reference; uncertainty in that reference is not jointly propagated.')
    source_table('cached_results/letter11_provenance_v1/summary.csv')
    add('Mapping controls supply the confidence value rather than ask the model to assess the answer. Every model/dataset/orientation/standalone cell is below. Contextual cells have 64 candidate records from 32 questions; standalone cells have 11 known values. Conditional correctness and first-token correctness under contextual reversal total 383/384. Standalone conditional accuracy can be perfect despite low mass or poor unconstrained first-token accuracy.')
    source_table('cached_results/letter11_mapping_control_v1/metrics.csv')
    source_table('cached_results/letter11_mapping_control_v1/paired_differences.csv')
    add('## Constructive Frame B: paired ordering and sets\n\nThese are **original conditional decoded expectations**, not actual emitted confidence. “Consistently incorrect” is an answer-assessment failure that survives both tested legends; “sign change” is convention-dependent ordering. Neither identifies what the number tracks instead. Do not promote this to “the state knows” without the matched correctness-state check listed above.')
    source_table(U+'frame_b_report_sets_v1/paired_ordering.csv')
    add('Separate candidate intervals permit unobserved cross-legend comparisons. The following counts include both correct-under-both and incorrect-under-both cases whose strict paired ordering is lost under the Cartesian difference of separate intervals. This is information preservation, not evidence of improved prediction.')
    source_table(U+'frame_b_report_sets_v1/pairing_preservation.csv')
    add('For actual emissions, `S={c_F,c_R}`, `L=min S`, `U=max S`: all accept if L≥t; all reject if U<t; legend-dependent if L<t≤U. An invalid component makes the bundle invalid. Every threshold/model/grid/contract cell is in Appendix A. The 13.3% historical six-mean flip figure and the 21.1% two-emission fine-transfer figure are different readouts, banks and samples—not a matched scale comparison.')
    source_table(U+'frame_b_report_sets_v1/actual_emission_decisions.csv',select=lambda r:r['contract']=='strict_v1' and r['threshold']=='0.75')
    add('Auxiliary p-boxes use per-question aligned CDFs: lower F=min_g F_g; upper F=max_g F_g. The supplied rsuq `core/beliefs.py::cluster_prob_mass` computes ordered-grid event masses. The original paired records are retained because the envelope admits distributions beyond the family. The exact stable/unstable equal-mixture example is saved. This is not an across-question histogram, a DS belief assignment, a true-correctness interval or a better predictor. At threshold t, P(V≥t)=1−F(t−), not 1−F(t). See [paired records]('+U+'frame_b_report_sets_v1/paired_answer_contrasts.jsonl), [p-box records]('+U+'frame_b_report_sets_v1/auxiliary_pboxes.jsonl), and [rsuq/source audit]('+U+'frame_b_report_sets_v1/AUDIT.json). Definitions follow [Troffaes–Destercke](https://arxiv.org/abs/1103.1805) and [Ferson et al.](https://www.cs.utep.edu/vladik/2003/sandia03.pdf).')
    add('## Actual emitted confidence and parser contracts\n\nThe old validation extraction uses the **same coarse correctness-only checkpoint on both grids**. It contains 1,000 deterministically selected validation questions, one fixed Llama answer each, and direct/normal/reversed generations. Coarse direct prompts list 0%, 50%, 100%; fine prompts list every 10% increment. Generation is not hard-constrained to that alphabet. All grids and raw outputs remain frozen.')
    audit=obj(U+'emitted_validation_v1/lenient_v1/direct_grid_audit.json')
    add('Direct values under secondary lenient_v1:\n\n'+table([{'grid':s,**{str(v):count for v,count in h.items()}} for s,h in audit['value_histograms'].items()],['grid','0.0','0.1','0.3','0.5','1.0'])+'\nOnly 60 questions use fine-only values, while 75 paired emissions change. Similar Brier does not imply identical emissions. The blank histogram entries above are absent values; the full histogram JSON is authoritative.\n\n'+table(audit['fine_brier_contrasts']))
    add('For this checkpoint, the normal-minus-direct Brier point estimate is +0.01288 with an interval crossing zero; reversal-minus-direct is +0.03410 [0.01239,0.05465]. This is a secondary-parser transfer result, not a claim about all fine-trained reporters. Strict direct invalidity remains in the primary tables.')
    source_table(U+'emitted_validation_v1/analysis/metrics.csv')
    source_table(U+'emitted_validation_v1/lenient_v1/analysis/metrics.csv')
    source_table(U+'emitted_validation_v1/analysis/stability.csv')
    add('## Completed matched training\n\nFive coarse and six fine arms are complete: 128 identical training questions, batch size 8, two views, 16 updates, rank 16, learning rate 1e-4, same initial QA weights with fresh optimizers, fixed consistency/supervision coefficients 1. Normal-only token Brier repeats the normal view. Other students see normal and reversal. Pure distillation receives no additional labels but inherits the original supervised checkpoint. Across grids, scored sequence counts and tokens differ; this is not FLOP parity. Evaluation uses the preserved 128-question exploratory test subset.\n\nTraining and conditional scoring use `training.prompt_text`; free-emission evaluation uses the common `emitted_reports.render` wrapper with different instruction placement. This is shared-interface transfer across methods, not byte-identical train/evaluation prompting.\n\nToken Brier = E_r[(V−y)^2]; mean Brier = (E_r[V]−y)^2; the difference is within-report variance. Consistency adds the squared normal/reversed mean difference. Distillation uses KL from the full numerically aligned six-report teacher mixture; supervised distillation adds the same mean-Brier correctness term. These are ConfTuner/Liusie-style adaptations, not exact original-method reproductions.\n\nPrimary actual emitted readouts, every arm:')
    source_table(F+'trained_emission_analysis/metrics.csv',columns=['scale','method','contract','report','calls','n','valid','invalid_rate','penalized_brier','brier_lo','brier_hi','valid_only_auroc','valid_only_aurc'],select=lambda r:r['contract']=='strict_v1' and r['report'] in ['direct','normal','reversed','six_report_average'])
    add('Secondary lenient actual emissions, every arm:')
    source_table(F+'trained_emission_analysis/metrics.csv',columns=['scale','method','report','calls','n','valid','invalid_rate','penalized_brier','brier_lo','brier_hi','valid_only_auroc','valid_only_aurc'],select=lambda r:r['contract']=='lenient_v1' and r['report'] in ['direct','normal','reversed','six_report_average'])
    add('All actual per-legend, mean/worst individual, two-report-average, paired objective contrasts, six-bank flips, invalidity and source histograms are included in the complete appendix. Zero observed flips do not establish zero population risk. A six-report ensemble is not a one-call individual report.\n\nAuxiliary teacher/student decoded-expectation cells:')
    source_table(F+'students/analysis/metrics.csv',select=lambda r:r['evaluation'] in ['normal','reversed','six_report_average','common_withheld_bank','fresh_beyond_teacher'])
    source_table(F+'students/analysis/imitation.csv')
    source_table(F+'students/analysis/scale_comparison.csv')
    source_table(F+'students/analysis/budget_audit.csv')
    add('Exposure: all six coarse mappings are teacher-seen; four are unseen by every student, with reversal additionally unseen by normal-only training. Fine teacher targets use identity, reversal and four random mappings; four further saved mappings are excluded from teacher targets and student training. `common_withheld_bank` excludes both normal and reversal for matched comparisons. Reordering the same cached reports is not a test of starting-legend robustness. The composed-bank test remains unrun.')
    add('## Averaging, family selection, DINCO and GEPA\n\nThe strongest previously tested coarse mean-Brier student has a completed six-report family comparison. Selectors fit on 128 validation questions and evaluate on 128 preserved test questions. No distinct richer-family benefit beyond mixture plus elementary spread is established; paired intervals cross zero. This is not yet the same analysis on the new ConfTuner control.')
    source_table(F+'students/coarse/both_mean_brier/family_analysis/metrics.csv')
    source_table(F+'students/coarse/both_mean_brier/family_analysis/paired_differences.csv')
    add('DINCO: 32-question supplied-answer development pilot, 28 valid distractor lists. Raw, normalized verbal confidence, SC and final mixture must stay separate. With frozen SC, the final score gap equals half the NVC gap algebraically; it does not show the raw verbal report was repaired. The following tables preserve source denominators. Cost is a recorded lower bound after interrupted calls, not exact realized resource parity.')
    source_table(U+'dinco_emission_v1/analysis/metrics.csv')
    source_table(U+'dinco_emission_v1/analysis/stability.csv')
    add('DSPy is the framework and GEPA the optimizer, not two independent methods. Current v2 searches use 16 training questions, 8 adaptive selection questions and 128 search-excluded validation questions for evaluation. Both searches were configured for 64 metric calls; quality executed 68 (204 target calls), joint executed 64 (192 target calls). Reflection calls were 6 vs 8. The joint arm selected the original prompt. This is not evidence that prompting cannot solve invariance. The old v1 reflector plumbing run is retained but superseded. Lenient tables reparse old outputs; the optimizer used strict parsing.')
    for arm in ['quality','quality_consistency']:
        for contract,suffix in [('strict_v1',''),('lenient_v1','lenient_v1/')]:
            add(f'GEPA {arm}, {contract}:')
            source_table(U+f'gepa_emission_coarse_v2/{arm}/evaluation/{suffix}analysis/metrics.csv')
    add('ADVICE, ADVICE+ConfTuner, ORCE and DirEAG do not have result cells. Their absence is recorded as **not run**, never zero or a failed method. SteerConf has a scalar-rule adaptation but no matched fresh steered-report experiment. Do not claim experimental superiority over these methods.')
    add('## Historical binary q/qa and conformal results\n\nThese support the research program but do not measure emitted percentages. `q` hides the candidate and asks about the fixed answer source; `qa` supplies the fixed candidate. The full answer collection uses source commit `aa7270c3c1089393eec8d8179d1581f3ec0c7b80`, source generator `llama3.1_8b_chat`, imported `correct_revisited` grades, and Qwen/Qwen3-8B as reporter. There were 58,546 source rows; remove 5 empty-answer rows and 5,214 duplicate-question rows to obtain 53,327 unique questions. The deterministic project split (salt `tinker-uq-v1`) is train 26,737 / validation 5,346 / calibration 10,597 / test 10,647. This is not the official TriviaQA test-set size.\n\nAll q/qa, base/trained, normal/reversed/averaged test cells:')
    source_table(U+'full_llama31_8b_qwen3_8b/qa_average_verified/q_qa_metrics.csv')
    source_table(U+'full_llama31_8b_qwen3_8b/qa_average_verified/q_qa_paired_differences.csv')
    add('The historical audit saved only the normal prompt text; reversed construction is supported by the runner. This provenance distinction remains. Fresh caches have literal prompts for both orientations.\n\nCompleted full-run conformal partition results are below. They concern the saved binary method and calibration experiment, not coverage of the new report-family interval. K=1 has the smallest average set size in the full run; finer partitions do not automatically improve efficiency. These are one fitted calibration draw and one test set.')
    source_table(U+'full_llama31_8b_qwen3_8b/aistats/summary.csv')
    add('## Audits, exclusions and manuscript readiness\n\nThe independent completed-pilot audit reconstructed all 53,327 source records, checked 11 matched training arms, 12,928 raw generations, 25,856 strict/lenient decode records, 13,568 distributions, 180 paired contrasts, 284 trained known-value controls, 834 original controls and 768 original contextual prompt/readout joins. It found no numerical discrepancy in the audited claims and made no new model calls. See [audit report](tinker_uq/audit_2026-09-15/REPORT.md) and [hashed source ledger](tinker_uq/audit_2026-09-15/AUDIT.json). This compiler does **not** extend that audit stamp to every legacy table in the appendix.\n\nLimits: Tinker exposes adapter paths but not an independently pinned backbone commit; the original control full-vocabulary logits were not saved, so their mass cannot be independently recomputed; parser chronology relies on session provenance rather than an externally timestamped registry. New sequence-cache mass is recomputed from saved likelihoods.\n\nWithdrawn/audit-only material is kept for completeness: legacy reverse/retained-risk caches; target-selected state-direction geometry; superseded specialized-reflector GEPA v1; incomplete initial matched-letter11 training; simulated/demo outputs; tokenizer-collision examples with unrecovered writer provenance. These are not additional successful or failed main-method replications. Frozen legacy retained risks use a preference target and must not become correctness claims.\n\nThe locked Figure 1 example remains pending in `main_fig.md` and `paper/figs/fig2_two_failures_manifest.txt`. Existing illustrations are not proof that the item selection has run. ICLR draft resources (the longer Frame B note must not be pasted wholesale under the one-sentence §7 decision): [mapping control and §7](ICLR_MAPPING_CONTROL_AND_SECTION7.md), [Frame B set interpretation](ICLR_FRAME_B_SET_INTERPRETATION.md). No local manuscript source was available to apply these edits or verify compilation.\n\nThe actionable claim currently supported by the new training pilot is that ordinary calibration training with legend variation can improve numerical agreement on trained conventions. Residual additional-legend variation remains. Stronger answers about unseen interfaces, independent replication, state correctness, or set-based prediction require the work listed at the top.\n\n### Raw artifacts and figures\n\nThe raw records are linked so no derived table substitutes for underlying evidence: [unified distribution predictions]('+F+'students/analysis/verified_reporting_predictions.jsonl), [actual trained emissions]('+F+'trained_emissions), [original TruthfulQA/ NLI distribution tensors](letter11_provenance_v1), [paired contrast records]('+U+'frame_b_report_sets_v1/paired_answer_contrasts.jsonl), [p-box families]('+U+'frame_b_report_sets_v1/auxiliary_pboxes.jsonl), [historical q/qa scatter]('+U+'full_llama31_8b_qwen3_8b/qa_average_verified/q_qa_scatter.png). Full source rows and seed tables follow; repeated views are labeled as overlapping evidence.')
    add('## Complete CSV appendix\n\nEvery row and column of every CSV found under `cached_results`, `token_mode_mean`, `verbalized_confidence`, and `tinker_uq/runs` is reproduced below at saved precision. This includes legacy, subset, seed and per-item tables; inclusion is not endorsement. Empty fields display “—”; `nan` means undefined, not zero. Large tables are collapsible but are fully present in this Markdown file. Each source has a SHA-256 entry in `iclr_report_sources.json`.')
    paths=sorted({p.relative_to(ROOT).as_posix() for base in ['cached_results','token_mode_mean','verbalized_confidence','tinker_uq/runs'] for p in (ROOT/base).rglob('*.csv')})
    registry=[]
    for j,path in enumerate(paths,1):
        rr=rows(path);sources[path].update(rows=len(rr),kind='csv',status=category(path));registry.append(dict(table=f'A{j:03d}',source=f'[{path}]({path})',rows=len(rr),status=category(path)))
    add(table(registry))
    for j,path in enumerate(paths,1):
        rr=rows(path)
        add(f'<details>\n<summary>A{j:03d} — {path} — {len(rr)} rows</summary>\n\n**Status:** {category(path)}. Source: [{path}]({path}).\n\n'+table(rr)+'\n</details>')
    add('## JSON result and protocol appendix\n\nThese are complete selected JSON summaries and protocols, not truncated snippets. Raw generations, token arrays, teacher targets, checkpoints and immutable input banks remain in linked source artifacts. Absence of a method result is not inferred from a missing scalar field.')
    names={'metrics.json','answer_gain.json','legend_transfer.json','summary.json','result.json','protocol.json','mapping_control_summary.json','individual_brier.json'}
    json_paths={p.relative_to(ROOT).as_posix() for p in (ROOT/'tinker_uq/runs').rglob('*.json') if p.name in names and '/gepa/' not in p.as_posix()}
    json_paths.update(['cached_results/partition_invariance/result.json','tinker_uq/EXPERIMENT_STATUS.json','tinker_uq/contracts/strict_v1.json','tinker_uq/contracts/lenient_v1.json',F+'manifest.json',F+'student_plan_conftuner.json',F+'TRAINING_BUDGET_AUDIT.json',F+'mappings.json',U+'emitted_validation_v1/lenient_v1/direct_grid_audit.json',U+'dinco_emission_v1/analysis/audit.json',U+'frame_b_report_sets_v1/PROTOCOL.json',U+'frame_b_report_sets_v1/toy_example.json'])
    for j,path in enumerate(sorted(json_paths),1):
        value=obj(path);sources[path].update(kind='json',status=category(path))
        add(f'<details>\n<summary>B{j:03d} — {path}</summary>\n\nSource: [{path}]({path}). Source-specific status applies; coarse-only/earlier files are not independent replications.\n\n```json\n'+json.dumps(value,indent=2)+'\n```\n\n</details>')
    add('## Prose-only audit appendix\n\nThese preserve numerical evidence that has no equivalent CSV here. Their original targets and status remain in force. Pending statements are historical; use this report’s requirements table for current status.')
    for path in ['p2_iclr/cached_results/z2_average_audit.md','p2_iclr/cached_results/legend_accounts_audit.md','main_fig.md','paper/figs/fig2_two_failures_manifest.txt']:
        content=track(path).read_text()
        add(f'<details>\n<summary>{path}</summary>\n\nSource: [{path}]({path}).\n\n'+content+'\n</details>')
    text='\n\n'.join(parts)+'\n'
    # Check internal links and complete appendix coverage before publishing.
    links=re.findall(r'\]\(([^)]+)\)',text)
    missing=[x for x in links if not x.startswith(('https://','http://','#')) and not (ROOT/x).exists()]
    assert not missing,missing
    assert text.count('<details>')==text.count('</details>')
    assert all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==entry['sha256'] for path,entry in sources.items())
    (ROOT/'iclr_report.md').write_text(text)
    manifest=dict(as_of='2026-09-15',report='iclr_report.md',scope='local saved tables; no new experiments',csv_tables=len(paths),csv_rows=sum(sources[p]['rows'] for p in paths),json_summaries=len(json_paths),all_local_links_resolve=True,source_files_unchanged=True,sources=sources)
    (ROOT/'iclr_report_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='sources'},indent=2));print('Markdown bytes',len(text.encode()))


if __name__=='__main__':main()
