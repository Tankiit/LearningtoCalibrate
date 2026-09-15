"""Build the research-status report from completed artifacts, without new fitting."""
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def csvrows(path):return list(csv.DictReader((ROOT/path).open()))
def read(path):return json.loads((ROOT/path).read_text())
def lines(path):return [json.loads(x) for x in (ROOT/path).read_text().splitlines()]

def main():
 coarse=csvrows('runs/scale_family_v1/students/analysis_coarse/individual_and_ensemble.csv');primary=csvrows('runs/emitted_validation_v1/analysis/metrics.csv');stability=csvrows('runs/emitted_validation_v1/analysis/numerical_stability.csv');family=csvrows('runs/scale_family_v1/students/coarse/both_mean_brier/family_analysis/paired_differences.csv')
 names={'teacher':'Frozen teacher','both_mean_brier':'Correctness-only + legends','aligned_distill':'Distillation only','aligned_distill_supervised':'Distillation + correctness'}
 text='''# Reporting-convention study: analysis and status, 15 September 2026

The coarse distillation comparison is complete. Ordinary correctness supervision with legend variation is the strongest completed coarse control. Its report family has not established a selective-decision benefit beyond the full average plus elementary disagreement features. Fresh actual emissions show materially greater normal/reversed variation when this coarse-trained checkpoint is used on the fine grid. These results do not establish failure of permutation averaging, distillation generally, or fine-grid supervised training.

## Deliverable status

| Deliverable | Verified status |
|---|---|
| Historical binary q/qa comparison | Supporting audit, averaging, paired intervals, stability and scatter artifacts available; binary probability readout kept separate |
| Shared reporting contract | Explicit mappings in prompt, batch, objectives and decoder; full conditional vector, decoded mean, allowed sequence mass and prompt provenance exported |
| Coarse teacher/student comparison | Complete: 128 matched test questions, six legends, three matched 16-update student arms |
| Coarse per-legend verification | 3,072 records checked for IDs/labels, literal prompts, tokenization, mappings, normalization and decoding |
| Fresh numerical interface | Complete: 1,000 validation questions × two grids × three formats = 6,000 actual generations, plus 42 known-value controls |
| Strong supervised family selection | Complete: fit on 128 validation questions, evaluate on preserved 128 test questions; paired question intervals |
| DINCO | 32-question supplied-answer implementation pilot complete; full main comparison and exact realized-cost run outstanding |
| DSPy–GEPA | Two coarse search arms and search-excluded development evaluation complete as a pilot; fine search and strict realized-budget matching outstanding |
| ConfTuner / ADVICE / fine supervised training | Not completed; these remain required controls before new training-method claims |
| Auxiliary 1,000-question distribution expansion | Paused after one completed coarse mapping; does not replace the completed 6,000-emission cache |

## Primary outcome: actual numerical emissions

Same coarse correctness-only Qwen3-8B checkpoint, same 1,000 fixed TriviaQA/Llama answer pairs, normal/reversed legends within each numerical grid. No calibration or test questions entered this extraction. Brier below uses actual generated values, not decoded expectations. Every normal/reversed report parsed successfully.

| Grid | Normal Brier | Reversed Brier | Two-report-average Brier | Mean absolute shift | Decision flips at 0.75 |
|---|---:|---:|---:|---:|---:|
'''
 for scale in ['coarse','fine']:
  mm={r['method']:r for r in primary if r['scale']==scale};ss=next(r for r in stability if r['scale']==scale)
  text+=f"| {scale} | {float(mm['normal']['penalized_brier']):.4f} | {float(mm['reversed']['penalized_brier']):.4f} | {float(mm['normal_reversed_average']['penalized_brier']):.4f} | {float(ss['mean_abs_shift']):.4f} | {float(ss['flip_0.75']):.1%} |\n"
 text+='''
The average is a two-call estimator; individual reports each use one call. The paired fine-minus-coarse mean absolute shift is 0.1762 (95% question-level interval [0.1550, 0.2006]). This is a transfer result for a coarse-trained checkpoint, not evidence that fine-grid training or Liusie-style distillation must fail.

Direct-format compliance is a separate finding: 99.8% of coarse and 91.3% of fine responses omitted the required `%` suffix under the frozen strict parser. Most were bare numbers such as `100`. These are syntax failures under that contract, not evidence that the outputs contained no numerical information. The parser was not changed after inspection. Invalid outcomes receive loss penalty 1 and are retained; valid-only discrimination has its denominator reported.

Known-value generation controls encoded all normal/reversed legend values correctly (6/6 coarse; 22/22 fine). All 14 direct controls omitted the required suffix. These small controls diagnose basic codebook following and do not guarantee correct confidence judgments on natural questions.

[Primary table and intervals](runs/emitted_validation_v1/analysis/REPORT.md) · [Paired differences](runs/emitted_validation_v1/analysis/paired_differences.csv) · [Stability, tails and rank agreement](runs/emitted_validation_v1/analysis/numerical_stability.csv) · [Retained-set risks](runs/emitted_validation_v1/analysis/retained_risk.csv) · [Scatter figure](runs/emitted_validation_v1/analysis/emitted_normal_reversed.png)

Retained sets use the top `ceil(coverage × n)` scores, resolving ties by question ID. Overlap is intersection size divided by retained-set size; Jaccard is intersection divided by union. Risk is the fraction of fixed supplied answers labeled incorrect. Paired intervals resample questions, keeping all reporting variants together. Primary AURC averages all prefix risks with expected handling of tied scores.

## Auxiliary outcome: coarse decoded-distribution training comparison

128 preserved test questions. All students inherit the same binary-trained checkpoint, reset the optimizer, and use the same 128 training questions, rank-16 adapter, two normal/reversed views, learning rate and 16 updates. All receive identical additional labels when a correctness term is present. Distillation-only still inherits the earlier checkpoint's correctness supervision.

| Method | Normal Brier | Reversed Brier | Mean per-report Brier | Worst-legend Brier | Six-report-average Brier | Mean pairwise gap |
|---|---:|---:|---:|---:|---:|---:|
'''
 for r in coarse:
  text+=f"| {names[r['method']]} | "+' | '.join(f"{float(r[k]):.4f}" for k in ['normal','reversed','average_per_report','worst_legend','six_report_average','mean_pairwise_disagreement'])+' |\n'
 text+='''
Worst legend means the largest dataset-average Brier among the six mappings, not the average questionwise maximum. The correctness-only control performs well individually as well as in its ensemble. Its mean gap is small, but 13.3% of these questions still cross threshold .75 somewhere among the six legends. Zero threshold flips for the distillation arms reflects all their reports staying below .75; it does not imply useful acceptance at that threshold.

Distillation-only matches the teacher mixture closely on the two training legends (mean KL 0.0099), with larger mismatch on the four student-unseen legends (0.0548). Teacher quality, student imitation and reporting invariance are distinct outcomes. All six coarse mappings occur in teacher target construction; only two occur as student training inputs. There are no globally unseen coarse permutations.

[Complete comparison and per-legend tables](runs/scale_family_v1/students/analysis_coarse/REPORT.md) · [Paired objective contrasts](runs/scale_family_v1/students/analysis_coarse/paired_differences.csv) · [Self-contained verified predictions](runs/scale_family_v1/students/analysis_coarse/verified_reporting_predictions.jsonl) · [Contract audit](runs/scale_family_v1/students/analysis_coarse/contract_audit.json)

## Does retaining the family help beyond simple disagreement?

The strongest coarse supervised student supplies all six reports to every family/mixture comparison. Fixed-complexity learned selectors are fitted on validation questions only. Richer family features are compared against the entire averaged distribution plus mean, range, variance and extrema.

'''
 r=family[0];text+=f"Richer-family minus full-mixture-plus-elementary AURC: {float(r['aurc']):.6f}, paired 95% interval [{float(r['aurc_lo']):.6f}, {float(r['aurc_hi']):.6f}]. Brier difference: {float(r['brier']):.6f}, interval [{float(r['brier_lo']):.6f}, {float(r['brier_hi']):.6f}]. Neither establishes an improvement.\n"
 text+='''
The exact toy example remains valid: identical average distributions can hide different component families. But the present predictive experiment does not justify a distinct richer-set benefit. A scalar unanimous-threshold rule is fully determined by the minimum and maximum report means. These are reporting-sensitivity bounds over evaluated legends, not calibrated bounds on correctness or conformal coverage.

[Family selection report](runs/scale_family_v1/students/coarse/both_mean_brier/family_analysis/REPORT.md) · [Paired family contrasts](runs/scale_family_v1/students/coarse/both_mean_brier/family_analysis/paired_differences.csv) · [Retained-coverage risks](runs/scale_family_v1/students/coarse/both_mean_brier/family_analysis/retained_risk.csv)

## DINCO and DSPy–GEPA pilots

DINCO now has separate raw-emission, normalized-confidence, self-consistency and final-score outputs on 32 development questions. Four candidate-list generations failed the strict auxiliary grammar, and some direct reports failed the percentage grammar; the table reports those denominators. Normalized confidence remains legend-sensitive. On matched valid fine-grid questions its mean shift is 0.2176 and the final DINCO shift is exactly 0.1088. Fixed SC explains that halving algebraically. This small fixed-answer adaptation is not evidence against the published method; its full matched main comparison remains outstanding. Interrupted attempts mean the recorded cost totals are lower bounds.

[DINCO pilot report](runs/dinco_emission_v1/analysis/REPORT.md) · [Verified DINCO cache](runs/dinco_emission_v1/verified_reports.jsonl)

The two coarse GEPA arms use one mutable instruction inside a fixed DSPy wrapper and the same untuned Qwen3-8B reflector. An earlier plumbing run with the confidence-specialized adapter as reflector is preserved separately. The search used 16 training questions and 8 adaptive validation questions. The quality arm used 68 realized metric calls; the joint arm used 64 despite the common configured budget of 64. Their respective target calls were 204 and 192, and reflection calls 6 and 8. Thus this is not an exactly equal realized-resource comparison. Evaluation uses 128 validation questions excluded from optimizer selection.

'''
 # Read the saved raw responses to compare the two frozen programs on identical evaluation IDs.
 losses={}
 for arm in ['quality','quality_consistency']:
  rs=lines(f'runs/gepa_emission_coarse_v2/{arm}/evaluation/reports.jsonl');qids=sorted({r['question_id'] for r in rs});by={i:[] for i in qids}
  for r in rs:by[r['question_id']].append((r['emitted_value']-r['correct'])**2 if r['valid'] else 1.)
  losses[arm]=np.array([np.mean(by[i]) for i in qids])
 delta=losses['quality']-losses['quality_consistency'];ix=np.random.default_rng(20260915).integers(0,len(delta),(2000,len(delta)));lo,hi=np.quantile(delta[ix].mean(1),[.025,.975])
 text+=f"On these search-excluded questions, mean per-format penalized Brier is {losses['quality'].mean():.4f} for the quality-selected prompt and {losses['quality_consistency'].mean():.4f} for the joint arm, which selected the original prompt. Their paired difference is {delta.mean():.4f} [{lo:.4f}, {hi:.4f}]. This pilot has not established a prompting remedy or a general impossibility result.\n"
 text+='''
[GEPA quality evaluation](runs/gepa_emission_coarse_v2/quality/evaluation/analysis/REPORT.md) · [GEPA joint evaluation](runs/gepa_emission_coarse_v2/quality_consistency/evaluation/analysis/REPORT.md) · [Recent-method specifications and remaining controls](RECENT_BASELINES.md)

## Protocol and interpretation limits

The full generated-answer source contains 53,327 unique TriviaQA questions and fixed Llama 3.1-8B answers, with imported `correct_revisited` labels. Existing project partitions remain train 26,737, validation 5,346, calibration 10,597, test 10,647. The new 128-question and 1,000-question subsets do not redefine those partitions. The 10,647-question historical binary result is a separate probability-readout experiment.

The exact adapter sampler paths and tokenizer/template hashes are recorded. An independent provider backbone commit is not exposed; the report does not invent one. The historical binary audit retained normal prompt evidence but not an independently saved reversed prompt. New coarse caches verify both literal prompts and tokenization.

The coarse test summary was inspected and informed the research direction and checkpoint choice. New selector fitting and prompt optimization use development questions only. These findings remain exploratory; an independent replication is required before treating an adaptively chosen method as confirmatory.

[Shared protocol](REPORTING_CONTRACT.md) · [Historical binary q/qa comparison](runs/full_llama31_8b_qwen3_8b/qa_average_verified/Q_QA_REPORT.md) · [QA provenance audit](runs/full_llama31_8b_qwen3_8b/qa_average_verified/REPORT.md) · [Fresh curated-candidate correctness analysis](../cached_results/fresh_candidate_correctness/REPORT.md)

The next substantial training comparison remains ConfTuner canonical versus matched legend augmentation, ADVICE with graded answer pairs and label-matched controls, and the fine-grid supervised/distillation counterparts. No new specialized objective or richer-set method should be promoted on the current evidence.
'''
 (ROOT/'ANALYSIS_2026-09-15.md').write_text(text)
 print(ROOT/'ANALYSIS_2026-09-15.md')
if __name__=='__main__':main()
