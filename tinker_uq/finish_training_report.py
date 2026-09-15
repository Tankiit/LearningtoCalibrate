"""Publish the completed expanded pilot without rewriting historical strict caches."""
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

from tinker_uq.trained_emission_summary import run as summarize
from tinker_uq.verify_coarse_contract import run as audit


def main(wait=False):
    root=Path(__file__).resolve().parent
    source=root/'runs/scale_family_v1'
    if wait:
        print('Waiting for completed collection and paired distribution analysis.',flush=True)
        deadline=time.monotonic()+1800
        while not all((source/name).exists() for name in ['STUDENTS_COMPLETE.json','TRAINED_EMISSIONS_FINE_COMPLETE.json']):
            if time.monotonic()>deadline:raise TimeoutError('Collection/analysis completion marker absent; no completion claim written')
            time.sleep(10)
    status=json.loads((source/'STUDENTS_COMPLETE.json').read_text())
    assert status.get('complete') and status.get('plan')=='student_plan_conftuner.json'
    assert json.loads((source/'TRAINED_EMISSIONS_FINE_COMPLETE.json').read_text())['complete']
    assert json.loads((source/'TRAINED_EMISSIONS_COARSE_COMPLETE.json').read_text())['complete']
    summary=summarize(source)
    for scale in ['coarse','fine']:audit(source,scale,'student_plan_conftuner.json')
    combined=source/'students/analysis/verified_reporting_predictions.jsonl'
    with combined.open('wb') as stream:
        for scale in ['coarse','fine']:
            with (source/f'students/contract_audit_{scale}/verified_reporting_predictions.jsonl').open('rb') as part:shutil.copyfileobj(part,stream)
    metrics=list(csv.DictReader((summary/'metrics.csv').open()))
    plan=json.loads((source/'student_plan_conftuner.json').read_text())
    table={(r['scale'],r['method'],r['report']):r for r in metrics if r['contract']=='strict_v1'}
    text='''# Completed ConfTuner-style and fine-grid pilot

All five coarse and six fine student arms completed training and evaluation. Every arm uses the same 128 fixed training answers, ordering, correctness labels, initial QA checkpoint, fresh optimizer, rank-16 adapter, 16 updates and learning rate 1e-4. Alphabet size changes scored sequence/token cost; this is not a FLOP-matched cross-grid experiment. The preserved 128-question test subset is exploratory: earlier coarse test results informed the research direction.

Primary results below are actual emitted confidence under strict_v1. Brier assigns invalid outputs loss 1; six-report averages are invalid if any component is invalid. Full tables give invalidity, valid-only AUROC/AURC, paired intervals, per-legend results, and 1/2/6-call budgets. Decoded expectations and full distributions are a separate auxiliary panel.

| Grid | Method | Normal Brier | Reversed Brier | Six-report-average Brier |
|---|---|---:|---:|---:|
'''
    for scale,methods in plan['methods'].items():
        for method in methods:
            values=[float(table[scale,method,name]['penalized_brier']) for name in ['normal','reversed','six_report_average']]
            text+=f'| {scale} | {method} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |\n'
    text+='''
The ConfTuner-style comparison is an adaptation to fixed supplied answers and conditional letter-plus-newline distributions, not an exact reproduction of percentage-token training. Free-emission evaluation uses a shared frozen wrapper with different instruction placement from the training prompt; this wording transfer is common across methods. Within each interface only the legend changes.

The fine-grid ConfTuner normal/reversal improvement is documented with paired intervals in [the focused update](CONFTUNER_FINE_UPDATE_2026-09-15.md). Agreement on these two trained mappings does not establish general invariance: additional-legend decision flips persist. Low disagreement alone is insufficient; retain predictive quality and invalidity alongside it. Zero observed flips and a degenerate bootstrap interval do not establish zero population risk.

Strict_v1 remains primary and historical strict caches are unchanged. Lenient_v1 is secondary: bare probabilities in [0,1], bare percentages in (1,100], and suffixed percentages must land exactly on the grid. Off-grid values, multiple numbers and explanatory text remain invalid. Both contracts preceded the new generations; the old 1,000-question lenient reanalysis is retrospective. Earlier GEPA search optimized strict parsing.

No richer-set advantage is inferred here. The earlier supervised-family comparison did not establish improvement beyond mixture plus elementary spread features. Any extension to the strongest new control requires validation-fitted selectors, same-call baselines and held-out questions. No objective is selected or retuned on these inspected test results.

The ICLR two-failure headline is unchanged. Its original-cell letter11 mapping control is complete; insertion text and §7 pointers are in [the ICLR note](../ICLR_MAPPING_CONTROL_AND_SECTION7.md). New training, the six-bank follow-up results and direct-format transfer remain separate.

Artifacts:

- [Actual-emission tables](runs/scale_family_v1/trained_emission_analysis/REPORT.md), [paired differences](runs/scale_family_v1/trained_emission_analysis/paired_differences.csv), [stability](runs/scale_family_v1/trained_emission_analysis/stability.csv), [direct histograms](runs/scale_family_v1/trained_emission_analysis/direct_histograms.json).
- [Auxiliary distribution analysis](runs/scale_family_v1/students/analysis/REPORT.md), including teacher fit, codebook following, held-out mappings and cross-grid disagreement.
- [Unified verified prediction file](runs/scale_family_v1/students/analysis/verified_reporting_predictions.jsonl), with scale-specific prompt/token/model audits beside the source files.
- [Training budget audit](runs/scale_family_v1/TRAINING_BUDGET_AUDIT.json), [reporting contract](REPORTING_CONTRACT.md), [old direct-grid audit](runs/emitted_validation_v1/lenient_v1/DIRECT_GRID_AUDIT.md).
'''
    report=root/'COMPLETED_TRAINING_ANALYSIS_2026-09-15.md'
    report.write_text(text)
    p=root/'CONFTUNER_FINE_UPDATE_2026-09-15.md'
    p.write_text(p.read_text().replace('Remaining fine objective readouts are still being collected.', 'All remaining fine objective readouts are complete; see COMPLETED_TRAINING_ANALYSIS_2026-09-15.md.'))
    p=root/'REPORTING_CONTRACT.md'
    p.write_text(p.read_text().replace('their emission and held-out distribution evaluations are in progress.', 'their emission and held-out distribution evaluations are complete.').replace('its fine-grid readouts are being collected.', 'its fine-grid readouts are complete.'))
    old=root/'ANALYSIS_2026-09-15.md'
    marker='> Subsequent completed training and parser update:'
    if marker not in old.read_text():old.write_text(marker+' [expanded pilot analysis](COMPLETED_TRAINING_ANALYSIS_2026-09-15.md). Earlier pending-training statements below describe the prior snapshot.\n\n'+old.read_text())
    state=json.loads((root/'EXPERIMENT_STATUS.json').read_text())
    state['completed']+=['coarse/fine ConfTuner-style and fine supervised/distillation training and evaluation','declared strict_v1 and lenient_v1 analysis','original ICLR-cell letter11 mapping control','direct-grid histogram and paired transfer-cost audit']
    state['completed']=list(dict.fromkeys(state['completed']))
    state['pending']=['ADVICE paired-answer training comparison','full-scale independent held-out replication','strict equal-realized-cost GEPA and DINCO main comparisons','family selection on the strongest new control beyond mixture and elementary spread']
    state['tests_passed']=29
    for filename in list(state['artifact_sha256'])+['COMPLETED_TRAINING_ANALYSIS_2026-09-15.md','runs/scale_family_v1/students/contract_audit_fine/contract_audit.json']:
        state['artifact_sha256'][filename]=hashlib.sha256((root/filename).read_bytes()).hexdigest()
    state['latest_report']=str(report.relative_to(root))
    (root/'EXPERIMENT_STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
    print('Completed expanded training report:',report,flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--wait',action='store_true');a=p.parse_args();main(a.wait)
