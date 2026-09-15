"""Matched actual-emission readouts; keep parser contracts and call budgets separate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .data import load_jsonl, write_json
from .emission_analysis import analyze as verify_and_analyze
from .matched_analysis import csv_save, metric_vector
from .emitted_reports import contract as render_contract, parse, POLICY
from .reporting import digest


def interval(values):
    return [float(x) for x in np.nanquantile(values, [.025, .975])]


def control_rows(source, scales, methods):
    rows=[]
    for scale in scales:
        for method in methods[scale]:
            folder=source/'trained_emissions'/scale/method
            if not (folder/'mapping_controls.jsonl').exists():continue
            manifest=json.loads((folder/'manifest.json').read_text())
            maps=json.loads((folder/'mappings.json').read_text())[scale]
            reports=load_jsonl(folder/'mapping_controls.jsonl')
            expected_keys={(fmt,v) for fmt in manifest['formats'] for v in maps['normal'].values()}
            if {(r['format'],r['known_value']) for r in reports}!=expected_keys or len(reports)!=len(expected_keys):raise ValueError('Incomplete or duplicate known-value control')
            for r in reports:
                mapping=maps['normal' if r['format']=='direct' else r['format']]
                prompt=f"The supplied probability is {r['known_value']*100:g}%.\nEncode only this known probability using the reporting contract.\n"+render_contract(mapping,r['format']=='direct')
                if r['mapping']!=mapping or r['prompt_text']!=prompt or r['prompt_text_sha256']!=hashlib.sha256(prompt.encode()).hexdigest():raise ValueError('Control prompt mismatch')
                if r['full_prompt_hash']!=digest(r['prompt_token_ids']) or r['model_revision']!=manifest['sampler_path'] or r['sampling_policy']!=POLICY:raise ValueError('Control provenance mismatch')
            for name in ['strict_v1','lenient_v1']:
                for fmt in manifest['formats']:
                    rs=[r for r in reports if r['format']==fmt]
                    vs=[parse(r['raw_response'],r['mapping'],fmt=='direct',contract_name=name) for r in rs]
                    rows.append(dict(scale=scale,method=method,contract=name,format=fmt,n=len(rs),correct=sum(v==r['known_value'] for v,r in zip(vs,rs)),valid=sum(v is not None for v in vs),calls=len(rs),input_tokens=sum(r['input_tokens'] for r in rs),generated_tokens=sum(r['generated_tokens'] for r in rs)))
    return rows


def run(source, scales=None, draws=2000, methods=None):
    source = Path(source)
    plan = json.loads((source / 'student_plan_conftuner.json').read_text())
    scales = scales or list(plan['methods'])
    if methods is not None:
        plan['methods']={s:[m for m in ms if m in methods] for s,ms in plan['methods'].items()}
    refs = {r['question_id']: r for r in load_jsonl(source / 'data.jsonl') if r['split'] == 'test'}
    ids = sorted(refs)
    y = np.array([refs[i]['correct'] for i in ids])
    n = len(ids)
    ix = np.random.default_rng(20260915).integers(0, n, (draws, n))
    metrics, stability, contrasts, histograms, exposures = [], [], [], [], []
    scores, ranges = {}, {}
    for scale in scales:
        for method in plan['methods'][scale]:
            base = source / 'trained_emissions' / scale / method
            for contract in ['strict_v1', 'lenient_v1']:
                folder = base if contract == 'strict_v1' else base / contract
                verify_and_analyze(folder, draws=draws)
                manifest = json.loads((folder / 'manifest.json').read_text())
                formats = manifest['formats']
                groups = {f: {} for f in formats}
                for r in load_jsonl(folder / 'reports.jsonl'):
                    groups[r['format']][r['question_id']] = r['emitted_value']
                p = np.array([[np.nan if groups[f][i] is None else groups[f][i] for f in formats] for i in ids])
                legends = [j for j, f in enumerate(formats) if f != 'direct']
                bundles = {f: [j] for j, f in enumerate(formats)}
                bundles.update(normal_reversed_average=[formats.index('normal'), formats.index('reversed')], six_report_average=legends[:6])
                for name, js in bundles.items():
                    value = p[:, js].mean(1)
                    valid = np.isfinite(value)
                    loss = np.where(valid, (value-y)**2, 1.)
                    scores[scale, method, contract, name] = loss
                    v = metric_vector(y[valid], value[valid]) if valid.any() else [np.nan]*3
                    lo, hi = interval(loss[ix].mean(1))
                    metrics.append(dict(scale=scale, method=method, contract=contract, report=name, calls=len(js), n=n, valid=int(valid.sum()), invalid_rate=float(1-valid.mean()), penalized_brier=float(loss.mean()), brier_lo=lo, brier_hi=hi, valid_only_brier=float(v[0]), valid_only_auroc=float(v[1]), valid_only_aurc=float(v[2])))
                # Individual reports remain distinct from the ensemble result.
                individual = np.where(np.isfinite(p[:, legends[:6]]), (p[:, legends[:6]]-y[:, None])**2, 1.)
                for name, value in [('mean_individual_brier', individual.mean(1))]:
                    lo, hi = interval(value[ix].mean(1))
                    metrics.append(dict(scale=scale, method=method, contract=contract, report=name, calls=6, n=n, valid=None, invalid_rate=None, penalized_brier=float(value.mean()), brier_lo=lo, brier_hi=hi))
                worst = int(np.argmax(individual.mean(0)))
                write_json(folder / 'analysis' / 'individual_brier.json', dict(mean=float(individual.mean()), worst_legend=formats[legends[worst]], worst=float(individual.mean(0)[worst]), six_report_average_is_distinct=True))
                for name, js in [('normal_reversed', bundles['normal_reversed_average']), ('first_six', legends[:6]), ('common_withheld_bank', legends[2:])]:
                    a = p[:, js]
                    valid = np.isfinite(a).all(1)
                    gap = np.full(n, np.nan)
                    flip = np.full(n, np.nan)
                    gap[valid] = np.ptp(a[valid], axis=1)
                    flip[valid] = (a[valid].min(1) < .75) & (a[valid].max(1) >= .75)
                    ranges[scale, method, contract, name] = (gap, flip)
                    gap_lo, gap_hi = interval(np.nanmean(gap[ix], axis=1)) if valid.any() else [np.nan]*2
                    flip_lo, flip_hi = interval(np.nanmean(flip[ix], axis=1)) if valid.any() else [np.nan]*2
                    stability.append(dict(scale=scale, method=method, contract=contract, bank=name, calls=len(js), n=n, complete=int(valid.sum()), excluded_invalid=int((~valid).sum()), mean_range=float(np.nanmean(gap)) if valid.any() else np.nan, range_lo=gap_lo, range_hi=gap_hi, p95_range=float(np.nanquantile(gap,.95)) if valid.any() else np.nan, flip_075=float(np.nanmean(flip)) if valid.any() else np.nan, flip_lo=flip_lo, flip_hi=flip_hi))
                direct = p[:, formats.index('direct')]
                values, counts = np.unique(direct[np.isfinite(direct)], return_counts=True)
                histograms.append(dict(scale=scale, method=method, contract=contract, histogram={str(v):int(c) for v,c in zip(values,counts)}, invalid=int(np.isnan(direct).sum())))
            for j, fmt in enumerate(formats):
                exposures.append(dict(scale=scale, method=method, format=fmt, student_training=fmt in (['normal'] if method=='normal_token_brier' else ['normal','reversed']), teacher_target=fmt != 'direct' and j <= 6, teacher_target_used_by_method=method in ('aligned_distill','aligned_distill_supervised'), note='All coarse mappings teacher-seen; fine target uses first six. Direct excluded from all student training.'))
    for scale in scales:
        comparisons = [('both_token_brier','normal_token_brier'),('both_mean_brier','both_token_brier'),('aligned_distill_supervised','aligned_distill'),('both_mean_brier','aligned_distill_supervised')]
        if scale == 'fine':
            comparisons += [('mean_consistency','both_mean_brier'),('mean_consistency','aligned_distill_supervised')]
        for a,b in comparisons:
            if a not in plan['methods'][scale] or b not in plan['methods'][scale]:continue
            for contract in ['strict_v1','lenient_v1']:
                for report in ['direct','normal','reversed','normal_reversed_average','six_report_average']:
                    delta = scores[scale,a,contract,report]-scores[scale,b,contract,report]
                    lo,hi = interval(delta[ix].mean(1))
                    contrasts.append(dict(scale=scale, contract=contract, comparison=a+'_minus_'+b, readout=report, metric='penalized_brier', n=n, delta=float(delta.mean()), lo=lo, hi=hi))
                for bank in ['normal_reversed','first_six']:
                    for k, metric in enumerate(['mean_range','flip_075']):
                        av,bv=ranges[scale,a,contract,bank][k],ranges[scale,b,contract,bank][k]
                        keep=np.isfinite(av)&np.isfinite(bv)
                        delta=np.where(keep,av-bv,np.nan)
                        lo,hi=interval(np.nanmean(delta[ix],axis=1)) if keep.any() else [np.nan]*2
                        contrasts.append(dict(scale=scale,contract=contract,comparison=a+'_minus_'+b,readout=bank,metric=metric,n=int(keep.sum()),delta=float(np.nanmean(delta)) if keep.any() else np.nan,lo=lo,hi=hi))
    out=source / ('trained_emission_analysis' if len(scales)==2 else 'trained_emission_analysis_'+'_'.join(scales))
    if methods is not None:out=out.with_name(out.name+'_selected_methods')
    out.mkdir(exist_ok=True)
    for name,rows in [('metrics',metrics),('stability',stability),('paired_differences',contrasts),('mapping_exposure',exposures)]:
        csv_save(out / (name+'.csv'),rows)
    write_json(out/'direct_histograms.json',histograms)
    controls=control_rows(source,scales,plan['methods'])
    if controls:csv_save(out/'mapping_controls.csv',controls)
    write_json(out/'protocol.json',dict(n=n,scales=scales,primary='strict_v1 actual emitted confidence',secondary='lenient_v1; declared before these generations',invalid='Brier loss 1; no imputation. Discrimination and stability conditional on validity, paired stability contrasts use common complete questions.',bootstrap='2000 paired question draws, fixed seed 20260915, fitted checkpoints and banks held fixed',scope='Exploratory 128-question preserved test subset. Earlier coarse results informed research direction. Not independent confirmation.',budget='1/2/6 deployment calls reported separately; training and teacher target costs in student analysis; same-call family and average comparisons only.'))
    text='# Matched trained numerical-emission pilot\n\nStrict_v1 remains primary; lenient_v1 is a declared secondary contract. Both were fixed before these fresh generations. These are actual outputs, not decoded expectations. Invalid reports receive Brier loss 1. AUROC/AURC and flip rates condition on validity; consult denominators. All intervals resample matched questions. This 128-question test pilot is exploratory, not independent confirmation.\n\n'
    for contract in ['strict_v1','lenient_v1']:
        text+=f'## {contract}\n\n| Grid | Student | Report | Calls | Brier [95% interval] | Invalid |\n|---|---|---|---:|---|---:|\n'
        for r in metrics:
            if r['contract']!=contract or r['report'] not in ['direct','normal','reversed','six_report_average']:continue
            text+=f"| {r['scale']} | {r['method']} | {r['report']} | {r['calls']} | {r['penalized_brier']:.4f} [{r['brier_lo']:.4f}, {r['brier_hi']:.4f}] | {r['invalid_rate']:.1%} |\n"
        text+='\n| Grid | Student | Bank | Complete / 128 | Mean range | Flip at .75 [95% interval] |\n|---|---|---|---:|---:|---|\n'
        for r in stability:
            if r['contract']!=contract or r['bank']=='common_withheld_bank':continue
            text+=f"| {r['scale']} | {r['method']} | {r['bank']} | {r['complete']} | {r['mean_range']:.4f} | {r['flip_075']:.1%} [{r['flip_lo']:.1%}, {r['flip_hi']:.1%}] |\n"
    text+='\nThe six-call coarse bank is exhaustive; fine uses six saved mappings. Additional fine mappings are excluded from teacher targets. Normal-only token Brier sees one mapping; all other students see normal/reversal. Low flip rates alone do not establish useful confidence. No richer-set gain is inferred from this table.\n'
    (out/'REPORT.md').write_text(text)
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('--scales',nargs='+',choices=['coarse','fine']);p.add_argument('--methods',nargs='+');a=p.parse_args();print(run(a.source,a.scales,methods=a.methods))
