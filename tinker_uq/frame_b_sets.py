"""Constructive report-set diagnostics; no new model calls or predictive claims."""
import collections
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from tinker_uq.emitted_reports import parse
from tinker_uq.report_sets import decisions,paired_ordering,pbox

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/frame_b_report_sets_v1'
RS=Path('/Users/tanmoy/research/Credal_Sets/Random_Sets/rsuq/rsuq')
HASHES={}


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    HASHES[str(p)]=sha(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def write(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def jsonl(p,rr):p.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rr))
def csvout(p,rr):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def fraction(mask,ix):
    low,high=np.quantile(np.asarray(mask,dtype=float)[ix].mean(1),[.025,.975])
    return dict(count=int(np.sum(mask)),fraction=float(np.mean(mask)),lo=float(low),hi=float(high))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('rsuq_report_mass',RS/'core/beliefs.py');rs=importlib.util.module_from_spec(spec);spec.loader.exec_module(rs)
    HASHES[str(RS/'core/beliefs.py')]=sha(RS/'core/beliefs.py')
    # Declare taxonomy before producing the new diagnostic outputs.
    write(OUT/'PROTOCOL.json',dict(primary_emissions='actual normal/reversed reports; invalid if either output invalid; strict_v1 primary and lenient_v1 secondary',
        paired_answer_panel='Original TruthfulQA conditional decoded expectations only; actual paired natural-confidence emissions are not available in these caches.',
        ordering='D_g=c_g(correct)-c_g(incorrect), same legend within each contrast; strict positive/negative, actual sign changes, and three tie subtypes are disjoint.',
        tolerance='Exact stored-value ties primary (epsilon=0); epsilon=1e-6 sensitivity reported separately without replacing primary.',
        thresholds=[.5,.75,.9],bootstrap=dict(draws=2000,seed=20260915,unit='question with both candidates and legends retained'),
        pbox='Auxiliary per-question aligned confidence distributions; rsuq cluster_prob_mass sums grid values <= t. Tiny saved normalization residuals removed only for CDF construction.',
        semantics='Finite report families, not truth bounds, not a predictive improvement, not a DS belief assignment. No across-question histogram treated as a per-question distribution.'))
    # Actual-report sets, first: original transfer cache and all completed students.
    bases=[('coarse_control_transfer',ROOT/'runs/emitted_validation_v1')]
    plan=json.loads((ROOT/'runs/scale_family_v1/student_plan_conftuner.json').read_text())
    bases += [(method,ROOT/'runs/scale_family_v1/trained_emissions'/scale/method) for scale,methods in plan['methods'].items() for method in methods]
    emission_stats=[];emission_records=[]
    for method,base in bases:
        rows=load(base/'reports.jsonl')
        for scale in sorted({r['scale'] for r in rows}):
            groups={g:{r['question_id']:r for r in rows if r['scale']==scale and r['format']==g} for g in ['normal','reversed']}
            ids=sorted(groups['normal']);assert set(ids)==set(groups['reversed'])
            ix=np.random.default_rng(20260915).integers(0,len(ids),(2000,len(ids)))
            for contract in ['strict_v1','lenient_v1']:
                values=[]
                for i in ids:
                    a,b=groups['normal'][i],groups['reversed'][i]
                    assert all(a[k]==b[k] for k in ['question','answer','correct','split','model_revision'])
                    v=[parse(r['raw_response'],r['mapping'],False,contract_name=contract) for r in [a,b]]
                    values.append([np.nan if x is None else x for x in v])
                    emission_records.append(dict(method=method,scale=scale,split=a['split'],contract=contract,question_id=i,fixed_answer_id=a['fixed_answer_id'],correctness_label=a['correct'],reports=v,lower=min(v) if None not in v else None,upper=max(v) if None not in v else None,valid=None not in v,model_revision=a['model_revision']))
                for threshold in [.5,.75,.9]:
                    result=decisions(values,threshold)
                    assert np.all(sum(result[k].astype(int) for k in ['all_accept','all_reject','legend_dependent','invalid'])==1)
                    for category in ['all_accept','all_reject','legend_dependent','invalid']:
                        emission_stats.append(dict(method=method,scale=scale,split=groups['normal'][ids[0]]['split'],contract=contract,threshold=threshold,category=category,n_questions=len(ids),**fraction(result[category],ix)))
    csvout(OUT/'actual_emission_decisions.csv',emission_stats);jsonl(OUT/'actual_emission_sets.jsonl',emission_records)
    # Frame B auxiliary paired-answer panel, preserving source candidate pairs.
    source=load(ROOT.parent/'cached_results/fresh_candidate_correctness/verified_predictions.jsonl')
    source=[r for r in source if r['dataset']=='truthfulqa']
    paired_stats=[];paired_records=[];envelopes=[];pairing_stats=[]
    for model in sorted({r['model'] for r in source}):
        rows=[r for r in source if r['model']==model];groups={}
        originals={}
        for g in ['forward','reversed']:
            path=ROOT.parent/f'letter11_provenance_v1/{model}/truthfulqa/{g}.pt';HASHES[str(path)]=sha(path)
            originals[g]=torch.load(path,map_location='cpu',weights_only=False)
        for r in rows:
            side='pos' if r['correct']==1 else 'neg';key=(r['question_id'],side,r['legend']);assert key not in groups;groups[key]=r
            original=originals[r['legend']];j=[str(x) for x in original['example_ids']].index(r['question_id'])
            assert original['prompt_hashes_'+side][j]==r['prompt_provenance']['text_sha256']
            assert np.allclose(np.asarray(original['Vdist_'+side][j]),r['probability_vector'],atol=1e-6)
        ids=sorted({r['question_id'] for r in rows});assert len(ids)==817 and len(groups)==4*len(ids)
        ix=np.random.default_rng(20260915).integers(0,len(ids),(2000,len(ids)))
        pos=np.array([[groups[i,'pos',g]['decoded_mean'] for g in ['forward','reversed']] for i in ids]);neg=np.array([[groups[i,'neg',g]['decoded_mean'] for g in ['forward','reversed']] for i in ids])
        primary=paired_ordering(pos,neg)
        for epsilon in [0.,1e-6]:
            result=paired_ordering(pos,neg,epsilon)
            for category in ['consistently_correct','consistently_incorrect','sign_change','tie_all','tie_and_correct','tie_and_incorrect']:
                paired_stats.append(dict(model=model,dataset='truthfulqa',readout='conditional_decoded_expectation',tolerance=epsilon,category=category,n_questions=len(ids),**fraction(result['category']==category,ix)))
            tied=np.isin(result['category'],['tie_all','tie_and_correct','tie_and_incorrect'])
            paired_stats.append(dict(model=model,dataset='truthfulqa',readout='conditional_decoded_expectation',tolerance=epsilon,category='ties_total',n_questions=len(ids),**fraction(tied,ix)))
        lost=((primary['category']=='consistently_correct')&(primary['cartesian_lower']<=0))|((primary['category']=='consistently_incorrect')&(primary['cartesian_upper']>=0))
        pairing_stats.append(dict(model=model,n_questions=len(ids),description='Strict paired ordering lost by separate candidate min/max intervals',**fraction(lost,ix)))
        for j,i in enumerate(ids):
            paired_records.append(dict(model=model,question_id=i,readout='conditional_decoded_expectation',correct_candidate_row_id=groups[i,'pos','forward']['row_id'],incorrect_candidate_row_id=groups[i,'neg','forward']['row_id'],legends=['forward','reversed'],correct_candidate_reports=pos[j].tolist(),incorrect_candidate_reports=neg[j].tolist(),contrasts=primary['contrasts'][j].tolist(),lower=float(primary['lower'][j]),upper=float(primary['upper'][j]),category=primary['category'][j],cartesian_lower=float(primary['cartesian_lower'][j]),cartesian_upper=float(primary['cartesian_upper'][j])))
        for side in ['pos','neg']:
            aligned=[]
            for i in ids:
                family=[]
                for g in ['forward','reversed']:
                    r=groups[i,side,g];order=np.argsort([r['mapping'][k] for k in r['letters']]);p=np.array(r['probability_vector'])[order]
                    assert abs(p@np.linspace(0,1,11)-r['decoded_mean'])<1e-6
                    family.append(p)
                aligned.append(family)
            box=pbox(aligned,np.linspace(0,1,11),rs.cluster_prob_mass)
            assert np.allclose(box['cdf'],np.cumsum(box['aligned'],axis=-1))
            assert np.all(box['pbox_mean_lower']<=box['family_mean_lower']+1e-12) and np.all(box['pbox_mean_upper']>=box['family_mean_upper']-1e-12)
            for j,i in enumerate(ids):
                envelopes.append(dict(model=model,question_id=i,candidate_side=side,grid=np.linspace(0,1,11).tolist(),legends=['forward','reversed'],aligned_distributions=box['aligned'][j].tolist(),cdf_by_legend=box['cdf'][j].tolist(),cdf_lower=box['cdf_lower'][j].tolist(),cdf_upper=box['cdf_upper'][j].tolist(),family_mean_interval=[float(box['family_mean_lower'][j]),float(box['family_mean_upper'][j])],pbox_admitted_mean_interval=[float(box['pbox_mean_lower'][j]),float(box['pbox_mean_upper'][j])],allowed_alphabet_mass=[groups[i,side,g]['allowed_alphabet_mass'] for g in ['forward','reversed']]))
    csvout(OUT/'paired_ordering.csv',paired_stats);csvout(OUT/'pairing_preservation.csv',pairing_stats);jsonl(OUT/'paired_answer_contrasts.jsonl',paired_records);jsonl(OUT/'auxiliary_pboxes.jsonl',envelopes)
    # The exact equal-mixture example; CDF envelopes differ despite identical mixtures.
    example=pbox(np.array([[[.5,.5],[.5,.5]],[[1.,0.],[0.,1.]]]),np.array([.6,1.]),rs.cluster_prob_mass)
    assert np.allclose(example['aligned'].mean(1),[[.5,.5],[.5,.5]])
    assert np.allclose(example['family_mean_lower'],[.8,.6]) and np.allclose(example['family_mean_upper'],[.8,1.])
    write(OUT/'toy_example.json',{k:v.tolist() for k,v in example.items()})
    assert all(sha(Path(p))==h for p,h in HASHES.items())
    write(OUT/'AUDIT.json',dict(status='PASS',source_hashes=HASHES,rsuq_reuse='core/beliefs.py::cluster_prob_mass for ordered-grid CDF events; no embedding clusters, credal_width or pignistic transformation used',paired_questions=len(paired_records),pboxes=len(envelopes),emitted_sets=len(emission_records),new_model_calls=0,paired_actual_emissions_available=False))
    text='# Report sets: conclusions that survive the tested legend\n\nActual emissions are analyzed first in `actual_emission_decisions.csv`: L>=t means all evaluated legends accept; U<t means all reject; L<t<=U means the decision depends on the legend. Invalid pairs are a fourth outcome and stay in the denominator. These statements concern observed reporting decisions, not correctness guarantees.\n\nThe original paired TruthfulQA caches contain conditional distributions only. The following Frame B table therefore uses their **decoded expectations**, not actual emitted confidences. There are 817 paired questions per model. Each contrast compares the curated correct and incorrect candidates under the same legend. Intervals are paired question bootstrap intervals, conditional on the fitted models and two saved legends.\n\n| Model | Consistently correct | Consistently incorrect | Sign change | Any tie |\n|---|---:|---:|---:|---:|\n'
    for model in sorted({r['model'] for r in paired_stats}):
        cells=[]
        for cat in ['consistently_correct','consistently_incorrect','sign_change','ties_total']:
            r=next(r for r in paired_stats if r['model']==model and r['category']==cat and r['tolerance']==0)
            cells.append(f"{r['count']}/817 ({r['fraction']:.1%}; {r['lo']:.1%}–{r['hi']:.1%})")
        text+='| '+model+' | '+' | '.join(cells)+' |\n'
    text+='\nExact stored-value ties are primary. `paired_ordering.csv` separately reports all-legends ties, ties with correct ordering, ties with incorrect ordering, and the declared 1e-6 sensitivity analysis. Ties are not sign reversals. `paired_answer_contrasts.jsonl` retains both candidates and the same-legend contrasts.\n\nIndependent candidate intervals can discard useful pairing: `pairing_preservation.csv` counts cases where strict same-legend ordering becomes unresolved after replacing paired contrasts by the Cartesian difference of candidate intervals. This is a representation diagnostic, not evidence of a better predictor.\n\nAuxiliary p-boxes retain per-question CDF lower/upper envelopes **and** the underlying paired distributions. The implementation reuses the supplied rsuq `cluster_prob_mass` function to aggregate probability on values <=t. The rsuq cluster-size width and DS mass constructions are not substituted for this ordered-grid report family. At acceptance threshold t, probability of V>=t uses 1-F(t-), not 1-F(t). Across-question histograms are not these per-question CDFs.\n\nThe exact stable/unstable equal-mixture example is saved in `toy_example.json`. The p-box can admit more distributions than the original family; both family and p-box mean intervals are exported. Neither bounds true correctness probabilities without additional assumptions. References: [Troffaes and Destercke](https://arxiv.org/abs/1103.1805), [Ferson et al.](https://www.cs.utep.edu/vladik/2003/sandia03.pdf).\n\nICLR scope: retain the two-failure structure. This paired conditional-readout table may support a constructive interpretation of Frame B, explicitly labeled by readout. An **actual-emission paired-answer table remains unavailable** without collecting natural-confidence generations for both original candidates. The current TriviaQA emission caches supply only one fixed candidate per question; known-value mapping controls do not fill this gap. Keep p-boxes in the auxiliary follow-up. No richer-set predictive advantage, calibration guarantee, conformal coverage or new learned objective is claimed.\n'
    excerpt='Actual normal/reversed report sets at t=0.75, strict_v1:\n\n| Reporter / sample | Grid | n | All accept | All reject | Legend-dependent | Invalid |\n|---|---|---:|---:|---:|---:|---:|\n'
    for method in ['coarse_control_transfer','both_token_brier']:
        for scale in ['coarse','fine']:
            rs={r['category']:r for r in emission_stats if r['method']==method and r['scale']==scale and r['contract']=='strict_v1' and r['threshold']==.75}
            label='Coarse-trained control / validation' if method=='coarse_control_transfer' else 'Grid-trained ConfTuner / exploratory test'
            excerpt+=f"| {label} | {scale} | {rs['all_accept']['n_questions']} | "+' | '.join(f"{rs[k]['fraction']:.1%}" for k in ['all_accept','all_reject','legend_dependent','invalid'])+' |\n'
    excerpt+='\nThe 1,000-question validation and 128-question exploratory test samples are different populations. Letter parsing is unchanged by lenient_v1, so these two-legend categories agree under both parser contracts. The full CSV gives paired question intervals and thresholds 0.5, 0.75 and 0.9.\n\n'
    text=text.replace('The original paired TruthfulQA caches contain conditional distributions only.',excerpt+'The original paired TruthfulQA caches contain conditional distributions only.')
    (OUT/'REPORT.md').write_text(text)
    print(text)


if __name__=='__main__':main()
