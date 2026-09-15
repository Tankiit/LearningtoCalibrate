# ICLR: evidence-matched wording and commissioned experiments

Recorded 15 September 2026 from the supplied research decisions. This document commissions work; it does not assert that the new input audits, extraction, correctness probe, or joint bootstrap have run. Saving this plan does not update an external manuscript or submit an abstract.

## Recommended decisions

Run original-model emissions and the candidate-correctness probe, subject to the concrete input checks below. Narrow/reword the registration abstract immediately. The abstract must stand on completed evidence; neither experiment is a promised positive result. If an input or deadline gate fails, retain the narrower wording and mark the experiment as **not run**.

| Issue | Wording supported now | Additional evidence to commission |
|---|---|---|
| Confidence readout | Conditional decoded expected confidence over the permitted alphabet | Actual confidence emissions on the original three TruthfulQA cells |
| State-probe target | Teacher-forced candidate preference | Candidate-level correctness probe with question-disjoint evaluation |
| Reversal reference | Agreement intervals fall below saved reliability point references | Joint uncertainty for reliability minus reversal agreement |

ICLR's deadlines are **18 September 2026, 23:59 AoE for abstracts** and **25 September 2026, 23:59 AoE for papers**. The official guidelines allow titles and abstracts to be edited before the submission deadline and require genuine, informative abstracts reflecting the full submission. Register a supported abstract and keep revisions close to the registered paper as a project constraint. [ICLR 2027 Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), checked 15 September 2026.

Suggested internal checkpoints: input/protocol decisions by **17 September**; evidence review by **22 September**; Tanmoy then integrates before the full-paper deadline. These are proposed project checkpoints, not conference requirements.

## 1. Original-model emissions and the readout boundary

### Current evidence and wording

Original reversal correlations, report-channel gaps and paired-ordering results use a conditional expectation of the next-symbol distribution. The supplied-value mapping control tests a different task. Its **383/384 reversed contextual accuracy** does not establish valid natural-confidence emissions or their invariance.

Wording now:

> Reversing the letter-to-value legend changes the confidence decoded from the model's conditional distribution over the permitted symbols.

For the answer-comparison result:

> On TruthfulQA, legend reversal changes which candidate receives higher decoded expected confidence in each of the three models.

The saved sign-change rates are **30.7%, 39.3% and 31.8%** for Llama, Mistral and Qwen. Keep the readout explicit. These changes cannot be explained by a common confidence offset, but do not establish emitted-confidence instability.

### Collection decision

Commission one matched emission study. Use the original pinned Llama 3.1 8B, Mistral 7B and Qwen 2.5 7B checkpoints; all **817 original TruthfulQA question pairs**; both candidates; and forward/reversed **letter11-v1 (A–L, skipping I)**. One response per cell gives:

**3 models × 817 questions × 2 candidates × 2 legends = 9,804 responses.**

This is a count, not a verified Modal runtime estimate. Do not substitute a Tinker pilot model and call it an original-cell replication.

Before collection:

1. Resolve exact checkpoint/tokenizer revisions and candidate IDs. Verify the literal natural-confidence prompt and rendered token prefix; a frozen known-value prompt is insufficient.
2. Freeze generation, stopping/truncation rules and whole-response grammar. Prefer one deterministic unconstrained generation per cell for this minimum study. A prompted alphabet is distinct from hard masking the vocabulary. If a new wrapper is necessary, version and disclose the change.
3. Run a fixed shard, such as eight questions per model: **96 responses**. Check joins, runtime, raw-response capture and parser behavior. Do not optimize the prompt against reversal results. If the protocol changes, recollect the shard under the final contract.

**Save:** raw text and token IDs; question/candidate IDs and external labels; model/prompt/parser/generation hashes; legend; parsed symbol and decoded value; validity and failure reason. Keep original conditional readouts separate. An emitted letter decoded to a number is a letter report, not a directly emitted numeral.

**Analyze:** report validity by legend, numerical changes on matched valid responses, and paired-ordering categories using all four responses per question. Separate strict sign changes, ties and invalid bundles. Give valid-bundle and all-question denominators; invalid responses are not zero confidence. Bootstrap questions with their candidates and legends kept together. Retain per-legend predictive quality alongside stability.

**Claim gate:** fix the endpoint before full-result inspection. Numerical change, ordering reversal and invalidity are different findings. Widen to an emitted-ordering claim only when valid emissions support that result, with denominators and intervals; scope replication to the models that support it. Stable emissions restrict Failure 2 to the conditional readout. Mostly invalid emissions support a compliance limitation. Neither licenses the original stronger wording.

**Deliverable:** protocol manifest, raw emissions, joined analysis table and a decision note: supported emission claim, readout-only claim, or inconclusive. Existing mapping-control and trained-pilot emissions cannot replace this study.

## 2. Candidate correctness and the honest S4

### Current target and wording

Current target: `1[logprob_pos_mean > logprob_neg_mean]`.

This labels whether the curated correct candidate has higher teacher-forced mean log-probability than the incorrect candidate. It is a question-level preference event, not a candidate-level correctness label. Matching the state/report target makes that comparison interpretable; it does not turn the target into correctness recognition.

Honest replacement for S4 now:

> Paired-state probes predict teacher-forced candidate preference better than question-only probes, providing an internal-state comparator for the confidence readout.

Document the gold-indexed paired inputs and target in methods. This premise does not establish a deployable detector that knows which answer is correct. If the additional experiment is unavailable, retain Failure 1 as the protocol-level identification argument with this preference comparator; do not describe it as an empirical demonstration of correctness knowledge.

### Probe decision

Commission a small frozen-feature correctness experiment. First inventory separately recoverable candidate states and external candidate labels. Reuse existing states if suitable. A summary metric or a target-selected direction is insufficient. If extraction is needed, specify that cost before calling the work cache-only.

Form rows `(question_id, candidate_id, h(q,a), correctness)` using curated candidate labels independent of model likelihood. The same scalar reader scores either candidate separately. Never provide a gold-ordered pair or a correct/incorrect position flag to this classifier.

Reuse existing question-disjoint split definitions and seeds **0–2** where applicable. Both candidates stay together. Freeze the cached layer/pooling and use a regularized linear reader. Fit preprocessing and choose regularization on training/development questions only. Report actual sample counts and overlap with the 817-pair report population.

Compare against each original legend's decoded confidence on the exact held-out candidates. Add actual emissions separately if available, with their validity masks. Include a cheap candidate-length baseline: paired texts can differ systematically, and state decodability alone does not identify factual reasoning.

Report candidate AUROC and within-question correct-candidate ordering accuracy, with ties worth one half. Give question-level intervals for reader-minus-report differences. Show seed variation separately; appearances in multiple splits are not independent examples.

Evaluate the predeclared wrong-under-both-legends subset after fitting. Keep its count and uncertainty explicit and label it a secondary analysis conditional on report failure. Do not select the reader on this subset.

**Question-only control:** with one correct and one incorrect candidate per question, a question-only score is identical for both. Its within-pair ordering accuracy is one half by construction. This is a design property, not evidence that question difficulty is generally uninformative. Keep this task distinct from the old question-level preference task.

**Claim gate:** a held-out correctness gain supports decodable information about curated candidate correctness on the tested population. Superiority over reporting additionally requires its matched comparison and uncertainty. Neither implies unique internal true confidence or causal use of the decoded information. If results are mixed, weak or unavailable, retain the honest S4; **not run is not failed**.

**Deliverable:** target/input/split manifest, held-out predictions with IDs/seeds, state-versus-report correctness table and evidence-matched S4. No broad new seed or backbone sweep.

## 3. Uncertainty in the reversal reference

The collaborator's gate check is correct relative to the saved point references. The supplied ledger gives:

| TruthfulQA model | Forward/reversed correlation | Reported 95% interval | Saved rel_fwd |
|---|---:|---|---:|
| Llama | 0.1227 | [0.0475, 0.1960] | 0.3510 |
| Mistral | 0.0583 | [-0.0094, 0.1223] | 0.3653 |
| Qwen | 0.3400 | [0.2755, 0.4037] | 0.4684 |

All three interval upper limits are below their references. Existing intervals propagate uncertainty in reversal agreement only. These ledger values are preserved here; this documentation update does not recompute them.

**Required:** recover the exact `rel_fwd` definition and question-level records. Verify comparable quantities, readouts and populations before estimating:

`D = forward reliability − forward/reversed agreement`.

Recompute both statistics in each bootstrap draw. Preserve shared question IDs and all associated records; account for the **817-versus-500 sample sizes** and actual overlap. If using only the common subset, label it a matched-subset analysis and retain the original full-population estimates separately. State whether inference conditions on fitted components.

**Interpretation:** an interval for D wholly above zero supports lower reversal agreement relative to this empirical benchmark. An interval crossing zero leaves the comparison inconclusive. Neither establishes an absolute measurement-validity threshold. Do not subtract separate interval endpoints or treat three models evaluated on the same questions as independent datasets.

**Fallback:** if reference records or a comparable definition are unavailable, state:

> The reversal-agreement intervals lie below the saved reliability point estimates; uncertainty in the reliability estimates is not jointly propagated.

Keep this comparison conditional/descriptive. Do not present it as a fully uncertainty-propagated falsification test.

**Deliverable:** source/overlap audit and a three-row table with both statistics, their difference and its interval—or the explicit fallback limitation.

## Code priorities and repository disposition

The following is an implementation inventory, not a completion claim for the commissioned experiments.

| Priority | Preserve/reuse | Required work and boundary |
|---|---|---|
| P0: contracts and provenance | `letter11_provenance_extract.py`, `letter11_mapping_control.py`, `tinker_uq/tinker_uq/reporting.py`, `emitted_reports.py` in that package, and both parser contract JSON files | Audit original-cell prompt/tokenizer/checkpoint/IDs; freeze the natural-confidence generation and whole-response grammar. Known-value control and pilot numeric parsers cannot silently replace that contract. |
| P1: original emissions | Original-cell extraction and mapping-control infrastructure; `letter11_mapping_control_analysis.py`; package `emission_analysis.py`, `emission_stability.py`, `report_sets.py` | Implement the matched original-model natural-confidence emission runner and its joined analysis after the input gate. Run the 96-response shard before the 9,804-response collection. Existing infrastructure is not the completed experiment. |
| P1: correctness probe | Existing state extraction, question-split definitions, candidate provenance and report caches | Inventory candidate states/labels first, then implement a separate candidate-level linear probe, length baseline and paired evaluation. `state_readout_test.py` is an input/method reference, not the commissioned correctness probe. |
| P1: reference uncertainty | Original reversal analysis and reference records, if recoverable | Implement a source/overlap audit and joint question bootstrap after establishing the exact reliability statistic; otherwise use the stated fallback. |
| Supporting reproducibility | Matched training/analysis, ConfTuner and fine-grid runners, DINCO/GEPA, family analyses, binary verification and completed-pilot audits | Keep reusable scientific code tracked. These controls remain useful supporting evidence; do not conflate their models, targets or readouts with the original ICLR cells. |

Retain `tinker_uq/tinker_uq/finish_coarse.py`: it collects existing checkpoints. Retain `tinker_uq/tinker_uq/coarse_summary.py`: it computes individual/ensemble statistics and question-bootstrap intervals. Neither is merely disposable prose generation.

Four one-off local report/status builders are excluded from tracking, with their working-tree copies preserved:

- `build_iclr_report.py`
- `compile_results.py`
- `tinker_uq/post_analysis.py`
- `tinker_uq/finish_training_report.py`

Generated result/cache/report paths and the four recent local tests remain ignored under the existing policy. Ignore rules do not remove historically tracked results; this change does not erase those files or rewrite repository history. `things2do.md` is an explicit exception to the Markdown ignore rule, alongside the root README.

### Execution checklist

- [ ] Integrate the narrow readout wording and honest S4 into the actual registration manuscript; do not promise either experiment will be positive.
- [ ] Freeze original-model identifiers, candidate joins, natural-confidence prompts, parser and generation manifest by the input checkpoint.
- [ ] Run and audit the 96-response shard; record measured runtime before committing to full collection.
- [ ] If gates pass, collect and analyze all 9,804 original-model responses; otherwise record not run and retain the conditional-readout claim.
- [ ] Audit separately recoverable candidate states, external labels, splits and extraction cost.
- [ ] If inputs pass, fit the frozen-feature correctness probe and length control; save held-out predictions and matched intervals.
- [ ] Recover reliability definitions, records and population overlap; compute joint intervals or retain the explicit limitation.
- [ ] Review evidence by 22 September; integrate only supported claims before the full-paper deadline.

The supplied plan is preserved above with formatting and explicit status notes. No new experiment is launched by this documentation/ignore update. The local comprehensive `iclr_report.md` remains a separate ignored report; it is not replaced by this task list.
