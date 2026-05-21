# Reviewer Attack Map

This map connects appendix tasks to reviewer concerns. Keep the appendix work
defensive: one task, one concern, one table or note.

## S-tier concerns owned by lead author

These should stay on the main critical path and should not depend on appendix
timing.

### S1: `y_expert` is not a real expert signal

Defense: y-expert semantic audit on showcase datasets. Native signal should
beat shuffled and constant variants. This is main-paper material.

### S2: Gap is not theoretically justified by the SCOD framing

Defense: SCOD diagnostics plus D2 appendix comparison. Lead owns diagnostics;
collaborator owns the optimal-selector appendix table.

### S3: The result is post-hoc selection

Defense: signed pre-registration and neutral reporting against locked
predictions.

### S4: TruthfulQA behavior is just topic confounding

Defense: lead-owned topic-confound test. Collaborator can cite the result in
W3 rebuttals but should not own the experiment.

## A-tier concerns delegated to appendix

### A1: Independent reproduction

Task R1. A collaborator reruns the headline matrix on separate hardware and
software stack with Qwen2.5-7B-Instruct.

### A2: Too few LLM uncertainty baselines

Task G1. Add SelfCheckGPT on ChaosNLI and TruthfulQA and compare to semantic
entropy.

### A3: Gap may leave information on the table

Task D2. Compare probe-only, gap, SCOD-additive, calibrated SCOD-additive, and
learned 2D selector. If gap is near learned 2D on showcase datasets, the claim
is stronger. If not, report the gap honestly.

### A4: The paper will need fast, precise review responses

Task W3. Draft one-paragraph internal rebuttals for S-tier and A-tier concerns.

## B-tier/stretch concerns

- OOD-analogy diagnostics. Stretch only after R1, D2, G1, and W3.
- Robustness sweeps. Defer unless the main paper needs them.
- Full y-expert variant matrix outside showcase datasets. Lead-owned if needed.

