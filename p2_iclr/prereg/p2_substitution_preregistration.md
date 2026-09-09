# P2/ICLR preregistration — letter-token substitution arm

Frozen 9 September 2026, before any substitution forward pass.

## Arm

- First cell: Llama/TruthfulQA. The arm may be extended to other cells only
  under a separately recorded amendment.
- Base: canonical `letter11-v1` forward legend, `A=0% ... L=100%`.
- The substituted position is `H`, the eighth label in the eleven-label list,
  denoting 70%. `H` was selected using training data only: it is the highest
  mean `Vdist_pos` mass on each of the three prespecified training folds
  (seed 0, 1, and 2; 491 items per fold). The fold means are `.179865`,
  `.179173`, and `.180288`; their mean is `.179776` and their sample SD is
  `.000563`.
- Change only the label occupying H's position in the legend. Keep its 70%
  value, list position, prompt template, candidate answers, and read position
  fixed. Record prompt hashes and the exact replacement text.
- Use two replacement letters, selected before the substitution forward pass
  from `{M,N,P,R,S,T,W,X,Y,Z}`. After the all-model single-token gate, choose
  the low-prior and high-prior replacements by minimum and maximum neutral-
  prompt probability, respectively; ties are broken lexicographically. The
  prior audit must publish per-letter raw mass before this selection. No
  replacement is selected from item-conditioned V values.

## Token and prior gates

- Each replacement must be a single token in the actual rendered template on
  all three tokenizers and must not collide with any base label.
- Run the content-free confidence-question prior first and save per-letter
  full-vocabulary probabilities and raw alphabet mass. The current prior
  artifact reports only total alphabet mass, so the substitution arm is not
  yet runnable.

## Outcomes and frozen rules

- Primary: paired per-item mass comparison `p(new)` versus `p(H)` at the
  confidence read position. Report mean difference and item-bootstrap CI for
  each replacement. The tolerance is `delta = 0.000563`, the sample SD of
  training-fold mean `p(H)` across the three fixed seeds. Invariance passes
  this descriptive tolerance when `abs(mean(p(new)-p(H))) < delta`; this is
  not a claim that item-level mass is identical.
- Secondary: Spearman correlation of decoded `V+_fwd` and `V+_sub` against
  the canonical `rel_fwd` rule.
- Report where H's mass goes using the transfer vector over the unchanged
  labels and the replacement label. Interpret token-tracking direction only
  if the pre-registered low/high-prior ordering is observed.
- Apply the existing 5-point adequacy gate to substituted `V+`.
- The arm has its own scheme count and cannot change letter11 counts.

## Current status

The training-fold H choice and tolerance are resolved. The per-letter neutral
prior column and the all-three-tokenizer replacement gate remain open. No
substitution result may be reported until both gates pass.
