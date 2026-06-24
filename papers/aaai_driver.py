"""papers/aaai_driver.py — The AAAI experiment as a thin RSUQ driver.

THE experiment (the one the paper rests on): a CONTROLLED frame comparison.
Same K clusters, same trained mass head signature, same width formula —
only the membership rule changes (fixed kappa vs context-selected kappa_t).
Evaluation stratified by position type:

    Stratum P (polysemous-token positions): where the fixed frame's width is
        predicted to be blunted and the context frame is predicted to fix it.
    Stratum M (monosemous positions):       where the two frames coincide by
        construction — the built-in no-regression control.

The headline shape: 'surgical improvement' — detection gains concentrated in
Stratum P, identity on Stratum M (up to head-training noise). A diffuse gain
everywhere would actually be SUSPICIOUS (frames coincide on M), so Stratum M
doubles as an implementation sanity check.

Run order (each step gates the next):
  0. Build FixedFrame; collect occurrence states; build sense inventory
  0.5 Sense sanity: print sense neighbourhoods for famous polysemes
      ('bank', 'spring', 'bat', ...) — the Figure-1 material
  1. Train mass head ONCE on fixed-frame targets; reuse for both frames
     (ablation: retrain per frame — report both, the shared head is cleaner)
  2. Compute W under both frames at all eval positions
  3. Stratified evaluation + redundancy_screen + ablations
"""

from __future__ import annotations
import torch
import numpy as np

# from rsuq.core.frame import FixedFrame, ContextFrame
# from rsuq.core.senses import (pool_last4, detect_polysemous, build_senses,
#                               cluster_prototypes, map_senses_to_clusters)
# from rsuq.core.beliefs import MassHead, within_cluster_q
# from rsuq.core.signals import all_signals, credal_width
# from rsuq.diagnostics import (redundancy_screen, smoking_gun_pairs,
#                               auroc, separation_ratio, bootstrap_ci)


# ----------------------------------------------------------------- ablations
ABLATIONS = {
    # The three controls reviewers will ask for, pre-committed:
    "random_sense": "select sense uniformly at random (mechanism check: "
                    "context selection must beat random reassignment)",
    "freq_matched_poly": "run the pipeline on a frequency-matched set of "
                         "MONOsemous tokens declared 'polysemous' (placebo: "
                         "gains must vanish)",
    "shared_vs_retrained_head": "mass head trained on fixed frame vs "
                                "retrained under context frame",
}

# ----------------------------------------------------- pre-registered claims
CLAIMS = {
    "C1_surgical": "AUROC / separation-ratio gain of context-W over fixed-W "
                   "is significantly positive on Stratum P and ~zero on "
                   "Stratum M (paired bootstrap over positions).",
    "C2_alignment": "context-W passes the redundancy_screen vs a correctness "
                    "probe on Stratum P where fixed-W is weakest (the "
                    "aligned-but-non-redundant claim, deferral-line bar).",
    "C3_mechanism": "context-W beats random_sense ablation; placebo gains "
                    "vanish (freq_matched_poly).",
    "C4_size_control": "C1 holds for BOTH raw W and size-controlled W_basecoef "
                       "(closes the 'the ruler moved' objection).",
}


def main():
    raise NotImplementedError(
        "Driver skeleton: wire model/tokenizer + substrate, then follow the "
        "run order in the module docstring. All machinery is in rsuq.*; "
        "this file should stay < 200 lines — if it grows, the abstraction "
        "leaked and the fix belongs in the library."
    )


if __name__ == "__main__":
    main()
