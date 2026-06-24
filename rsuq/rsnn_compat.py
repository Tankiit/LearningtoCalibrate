"""rsuq.rsnn_compat — Compatibility layer with RS-NN (Manchingal et al. 2025).

Adopts the TRANSFERABLE pieces of the RS-NN reference implementation
(github.com/shireenkmanch/Random-Set-Neural-Networks), giving citeable
lineage and an overlapping-focal-set path for Stage III. Does NOT adopt the
softmax-replaced pieces.

WHAT TRANSFERS (here):
  betp_matrix / final_betp   matrix-form pignistic; entry 1/|A| if c in A.
                             For DISJOINT frames this equals RSUQ's scatter
                             pignistic exactly (tested). For OVERLAPPING sets
                             it is the correct general form Stage III needs.
  groundtruth_belief_encode  mark every focal set containing the label; the
                             general (overlapping) form of RSUQ's cluster-CE
                             target, which is its disjoint special case.

WHAT DOES NOT TRANSFER (intentionally omitted from the active path):
  belief_to_mass / mass_coeff  Mobius inversion over the powerset lattice +
                             negative-mass clipping + universal-set residual.
                             The NeurIPS architecture REPLACED this with
                             softmax-over-mass-logits (valid simplex by
                             construction, no O(K^2) lattice op, no clipping).
                             Provided ONLY under mobius_inverse() for Stage III
                             (overlapping sets), clearly fenced, not used by the
                             disjoint Stage I/II/AAAI pipeline.

Citation register: these are Shireen's operations; cite RS-NN for the
pignistic-matrix and belief-encoding lineage. The softmax-mass head is the
RSUQ departure, stated as such.
"""

from __future__ import annotations
import numpy as np
import torch


# --------------------------------------------------------- pignistic (matrix)
def betp_matrix(focal_sets: list[set], classes: list) -> np.ndarray:
    """(|focal_sets|, |classes|) with entry 1/|A| if class c in A else 0.
    RS-NN final_betp form. Works for disjoint OR overlapping focal sets."""
    M = np.zeros((len(focal_sets), len(classes)))
    for j, A in enumerate(focal_sets):
        if not A:
            continue
        inv = 1.0 / len(A)
        for i, c in enumerate(classes):
            if c in A:
                M[j, i] = inv
    return M


def final_betp(mass: np.ndarray, betp_mat: np.ndarray) -> np.ndarray:
    """Pignistic probability: mass @ betp_matrix. (N, |F|) @ (|F|, V) -> (N, V).
    RS-NN-faithful; the general (overlapping-capable) pignistic."""
    return mass @ betp_mat


# ------------------------------------------------ ground-truth belief encoding
def groundtruth_belief_encode(labels, focal_sets: list[set]) -> np.ndarray:
    """RS-NN groundtruthmod: y[i, j] = 1 if label_i in focal_set_j.
    For disjoint frames each row is one-hot (= RSUQ's cluster-CE target);
    for overlapping frames it is multi-hot (the general belief-encoding)."""
    Y = np.zeros((len(labels), len(focal_sets)), dtype=np.int64)
    for i, lab in enumerate(labels):
        for j, A in enumerate(focal_sets):
            if lab in A:
                Y[i, j] = 1
    return Y


# -------------------------------------------- Mobius (Stage III ONLY; fenced)
def mobius_inverse(belief_preds: np.ndarray, focal_sets: list[set],
                   add_universal: bool = True) -> np.ndarray:
    """RS-NN belief_to_mass: Mobius inversion belief->mass over the subset
    lattice, clip negatives, residual to the universal set.

    DO NOT USE in the disjoint Stage I/II/AAAI pipeline — there mass == belief
    and the softmax-mass head already produces a valid simplex. This exists
    for Stage III (overlapping/nested focal sets) where the lattice is
    non-trivial. Kept faithful to the reference for citeability.
    """
    n = len(focal_sets)
    coeff = np.zeros((n, n))
    for i, A in enumerate(focal_sets):
        for j, B in enumerate(focal_sets):
            if B.issubset(A):
                coeff[j, i] = (-1) ** (len(A) - len(B))
    mass = belief_preds @ coeff
    mass[mass < 0] = 0
    if add_universal:
        resid = np.clip(1 - mass.sum(-1), 0, None)
        mass = np.concatenate([mass, resid[:, None]], -1)
    return mass / mass.sum(-1, keepdims=True)


# ------------------------------------------- disjoint-equivalence check (test)
def assert_matches_scatter_pignistic(frame, m: torch.Tensor, classes=None):
    """For a DISJOINT frame, RS-NN matrix pignistic == RSUQ scatter pignistic.
    Guards the compatibility claim; called in tests."""
    from rsuq.core.beliefs import pignistic
    V = frame.V
    classes = classes or list(range(V))
    fs = [set(frame.members[k].tolist()) for k in range(frame.K)]
    Bm = betp_matrix(fs, classes)
    bp_matrix = final_betp(m.detach().cpu().numpy(), Bm)
    bp_scatter = pignistic(m, frame.kappa, frame.sizes).detach().cpu().numpy()
    assert np.allclose(bp_matrix, bp_scatter, atol=1e-5), \
        "RS-NN matrix pignistic disagrees with RSUQ scatter on a disjoint frame"
    return True
