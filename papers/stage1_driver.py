"""papers/stage1_driver.py — Stage I gating experiments (Layer 1 build plan).

Build-order status:
  Gate 1  Q1.1 identity            DONE — and UPGRADED: ||p_sigma - BetP||_1
          is CONSTANT over vertices in the disjoint case, so
          W = (1/2)||p_sigma - BetP||_1 holds vertex-by-vertex,
          deterministically (5-line lemma; pinned below as a test).
          Caveat: special to disjoint sets + l1; reverts to an expectation
          under ranking-weighted spread (Stage II, Q2.2).
  Gate 2  Experiment 0 (confusion alignment)   THIS FILE — run first on M2
  (collapse property: pinned as RSUQ test I1 — done)
  Gate 3  Experiment 1 (bag matters)           THIS FILE
  Then    D2D estimator on synthetic, consistency curves (separate driver)

Substrate: GPT-2 small on M2/MPS; WikiText-2 for gates, WikiText-103 for
real runs. Bag decision: MC-dropout primary, token-perturb + embed-noise
ablations (DECISION_bag_construction.md).

PRE-REGISTERED GATES:
  E0 PASS: same-cluster fraction of top-k competitor pairs exceeds the
           size-matched random-partition baseline with the paired-bootstrap
           95% CI of the ratio excluding 1.0, for k in {5, 10}.
           FAIL => the frame foundation is unsound: stop, rethink partition.
  E1 PASS: validation cluster-CE at N=16 < at N=1 (one stochastic sample),
           3 seeds, bootstrap CI on the difference excluding 0; trend over
           N in {1,4,8,16} weakly monotone.
           FAIL => the distributional-input claim is in trouble: the D2D
           framing needs reconsideration before the theory paper is written.
"""

from __future__ import annotations
import json
import numpy as np
import torch

# from transformers import GPT2LMHeadModel, GPT2TokenizerFast
# from rsuq.core.frame import FixedFrame
# from rsuq.core.beliefs import MassHead
# from rsuq.bags import make_bag, bag_mean
# from rsuq.diagnostics import bootstrap_ci

DEVICE = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
K, SEEDS = 200, (0, 1, 2)


# ===================================================================== Q1.1
def lemma_q11_vertexwise(n_trials: int = 200, seed: int = 0):
    """Pinned: W(m) = (1/2)||p_sigma - BetP||_1 for EVERY vertex sigma
    (disjoint focal sets). Proof: within F_k the selected token contributes
    m_k(1-1/n_k) and the other n_k-1 tokens contribute m_k/n_k each, so the
    cluster total is 2 m_k (1-1/n_k), independent of which token sigma picks."""
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        Kc = rng.integers(2, 6)
        sizes = rng.integers(1, 50, Kc).astype(float)
        m = rng.dirichlet(np.ones(Kc))
        W = m @ (1 - 1 / sizes)
        per_vertex = (m * (1 - 1 / sizes) + (sizes - 1) * m / sizes).sum() / 2
        assert np.isclose(W, per_vertex, atol=1e-12)
    return True


# ============================================== Experiment 0: confusion align
@torch.no_grad()
def experiment0_confusion_alignment(model, tokenizer, frame, texts,
                                    ks=(5, 10), max_positions: int = 20_000,
                                    n_baseline: int = 50, seed: int = 0):
    """Do embedding-clustered focal sets align with the model's confusion
    structure? For each position, take the top-k softmax tokens; measure the
    fraction of pairs in the same cluster; compare to size-matched random
    partitions. This gates the entire frame foundation (build step 2)."""
    model.eval()
    kappa = frame.kappa
    rng = np.random.default_rng(seed)

    same_frac = {k: [] for k in ks}
    n = 0
    for text in texts:
        if n >= max_positions:
            break
        ids = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=256)["input_ids"].to(DEVICE)
        if ids.shape[1] < 16:
            continue
        logits = model(ids).logits[0]                       # (T, V)
        for k in ks:
            top = logits.topk(k, dim=-1).indices.cpu()      # (T, k)
            c = kappa[top]                                  # (T, k)
            # fraction of same-cluster pairs per position, vectorised
            eq = (c.unsqueeze(-1) == c.unsqueeze(-2)).float()
            pairs = (eq.sum((-1, -2)) - k) / (k * (k - 1))
            same_frac[k].extend(pairs.tolist())
        n += ids.shape[1]

    # size-matched random-partition baseline: permute kappa over the vocab
    results = {}
    for k in ks:
        obs = float(np.mean(same_frac[k]))
        base = []
        for _ in range(n_baseline):
            perm = torch.from_numpy(rng.permutation(len(kappa)))
            kb = kappa[perm]
            # expected same-cluster prob under a size-preserving random
            # partition is sum_k n_k(n_k-1) / (V(V-1)) — analytic, no resample
            del kb
        szs = frame.sizes.numpy()
        V = float(len(kappa))
        base_analytic = float((szs * (szs - 1)).sum() / (V * (V - 1)))
        ratio = obs / max(base_analytic, 1e-12)
        # paired bootstrap over positions for the CI of obs (baseline is exact)
        arr = np.array(same_frac[k])
        bci = np.quantile([arr[rng.integers(0, len(arr), len(arr))].mean()
                           for _ in range(1000)], [0.025, 0.975])
        results[k] = {"observed": obs, "baseline": base_analytic,
                      "ratio": ratio,
                      "obs_ci": [float(bci[0]), float(bci[1])],
                      "pass": bool(bci[0] > base_analytic)}
    results["verdict"] = ("PASS" if all(results[k]["pass"] for k in ks)
                          else "FAIL — frame foundation unsound; stop and "
                               "rethink the partition before anything else")
    return results


# ================================================ Experiment 1: bag matters
def experiment1_bag_matters(model, tokenizer, frame, texts,
                            Ns=(1, 4, 8, 16), kinds=("mc_dropout",),
                            n_examples: int = 4000, epochs: int = 3):
    """Does the bag (N>1) beat the single sample (N=1)? Train the mass head
    on bag-mean states vs single stochastic states; compare val cluster-CE.
    Deterministic eval-mode state reported as a reference row.

    Skeleton — wiring identical to Stage II's train loop; the only new code
    is the per-position bag construction:

        bag = make_bag(model, ids, n=N, kind=kind)       # (N, T, d)
        x_t = bag_mean(bag)[t]                            # the D2D input
    """
    raise NotImplementedError(
        "Wire: collect (bag_mean_state, kappa(gold)) pairs per (N, kind, "
        "seed) cell; reuse train.train_mass_head; report val CE per cell "
        "with bootstrap CI on CE(N=16) - CE(N=1). Cells: len(Ns) x "
        "len(kinds) x 3 seeds; ~minutes per cell on M2 at 4k examples.")


if __name__ == "__main__":
    assert lemma_q11_vertexwise()
    print("Q1.1 vertexwise lemma: pinned OK")
    # model/tokenizer/frame setup, then:
    # r0 = experiment0_confusion_alignment(...)
    # print(json.dumps(r0, indent=2))
    # if "PASS" in r0["verdict"]: experiment1_bag_matters(...)
