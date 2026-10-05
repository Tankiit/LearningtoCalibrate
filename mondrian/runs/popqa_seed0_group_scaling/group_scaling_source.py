"""Paired optimal-group-count scaling experiment using TorchCP calibration."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import inspect
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering
import torch
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import LAC

from .data import digest, write_json
from .graph_memory import FrozenPartitions, load_cache, nearest

GOALS = ('average', 'spread', 'worst_group')


def paired_draws(n_pool, sizes, repetitions, seed):
    if min(sizes) < 1 or max(sizes) > n_pool or repetitions < 2:
        raise ValueError('Sizes must fit the calibration pool, with at least two repetitions')
    # A single permutation per repetition; prefixes are nested across sizes.
    return np.stack([np.random.default_rng(np.random.SeedSequence([seed, rep])).permutation(n_pool)
                     for rep in range(repetitions)])


def objective_values(sets, labels, audit, n_groups, alpha):
    sets = np.asarray(sets, dtype=bool)
    covered = sets[np.arange(len(labels)), labels]
    counts = np.bincount(audit, minlength=n_groups)
    if np.any(counts == 0):
        raise ValueError('Every fixed audit group must have evaluation questions')
    weights = counts/counts.sum()
    coverage = np.bincount(audit, weights=covered, minlength=n_groups)/counts
    error = np.abs(coverage-(1-alpha))
    average = float(weights @ error)
    spread = float(np.sqrt(weights @ ((coverage-weights @ coverage)**2)))
    worst = float(error.max())
    return np.array([average, spread, worst]), coverage


def smallest_minimizer(means, m_grid):
    """Last axis is M; all numeric ties within 1e-12 choose smallest M."""
    tied = np.abs(means-np.min(means, axis=-1, keepdims=True)) <= 1e-12
    return np.asarray(m_grid)[np.argmax(tied, axis=-1)], tied


def loglog_slope(n_grid, m_star):
    x = np.log(np.asarray(n_grid, dtype=float))
    centered = x-x.mean()
    return np.log(m_star) @ centered / (centered @ centered)


def bootstrap_optima(loss, n_grid, m_grid, draws, seed):
    """loss[goal,n,M,rep]; resample whole paired rep blocks everywhere."""
    point, ties = smallest_minimizer(loss.mean(axis=-1), m_grid)
    rng = np.random.default_rng(seed)
    repeats = loss.shape[-1]
    optima = np.empty((draws, loss.shape[0], loss.shape[1]), dtype=int)
    replicate_draws = rng.integers(0, repeats, size=(draws, repeats))
    for b, indices in enumerate(replicate_draws):
        optima[b], _ = smallest_minimizer(loss[..., indices].mean(axis=-1), m_grid)
    slopes = loglog_slope(n_grid, optima)
    return point, ties, optima, slopes, replicate_draws


def torchcp_sets(predictor, cal_probs, cal_labels, cal_cells, test_probs, test_cells, m):
    thresholds = torch.empty(m, dtype=torch.float64)
    occupancy = np.bincount(cal_cells, minlength=m)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module=r'torchcp\.utils\.common')
        for cell in range(m):
            mask = torch.as_tensor(cal_cells == cell)
            predictor.calculate_threshold(cal_probs[mask], cal_labels[mask])
            thresholds[cell] = torch.as_tensor(predictor.q_hat, dtype=torch.float64)
    # TorchCP's public prediction API broadcasts one group threshold per row.
    sets = predictor.predict_with_logits(test_probs, q_hat=thresholds[test_cells, None]).bool()
    return sets.numpy(), thresholds.numpy(), occupancy


def geometry(cache, reference, m_grid, seed):
    fit_idx = cache.indices('fit')
    x = cache.x[fit_idx]
    np.testing.assert_array_equal(x, reference.reference)
    d2, neighbors = nearest(x, x, reference.graph_k, exclude_self=True)
    weights = np.exp(-d2/(2*reference.bandwidth**2))
    graph = csr_matrix((weights.ravel(), (np.repeat(np.arange(len(x)), reference.graph_k), neighbors.ravel())),
                       shape=(len(x), len(x)))
    graph = graph.maximum(graph.T)
    component_count = int(connected_components(graph, directed=False, return_labels=False))
    cal_idx, test_idx = cache.indices('cal'), cache.indices('test')
    _, cal_nearest = nearest(cache.x[cal_idx], x, 1)
    _, test_nearest = nearest(cache.x[test_idx], x, 1)
    labels = []
    for m in m_grid:
        if m == 1:
            current = np.zeros(len(x), dtype=int)
        elif m == reference.bins:
            current = reference.graph_labels.copy()
        else:
            current = SpectralClustering(n_clusters=m, affinity='precomputed', assign_labels='kmeans',
                                        random_state=seed, n_init=10).fit_predict(graph)
        if len(np.unique(current)) != m:
            raise ValueError('Requested number of groups was not realized')
        labels.append(current)
        print(f'Frozen fit-only graph partition M={m}', flush=True)
    labels = np.stack(labels)
    return dict(fit_labels=labels, cal_cells=labels[:, cal_nearest[:, 0]],
                test_cells=labels[:, test_nearest[:, 0]], cal_nearest_fit=cal_nearest[:, 0],
                test_nearest_fit=test_nearest[:, 0], components=np.array(component_count))


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'Output must be new: {out}')
    n_grid, m_grid = sorted(set(args.n_cal)), sorted(set(args.groups))
    if len(n_grid) < 5 or len(m_grid) < 3 or min(m_grid) < 1 or args.bootstrap < 100:
        raise ValueError('Need >=5 calibration sizes, >=3 positive M values, and >=100 bootstrap draws')
    if min(args.seed, args.bootstrap_seed) < 0:
        raise ValueError('Seeds must be nonnegative')
    cache = load_cache(args.cache)
    fit_state = json.loads((Path(args.fitted)/'calibration.json').read_text())
    if cache.hashes != fit_state['cache_hashes']:
        raise ValueError('Cache differs from original fit')
    if digest(Path(args.fitted)/'partitions.npz') != fit_state['partition_sha256']:
        raise ValueError('Reference graph artifact changed')
    reference = FrozenPartitions.load(Path(args.fitted)/'partitions.npz')
    if max(m_grid) >= len(reference.reference):
        raise ValueError('M grid must be smaller than fit pool')
    cal_idx, test_idx = cache.indices('cal'), cache.indices('test')
    if cache.ids(test_idx) != fit_state['test_row_ids']:
        raise ValueError('Original test IDs changed')
    permutations = paired_draws(len(cal_idx), n_grid, args.repetitions, args.seed)
    audit_names = sorted({cache.rows[i]['audit_group'] for i in cache.indices('fit')})
    audit_lookup = {name: i for i, name in enumerate(audit_names)}
    audit = np.array([audit_lookup[cache.rows[i]['audit_group']] for i in test_idx])
    if len(np.unique(audit)) != len(audit_names):
        raise ValueError('An audit group has no test support')
    torch.set_num_threads(2)
    predictor = SplitPredictor(LAC(score_type='identity'), alpha=args.alpha, device='cpu')
    out.mkdir(parents=True)
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), experiment='paired optimal-M scaling pilot',
        n_cal_grid=n_grid, M_grid=m_grid, repetitions=args.repetitions, alpha=args.alpha,
        goals=list(GOALS), objectives={
            'average': 'sum_g w_g * abs(coverage_g - (1-alpha))',
            'spread': 'sqrt(sum_g w_g * (coverage_g - sum_h w_h coverage_h)**2)',
            'worst_group': 'max_g abs(coverage_g - (1-alpha))'},
        audit_groups=audit_names, audit_weights='Fixed test question frequencies; groups are source relation metadata, independent of M.',
        predictor='Frozen anchor_p only, held constant across M, n_cal and repetition. No memory updates.',
        grouping='Fit-only spectral partitions of the same graph; nearest-fit-node assignment; graph k and bandwidth fixed.',
        graph_k=reference.graph_k, graph_seed=fit_state['config']['graph_seed'], bandwidth=reference.bandwidth,
        calibration_pool='Original cal role only; update and fit questions excluded.',
        draw_seed=args.seed, draw_rule='Independent seeded permutations of the calibration pool; nested prefixes across n_cal; '
            'same draw for every M and goal. All historical test questions fixed across repetitions.',
        bootstrap_draws=args.bootstrap, bootstrap_seed=args.bootstrap_seed,
        bootstrap_rule='Resample whole repetition blocks jointly across all sizes, M values and goals; recompute mean losses, '
            'argmin with smallest-M tie break, and OLS log(M_star) on log(n_cal) in every bootstrap draw.',
        headline_pair=args.headline, decision_rule='Headline statistical criterion holds iff the two pointwise percentile '
            '95% slope CIs do not overlap. No direction selected. Paired slope-difference CI also reported.',
        interpretability_guard='For both headline goals, point M_star must be interior to the M grid at >=3 of 5 sizes '
            '(or ceiling(0.6*n_sizes)), vary across sizes, and never yield mean full-set rate >=0.99. '
            'Headline supported requires both statistical criterion and this guard.',
        tie_rule='Choose smallest M among mean losses within absolute 1e-12 of the minimum.',
        sparse_cell_rule='TorchCP finite-sample threshold; empty/too-small calibration cells get infinity and full sets. No merging.',
        selection_scope='Evaluation-optimal M on the already-inspected test set, descriptive only; no deployed M selected.',
        limitations=['Rep-bootstrap CIs condition on the finite calibration pool, test data, predictor and partitions; '
            'they measure repeated-draw/Monte Carlo variability, not population or new-dataset uncertainty.',
            'Spread alone can prefer all-full sets; efficiency/full-set diagnostics are mandatory.',
            'Grid-boundary minima and discrete unstable optima cannot establish an asymptotic power law.',
            'Supplied-candidate PopQA labels and historical-test limitations remain as in the cache manifest.'],
        cache_path=str(Path(args.cache).resolve()), cache_hashes=cache.hashes,
        reference_fit=str(Path(args.fitted).resolve()), reference_fit_sha256=digest(Path(args.fitted)/'calibration.json'),
        code_sha256=digest(__file__),
        torchcp_api='SplitPredictor(LAC(score_type="identity")); calculate_threshold per feature cell; predict_with_logits.',
        torchcp_sources={name: {'path': inspect.getfile(cls), 'sha256': digest(inspect.getfile(cls))}
                         for name, cls in [('SplitPredictor', SplitPredictor), ('LAC', LAC)]},
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'torch', 'torchcp')})
    write_json(out/'plan.json', plan)  # Frozen before computing any new loss curve.
    partitions = geometry(cache, reference, m_grid, fit_state['config']['graph_seed'])
    np.savez_compressed(out/'partitions.npz', **partitions, M_grid=m_grid,
                        fit_row_ids=np.array(cache.ids(cache.indices('fit'))),
                        cal_row_ids=np.array(cache.ids(cal_idx)), test_row_ids=np.array(cache.ids(test_idx)))
    np.savez_compressed(out/'draws.npz', permutations=permutations, n_cal_grid=n_grid,
                        cal_pool_row_ids=np.array(cache.ids(cal_idx)), test_row_ids=np.array(cache.ids(test_idx)))
    def probs(indices):
        p = np.array([cache.rows[i]['anchor_p'] for i in indices])
        return torch.tensor(np.column_stack((1-p, p)), dtype=torch.float64)
    cal_probs, test_probs = probs(cal_idx), probs(test_idx)
    cal_y = torch.tensor([cache.rows[i]['correct'] for i in cal_idx], dtype=torch.long)
    test_y = np.array([cache.rows[i]['correct'] for i in test_idx])
    shape = (len(n_grid), len(m_grid), args.repetitions)
    loss = np.empty((len(GOALS), *shape))
    covered = np.empty(shape)
    size = np.empty(shape)
    full = np.empty(shape)
    group_coverage = np.empty((*shape, len(audit_names)))
    rows = []
    for mi, m in enumerate(m_grid):
        cc, tc = partitions['cal_cells'][mi], partitions['test_cells'][mi]
        qhat = np.empty((len(n_grid), args.repetitions, m))
        occupancy = np.empty(qhat.shape, dtype=int)
        packed_sets = np.empty((len(n_grid), args.repetitions, len(test_idx)), dtype=np.uint8)
        for ni, n in enumerate(n_grid):
            for rep in range(args.repetitions):
                draw = permutations[rep, :n]
                sets, q, counts = torchcp_sets(predictor, cal_probs[draw], cal_y[draw], cc[draw], test_probs, tc, m)
                values, groups = objective_values(sets, test_y, audit, len(audit_names), args.alpha)
                loss[:, ni, mi, rep] = values
                group_coverage[ni, mi, rep] = groups
                coverage = float(sets[np.arange(len(test_y)), test_y].mean())
                sizes = sets.sum(axis=1)
                covered[ni, mi, rep], size[ni, mi, rep], full[ni, mi, rep] = coverage, sizes.mean(), (sizes == 2).mean()
                qhat[ni, rep], occupancy[ni, rep] = q, counts
                packed_sets[ni, rep] = sets[:, 0].astype(np.uint8)+2*sets[:, 1].astype(np.uint8)
                rows.append(dict(n_cal=n, M=m, rep=rep, **dict(zip(GOALS, values)), coverage=coverage,
                    mean_set_size=float(sizes.mean()), full_set_rate=float((sizes == 2).mean()),
                    empty_set_rate=float((sizes == 0).mean()), fallback_cells=int(np.isinf(q).sum()),
                    fallback_test_fraction=float(np.isinf(q)[tc].mean()), min_cell_cal_count=int(counts.min())))
            print(f'Completed M={m}, n_cal={n}, {args.repetitions} paired draws', flush=True)
        np.savez_compressed(out/f'M{m}_sets.npz', packed_sets=packed_sets, qhat=qhat, occupancy=occupancy,
                            n_cal_grid=n_grid, test_row_ids=np.array(cache.ids(test_idx)))
    write_csv(out/'losses.csv', rows)
    np.savez_compressed(out/'losses.npz', loss=loss, coverage=covered, mean_set_size=size, full_set_rate=full,
                        group_coverage=group_coverage, goals=np.array(GOALS), n_cal_grid=n_grid, M_grid=m_grid,
                        audit_groups=np.array(audit_names), test_audit_group=audit, test_labels=test_y)
    point, ties, stars, slopes, boot_draws = bootstrap_optima(loss, n_grid, m_grid, args.bootstrap, args.bootstrap_seed)
    point_slopes = loglog_slope(n_grid, point)
    np.savez_compressed(out/'bootstrap.npz', M_star=stars, slopes=slopes, rep_indices=boot_draws,
                        goals=np.array(GOALS), n_cal_grid=n_grid, M_grid=m_grid)
    summaries = []
    optimum_rows = []
    for gi, goal in enumerate(GOALS):
        y = np.log(point[gi]); x = np.log(n_grid)
        intercept = float(y.mean()-point_slopes[gi]*x.mean())
        total = float(np.sum((y-y.mean())**2))
        interior = int(((point[gi] > min(m_grid)) & (point[gi] < max(m_grid))).sum())
        degenerate = False
        for ni, n in enumerate(n_grid):
            mi = m_grid.index(int(point[gi, ni]))
            full_rate = float(full[ni, mi].mean())
            degenerate |= full_rate >= .99
            optimum_rows.append(dict(goal=goal, n_cal=n, M_star=int(point[gi, ni]),
                bootstrap_M_low=float(np.quantile(stars[:, gi, ni], .025)),
                bootstrap_M_high=float(np.quantile(stars[:, gi, ni], .975)),
                tied_M_values=[m_grid[i] for i in np.flatnonzero(ties[gi, ni])],
                bootstrap_boundary_fraction=float(np.isin(stars[:, gi, ni], [min(m_grid), max(m_grid)]).mean()),
                mean_loss=float(loss[gi, ni, mi].mean()), mean_coverage=float(covered[ni, mi].mean()),
                mean_set_size=float(size[ni, mi].mean()), mean_full_set_rate=full_rate))
        summaries.append(dict(goal=goal, slope=float(point_slopes[gi]),
            slope_ci95=np.quantile(slopes[:, gi], [.025, .975]).tolist(), intercept=intercept,
            loglog_r_squared=None if total == 0 else float(1-np.sum((y-(intercept+point_slopes[gi]*x))**2)/total),
            M_star=point[gi].tolist(), interior_sizes=interior,
            all_full_degeneracy=bool(degenerate),
            interpretable=interior >= int(np.ceil(.6*len(n_grid))) and len(set(point[gi])) > 1 and not degenerate))
    contrast = []
    for a in range(len(GOALS)):
        for b in range(a+1, len(GOALS)):
            ci = np.quantile(slopes[:, a]-slopes[:, b], [.025, .975]).tolist()
            contrast.append(dict(left=GOALS[a], right=GOALS[b], difference=float(point_slopes[a]-point_slopes[b]),
                                 paired_ci95=ci, paired_ci_excludes_zero=ci[0] > 0 or ci[1] < 0))
    lookup = {r['goal']: r for r in summaries}
    left, right = [lookup[g] for g in args.headline]
    a, b = left['slope_ci95'], right['slope_ci95']
    separated = a[1] < b[0] or b[1] < a[0]
    guard = bool(left['interpretable'] and right['interpretable'])
    result = dict(status='complete', n_cal_grid=n_grid, M_grid=m_grid,
        repetitions=args.repetitions, bootstrap_draws=args.bootstrap, goals=summaries,
        contrasts=contrast, headline_pair=args.headline, slope_CIs_nonoverlapping=bool(separated),
        interpretability_guard_passed=guard, headline_supported=bool(separated and guard),
        inference_scope='Conditional finite-cache, calibration-draw pilot; not population scaling evidence.',
        plan_sha256=digest(out/'plan.json'), trial_conditions=len(rows))
    write_json(out/'results.json', result)
    write_csv(out/'M_star.csv', optimum_rows)
    write_json(out/'manifest.json', dict(status='complete', plan_sha256=digest(out/'plan.json'),
        output_sha256={p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--fitted', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--n-cal', type=int, nargs='+', default=[50, 100, 200, 400, 800])
    parser.add_argument('--groups', type=int, nargs='+', default=[1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    parser.add_argument('--repetitions', type=int, default=200)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=271828)
    parser.add_argument('--bootstrap-seed', type=int, default=314159)
    parser.add_argument('--alpha', type=float, default=.1)
    parser.add_argument('--headline', nargs=2, choices=GOALS, default=['average', 'worst_group'])
    run(parser.parse_args())


if __name__ == '__main__':
    main()
