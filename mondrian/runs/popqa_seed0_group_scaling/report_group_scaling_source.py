"""Render the completed paired group-scaling pilot, without changing selection."""
import argparse
import json
from pathlib import Path

import numpy as np


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    result = json.loads((root/'results.json').read_text())
    plan = json.loads((root/'plan.json').read_text())
    with np.load(root/'bootstrap.npz') as a:
        stars = a['M_star'].copy()
    with np.load(root/'losses.npz') as a:
        loss = a['loss'].copy()
    n, m = np.array(result['n_cal_grid']), np.array(result['M_grid'])
    colors = ['#26638e', '#a76d17', '#6a4b8a']
    titles = ['Average absolute coverage error', 'Coverage spread', 'Worst-group absolute error']
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5), sharey=True)
    for gi, (ax, g) in enumerate(zip(axes, result['goals'])):
        point = g['M_star']
        low, high = np.quantile(stars[:, gi], [.025, .975], axis=0)
        ax.vlines(n, low, high, color=colors[gi], alpha=.6, linewidth=5)
        ax.plot(n, point, 'o-', color=colors[gi], linewidth=1.6, markersize=5)
        ax.set_xscale('log', base=2); ax.set_yscale('log', base=2)
        ax.set_xticks(n, [str(x) for x in n]); ax.set_yticks([1, 2, 4, 8, 16, 32], ['1', '2', '4', '8', '16', '32'])
        ax.set_ylim(.8, 40); ax.set_xlim(43, 930)
        ax.set_xlabel('Calibration questions')
        lo, hi = g['slope_ci95']
        ax.set_title(f'{titles[gi]}\nslope {g["slope"]:.3f} [{lo:.3f}, {hi:.3f}]', fontsize=10)
        ax.grid(axis='y', color='#dddddd', linewidth=.5)
        if not g['interpretable']:
            ax.text(.5, .03, 'Grid-boundary / unstable optima', transform=ax.transAxes,
                    ha='center', fontsize=8, color='#825519')
    axes[0].set_ylabel('Selected group count M*')
    fig.suptitle('PopQA: optimal group count under three coverage objectives', fontsize=13, y=.98)
    fig.text(.5, .02, '200 paired draws; bars show 95% bootstrap ranges for discrete M*. Fixed-cache conditional inference.',
             ha='center', fontsize=9, color='#444444')
    fig.tight_layout(rect=(0, .055, 1, .93))
    for suffix in ('png', 'svg'):
        fig.savefig(root/f'group_scaling.{suffix}', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2))
    palette = plt.get_cmap('viridis')(np.linspace(.05, .85, len(n)))
    for gi, ax in enumerate(axes):
        for ni, value in enumerate(n):
            ax.plot(m, loss[gi, ni].mean(-1), 'o-', color=palette[ni], markersize=3, linewidth=1.2, label=f'n={value}')
        ax.set_xscale('log', base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32], ['1', '2', '4', '8', '16', '32'])
        ax.set_xlabel('Number of calibration groups M')
        ax.set_title(titles[gi], fontsize=10)
        ax.grid(axis='y', color='#dddddd', linewidth=.5)
    axes[0].set_ylabel('Mean loss across calibration draws')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(n), frameon=False)
    fig.suptitle('Full loss curves: no M values discarded after evaluation', fontsize=13)
    fig.tight_layout(rect=(0, .08, 1, .94))
    fig.savefig(root/'loss_curves.png', dpi=180)
    plt.close(fig)

    pair = result['headline_pair']
    decision = 'passes' if result['headline_supported'] else 'does not pass'
    overlap = 'do not overlap' if result['slope_CIs_nonoverlapping'] else 'overlap'
    guard = 'passes' if result['interpretability_guard_passed'] else 'fails'
    lines = ['# Paired optimal-group-count scaling: PopQA pilot', '',
        f'**The predeclared {pair[0]}-versus-{pair[1]} pilot criterion {decision}.** '
        f'Their conditional 95% slope intervals {overlap}; the declared interior-grid/nondegeneracy guard {guard}. '
        'This is descriptive evidence on a fixed cache, not confirmation of population power-law exponents.', '',
        f'TorchCP {plan["versions"]["torchcp"]}; {len(n)} calibration sizes; {len(m)} M values; '
        f'{result["repetitions"]} paired calibration repetitions; {result["bootstrap_draws"]:,} joint repetition-bootstrap draws. '
        'All original test questions are preserved. '
        'The same frozen anchor and fit-only geometry underlie every goal.', '',
        '## Objectives and estimates', '',
        'For the same eight fixed relation audit groups and target coverage 90%: '
        'average = frequency-weighted mean absolute group coverage error; spread = frequency-weighted SD of group coverage; '
        'worst_group = maximum absolute group coverage error. Set size is a separate diagnostic.', '',
        '| Goal | M* at n=50,100,200,400,800 | Slope [conditional 95% CI] | Log-log R² |',
        '|---|---|---|---:|']
    for g in result['goals']:
        lo, hi = g['slope_ci95']
        r2 = 'undefined' if g['loglog_r_squared'] is None else f'{g["loglog_r_squared"]:.3f}'
        lines.append(f'| {g["goal"]} | {", ".join(map(str, g["M_star"]))} | {g["slope"]:.3f} [{lo:.3f}, {hi:.3f}] | {r2} |')
    comparison = next(c for c in result['contrasts'] if {c['left'], c['right']} == set(pair))
    lo, hi = comparison['paired_ci95']
    lines += ['', f'The paired slope difference ({comparison["left"]} minus {comparison["right"]}) is **{comparison["difference"]:.3f} '
              f'[{lo:.3f}, {hi:.3f}]**. Each bootstrap draw reruns mean-loss minimization and log-log fitting, '
              'using the same resampled repetition blocks for every size, M and goal.', '',
              '![Selected group counts and bootstrap ranges](group_scaling.png)', '',
              '## Interpretation', '',
              'Read the optimum sequences and log-log R² alongside the slopes. Plateaus, jumps, '
              'grid-boundary minima and nonmonotone sequences weaken a smooth power-law interpretation. '
              'The spread objective can favor high full-set rates; inspect the selected-point diagnostics in M_star.csv.', '',
              'The finite-grid criterion does not establish distinct asymptotic exponents. '
              'All intervals condition on this finite calibration pool, fixed test set, '
              'frozen head, and geometry. They describe repeated calibration draws and their Monte Carlo mean. '
              'Increasing repetitions can narrow them without adding independent questions. '
              'Independent data and a wider sample-size range are needed for a scaling-law claim.', '',
              'This run uses evaluation outcomes to describe optimal M; no deployment M was selected. '
              'The prior supplied-candidate and alias-label limitations still apply.', '',
              '![Complete mean loss curves](loss_curves.png)', '',
              '## Artifacts', '',
              '- [Protocol recorded before new losses](plan.json)',
              '- [Machine-readable slope results and decision](results.json)',
              '- [Selected M and diagnostics](M_star.csv)',
              '- [Every trial loss and occupancy diagnostic](losses.csv)',
              '- [Replay validation](validation.json)', '',
              'Packed sets, thresholds, permutations and bootstrap draws are saved alongside these reports. '
              'The implementation uses TorchCP SplitPredictor/LAC within feature-defined cells: '
              '[official API documentation](https://torchcp.readthedocs.io/en/latest/torchcp.classification.html).']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(f'Report and figures saved to {root}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    report(parser.parse_args().run)
