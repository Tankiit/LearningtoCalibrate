# Validation — 7 September 2026

- `python -m unittest discover -s tests -v`: 14 tests discovered; 13 passed, one skipped (optional torch gradient check; torch unavailable).
- Plain-Python demo: 250 simulated questions per split, four splits, q/qa and base/trained scores under two legends.
- ICLR analysis: completed with 200 bootstrap resamples in the bundled demo (CLI default is 1000).
- Mondrian fitting: ten fixed calibrators, q/qa partition signals and K = 1, 2, 4, 8, 16 at alpha = 0.1.
- Conformal evaluation: completed; both K=1 baselines agreed in coverage, mean size, empty, singleton and full-set rates.
- Source syntax and package metadata parsed successfully.

The demo values are simulated plumbing checks, not real experimental results.
Live Tinker API calls, remote training, and the torch backward path were not executed.
No authentication credentials or model weights are included.
