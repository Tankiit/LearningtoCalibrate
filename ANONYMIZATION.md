# Anonymous Release Notes

This is an anonymous, derived-artifact reproducibility repository for
`OVA deferral`. It intentionally excludes:

- cloud/Modal execution code
- raw LLM hidden states and generation caches
- model checkpoints and trained probe weights
- logs, local absolute paths, and private workflow notes

The included CSV files are derived result artifacts used by the paper
figures and tables. The scripts in `scripts/` regenerate the packaged
figures from those CSVs only.
