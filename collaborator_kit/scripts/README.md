# Collaborator Scripts

The appendix tasks should use the repository's canonical scripts by default:

- `../../scripts/scod_optimal_selector.py`
- `../../scripts/scod_diagnostics.py`
- `../../scripts/y_expert_semantic_audit.py`
- `../../scripts/id_ood_analogy.py`

Do not edit those shared scripts for appendix-only work. If a task needs a
variant, add it here with a task prefix, for example:

```text
G1_selfcheckgpt.py
D2_collect_selector_results.py
```

The current kit intentionally keeps this directory thin until a task-specific
variant is needed.

