"""Étape 4, vérification locale sans GPU, avant le shard.

Contrôle le parser, la jointure des questions avec les caches provenance-v1 (817 identifiants
dans le même ordre pour les trois modèles), les textes des candidats (les 32 questions du
mapping control) et le nombre de réponses prévu.

Usage : ~/Documents/hf/bin/python naim/etape4_verifier_protocole.py
"""
from __future__ import annotations

import json

from commun import MODELS, example_ids, fetch, provenance
from etape4_emissions_modal import (LEGENDS, N_QUESTIONS, PROTOCOL_SHA256, SHARD_QUESTIONS,
                                    load_truthfulqa, parse_response, prompt_for)

PARSER_CASES = [
    ((" B", True), ("B", None, "B")),
    ((" B\nQuestion: ...", False), ("B", None, "B")),
    ((" L.", True), ("L", None, "L")),
    (("", True), (None, "vide", None)),
    ((" I", True), (None, "lettre_hors_legende", None)),
    ((" B (10%)", True), (None, "texte_en_trop", "B")),
    ((" Based on", True), (None, "texte_en_trop", None)),
    ((" B", False), (None, "tronque", "B")),
    ((" 70%", True), (None, "texte_en_trop", None)),
    # Réponses réelles de Mistral dans le shard v1 (motif du passage en v2).
    (("\nA\nExplanation:\n", False), ("A", None, "A")),
    (("\n\nThe idea of left-br", False), (None, "texte_en_trop", None)),
    (("\nA=0%\nB=", False), (None, "texte_en_trop", "A")),
    ((" \n L\n", False), ("L", None, "L")),
    (("\n\n", True), (None, "vide", None)),
]


def main() -> None:
    for (text, ended), expected in PARSER_CASES:
        got = parse_response(text, ended)
        assert got == expected, f"parser : {text!r} -> {got}, attendu {expected}"
    print(f"parser : {len(PARSER_CASES)} cas OK")

    records = load_truthfulqa()
    assert len(records) == N_QUESTIONS, len(records)
    ids = [r["question_id"] for r in records]
    for model in MODELS:
        assert ids == list(example_ids(provenance(model, "forward"))), f"jointure {model}"
    print(f"jointure : {len(ids)} identifiants identiques au cache pour {len(MODELS)} modèles")

    control = json.loads(fetch("letter11_mapping_control_v1/llama3_8b.json").read_text())
    by_id = {r["question_id"]: r for r in records}
    checked = 0
    for row in control["rows"]:
        if row["kind"] == "contextual" and row["dataset"] == "truthfulqa":
            rec = by_id[row["question_id"]]
            assert rec["question"] == row["question"] and rec[row["candidate_side"]] == row["answer"]
            checked += 1
    print(f"candidats : {checked} textes identiques à ceux du mapping control")

    per_question = 2 * len(LEGENDS)
    print(f"réponses : shard = {len(MODELS) * SHARD_QUESTIONS * per_question}, "
          f"complet = {len(MODELS) * N_QUESTIONS * per_question}")
    print(f"protocol_sha256 = {PROTOCOL_SHA256}")
    print("exemple de prompt (reversed) :\n" + prompt_for(records[0]["question"], records[0]["neg"], "reversed"))


if __name__ == "__main__":
    main()
