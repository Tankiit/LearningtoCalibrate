"""Convert the repo's TriviaQA generation CSVs into the tinker_uq input contract.

Source: evaluations_revisited/<generator>/trivia_qa_60k/base/main_generations_evaluated_revisited.csv
Columns: prompt (few-shot preamble + question), answer (generation), ground_truth, correct,
correct_revisited.

The exported ``question`` is the final TriviaQA question only. The few-shot preamble and the
family-specific EOS markers are dropped so the reporter prompt does not depend on which model
generated the candidate. ``ground_truth`` is never exported: the gold answer must not reach the
model. Question IDs hash the normalized question text, so the same question lands in the same
split for every generator model.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

SPLITS = ('train', 'val', 'cal', 'test')
DEFAULT_FRACTIONS = (0.50, 0.10, 0.20, 0.20)
EOS = ('<|end_of_text|>', '</s>', '<|eot_id|>', '[/INST]')


def normalize(text):
    return ' '.join(str(text).split()).casefold()


def stable_hash(text, salt=''):
    return hashlib.sha256(f'{salt}{text}'.encode()).hexdigest()


def extract_question(prompt):
    """Take the final 'Question: ... \\nAnswer:' block, not the few-shot examples."""
    tail = str(prompt).rsplit('Question:', 1)[-1]
    return re.split(r'\nAnswer:', tail, maxsplit=1)[0].strip()


def strip_eos(text):
    out = str(text)
    for marker in EOS:
        out = out.replace(marker, ' ')
    return ' '.join(out.split())


def assign_split(question_id, fractions, salt):
    bucket = int(stable_hash(question_id, salt)[:8], 16) / 0x100000000
    total = 0.0
    for split, frac in zip(SPLITS, fractions):
        total += frac
        if bucket < total:
            return split
    return SPLITS[-1]


def build(csv_path, generator, dataset, grading, fractions, salt, pilot, pilot_seed):
    df = pd.read_csv(csv_path)
    stats = {'csv_rows': int(len(df))}

    df = df.assign(question=df.prompt.map(extract_question), answer_text=df.answer.map(strip_eos))
    dropped_answer = int(df.answer.isna().sum() + (df.answer_text.str.len() == 0).sum())
    df = df[df.answer.notna() & (df.answer_text.str.len() > 0) & (df.question.str.len() > 0)]
    stats['dropped_empty_answer'] = dropped_answer

    df = df.assign(norm=df.question.map(normalize))
    before = len(df)
    df = df.drop_duplicates('norm', keep='first')
    stats['dropped_duplicate_question'] = before - len(df)

    df = df.assign(question_id='tqa-' + df.norm.map(lambda t: stable_hash(t)[:16]))
    if df.question_id.duplicated().any():
        raise SystemExit('question_id collision; widen the hash prefix')
    df = df.assign(split=df.question_id.map(lambda q: assign_split(q, fractions, salt)))

    if pilot:
        total = len(df)
        keep = []
        for split, group in df.groupby('split', sort=True):
            n = min(len(group), max(1, round(pilot * len(group) / total)))
            keep.extend(group.sample(n=n, random_state=pilot_seed).index)
        df = df.loc[sorted(keep)]

    label = df[grading].astype(int)
    if not label.isin((0, 1)).all():
        raise SystemExit(f'{grading} is not binary')

    rows = [{'row_id': f'{generator}-{qid}', 'question_id': qid, 'question': q, 'answer': a,
             'correct': int(c), 'split': s, 'dataset': dataset, 'generator_model': generator}
            for qid, q, a, c, s in zip(df.question_id, df.question, df.answer_text, label, df.split)]
    stats['exported_rows'] = len(rows)
    stats['accuracy'] = round(float(label.mean()), 4)
    stats['by_split'] = {s: {'rows': int((df.split == s).sum()),
                             'accuracy': round(float(label[df.split == s].mean()), 4)}
                         for s in SPLITS}
    return rows, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generator', required=True,
                   choices=['llama3.1_8b_chat', 'llama3.3_70b', 'mistral_7b_instruct'])
    p.add_argument('--root', default='../evaluations_revisited')
    p.add_argument('--dataset', default='trivia_qa_60k')
    p.add_argument('--grading', default='correct_revisited', choices=['correct', 'correct_revisited'])
    p.add_argument('--out', required=True, help='output directory (must not exist)')
    p.add_argument('--fractions', type=float, nargs=4, default=list(DEFAULT_FRACTIONS),
                   metavar=('TRAIN', 'VAL', 'CAL', 'TEST'))
    p.add_argument('--split-salt', default='tinker-uq-v1',
                   help='fixed salt for the question-level split hash; changing it reshuffles splits')
    p.add_argument('--pilot', type=int, default=0, help='if >0, keep about this many questions')
    p.add_argument('--pilot-seed', type=int, default=0)
    a = p.parse_args()

    if abs(sum(a.fractions) - 1) > 1e-9:
        raise SystemExit('fractions must sum to 1')
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'{out} exists; output directories must be new')
    csv_path = Path(a.root) / a.generator / a.dataset / 'base' / 'main_generations_evaluated_revisited.csv'
    if not csv_path.exists():
        raise SystemExit(f'missing {csv_path}')

    rows, stats = build(csv_path, a.generator, a.dataset, a.grading,
                        a.fractions, a.split_salt, a.pilot, a.pilot_seed)
    out.mkdir(parents=True)
    with (out / 'cached_answers.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    provenance = {'source_csv': str(csv_path),
                  'source_sha256': hashlib.sha256(csv_path.read_bytes()).hexdigest(),
                  'generator_model': a.generator, 'dataset': a.dataset, 'grading_column': a.grading,
                  'split_fractions': dict(zip(SPLITS, a.fractions)), 'split_salt': a.split_salt,
                  'pilot_questions': a.pilot, 'pilot_seed': a.pilot_seed, 'stats': stats}
    (out / 'prepare_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(stats, indent=2))
    print(f'wrote {out}/cached_answers.jsonl')


if __name__ == '__main__':
    main()
