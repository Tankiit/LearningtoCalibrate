"""Input validation and reproducible question-level sampling."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import warnings

SPLITS = ('train', 'val', 'cal', 'test')


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def write_jsonl(path, rows):
    with Path(path).open('w') as f:
        for r in rows:
            f.write(json.dumps(r, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    rows = load_jsonl(path)
    if not rows:
        raise ValueError('Empty dataset')
    ids, groups, texts, candidates = set(), {}, {}, set()
    for r in rows:
        for k in ('row_id', 'question_id', 'question', 'answer', 'correct', 'split'):
            if k not in r:
                raise ValueError(f'Missing field {k}')
        for k in ('row_id', 'question_id', 'question', 'answer'):
            if not isinstance(r[k], str) or not r[k].strip():
                raise ValueError(f'{k} must be a nonempty string')
        if r['row_id'] in ids:
            raise ValueError('Duplicate row_id')
        ids.add(r['row_id'])
        if type(r['correct']) not in (int, bool) or r['correct'] not in (0, 1):
            raise ValueError('correct must be independently graded binary 0/1')
        if r['split'] not in SPLITS:
            raise ValueError('Use train/val/cal/test splits')
        signature = (r['question'], r['split'])
        if groups.setdefault(r['question_id'], signature) != signature:
            raise ValueError('Question ID has inconsistent text or crosses splits')
        normalized = ' '.join(r['question'].split()).casefold()
        if texts.setdefault(normalized, r['question_id']) != r['question_id']:
            raise ValueError('Duplicate question text under different IDs; unify IDs')
        candidate = (r['question_id'], r['answer'])
        if candidate in candidates:
            warnings.warn('Repeated identical candidate within a question; confirm these are intended repeated draws.')
        candidates.add(candidate)
    if {r['split'] for r in rows} != set(SPLITS):
        raise ValueError('All four splits must be nonempty')
    for key in ('dataset', 'generator_model'):
        values = {r.get(key) for r in rows}
        if len(values)>1:
            raise ValueError(f'Use a separate run per {key}; mixed/missing values found')
    labels = defaultdict(list)
    for r in rows:
        labels[r['question_id']].append(r['correct'])
    if all(sum(v)*2 == len(v) for v in labels.values()):
        warnings.warn('Every question is label-balanced: q-only target is 0.5 by construction.')
    return rows


def audit(rows):
    return {split: {'rows': len(rs), 'questions': len({r['question_id'] for r in rs}),
                    'accuracy': sum(r['correct'] for r in rs)/len(rs)}
            for split in SPLITS for rs in [[r for r in rows if r['split']==split]]}


def read_predictions(path):
    rows = load_jsonl(path)
    seen, metadata, q_splits = set(), {}, {}
    if not rows:
        raise ValueError('Empty predictions')
    for r in rows:
        for k in ('row_id','question_id','split','correct','arm','stage','legend','p_correct'):
            if k not in r:
                raise ValueError(f'Missing prediction field: {k}')
        if r['split'] not in SPLITS or r['correct'] not in (0,1):
            raise ValueError('Invalid split or target')
        if not isinstance(r['p_correct'], (float,int)) or not math.isfinite(r['p_correct']) or not 0<=r['p_correct']<=1:
            raise ValueError('p_correct must be finite and in [0,1]')
        key = (r['row_id'],r['arm'],r['stage'],r['legend'])
        if key in seen:
            raise ValueError(f'Duplicate prediction: {key}')
        seen.add(key)
        meta = (r['question_id'],r['split'],r['correct'])
        if metadata.setdefault(r['row_id'],meta)!=meta:
            raise ValueError('Inconsistent labels/IDs across prediction arms')
        if q_splits.setdefault(r['question_id'],r['split'])!=r['split']:
            raise ValueError('Question crosses splits')
    return rows


def join_signals(rows, stage='trained', legend='normal', anchor='qa'):
    """Hold the anchor score fixed while comparing q versus qa partition signals."""
    maps = {arm: {r['row_id']:r for r in rows if r['arm']==arm and r['stage']==stage and r['legend']==legend}
            for arm in {'q','qa',anchor}}
    ids = set(maps[anchor])
    if not ids or any(set(v)!=ids for v in maps.values()):
        raise ValueError('q/qa/anchor exports must contain exactly the same row IDs for this stage/legend')
    result=[]
    for row_id in sorted(ids):
        r = maps[anchor][row_id]
        value = {k:r[k] for k in ('row_id','question_id','split','correct')}
        value.update(anchor_p=r['p_correct'], q=maps['q'][row_id]['p_correct'], qa=maps['qa'][row_id]['p_correct'])
        result.append(value)
    return result


def one_per_question(rows, policy='error', seed=0):
    """Sampling uses only IDs, never correctness or model scores."""
    if policy not in ('error','sample'):
        raise ValueError('Unknown candidate policy')
    groups=defaultdict(list)
    for r in rows:
        groups[r['question_id']].append(r)
    chosen=[]
    for qid, rs in sorted(groups.items()):
        if len(rs)>1 and policy=='error':
            raise ValueError('Multiple candidates per question: use --candidate-policy sample or supply one candidate per question')
        rs=sorted(rs,key=lambda r:r['row_id'])
        entropy=int.from_bytes(hashlib.sha256(f'{seed}:{qid}'.encode()).digest(),'big')
        chosen.append(random.Random(entropy).choice(rs))
    return chosen
