"""Shared letter/value reporting contract. Numeric values always lie in [0, 1]."""
import hashlib
import json
import math
import random
from decimal import Decimal

LETTERS = tuple('ABCDEFGHJKL')
SCHEMA = 'letter-value-report-v1'


def percentage(value):
    text=format(Decimal(str(value))*100,'f')
    if '.' in text:text=text.rstrip('0').rstrip('.')
    return text+'%'


def mapping_items(mapping):
    if isinstance(mapping, str):
        if mapping not in ('normal', 'reversed'):
            raise ValueError('Unknown binary legend')
        mapping = {'A': 1., 'B': 0.} if mapping == 'normal' else {'A': 0., 'B': 1.}
    if not isinstance(mapping, dict) or len(mapping) < 2:
        raise ValueError('An explicit letter-value mapping is required')
    items = sorted(mapping.items())  # Fixed glyph/list order; permute values only.
    if any(not isinstance(k, str) or len(k) != 1 or not k.isascii() or not k.isupper() for k, _ in items):
        raise ValueError('Labels must be single uppercase ASCII letters')
    values = [v for _, v in items]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError('Values must be finite probabilities in [0,1]')
    if len(set(values)) != len(values) or min(values) != 0 or max(values) != 1:
        raise ValueError('Mapping must be one-to-one and include endpoints 0 and 1')
    return [(k, float(v)) for k, v in items]


def mapping11(reverse=False):
    values = [i/10 for i in range(11)]
    return dict(zip(LETTERS, values[::-1] if reverse else values))


def freeze_mappings(seed=20260914, count=8):
    family = {'normal': mapping11(), 'reversed': mapping11(True)}
    rng = random.Random(seed)
    seen = {tuple(m.values()) for m in family.values()}
    while len(family) < count+2:
        values = [i/10 for i in range(11)]; rng.shuffle(values)
        if tuple(values) in seen: continue
        seen.add(tuple(values)); family[f'heldout_{len(family)-2:02d}'] = dict(zip(LETTERS, values))
    return family


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def decode(scores, mapping):
    items = mapping_items(mapping)
    if len(scores) != len(items) or any(not math.isfinite(x) for x in scores):
        raise ValueError('Invalid sequence log probabilities')
    m = max(scores)
    logmass = m + math.log(sum(math.exp(x-m) for x in scores))
    if logmass > 1e-5:
        raise ValueError('Allowed sequence mass exceeds one')
    p = [math.exp(x-logmass) for x in scores]
    return dict(schema_version=SCHEMA, mapping=dict(items), letters=[k for k, _ in items],
                probability_vector=p, decoded_mean=sum(q*v for q, (_, v) in zip(p, items)),
                allowed_alphabet_mass=math.exp(logmass), allowed_label_logmass=logmass,
                label_logprobs=list(scores), distribution_semantics='conditional_on_allowed_completion_sequences',
                mass_semantics='probability_of_allowed_letter_plus_newline_sequences')


def align_to_values(probability_vector, mapping):
    items = mapping_items(mapping)
    if len(probability_vector) != len(items): raise ValueError('Probability/mapping size mismatch')
    return [probability_vector[i] for i in sorted(range(len(items)), key=lambda i: items[i][1])]


def codebook_prompt(mapping, value):
    legend = '  '.join(f'{k}={percentage(v)}' for k, v in mapping_items(mapping))
    return (f'Report legend: {legend}\nThe supplied probability is {percentage(value)}.\n'
            'Encode this probability using the report legend.\nReply with one letter followed by a newline.')
