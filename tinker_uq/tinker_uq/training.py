"""Tinker binary-report training. See README.md for protocol and limitations."""
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import warnings


from .data import read_rows, audit
from .reporting import mapping_items, decode, digest, percentage


def prompt_text(row, arm, legend='normal', *, mapping=None):
    if arm not in ('q', 'qa'): raise ValueError('arm must be q or qa')
    mapping = legend if mapping is None else mapping
    items = mapping_items(mapping)
    task = ('Predict whether a candidate answer drawn from the fixed answer source is correct. '
            'The candidate is hidden.' if arm == 'q' else
            'Judge whether the supplied candidate answer is correct.')
    body = task + '\nQuestion:\n' + row['question']
    if arm == 'qa': body += '\nCandidate answer:\n' + row['answer']
    if isinstance(mapping, str):  # Preserve historical binary prompt byte-for-byte.
        positive = next(k for k, v in items if v == 1)
        negative = next(k for k, v in items if v == 0)
        codebook = f'{positive} = correct; {negative} = incorrect.'
    else:
        body += '\nReport the probability that this answer is correct.'
        codebook = '  '.join(f'{k}={percentage(v)}' for k, v in items)
    return body + f'\nReport legend: {codebook}\nReply with one letter followed by a newline.'


def encode_prompt(tokenizer, row, arm, legend='normal', *, mapping=None):
    # Qwen non-thinking template; prompt and candidate completion are explicitly
    # separate token sequences, identically in training and evaluation.
    # transformers>=5 returns a BatchEncoding unless return_dict is disabled; a bare
    # list() over that yields the dict keys, so the shape is checked rather than assumed.
    encoded = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt_text(row, arm, legend, mapping=mapping)}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if not isinstance(encoded, list):
        encoded = encoded['input_ids']
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError('Chat template returned more than one sequence')
        encoded = encoded[0]
    if not encoded or not all(isinstance(t, int) for t in encoded):
        raise ValueError('Chat template did not return a flat list of token IDs')
    return list(encoded)


def completions(tokenizer, mapping='normal'):
    result = [list(tokenizer.encode(k+'\n', add_special_tokens=False)) for k, _ in mapping_items(mapping)]
    if not all(result) or len({tuple(x) for x in result}) != len(result):
        raise ValueError('Invalid label tokenization')
    for i, x in enumerate(result):
        for j, y in enumerate(result):
            if i != j and y[:len(x)] == x: raise ValueError('Report sequences must be prefix-free')
    return result


def sequence_parts(prompt, completion):
    if not prompt or not completion:
        raise ValueError('Empty prompt or label')
    tokens = prompt + completion
    # Target index len(prompt)-1 is the FIRST label token.
    return tokens[:-1], tokens[1:], len(prompt)-1


def make_batch(rows, tokenizer, arm, tinker, mapping=None, *, mappings=None):
    # Explicit mappings use correctness labels; legacy binary callers use class indices.
    legacy = mapping is None and mappings is None
    maps = mappings if mappings is not None else [mapping if mapping is not None else 'normal']
    if not maps: raise ValueError('At least one mapping required')
    data, starts, labels = [], [], []
    for row in rows:
        for codebook in maps:
            prompt = encode_prompt(tokenizer, row, arm, mapping=codebook)
            for completion in completions(tokenizer, codebook):
                inputs, targets, start = sequence_parts(prompt, completion)
                data.append(tinker.Datum(
                    model_input=tinker.ModelInput.from_ints(inputs),
                    loss_fn_inputs={'target_tokens': tinker.TensorData(data=targets, dtype='int64', shape=[len(targets)])}))
                starts.append(start)
            labels.append((0 if row['correct'] else 1) if legacy else row['correct'])
    return data, starts, labels


def objective_value(scores, labels, mappings, objective, consistency_weight=1., teacher=None, supervision_weight=1.):
    import torch
    items = [mapping_items(m) for m in mappings]
    if any([k for k, _ in x] != [k for k, _ in items[0]] for x in items):
        raise ValueError('Mappings must share the same alphabet')
    if any(sorted(v for _, v in x) != sorted(v for _, v in items[0]) for x in items):
        raise ValueError('Mappings must permute the same value set')
    nv, nc = len(items), len(items[0])
    scores = scores.reshape(-1, nv, nc)
    y = torch.as_tensor(labels, dtype=scores.dtype, device=scores.device).reshape(-1, nv)
    if not torch.all(y == y[:, :1]): raise ValueError('Paired views have different labels')
    values = torch.tensor([[v for _, v in x] for x in items], dtype=scores.dtype, device=scores.device)
    p = scores.softmax(-1); mu = (p*values).sum(-1)
    mean_brier = ((mu-y)**2).mean()
    token_brier = (p*(values-y[..., None])**2).sum(-1).mean()
    consistency = ((mu[:, 0]-mu[:, 1])**2).mean() if nv == 2 else mu.sum()*0
    if objective == 'token_brier': loss = token_brier
    elif objective == 'mean_brier': loss = mean_brier
    elif objective == 'mean_consistency':
        if nv != 2: raise ValueError('Consistency requires two matched views')
        loss = mean_brier + consistency_weight*consistency
    elif objective in ('distill', 'distill_supervised'):
        if teacher is None: raise ValueError('Aligned teacher distribution required')
        target = torch.as_tensor(teacher, dtype=scores.dtype, device=scores.device)
        if target.shape != (scores.shape[0], nc) or not torch.allclose(target.sum(-1), torch.ones(scores.shape[0],device=scores.device,dtype=scores.dtype)) or (target<0).any():
            raise ValueError('Invalid value-aligned teacher')
        order = torch.argsort(values, dim=-1).unsqueeze(0).expand(scores.shape[0], -1, -1)
        logp = torch.gather(scores.log_softmax(-1), -1, order)
        kl = (target[:,None,:]*(target[:,None,:].clamp_min(1e-30).log()-logp)).sum(-1).mean()
        loss = kl + (supervision_weight*mean_brier if objective == 'distill_supervised' else 0)
    else: raise ValueError('Unknown objective')
    return loss, {'loss':float(loss.detach()), 'token_brier':float(token_brier.detach()),
                  'mean_brier':float(mean_brier.detach()), 'numerical_consistency':float(consistency.detach())}


def loss_function(starts, labels, mapping=None, *, mappings=None, objective='binary_nll',
                  consistency_weight=1., teacher=None, supervision_weight=1.):
    import torch
    maps = mappings if mappings is not None else [mapping if mapping is not None else 'normal']
    nc = len(mapping_items(maps[0]))
    def loss(data, logprobs):
        if len(logprobs) != len(starts): raise ValueError('Logprob count mismatch')
        scores = torch.stack([lp[s:].sum() for lp, s in zip(logprobs, starts)]).reshape(-1, nc)
        if objective == 'binary_nll':
            if nc != 2: raise ValueError('binary_nll requires binary alphabet')
            target = labels
            if mapping is not None or mappings is not None:
                target = [next(i for i, (_, v) in enumerate(mapping_items(maps[j%len(maps)])) if v == y) for j,y in enumerate(labels)]
            value = torch.nn.functional.cross_entropy(scores, torch.tensor(target, dtype=torch.long, device=scores.device))
            return value, {'binary_nll':float(value.detach())}
        return objective_value(scores, labels, maps, objective, consistency_weight, teacher, supervision_weight)
    return loss


def normalize_scores(scores, legend='normal', *, mapping=None, full=False):
    result = decode(scores, legend if mapping is None else mapping)
    return result if full else (result['decoded_mean'], result['allowed_label_logmass'])


def score_rows(client, rows, tokenizer, arm, stage, output, tinker, window=256, mappings=None):
    """Score every row under both legends.

    Requests are submitted in bounded windows rather than awaited one row at a time.
    The set of requests, the prompt cache and the output order are unchanged; only the
    point at which each future is resolved moves, so the exported scores are identical.
    """
    # Cache equal prompts so q-only scores are exactly identical within question.
    cache = {}
    mappings = mappings or {'normal':'normal', 'reversed':'reversed'}
    for legend, mapping in mappings.items():
        cs = completions(tokenizer, mapping)
        prompts = [encode_prompt(tokenizer, row, arm, mapping=mapping) for row in rows]
        pending = []
        for prompt in prompts:
            key = tuple(prompt)
            if key not in cache and key not in set(pending):
                pending.append(key)
        for start in range(0, len(pending), window):
            chunk = pending[start:start+window]
            futures = [(key, [client.compute_logprobs(tinker.ModelInput.from_ints(list(key)+c)) for c in cs])
                       for key in chunk]
            for key, fs in futures:
                scores = []
                for c, future in zip(cs, fs):
                    lp = future.result()
                    if len(lp) != len(key)+len(c):
                        raise ValueError('Unexpected logprob alignment')
                    tail = lp[len(key):]
                    if any(x is None or not math.isfinite(x) for x in tail):
                        raise ValueError('Missing/nonfinite report logprobs')
                    scores.append(sum(tail))
                cache[key] = scores
            print(f'  {arm}/{stage}/{legend}: scored {min(start+window,len(pending))}/{len(pending)} unique prompts',
                  flush=True)
        for row, prompt in zip(rows, prompts):
            scores = cache[tuple(prompt)]
            decoded = normalize_scores(scores, mapping=mapping, full=True)
            result = {k: row[k] for k in ('row_id','question_id','split','correct')}
            result.update(decoded)
            result.update(arm=arm, stage=stage, legend=legend, p_correct=decoded['decoded_mean'],
                          completion_token_ids=cs, prompt_provenance={
                              'text':prompt_text(row, arm, mapping=mapping),
                              'text_sha256':hashlib.sha256(prompt_text(row, arm, mapping=mapping).encode()).hexdigest(),
                              'token_ids_sha256':digest(prompt),
                              'chat_template_sha256':hashlib.sha256(str(getattr(tokenizer, 'chat_template', '')).encode()).hexdigest(),
                              'thinking':False, 'generation_prompt':True})
            output.write(json.dumps(result)+'\n')
        output.flush()


def summarize(path):
    from sklearn.metrics import roc_auc_score
    groups = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        groups[(r['arm'],r['stage'],r['legend'],r['split'])].append(r)
    result = {}
    for key, rs in groups.items():
        y, p = [r['correct'] for r in rs], [r['p_correct'] for r in rs]
        pair_groups = defaultdict(list)
        for r in rs:
            pair_groups[r['question_id']].append(r)
        pair = []
        for values in pair_groups.values():
            pos = [r['p_correct'] for r in values if r['correct']]
            neg = [r['p_correct'] for r in values if not r['correct']]
            if pos and neg:
                pair.append(sum((a>b)+0.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg)))
        result['/'.join(key)] = {
            'n': len(rs), 'brier': sum((a-b)**2 for a,b in zip(y,p))/len(y),
            'nll': -sum(a*math.log(max(b,1e-12))+(1-a)*math.log(max(1-b,1e-12)) for a,b in zip(y,p))/len(y),
            'auroc': float(roc_auc_score(y,p)) if len(set(y))==2 else None,
            'within_question_pair_accuracy': sum(pair)/len(pair) if pair else None,
            'paired_questions': len(pair)}
    return result


def run(args, rows):
    import tinker
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    manifest = vars(args).copy()
    manifest.update(data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                    audit=audit(rows), versions={p: importlib.metadata.version(p) for p in ('tinker','torch','transformers')})
    service = tinker.ServiceClient()
    supported = {x.model_name for x in service.get_server_capabilities().supported_models}
    if args.model not in supported:
        raise ValueError(f'Model unavailable: {args.model}. Available: {sorted(supported)}')
    base = service.create_sampling_client(base_model=args.model)
    tokenizer = base.get_tokenizer()
    if not getattr(tokenizer, 'chat_template', None):
        raise ValueError('This v1 runner requires a chat-template tokenizer; base models need an explicit renderer.')
    cs = completions(tokenizer)
    manifest['label_token_ids'] = cs
    template = str(getattr(tokenizer, 'chat_template', ''))
    manifest['chat_template_sha256'] = hashlib.sha256(template.encode()).hexdigest()
    train = [r for r in rows if r['split']=='train']
    heldout = rows  # Export train scores for partition fitting as well as all held-out splits.
    # Fail before training on overlong input; no silent truncation of answers.
    for r in rows:
        for arm in ('q','qa'):
            for legend in ('normal','reversed'):
                if len(encode_prompt(tokenizer,r,arm,legend))+max(map(len,cs)) > args.max_tokens:
                    raise ValueError(f'Overlong row {r["row_id"]}; explicitly revise data/max-tokens')
    (out/'prompt_audit.json').write_text(json.dumps({
        arm: {'text': prompt_text(rows[0], arm, 'normal'),
              'token_ids': encode_prompt(tokenizer, rows[0], arm, 'normal'),
              'rendered': tokenizer.decode(encode_prompt(tokenizer, rows[0], arm, 'normal'))}
        for arm in ('q','qa')}, indent=2))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with (out/'predictions.jsonl').open('w') as predictions:
        for arm in ('q','qa'):
            print(f'{arm}: evaluating base (all splits)' , flush=True)
            score_rows(base,heldout,tokenizer,arm,'base',predictions,tinker,getattr(args,'score_window',256))
            client = service.create_lora_training_client(base_model=args.model,rank=args.rank,seed=args.seed)
            rng = random.Random(args.seed)
            for epoch in range(args.epochs):
                shuffled = list(train)
                rng.shuffle(shuffled)
                for offset in range(0,len(shuffled),args.batch_size):
                    data, starts, labels = make_batch(shuffled[offset:offset+args.batch_size],tokenizer,arm,tinker)
                    result = client.forward_backward_custom(data,loss_function(starts,labels)).result()
                    client.optim_step(tinker.AdamParams(learning_rate=args.lr)).result()
                    with (out/'training.jsonl').open('a') as log:
                        log.write(json.dumps({'arm':arm,'epoch':epoch+1,'offset':offset,'metrics':result.metrics})+'\n')
                state = client.save_state(f'{arm}-epoch-{epoch+1}').result()
                with (out/'checkpoints.jsonl').open('a') as log:
                    log.write(json.dumps({'arm':arm,'epoch':epoch+1,'state_path':state.path})+'\n')
                print(f'{arm}: completed epoch {epoch+1}',flush=True)
            saved = client.save_weights_for_sampler(f'{arm}-final').result()
            with (out/'checkpoints.jsonl').open('a') as log:
                log.write(json.dumps({'arm':arm,'sampler_path':saved.path})+'\n')
            sampler = service.create_sampling_client(model_path=saved.path)
            score_rows(sampler,heldout,tokenizer,arm,'trained',predictions,tinker,getattr(args,'score_window',256))
    (out/'metrics.json').write_text(json.dumps(summarize(out/'predictions.jsonl'),indent=2))
    print(f'Finished: {out.resolve()}')

