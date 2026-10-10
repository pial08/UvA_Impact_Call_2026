"""Compute how much of the history sent to the model is not needed, from turn-level labels.

Usage: python -I compute.py <data_dir> <labels_dir> <out_dir>
"""
import json, os, re, sys, glob, random, csv
from tokenizers import Tokenizer

data_dir, labels_dir, out_dir = sys.argv[1:4]
os.makedirs(out_dir, exist_ok=True)
convs = {c['cid']: c for c in json.load(open(os.path.join(data_dir, 'conversations_indexed.json')))}
tok = Tokenizer.from_file(os.path.join(data_dir, 'claude_legacy_tokenizer.json'))
ntok = lambda s: len(tok.encode(s).ids) if s else 0

# ---- parse labels ----
labels = {}
for f in sorted(glob.glob(os.path.join(labels_dir, 'batch_*.txt'))):
    for line in open(f):
        line = line.strip()
        if not line: continue
        cid, rest = line.split(':', 1)
        labels[cid.strip()] = rest.split()
missing = sorted(set(convs) - set(labels))
assert not missing, f'unlabelled: {missing}'

def parse_list(code, t):
    out = set()
    for part in code.split(','):
        if '-' in part:
            a, b = map(int, part.split('-')); out.update(range(a, b + 1))
        else:
            out.add(int(part))
    assert all(1 <= k < t for k in out), (code, t)
    return out

def needed_sets(codes, mode='main'):
    """codes[0] is '-' for turn 1. Returns list of sets for t=1..T (1-indexed turns)."""
    T = len(codes); res = [set()]; last_p = 1
    for t in range(2, T + 1):
        c = codes[t - 1]
        if c == 'P':
            need = set()
        elif c == 'S':
            need = set(range(last_p, t))
        elif c == 'A':
            need = set(range(1, t))
        else:
            need = parse_list(c, t)
        if mode == 'pivot_keeps_prev' and c == 'P':
            need = {t - 1}
        res.append(need)
        if c == 'P':
            last_p = t
    return res

rows = []  # one row per user turn
for cid, c in convs.items():
    codes = labels[cid]
    assert len(codes) == c['n_ex'], (cid, len(codes), c['n_ex'])
    assert codes[0] == '-', cid
    ex_tok = [ntok(e['u']) + ntok(e['a']) for e in c['exchanges']]
    u_tok = [ntok(e['u']) for e in c['exchanges']]
    u_txt = [' '.join(e['u'].split()) for e in c['exchanges']]
    for mode in ('main', 'pivot_keeps_prev'):
        need = needed_sets(codes, mode)
        for t in range(1, c['n_ex'] + 1):
            H = sum(ex_tok[:t - 1]); N = sum(ex_tok[k - 1] for k in need[t - 1])
            rows.append(dict(mode=mode, cid=cid, n_turns=c['n_ex'], turn=t, code=codes[t - 1],
                             hist_tokens=H, needed_tokens=N, user_tokens=u_tok[t - 1],
                             duplicate=u_txt[t - 1] in u_txt[:t - 1]))

with open(os.path.join(out_dir, 'turn_level_results.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(r for r in rows if r['mode'] == 'main')

def summarise(rs):
    H = sum(r['hist_tokens'] for r in rs); N = sum(r['needed_tokens'] for r in rs)
    U = sum(r['user_tokens'] for r in rs)
    later = [r for r in rs if r['turn'] >= 2]
    return dict(
        share_history_unneeded=(H - N) / H if H else float('nan'),
        share_input_unneeded=(H - N) / (H + U) if H + U else float('nan'),
        pivot_rate=sum(r['code'] == 'P' for r in later) / len(later),
        turns_with_some_unneeded=sum(r['needed_tokens'] < r['hist_tokens'] for r in later) / len(later),
        hist_tokens=H, needed_tokens=N, user_tokens=U, n_turns_ge2=len(later),
        n_convs=len({r['cid'] for r in rs}))

def boot(rs, key, B=10000, seed=1):
    by = {}
    for r in rs: by.setdefault(r['cid'], []).append(r)
    ids = list(by); rng = random.Random(seed); vals = []
    for _ in range(B):
        sample = [r for i in (rng.choice(ids) for _ in ids) for r in by[i]]
        vals.append(summarise(sample)[key])
    vals.sort()
    return vals[int(.025 * B)], vals[int(.975 * B) - 1]

main = [r for r in rows if r['mode'] == 'main']
alt = [r for r in rows if r['mode'] == 'pivot_keeps_prev']
res = {'overall': summarise(main)}
for k in ('share_history_unneeded', 'share_input_unneeded', 'pivot_rate'):
    res['overall'][k + '_ci95'] = boot(main, k)

def binlen(n): return '4-5' if n <= 5 else ('6-9' if n <= 9 else '10+')
res['by_session_length'] = {}
for b in ('4-5', '6-9', '10+'):
    rs = [r for r in main if binlen(r['n_turns']) == b]
    s = summarise(rs); s['share_history_unneeded_ci95'] = boot(rs, 'share_history_unneeded', B=4000)
    res['by_session_length'][b] = s

def binpos(t): return '2-3' if t <= 3 else ('4-6' if t <= 6 else ('7-10' if t <= 10 else '11+'))
res['by_turn_position'] = {}
for b in ('2-3', '4-6', '7-10', '11+'):
    rs = [r for r in main if r['turn'] >= 2 and binpos(r['turn']) == b]
    res['by_turn_position'][b] = summarise(rs)

# robustness checks
res['robustness'] = {
    'every_pivot_keeps_previous_exchange': summarise(alt),
    'excluding_identical_resubmissions': summarise([r for r in main if not r['duplicate']]),
}
res['robustness']['every_pivot_keeps_previous_exchange']['share_history_unneeded_ci95'] = boot(alt, 'share_history_unneeded', B=4000)

# per-conversation distribution
per = {}
for r in main: per.setdefault(r['cid'], []).append(r)
shares = sorted(summarise(v)['share_history_unneeded'] for v in per.values())
res['per_conversation'] = dict(
    median=shares[len(shares) // 2],
    share_of_convs_with_zero_unneeded=sum(s == 0 for s in shares) / len(shares),
    share_of_convs_with_ge50pct_unneeded=sum(s >= .5 for s in shares) / len(shares))
res['code_counts'] = {k: sum(1 for r in main if r['turn'] >= 2 and (r['code'] == k if k in 'PSA' else r['code'] not in 'PSA')) for k in ('P', 'S', 'A', 'explicit')}
res['models'] = {}
for c in convs.values(): res['models'][c['model']] = res['models'].get(c['model'], 0) + 1
res['tokenizer'] = 'Claude legacy BPE tokenizer (bundled with anthropic-sdk-python 0.34.2); ratios are insensitive to the tokenizer choice'
json.dump(res, open(os.path.join(out_dir, 'results.json'), 'w'), indent=2)
print(json.dumps(res, indent=2))
