import json, random, sys, os
data_dir, out_dir = sys.argv[1], sys.argv[2]
convs = [json.loads(l) for l in open(os.path.join(data_dir, 'eligible.jsonl'))]
random.seed(7); random.shuffle(convs)
def clip(s, head, tail=0):
    s = ' '.join(s.split())
    if len(s) <= head + tail + 20: return s
    return s[:head] + ' … ' + (s[-tail:] if tail else '')
index = []
for i, c in enumerate(convs, 1):
    cid = f'C{i:03d}'
    msgs = c['messages']
    # pair into exchanges
    ex = []; cur = None
    for m in msgs:
        if m['role'] == 'user':
            cur = {'u': m['content'], 'a': ''}; ex.append(cur)
        elif cur is not None:
            cur['a'] += m['content']
    index.append({'cid': cid, 'id': c['id'], 'row_idx': c['row_idx'], 'model': c['model'], 'n_ex': len(ex), 'exchanges': ex})
json.dump(index, open(os.path.join(data_dir, 'conversations_indexed.json'), 'w'))
batch, bi, size = [], 1, 0
def flush():
    global batch, bi, size
    if batch:
        open(os.path.join(out_dir, f'batch_{bi:02d}.txt'), 'w').write('\n'.join(batch)); bi += 1; batch, size = [], 0
for c in index:
    lines = [f"### {c['cid']} ({c['n_ex']} turns, {c['model']})"]
    seen = {}
    for k, e in enumerate(c['exchanges'], 1):
        u = ' '.join(e['u'].split())
        hint = ''
        if u in seen: hint = f' [IDENTICAL to U{seen[u]}]'
        else:
            for j, prev in seen.items():
                if len(u) > 150 and len(j) > 150 and u[:150] == j[:150]: hint = f' [same first 150 chars as U{prev}]'; break
            seen[u] = k
        lines.append(f"[{k}] U ({len(e['u'])}ch){hint}: {clip(e['u'], 420, 120)}")
        lines.append(f"    A ({len(e['a'])}ch): {clip(e['a'], 170)}")
    block = '\n'.join(lines) + '\n'
    batch.append(block); size += len(block)
    if size > 45000: flush()
flush()
print('batches', bi - 1, 'convs', len(index), 'turns', sum(c['n_ex'] for c in index))
