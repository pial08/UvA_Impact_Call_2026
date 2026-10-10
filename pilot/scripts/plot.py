"""Figure for the context-relevance experiment. Usage: python -I plot.py <results_dir>"""
import csv, json, os, sys, random, collections
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

res_dir = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(res_dir, 'turn_level_results.csv'))))
for r in rows:
    for k in ('turn', 'n_turns', 'hist_tokens', 'needed_tokens'):
        r[k] = int(r[k])
res = json.load(open(os.path.join(res_dir, 'results.json')))

ACCENT, NEUTRAL, INK, MUTED, GRID = '#2a78d6', '#b9b8b2', '#0b0b0b', '#52514e', '#e6e5e0'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.spines.top': False,
                     'axes.spines.right': False})

def share(rs):
    h = sum(r['hist_tokens'] for r in rs); n = sum(r['needed_tokens'] for r in rs)
    return (h - n) / h

def boot_ci(rs, B=4000, seed=3):
    by = collections.defaultdict(list)
    for r in rs: by[r['cid']].append(r)
    ids = list(by); rng = random.Random(seed); v = []
    for _ in range(B):
        v.append(share([r for i in (rng.choice(ids) for _ in ids) for r in by[i]]))
    v.sort(); return v[int(.025 * B)], v[int(.975 * B) - 1]

bins = [('2-3', 2, 3), ('4-6', 4, 6), ('7-10', 7, 10), ('11+', 11, 999)]
pts = []
for lab, lo, hi in bins:
    rs = [r for r in rows if lo <= r['turn'] <= hi]
    s = share(rs); lo_ci, hi_ci = boot_ci(rs)
    pts.append((lab, s, lo_ci, hi_ci, len(rs)))

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={'width_ratios': [1, 1.15]})

# (a) share of history not needed, by turn position
x = range(len(pts))
a.errorbar(x, [p[1] * 100 for p in pts],
           yerr=[[100 * (p[1] - p[2]) for p in pts], [100 * (p[3] - p[1]) for p in pts]],
           fmt='o', color=ACCENT, ecolor=ACCENT, elinewidth=2, capsize=0, markersize=8, zorder=3)
for i, p in enumerate(pts):
    a.annotate(f'{p[1]*100:.0f}%', (i, p[1] * 100), xytext=(10, 0), textcoords='offset points',
               va='center', color=INK, fontsize=10)
overall = res['overall']['share_history_unneeded'] * 100
a.axhline(overall, color=MUTED, lw=1, ls='--', zorder=1)
a.text(-0.45, overall + 2, f'all turns {overall:.0f}%', color=MUTED, ha='left', fontsize=9)
a.set_xticks(list(x)); a.set_xticklabels([f'{p[0]}\n(n={p[4]})' for p in pts])
a.set_xlabel('User turn number in the conversation')
a.set_ylabel('History tokens not needed (%)')
a.set_ylim(0, 85); a.set_xlim(-0.5, len(pts) - 0.3)
a.yaxis.grid(True, color=GRID); a.set_axisbelow(True)
a.set_title('(a) The later the turn, the more history is unnecessary', loc='left', fontsize=11, color=INK)

# (b) distribution across conversations
per = collections.defaultdict(list)
for r in rows: per[r['cid']].append(r)
shares = [share(v) * 100 for v in per.values()]
zero = sum(1 for s in shares if s == 0)
counts = [0] * 10
for s in shares:
    if s > 0:
        counts[min(int((s - 1e-9) // 10), 9)] += 1
b.bar([-6], [zero], width=8.6, color=NEUTRAL, zorder=3)
b.bar([e * 10 + 5 for e in range(10)], counts, width=8.6, color=ACCENT, zorder=3)
b.set_xticks([-6] + list(range(0, 101, 10)))
b.set_xticklabels(['none'] + [str(e) for e in range(0, 101, 10)])
b.set_xlim(-12, 102)
b.set_xlabel('History tokens not needed in the conversation (%)')
b.set_ylabel('Conversations')
b.yaxis.grid(True, color=GRID); b.set_axisbelow(True)
z = res['per_conversation']['share_of_convs_with_zero_unneeded'] * 100
h = res['per_conversation']['share_of_convs_with_ge50pct_unneeded'] * 100
b.text(1, zero - 4, f'{z:.0f}% of conversations:\nall history needed', color=MUTED, fontsize=9, va='top')
b.text(97, max(counts) + 14, f'{h:.0f}% of conversations:\nhalf or more not needed', color=INK,
       fontsize=9, ha='right')
b.set_title('(b) Savings are concentrated in some conversations', loc='left', fontsize=11, color=INK)

n = res['overall']['n_convs']; t = res['overall']['n_turns_ge2']
fig.text(0.01, -0.04, f'{n} English WildChat conversations with 4+ user turns ({t} labelled turns). '
         'Whiskers: 95% bootstrap CI over conversations. Conservative labels: a lower bound on what can be removed.',
         fontsize=8.5, color=MUTED)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(res_dir, f'unneeded_context.{ext}'), dpi=200, bbox_inches='tight')
print(pts)
