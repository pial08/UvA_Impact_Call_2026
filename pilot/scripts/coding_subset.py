"""Compare conversations that contain code with all other conversations.

A conversation counts as a coding conversation when any user message or assistant
reply contains a fenced code block (```).

Usage:  python coding_subset.py <data_dir> <results_dir>
Reads   data/conversations_indexed.json, data/claude_legacy_tokenizer.json,
        results/turn_level_results.csv
Writes  results/coding_subset.json
"""
import collections
import csv
import json
import os
import random
import sys

from tokenizers import Tokenizer

data_dir, res_dir = sys.argv[1:3]
convs = json.load(open(os.path.join(data_dir, "conversations_indexed.json")))
tok = Tokenizer.from_file(os.path.join(data_dir, "claude_legacy_tokenizer.json"))
ntok = lambda s: len(tok.encode(s).ids) if s else 0

rows = list(csv.DictReader(open(os.path.join(res_dir, "turn_level_results.csv"))))
for r in rows:
    for k in ("turn", "hist_tokens", "needed_tokens"):
        r[k] = int(r[k])

is_code = {c["cid"]: any("```" in e["u"] or "```" in e["a"] for e in c["exchanges"]) for c in convs}
total_tokens = {c["cid"]: sum(ntok(e["u"]) + ntok(e["a"]) for e in c["exchanges"]) for c in convs}
n_turns = {c["cid"]: c["n_ex"] for c in convs}


def share(rs):
    h = sum(r["hist_tokens"] for r in rs)
    n = sum(r["needed_tokens"] for r in rs)
    return (h - n) / h if h else float("nan")


def boot_ci(rs, B=4000, seed=1):
    by = collections.defaultdict(list)
    for r in rs:
        by[r["cid"]].append(r)
    ids, rng, vals = list(by), random.Random(seed), []
    for _ in range(B):
        vals.append(share([r for i in (rng.choice(ids) for _ in ids) for r in by[i]]))
    vals.sort()
    return vals[int(.025 * B)], vals[int(.975 * B) - 1]


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


out = {}
all_hist = sum(r["hist_tokens"] for r in rows if r["turn"] >= 2)
for name, flag in (("coding", True), ("other", False)):
    cids = [c for c, v in is_code.items() if v == flag]
    rs = [r for r in rows if r["turn"] >= 2 and is_code[r["cid"]] == flag]
    late = [r for r in rs if r["turn"] >= 11]
    out[name] = dict(
        n_conversations=len(cids),
        median_turns=median(n_turns[c] for c in cids),
        share_of_all_history_tokens=sum(r["hist_tokens"] for r in rs) / all_hist,
        share_exceeding_4096_tokens=sum(total_tokens[c] > 4096 for c in cids) / len(cids),
        share_exceeding_8192_tokens=sum(total_tokens[c] > 8192 for c in cids) / len(cids),
        share_history_unneeded=share(rs),
        share_history_unneeded_ci95=boot_ci(rs),
        share_history_unneeded_after_turn_10=share(late) if late else None,
    )

json.dump(out, open(os.path.join(res_dir, "coding_subset.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
