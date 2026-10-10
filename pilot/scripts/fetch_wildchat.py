"""Download the WildChat-1M conversations used in the pilot (or draw a new sample).

WildChat-1M (Zhao et al., ICLR 2024) is published on Hugging Face under ODC-BY.
Rows are fetched from the public datasets-server API, so no account is needed.

Usage
-----
Reproduce the pilot exactly (204 conversations listed in data/conversation_ids.csv):

    python fetch_wildchat.py --ids ../data/conversation_ids.csv --out ../data

Draw a new random sample of blocks of consecutive rows and apply the same filter:

    python fetch_wildchat.py --sample --blocks 40 --block-size 100 --seed 42 --out ../data

Both modes write data/eligible.jsonl, one conversation per line:
    {"id": conversation_hash, "row_idx": ..., "model": ..., "turn": ..., "messages": [{"role", "content"}, ...]}

Filter (same as the pilot): English, not flagged toxic, not redacted, at least 4 user turns.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
import urllib.parse
import urllib.request

API = "https://datasets-server.huggingface.co/rows"
DATASET = "allenai/WildChat-1M"
N_ROWS = 837_989  # rows in the train split of WildChat-1M


def fetch_rows(offset, length, retries=6):
    query = urllib.parse.urlencode(dict(dataset=DATASET, config="default", split="train",
                                        offset=offset, length=length))
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{API}?{query}", timeout=60) as r:
                payload = json.load(r)
            rows = payload.get("rows", [])
            if any(row.get("truncated_cells") for row in rows):
                print(f"  warning: some cells were truncated by the API at offset {offset}", file=sys.stderr)
            return rows
        except Exception as e:  # rate limits (429) and transient errors
            wait = 5 * 2 ** attempt
            print(f"  offset {offset}: {e}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    print(f"  offset {offset}: giving up", file=sys.stderr)
    return []


def to_record(row_idx, row):
    return dict(id=row["conversation_hash"], row_idx=row_idx, model=row["model"], turn=row["turn"],
                messages=[dict(role=m["role"], content=m["content"]) for m in row["conversation"]])


def eligible(row):
    return (row.get("language") == "English" and not row.get("toxic") and not row.get("redacted")
            and row.get("turn", 0) >= 4)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ids", help="CSV with row_idx and conversation_hash columns (exact reproduction)")
    mode.add_argument("--sample", action="store_true", help="draw a new random sample of row blocks")
    p.add_argument("--blocks", type=int, default=40)
    p.add_argument("--block-size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sleep", type=float, default=1.0, help="seconds between requests")
    p.add_argument("--out", default="../data")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    records = []

    if args.ids:
        wanted = sorted(csv.DictReader(open(args.ids)), key=lambda r: int(r["order"]))
        for i, w in enumerate(wanted, 1):
            idx = int(w["row_idx"])
            rows = fetch_rows(idx, 1)
            if not rows:
                continue
            row = rows[0]["row"]
            if row["conversation_hash"] != w["conversation_hash"]:
                print(f"  row {idx}: hash mismatch, skipped", file=sys.stderr)
                continue
            records.append(to_record(idx, row))
            if i % 20 == 0:
                print(f"  {i}/{len(wanted)} fetched")
            time.sleep(args.sleep)
    else:
        rng = random.Random(args.seed)
        offsets = sorted(rng.sample(range(0, N_ROWS - args.block_size), args.blocks))
        for b, off in enumerate(offsets, 1):
            rows = fetch_rows(off, args.block_size)
            kept = [to_record(r["row_idx"], r["row"]) for r in rows if eligible(r["row"])]
            records += kept
            print(f"  block {b}/{len(offsets)} at {off}: {len(rows)} rows, {len(kept)} eligible")
            time.sleep(args.sleep)

    path = os.path.join(args.out, "eligible.jsonl")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(records)} conversations to {path}")


if __name__ == "__main__":
    main()
