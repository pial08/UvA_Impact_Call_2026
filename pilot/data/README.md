# Pilot data

The raw conversations are not stored in this repository. They come from
[WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) (ODC-BY licence)
and are downloaded on demand.

`conversation_ids.csv` lists the 204 conversations used in the pilot:

| Column | Meaning |
| --- | --- |
| `order` | Position in `eligible.jsonl` (the analysis shuffles this list with seed 7 to assign IDs) |
| `row_idx` | Row in the WildChat-1M train split |
| `conversation_hash` | WildChat hash, used to verify the downloaded row (it is not a unique key) |
| `model` | ChatGPT model that answered |
| `turns` | Number of user turns |
| `cid` | Pilot ID (`C001`–`C204`) used in `../labels/` and `../results/` |

To rebuild the data folder:

```bash
cd pilot/scripts
python fetch_wildchat.py --ids ../data/conversation_ids.csv --out ../data   # eligible.jsonl
python get_tokenizer.py --out ../data                                     # claude_legacy_tokenizer.json
python build_views.py ../data ../views                                    # conversations_indexed.json + views/
```
