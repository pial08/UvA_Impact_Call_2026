# How much chat history is actually needed? Pilot results

**Headline.** In 204 real multi-turn conversations from WildChat (1,270 user turns with history), **45% of the history tokens sent with each turn were not needed to answer it (95% CI 37–53%)**. Counted against all input tokens (history plus the current message, excluding any system prompt), it is **42% (34–49%)**. The share grows as a conversation gets longer: **62% after the tenth turn**.

## Data

- Source: WildChat-1M (Zhao et al., ICLR 2024), real ChatGPT conversations from 2023 (GPT-3.5 and GPT-4).
- Sampling: 40 blocks of 100 consecutive rows at random positions (seed 42); 26 blocks downloaded (14 were rate-limited), giving 2,600 conversations.
- Filter: English, not flagged toxic, not redacted, at least 4 user turns → **204 conversations, 1,474 user turns** (median 6 turns, max 34).

## Method

1. For every user turn after the first, the labeller recorded which earlier exchanges (user message + reply) the model needs in its context to answer as the user intends. Protocol: `labels/PROTOCOL.md`; labels: `labels/batch_*.txt`.
2. Labels are conservative: when in doubt, history counts as needed; follow-ups ("continue", "shorter", "thanks") keep their whole thread; indirect dependencies are included. The result is therefore a lower bound on what could be removed.
3. Tokens were counted with an offline BPE tokenizer (Claude legacy tokenizer). Ratios are insensitive to the tokenizer.
4. Share not needed = Σ(history tokens − needed tokens) / Σ history tokens, over all turns. Confidence intervals: cluster bootstrap over conversations (10,000 resamples).

## Results

| Measure | Value (95% CI) |
| --- | --- |
| History tokens not needed | 45% (37–53%) |
| All input tokens not needed | 42% (34–49%) |
| Turns that need no earlier turn at all | 33% (27–40%) |
| Turns with at least some history not needed | 56% |

By position of the turn in the conversation:

| User turn | Turns | History not needed |
| --- | --- | --- |
| 2–3 | 408 | 33% (26–41%) |
| 4–6 | 471 | 36% (29–43%) |
| 7–10 | 250 | 47% (34–59%) |
| 11+ | 141 | 62% (47–81%) |

By conversation length: 30% for 4–5 turns, 34% for 6–9 turns, 55% for 10+ turns.

Across conversations the picture is two-sided: **39% of conversations need all their history** (one coherent task), while **47% have half or more of their history not needed** (median 45%).

## Coding conversations

60 of the 204 conversations contain a fenced code block. They are longer (median 8 user turns versus 5), hold 47% of all history tokens and exceed 4,096 tokens twice as often (42% versus 19%). A smaller share of their history is not needed: **34% (95% CI 19–45%)**, against 55% in other conversations, rising to 50% after the tenth turn. Coding sessions are usually one continuous task, and the turn-level labels count outdated code inside a needed turn as needed, so for code the estimate is especially conservative.

## Robustness

| Variant | History not needed |
| --- | --- |
| Main analysis | 45% |
| Every topic switch still keeps the previous exchange | 39% (31–46%) |
| Identical resubmissions of a message excluded | 44% |
| The 10 conversations contributing most removed | 34% |

## Limitations

- **One LLM labeller.** Labels were made by Claude (claude-opus-5-5) from clipped views of each conversation. A human check of a random subset (about 30 conversations) is needed before publication. A spot check of 18 turns labelled "needs no history" found all 18 correct.
- **WildChat is not ChatGPT or Claude usage in general.** It is a free 2023 GPT proxy; it over-represents homework question series and repeated prompt templates, which raise the share. It also under-represents long professional sessions with uploaded documents.
- **No system prompt.** Real deployments add a fixed system prompt to every turn, which lowers the share of total input that can be removed.
- **Tokens, not energy.** With prefix caching, unneeded history is not recomputed, but it still occupies KV-cache memory and is read at every generated token. Energy savings must be measured, not inferred from these shares.
- **Answer quality not tested.** The labels say what *should* be sufficient; whether answers are equally good with only that context is the next experiment.

## Files

- `results.json`: all numbers above; `turn_level_results.csv`: one row per user turn
- `unneeded_context.png` / `.pdf`: figure
- `labels/`: protocol and labels; `scripts/`: `build_views.py`, `compute.py`, `plot.py`
- `coding_subset.json`: coding conversations compared with all others
- Reproduce: see the repository README (`pilot/scripts`)
