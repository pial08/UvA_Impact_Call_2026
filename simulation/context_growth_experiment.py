#!/usr/bin/env python3
"""
context_growth_experiment.py

Tests the "context used vs. time" figure for several LLMs and plots it:

  (a) Unmanaged growth  - a fan of lines that stop when they hit the context limit
  (b) Management strategies for one model:
        none        - grow until overflow
        sliding     - drop the oldest turns (FIFO), plateau at the limit
        compaction  - summarize history at a threshold, sawtooth below the limit
  (c) KV-cache memory (GB) vs. time per model
  (d) Context as % of each model's own limit, with compaction

Two modes
---------
analytic  No downloads, no GPU. Simulated conversations (log-normal message
          lengths) + published model configs. KV bytes are computed as
              2 * layers * kv_heads * head_dim * tokens * bytes_per_value

measure   Runs real Hugging Face chat models turn by turn. Tokens are counted
          with each model's own chat template; KV-cache bytes are measured
          from the actual cache returned by generate(); peak GPU memory is
          recorded when CUDA is available.

Examples
--------
  pip install numpy matplotlib                      # analytic mode
  python context_growth_experiment.py --mode analytic

  pip install torch transformers accelerate         # measure mode
  python context_growth_experiment.py --mode measure \
      --models Qwen/Qwen2.5-0.5B-Instruct HuggingFaceTB/SmolLM2-360M-Instruct \
      --limit 2048 --turns 30 --max-new-tokens 256

  # re-plot an existing run
  python context_growth_experiment.py --plot-only results/records.csv

Outputs (in --out): records.csv, context_growth.png, context_growth.pdf
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ----------------------------------------------------------------------------
# Model configurations for analytic mode (from the published model configs)
# ----------------------------------------------------------------------------
MODEL_CONFIGS = {
    # name:            layers, KV heads, head dim, max context
    "Llama-2-7B":      dict(layers=32, kv_heads=32, head_dim=128, max_ctx=4096),
    "Llama-3-8B":      dict(layers=32, kv_heads=8,  head_dim=128, max_ctx=8192),
    "Mistral-7B-v0.3": dict(layers=32, kv_heads=8,  head_dim=128, max_ctx=32768),
    "Qwen2.5-7B":      dict(layers=28, kv_heads=4,  head_dim=128, max_ctx=32768),
}

# Conversation styles: (median user tokens, median assistant tokens, P(long paste))
STYLES = {
    "short Q&A":      (40, 150, 0.00),
    "normal":         (80, 400, 0.00),
    "verbose":        (150, 900, 0.00),
    "document-heavy": (100, 500, 0.15),
}

STRATEGY_LABELS = {
    "none": "No management",
    "sliding": "Sliding window (FIFO)",
    "compaction": "Compaction (summarize)",
}

SYSTEM_PROMPT = "You are a helpful research assistant. Answer clearly and in detail."

USER_PROMPTS = [
    "Why do longer inputs make LLM inference use more memory?",
    "Explain how the KV cache works during prefill and decode.",
    "Compute the KV cache size per token for a 32-layer model with 32 KV heads of dimension 128 in fp16.",
    "How does grouped-query attention reduce KV cache memory?",
    "What is the difference between compute-bound and memory-bound inference?",
    "Why is decoding usually memory-bandwidth bound?",
    "How does FlashAttention reduce attention memory?",
    "What are the downsides of quantizing the KV cache?",
    "Compare sliding-window truncation with summarization for long chats.",
    "How would you measure energy per generated token on a GPU?",
    "What is PagedAttention and why does it help serving throughput?",
    "Summarize the main trade-offs we have discussed so far.",
]

PASTE_TEXT = (
    "Large language models store a key and a value vector for every token in every "
    "layer. During decoding, each new token attends over all cached keys and values, "
    "so the cache grows linearly with the context length and must be read in full "
    "at every step. "
) * 40

TOKENS_PER_S = 30.0  # simulated generation speed for the analytic time axis


def kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes=2):
    return 2 * layers * kv_heads * head_dim * dtype_bytes


# ----------------------------------------------------------------------------
# Analytic mode
# ----------------------------------------------------------------------------
def make_conversation(style, n_turns, rng):
    """Return a list of (user_tokens, assistant_tokens, seconds_elapsed)."""
    u_med, a_med, p_paste = STYLES[style]
    turns = []
    for _ in range(n_turns):
        u = int(rng.lognormal(math.log(u_med), 0.5))
        if rng.random() < p_paste:
            u += int(rng.uniform(1500, 5000))  # user pastes a document
        a = int(rng.lognormal(math.log(a_med), 0.4))
        think = rng.uniform(15, 90)  # seconds the user spends reading and typing
        turns.append((u, a, think + a / TOKENS_PER_S))
    return turns


def simulate(conv, limit, strategy, sys_tokens=200, threshold=0.75,
             summary_ratio=0.10, summary_min=300, keep_recent=2):
    """Simulate context size over a conversation under one strategy."""
    hist, t = [], 0.0
    rows = [dict(turn=0, time_s=0.0, context_tokens=sys_tokens, event="start")]
    for i, (u, a, dt) in enumerate(conv, 1):
        cur = sys_tokens + sum(hist)

        if strategy == "compaction" and hist and cur + u + a > threshold * limit:
            summary = max(summary_min, int(summary_ratio * sum(hist)))
            hist = [summary] + (hist[-keep_recent:] if keep_recent else [])
            if sys_tokens + sum(hist) + u + a > threshold * limit:
                hist = [summary]
            # vertical drop just before this turn
            rows.append(dict(turn=i - 1, time_s=t, context_tokens=sys_tokens + sum(hist),
                             event="compact"))

        t += dt
        hist += [u, a]
        ctx = sys_tokens + sum(hist)
        event = ""

        if strategy == "none" and ctx > limit:
            rows.append(dict(turn=i, time_s=t, context_tokens=limit, event="overflow"))
            break
        if strategy == "sliding" and ctx > limit:
            while ctx > limit and len(hist) > 1:
                hist.pop(0)
                ctx = sys_tokens + sum(hist)
            event = "truncate"
        if ctx > limit:  # a single message larger than the window: truncate it
            hist[-1] -= ctx - limit
            ctx = limit
            event = "truncate"

        rows.append(dict(turn=i, time_s=t, context_tokens=ctx, event=event))
    return rows


def run_analytic(args):
    rng_master = np.random.default_rng(args.seed)
    convs = {s: make_conversation(s, args.turns, np.random.default_rng(rng_master.integers(1 << 31)))
             for s in args.styles}
    records = []
    for name in args.models:
        cfg = MODEL_CONFIGS[name]
        per_tok = kv_bytes_per_token(cfg["layers"], cfg["kv_heads"], cfg["head_dim"])
        limit = cfg["max_ctx"]
        for style, conv in convs.items():
            for strategy in args.strategies:
                for r in simulate(conv, limit, strategy, threshold=args.compact_threshold):
                    kv = r["context_tokens"] * per_tok
                    r.update(model=name, conversation=style, strategy=strategy, limit=limit,
                             kv_bytes=kv, kv_bytes_formula=kv, peak_mem_bytes="")
                    records.append(r)
    return records


# ----------------------------------------------------------------------------
# Measure mode (real models via Hugging Face transformers)
# ----------------------------------------------------------------------------
def cache_nbytes(cache):
    """Bytes held by a transformers KV cache (works across library versions)."""
    total = 0

    def add(t):
        nonlocal total
        if t is not None and hasattr(t, "numel"):
            total += t.numel() * t.element_size()

    if cache is None:
        return 0
    if hasattr(cache, "layers"):  # transformers >= 4.56
        for layer in cache.layers:
            add(getattr(layer, "keys", None))
            add(getattr(layer, "values", None))
    elif hasattr(cache, "key_cache"):  # DynamicCache, older versions
        for k, v in zip(cache.key_cache, cache.value_cache):
            add(k)
            add(v)
    else:  # legacy tuple of (k, v) per layer
        for layer in cache:
            for t in layer[:2]:
                add(t)
    return total


def encode(tok, messages):
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids


def generate(model, tok, ids, max_new_tokens):
    import torch
    with torch.no_grad():
        return model.generate(
            ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
            do_sample=False, use_cache=True, return_dict_in_generate=True,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
        )


def summarize(model, tok, device, hist, max_tokens):
    convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in hist)
    msgs = [
        {"role": "system", "content": "You compress conversations faithfully."},
        {"role": "user", "content": "Summarize the conversation below. Keep decisions, "
                                    "constraints, numbers and open questions. Be concise.\n\n" + convo},
    ]
    ids = encode(tok, msgs).to(device)
    out = generate(model, tok, ids, max_tokens)
    return tok.decode(out.sequences[0, ids.shape[-1]:], skip_special_tokens=True)


def run_chat(model, tok, device, limit, per_tok, strategy, args):
    import torch
    cuda = device == "cuda"
    budget = limit - args.max_new_tokens  # leave room for the reply
    system = {"role": "system", "content": SYSTEM_PROMPT}
    hist, rows, t0 = [], [], time.time()
    n_sys = encode(tok, [system]).shape[-1]
    rows.append(dict(turn=0, time_s=0.0, context_tokens=n_sys, kv_bytes=0,
                     kv_bytes_formula=0, peak_mem_bytes="", event="start"))

    for turn in range(1, args.turns + 1):
        user_text = USER_PROMPTS[(turn - 1) % len(USER_PROMPTS)]
        if args.paste_every and turn % args.paste_every == 0:
            user_text = "Here is a document:\n" + PASTE_TEXT + "\n\n" + user_text
        user = {"role": "user", "content": user_text}
        msgs = [system] + hist + [user]
        n = encode(tok, msgs).shape[-1]
        event = ""

        if strategy == "compaction" and hist and n > args.compact_threshold * limit:
            summary = summarize(model, tok, device, hist, args.summary_tokens)
            system = {"role": "system",
                      "content": SYSTEM_PROMPT + "\n\nSummary of the earlier conversation:\n" + summary}
            hist = []
            n_after = encode(tok, [system]).shape[-1]
            rows.append(dict(turn=turn - 1, time_s=time.time() - t0, context_tokens=n_after,
                             kv_bytes=n_after * per_tok, kv_bytes_formula=n_after * per_tok,
                             peak_mem_bytes="", event="compact"))
            msgs = [system, user]
            n = encode(tok, msgs).shape[-1]

        if n > budget and strategy == "sliding":
            while n > budget and hist:
                hist = hist[2:]  # drop the oldest user/assistant pair
                msgs = [system] + hist + [user]
                n = encode(tok, msgs).shape[-1]
            event = "truncate"

        if n > budget:  # "none" strategy, or a single message too large to fit
            rows.append(dict(turn=turn, time_s=time.time() - t0, context_tokens=limit,
                             kv_bytes=limit * per_tok, kv_bytes_formula=limit * per_tok,
                             peak_mem_bytes="", event="overflow"))
            break

        ids = encode(tok, msgs).to(device)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        out = generate(model, tok, ids, args.max_new_tokens)
        new = out.sequences[0, ids.shape[-1]:]
        reply = tok.decode(new, skip_special_tokens=True)
        ctx = ids.shape[-1] + new.numel()
        hist += [user, {"role": "assistant", "content": reply}]
        rows.append(dict(
            turn=turn, time_s=time.time() - t0, context_tokens=ctx,
            kv_bytes=cache_nbytes(out.past_key_values),
            kv_bytes_formula=ctx * per_tok,
            peak_mem_bytes=torch.cuda.max_memory_allocated() if cuda else "",
            event=event,
        ))
        print(f"    turn {turn:3d}  context={ctx:6d} tokens  {event}")
    return rows


def run_measure(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.cuda.is_available():
        device, dtype = "cuda", torch.float16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float16
    else:
        device, dtype = "cpu", torch.float32
    dtype_bytes = 2 if dtype == torch.float16 else 4

    records = []
    for model_id in args.models:
        print(f"Loading {model_id} on {device} ...")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
        cfg = model.config
        n_heads = cfg.num_attention_heads
        kv_heads = getattr(cfg, "num_key_value_heads", None) or n_heads
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
        per_tok = kv_bytes_per_token(cfg.num_hidden_layers, kv_heads, head_dim, dtype_bytes)
        max_pos = getattr(cfg, "max_position_embeddings", None) or 4096
        limit = min(args.limit, max_pos) if args.limit else max_pos
        name = model_id.split("/")[-1]
        print(f"  limit={limit} tokens, KV per token={per_tok / 1024:.1f} KB")

        for strategy in args.strategies:
            print(f"  strategy: {strategy}")
            for r in run_chat(model, tok, device, limit, per_tok, strategy, args):
                r.update(model=name, conversation="scripted chat", strategy=strategy, limit=limit)
                records.append(r)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    return records


# ----------------------------------------------------------------------------
# I/O and plotting
# ----------------------------------------------------------------------------
FIELDS = ["model", "conversation", "strategy", "turn", "time_s", "context_tokens",
          "limit", "kv_bytes", "kv_bytes_formula", "peak_mem_bytes", "event"]


def save_csv(records, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def load_csv(path):
    records = []
    with open(path) as f:
        for r in csv.DictReader(f):
            for k in ("turn", "time_s", "context_tokens", "limit", "kv_bytes", "kv_bytes_formula"):
                r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
            records.append(r)
    return records


def select(records, **filt):
    return [r for r in records if all(r[k] == v for k, v in filt.items())]


def unique(records, key):
    seen = []
    for r in records:
        if r[key] not in seen:
            seen.append(r[key])
    return seen


def xs(rows, x):
    return [r["time_s"] / 60 if x == "time" else r["turn"] for r in rows]


def draw_line(ax, rows, x, y_fn, label, **kw):
    if not rows:
        return
    X, Y = xs(rows, x), [y_fn(r) for r in rows]
    line, = ax.plot(X, Y, label=label, lw=1.8, **kw)
    if rows[-1]["event"] == "overflow":
        ax.plot(X[-1], Y[-1], "x", color=line.get_color(), ms=9, mew=2)


def limit_line(ax, y, text):
    ax.axhline(y, ls="--", color="black", lw=1)
    ax.annotate(text, xy=(1, y), xytext=(-4, 4), textcoords="offset points",
                xycoords=("axes fraction", "data"), fontsize=9, ha="right")


def plot(records, args, out_dir):
    x = args.x
    xlabel = "Time (minutes)" if x == "time" else "Turn"
    models = unique(records, "model")
    convs = unique(records, "conversation")
    strategies = unique(records, "strategy")
    focus_model = args.focus_model if args.focus_model in models else models[min(1, len(models) - 1)]
    focus_conv = args.focus_conversation if args.focus_conversation in convs else convs[0]
    limit_of = {m: select(records, model=m)[0]["limit"] for m in models}

    # distinct colour + line style per series, so overlapping lines stay visible
    from cycler import cycler
    plt.rcParams["axes.prop_cycle"] = (
        cycler(color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
        + cycler(linestyle=["-", "--", "-.", ":", "-", "--"])
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    (a, b), (c, d) = axes

    # (a) unmanaged fan: one line per conversation (or per model if only one conversation)
    if "none" in strategies:
        if len(convs) > 1:
            for conv in convs:
                draw_line(a, select(records, model=focus_model, conversation=conv, strategy="none"),
                          x, lambda r: r["context_tokens"], conv)
            a.set_title(f"(a) Unmanaged growth, {focus_model}: different conversations")
        else:
            for m in models:
                draw_line(a, select(records, model=m, conversation=focus_conv, strategy="none"),
                          x, lambda r: r["context_tokens"], m)
            a.set_title("(a) Unmanaged growth: different models")
        limit_line(a, limit_of[focus_model], f"Limit ({int(limit_of[focus_model])} tokens)")
    a.set_ylabel("Context used (tokens)")

    # (b) strategies for one model and conversation
    for s in strategies:
        draw_line(b, select(records, model=focus_model, conversation=focus_conv, strategy=s),
                  x, lambda r: r["context_tokens"], STRATEGY_LABELS.get(s, s))
    limit_line(b, limit_of[focus_model], "Limit")
    if "compaction" in strategies:
        b.axhline(args.compact_threshold * limit_of[focus_model], ls=":", color="gray", lw=1)
    b.set_title(f"(b) Context management, {focus_model} ({focus_conv})")
    b.set_ylabel("Context used (tokens)")

    # (c) KV-cache memory per model, unmanaged
    s_c = "none" if "none" in strategies else strategies[0]
    for m in models:
        draw_line(c, select(records, model=m, conversation=focus_conv, strategy=s_c), x,
                  lambda r: (r["kv_bytes"] if r["kv_bytes"] == r["kv_bytes"] else r["kv_bytes_formula"]) / 1e9,
                  m)
    c.set_title(f"(c) KV-cache memory per model ({STRATEGY_LABELS.get(s_c, s_c).lower()})")
    c.set_ylabel("KV cache (GB, fp16)")

    # (d) % of each model's own limit, with compaction
    s_d = "compaction" if "compaction" in strategies else strategies[0]
    for m in models:
        draw_line(d, select(records, model=m, conversation=focus_conv, strategy=s_d), x,
                  lambda r, m=m: 100 * r["context_tokens"] / limit_of[m], m)
    limit_line(d, 100, "Limit (100%)")
    d.set_ylim(0, 110)
    d.set_title(f"(d) Context as % of each model's limit ({STRATEGY_LABELS.get(s_d, s_d).lower()})")
    d.set_ylabel("Context used (% of limit)")

    for ax in axes.flat:
        ax.set_xlabel(xlabel)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, loc="best", framealpha=0.9)

    mode = "simulated" if args.mode == "analytic" else "measured"
    fig.suptitle(f"Context growth over time ({mode}); x marks a context overflow", fontsize=12)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"context_growth.{ext}"), dpi=200)
    print(f"Saved figures to {out_dir}/context_growth.png and .pdf")


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["analytic", "measure"], default="analytic")
    p.add_argument("--models", nargs="+", default=None,
                   help="analytic: names from MODEL_CONFIGS; measure: Hugging Face model ids")
    p.add_argument("--styles", nargs="+", default=list(STYLES), help="analytic conversation styles")
    p.add_argument("--strategies", nargs="+", default=["none", "sliding", "compaction"],
                   choices=["none", "sliding", "compaction"])
    p.add_argument("--turns", type=int, default=40)
    p.add_argument("--limit", type=int, default=2048,
                   help="measure mode: context limit to enforce (capped at the model's maximum)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--summary-tokens", type=int, default=200)
    p.add_argument("--compact-threshold", type=float, default=0.75)
    p.add_argument("--paste-every", type=int, default=0, help="measure mode: paste a document every N turns")
    p.add_argument("--focus-model", default="Llama-3-8B")
    p.add_argument("--focus-conversation", default="normal")
    p.add_argument("--x", choices=["time", "turn"], default="time")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results")
    p.add_argument("--plot-only", default=None, help="path to an existing records.csv")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.plot_only:
        records = load_csv(args.plot_only)
    else:
        if args.mode == "analytic":
            args.models = args.models or list(MODEL_CONFIGS)
            records = run_analytic(args)
        else:
            args.models = args.models or ["Qwen/Qwen2.5-0.5B-Instruct",
                                          "HuggingFaceTB/SmolLM2-360M-Instruct",
                                          "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
            records = run_measure(args)
        save_csv(records, os.path.join(args.out, "records.csv"))
        records = load_csv(os.path.join(args.out, "records.csv"))
    plot(records, args, args.out)


if __name__ == "__main__":
    main()
