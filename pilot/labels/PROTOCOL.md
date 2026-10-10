# Labelling protocol

Unit: every user turn t >= 2 of a conversation. Question: which earlier exchanges (user message + assistant reply) does the model need in its context to answer turn t as the user intends?

One line per conversation: `C001: - S P 1,3 S`. The first entry is always `-` (turn 1 has no history).

| Code | Meaning | Earlier exchanges counted as needed |
| --- | --- | --- |
| `P` | Pivot or self-contained: a new topic, a request that carries all its own information (e.g. a full prompt template), or an identical resubmission | none |
| `S` | Continues the current thread | all exchanges from the most recent `P` turn (or turn 1) up to t-1 |
| `A` | Needs the whole conversation | all exchanges 1..t-1 |
| `1,3-4` | Needs specific earlier exchanges (e.g. returns to an earlier topic, or a shared instruction given earlier) | exactly those |

Conservative rules (the estimate is a lower bound on what could be removed):
- When in doubt whether earlier content is needed, include it.
- Short follow-ups ("continue", "more", "shorter", "why?", "another one", "rewrite it") and acknowledgements ("thanks") are `S`.
- Indirect dependencies are included: if turn t refines a draft that was built over several turns, all of those turns count.
- A series of self-contained questions is `P` each, unless an earlier turn set an instruction that still applies (e.g. "answer in one sentence"), in which case that turn is listed.

Labeller: Claude (claude-opus-5-5), reading compact views (user messages clipped to ~540 characters, assistant replies to 170 characters). A human check of a random subset is recommended before publication.
