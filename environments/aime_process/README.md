# aime-process

AIME process-reward evaluation environment. Grades each completion on three
dimensions, not just one:

| Signal | Source | Default weight |
|---|---|---|
| `correct_answer` | `math_verify` integer match of `\boxed{...}` | 1.0 |
| `step_validity` | LLM-as-judge over each CoT step | 0.5 |
| `completion_chars` | `len(completion_text)` | 0.0 (metric only) |
| `step_count` | Steps after `\n\n` segmentation | 0.0 (metric only) |

The point isn't to produce one number — AIME pass@1 already does that. The
point is the decomposition. With per-step grading you can answer questions
single-number benchmarks can't:

- Right answer, wrong reasoning (lucky guess / memorization)?
- Wrong answer, mostly right reasoning (one arithmetic slip)?
- Where in the chain does each model first go wrong?
- How many tokens does the model spend per correct answer?

## Quickstart

```bash
export OPENROUTER_API_KEY=sk-or-...

# Smoke test via prime CLI (uses defaults in pyproject.toml)
prime eval run aime-process --model openai/gpt-4.1-mini

# Or load programmatically
python - <<'PY'
import asyncio
from openai import AsyncOpenAI
import aime_process

env = aime_process.load_environment(num_eval_examples=2)
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=__import__("os").environ["OPENROUTER_API_KEY"],
)
out = asyncio.run(env.evaluate(
    client=client, model="openai/gpt-4.1-mini",
    num_examples=2, rollouts_per_example=1,
))
print(out)
PY
```

## Knobs (load_environment kwargs)

- `judge_model` — OpenRouter slug; default `anthropic/claude-sonnet-4.6`.
- `judge_base_url` — defaults to OpenRouter; any OpenAI-compatible endpoint works.
- `answer_weight`, `step_weight` — reward composition.
- `num_train_examples`, `num_eval_examples` — slice the AIME splits.

## Design notes

- **v0 verifiers idiom**: `SingleTurnEnv` + `Parser` + `Rubric`. The rubric
  subclasses `vf.MathRubric` so `math_verify` correctness comes for free.
- **Step segmentation**: `\n\n` heuristic on the post-`</think>` body, drop
  the final `\boxed{}` line. No LLM segmenter — too noisy and expensive for
  the value it adds.
- **Judge client**: any `AsyncOpenAI`-compatible client. We default to
  OpenRouter so a single API key covers all leaderboard models.
- **State caching**: per-rollout `step_validity_score` is cached on `state`,
  so the rubric is safe to call repeatedly without re-judging.

## Limitations (TODO: fill in after benchmark + audit run)

This section is the recruiter-facing differentiator and is intentionally
incomplete in the scaffold. After the first benchmark run, fill in with:

- Inter-judge agreement (Cohen's κ between `anthropic/claude-sonnet-4.6` and
  a second judge, e.g. `openai/gpt-4.1-mini`).
- Adversarial step-injection recall (synthetic corruptions of known-good
  chains; does the judge catch them?).
- Filler-resistance (vacuous "let me re-examine" steps; do they game the
  validity rate upward?).
- Domains where the judge is unreliable (geometry? combinatorics?) — split κ
  by tag.

## Results

TODO after the first multi-model run.
