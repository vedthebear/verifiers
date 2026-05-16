# AIME Process-Reward Benchmark — Plan

## Context

The user is interviewing for an RL engineer position at AfterQuery and wants a portfolio
project that demonstrates fluency with the `verifiers` library — the lightweight
RL-environments framework AfterQuery-style work is built on.

**The artifact**: an in-tree environment, `environments/aime_process/`, added to a
fork of `PrimeIntellect-ai/verifiers`, that grades AIME math completions on **three
dimensions**:

1. **Final-answer correctness** — deterministic integer match via `math_verify`
   (reused from `vf.MathRubric`).
2. **Per-step reasoning validity** — LLM-as-judge over each step in the chain of
   thought.
3. **Efficiency** — completion size (chars) and step count, reported as
   `weight=0.0` metrics so they don't contaminate the reward signal by default.

It is **eval-first**: the deliverable is a benchmark, not a trained model. The
verifiers eval loop (`env.evaluate(...)` / `prime eval run`) is the entry point.
Training is left as future work and explicitly out of scope.

**Why this shape wins the interview**

- Mirrors verifiers' own house style exactly (in-tree env, `load_environment()`
  function, `Parser` + `Rubric` + `SingleTurnEnv` idiom) — signals the candidate
  read the library deeply.
- Process-reward grading is a genuine open research direction (Math-Shepherd,
  PRM800K). It produces *new* information current benchmarks don't expose: the
  answer/reasoning decomposition (lucky guesses vs. honest-but-buggy chains).
- Pressure-testing the grader (inter-judge agreement, adversarial injection,
  filler resistance) signals AfterQuery-relevant taste: anyone can build an
  LLM-judge benchmark, almost nobody audits their own.

## Decisions (committed)

| Decision | Choice | Rationale |
|---|---|---|
| Project shape | In-tree env in fork of `PrimeIntellect-ai/verifiers` at `environments/aime_process/` | Matches the convention every community env follows (gsm8k, math500, …). The diff IS the portfolio artifact. |
| Verifiers idiom | **v0 path**: `SingleTurnEnv` + `Parser` + `Rubric` (specifically subclassing `MathRubric`) | User-chosen. `AGENTS.md` confirms v0 stays supported indefinitely; v1 (`Taskset`/`Harness`) is the newer style for harness-shaped envs but not needed here. |
| Judge backend | **OpenRouter** via `openai.AsyncOpenAI(base_url=..., api_key=OPENROUTER_API_KEY)` | OpenAI-compatible — no adapter code. One key, many models (Anthropic/OpenAI/open-weight) selectable via the `judge_model` slug. |
| Default judge | `anthropic/claude-sonnet-4.6` (OpenRouter slug) | Strong math judge. Trivially swappable. |
| Dataset | `vf.load_example_dataset("aime2024")` (train), `vf.load_example_dataset("aime2025")` (eval) | Already supported in `verifiers/utils/data_utils.py`. Preprocessing to `{question, answer}` is already wired. AIME 2025 as eval = clean, no contamination. |
| Scope | Eval-only | "Quick + impressive" trumps a training narrative for this timeline. |

## Architecture

### Reference files (read for idiom alignment)

- `verifiers/envs/environment.py` — `Environment.evaluate()` signature.
- `verifiers/envs/singleturn_env.py` — `SingleTurnEnv` base.
- `verifiers/parsers/parser.py` — `Parser` base; `parse_answer()` contract.
- `verifiers/rubrics/rubric.py` — `add_reward_func`, `add_metric`, async fan-out
  in `score_group`.
- `verifiers/rubrics/math_rubric.py` — answer correctness via `math_verify` in
  a `ProcessPoolExecutor`. Subclass this.
- `verifiers/rubrics/judge_rubric.py` — for reference only; we do NOT subclass
  (its `judge()` is one call per completion; we need one call per step, which is
  cleaner via a dedicated helper class).
- `verifiers/utils/data_utils.py` — `load_example_dataset("aime2024"/"aime2025")`
  and `extract_boxed_answer`.
- `environments/gsm8k/gsm8k.py` — canonical `load_environment()` example.

### Files to create

```
environments/aime_process/
├── aime_process.py    # load_environment() — user-facing entry
├── parsers.py         # AIMECoTParser (answer extraction + step segmentation)
├── rubrics.py         # StepJudge (per-step async fan-out) + AIMEProcessRubric
├── prompts.py         # SYSTEM_PROMPT, STEP_JUDGE_PROMPT
├── pyproject.toml     # Package metadata; deps verifiers, openai
└── README.md          # Design choices, results placeholder, audit section
```

Audit scripts (`scripts/judge_audit.py`, `scripts/run_benchmark.py`,
`scripts/analyze.py`) are deferred to a follow-up commit — they're consumers
of the env, not part of the env package itself.

### Style adherence (from `AGENTS.md`)

- Env modules avoid global helper functions — utility logic lives on classes
  (`AIMECoTParser.completion_text`, `AIMEProcessRubric._problem_text`).
- Basic env fits in a few dozen lines (`aime_process.py` is ~50 LOC).
- `vf.ensure_keys(["OPENROUTER_API_KEY"])` in `load_environment` for early,
  explicit failure.
- Reuses `vf.extract_boxed_answer`, `vf.MathRubric.correct_answer`,
  `vf.load_example_dataset` — no plumbing duplication.

### Reward composition (the heart of the project)

`AIMEProcessRubric` subclasses `vf.MathRubric`. Parent registers
`correct_answer` (weight 1.0); we rebind that weight via `self.weights[0]` and
add three more entries:

1. **`correct_answer`** (parent; weight `answer_weight=1.0`) — integer match.
2. **`step_validity`** (weight `step_weight=0.5`) — fraction of CoT steps the
   judge labels "valid". Cached on `state` to avoid re-judging.
3. **`completion_chars`** (metric; weight 0.0) — efficiency proxy.
4. **`step_count`** (metric; weight 0.0) — efficiency proxy.

All four surface in `state["metrics"]` per rollout, so post-hoc analysis can
decompose every score (pass@1 vs. clean-chain rate vs. tokens-per-correct).

### Step segmentation

Simple-first: split the post-`</think>` body on `\n\n`, drop empties, drop the
final `\boxed{N}` line (graded by correctness, not by step judge). System
prompt explicitly asks the model to use blank lines between steps. If this
proves too coarse on inspection, escalate to an equation-line regex; do NOT use
an LLM segmenter.

## The "audit your own grader" deliverable

The README section that wins the interview (deferred, but planned):

1. **Inter-judge agreement** — re-run `step_validity` with a second OpenRouter
   model (e.g., `openai/gpt-4.1-mini`); report Cohen's κ. Flag domains where
   κ < 0.6.
2. **Adversarial step injection** — corrupt one step in 20 known-good chains
   (swap integer, drop sign). Report judge recall.
3. **Filler resistance** — pad reasoning with vacuous "let me re-examine"
   steps; report rate of vacuous-step false-positives.

## Verification (end-to-end test plan)

After scaffolding:

1. **Import sanity** (no network): `python -c "import aime_process; print(aime_process.load_environment.__doc__)"`.
2. **Env construction** (requires `OPENROUTER_API_KEY` set):
   `python -c "import aime_process; env = aime_process.load_environment(num_eval_examples=2); print(env)"`.
3. **Parser round-trip**: hand-craft a completion with 3 `\n\n`-separated steps
   + `\boxed{42}`; assert `parser.parse_answer()` returns `"42"` and
   `parser.segment_steps()` returns 3 strings.
4. **One-rollout smoke**: `env.evaluate_sync(client, model="openai/gpt-4.1-mini",
   num_examples=1, rollouts_per_example=1)` returns a `GenerateOutputs` with
   `correct_answer`, `step_validity`, `completion_chars`, `step_count` all
   present in `metrics`.
5. **Multi-model benchmark** (later): `prime eval run aime-process
   --num-examples 30 --rollouts 4 --model <slug>` for 5+ models; produces a
   results table in the README.
6. **Audit** (later): three scripts above; numbers in README "Limitations".

## Out of scope (explicitly)

- GRPO / RL training loop.
- Tool use / Python sandbox variant.
- Local vLLM judge infra (cloud OpenRouter only).
- LLM-driven step segmentation.
- New paper / arxiv writeup.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Judge cost overruns | Cap `rollouts_per_example=4`, `num_eval_examples=30`, cache verdicts on `state`, `max_tokens=8` for the judge call. Ceiling ~$30 for the full multi-model run. |
| Step segmentation too coarse | Inspect 5 completions early; escalate to equation-line regex only if `\n\n` splitting visibly merges distinct steps. |
| Judge variance dominates signal | The audit deliverable IS the mitigation — report κ openly. If κ < 0.5, paper over with `n_judges=3` voting. |
| Verifiers API drift | Pin `verifiers>=0.1.15,<0.2.0` in `pyproject.toml`. |
| OpenRouter rate limits | `judge_sampling_args` caps tokens; concurrency limited by `max_concurrent` in `env.evaluate()`. |

## Next steps (step-by-step walkthrough)

1. **prompts.py** — system prompt + step-judge prompt template.
2. **parsers.py** — `AIMECoTParser` (answer extraction + step segmentation).
3. **rubrics.py** — `StepJudge` helper + `AIMEProcessRubric` subclass.
4. **aime_process.py** — `load_environment()` entry point.
5. **pyproject.toml** + **README.md** — packaging + docs.
6. Import sanity check (no network).
7. Env construction check (needs `OPENROUTER_API_KEY`).
8. One-rollout smoke test against `openai/gpt-4.1-mini`.
9. Multi-model benchmark + audit scripts (separate follow-up).
