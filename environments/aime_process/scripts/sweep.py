"""Run aime_process eval across multiple OpenRouter models.

Saves per-model results via verifiers' default save path
(outputs/evals/<model-slug>/<hash>/) and a cross-model leaderboard at
outputs/sweeps/<utc-timestamp>/leaderboard.json.

Usage:
    python environments/aime_process/scripts/sweep.py
    python environments/aime_process/scripts/sweep.py --num-examples 10 --rollouts 1
    python environments/aime_process/scripts/sweep.py --models openai/gpt-4.1 deepseek/deepseek-r1

Requires OPENROUTER_API_KEY in the environment.
"""

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

import aime_process
import verifiers as vf


DEFAULT_MODELS = [
    "openai/gpt-3.5-turbo",
    "meta-llama/llama-3.1-8b-instruct",
    "qwen/qwen-2.5-72b-instruct",
    "openai/gpt-4.1",
    "deepseek/deepseek-r1",
]


def _row(model: str, meta: dict | None, error: str | None = None) -> dict:
    if error or meta is None:
        return {"model": model, "error": error}
    m = meta["avg_metrics"]
    return {
        "model": model,
        "correct_answer": m["correct_answer"],
        "step_validity": m["step_validity"],
        "step_count": m["step_count"],
        "completion_chars": m["completion_chars"],
        "avg_reward": meta["avg_reward"],
        "time_s": meta["time"],
        "save_path": str(meta["path_to_save"]),
    }


def _print_leaderboard(rows: list[dict]) -> None:
    bar = "=" * 92
    print("\n" + bar)
    print(
        f"{'MODEL':40s}  {'CORRECT':>8s}  {'VALIDITY':>9s}  "
        f"{'STEPS':>6s}  {'CHARS':>7s}  {'TIME':>6s}"
    )
    print("-" * 92)
    for r in rows:
        if "error" in r and r["error"]:
            print(f"{r['model']:40s}  ERROR: {r['error'][:40]}")
            continue
        print(
            f"{r['model']:40s}  {r['correct_answer']:8.3f}  "
            f"{r['step_validity']:9.3f}  {r['step_count']:6.1f}  "
            f"{r['completion_chars']:7.0f}  {r['time_s']:5.0f}s"
        )
    print(bar)


async def main(args: argparse.Namespace) -> None:
    env = aime_process.load_environment(
        num_eval_examples=args.num_examples,
        judge_model=args.judge_model,
    )
    raw_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    client = vf.OpenAIChatCompletionsClient(raw_client)

    print(
        f"Sweep: {len(args.models)} models × {args.num_examples} problems × "
        f"{args.rollouts} rollouts/example  (judge={args.judge_model})"
    )

    sampling_args = (
        {"max_tokens": args.max_tokens} if args.max_tokens else None
    )

    rows: list[dict] = []
    sweep_t0 = time.time()
    for i, model in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] {model}")
        try:
            results = await env.evaluate(
                client=client,
                model=model,
                sampling_args=sampling_args,
                num_examples=args.num_examples,
                rollouts_per_example=args.rollouts,
                max_concurrent=args.max_concurrent,
                save_results=True,
                state_columns=["step_audit"],
            )
            meta = results["metadata"]
            m = meta["avg_metrics"]
            print(
                f"  correct={m['correct_answer']:.3f}  "
                f"validity={m['step_validity']:.3f}  "
                f"steps={m['step_count']:.1f}  ({meta['time']:.0f}s)"
            )
            rows.append(_row(model, meta))
        except Exception as exc:  # noqa: BLE001 — per-model isolation
            print(f"  FAILED: {exc!r}")
            rows.append(_row(model, None, error=repr(exc)))

    _print_leaderboard(rows)

    sweep_dir = Path("outputs/sweeps") / datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    sweep_dir.mkdir(parents=True, exist_ok=True)
    out_path = sweep_dir / "leaderboard.json"
    out_path.write_text(
        json.dumps(
            {
                "judge_model": args.judge_model,
                "num_examples": args.num_examples,
                "rollouts_per_example": args.rollouts,
                "max_tokens": args.max_tokens,
                "sweep_time_s": time.time() - sweep_t0,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nLeaderboard → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--num-examples", type=int, default=30)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--max-concurrent", type=int, default=10)
    parser.add_argument("--judge-model", default="anthropic/claude-haiku-4.5")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override the model-under-test's max output tokens (default: provider default, often 4096).",
    )
    asyncio.run(main(parser.parse_args()))
