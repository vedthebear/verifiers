"""Run a batch of rollouts against the aime_process environment.

Usage:
    python environments/aime_process/scripts/smoke.py
    python environments/aime_process/scripts/smoke.py --num-examples 10 --rollouts 4
    python environments/aime_process/scripts/smoke.py --model deepseek/deepseek-r1

Requires OPENROUTER_API_KEY in the environment.
"""

import argparse
import asyncio
import os

from openai import AsyncOpenAI

import aime_process
import verifiers as vf


async def main(args: argparse.Namespace) -> None:
    env = aime_process.load_environment(num_eval_examples=args.num_examples)
    raw_client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    client = vf.OpenAIChatCompletionsClient(raw_client)
    total = args.num_examples * args.rollouts
    print(
        f"Running {total} rollouts ({args.num_examples} examples × "
        f"{args.rollouts} rollouts) with {args.model}, "
        f"max_concurrent={args.max_concurrent} ..."
    )
    results = await env.evaluate(
        client=client,
        model=args.model,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts,
        max_concurrent=args.max_concurrent,
        save_results=True,
        state_columns=["step_audit"],
    )
    meta = results["metadata"]
    print("=" * 60)
    print(f"Avg reward:      {meta['avg_reward']:.3f}")
    for k, v in meta["avg_metrics"].items():
        print(f"  {k:18s} {v:.3f}")
    print(f"Wall time:       {meta['time']:.1f}s")
    print(f"Tokens (model):  in={meta['usage']['input_tokens']:.0f}  out={meta['usage']['output_tokens']:.0f}")
    print(f"Saved to:        {meta['path_to_save']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--max-concurrent", type=int, default=10)
    asyncio.run(main(parser.parse_args()))
