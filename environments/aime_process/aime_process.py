import os

from openai import AsyncOpenAI

import verifiers as vf
from verifiers.utils.data_utils import load_example_dataset

from aime_parsers import AIMECoTParser
from aime_prompts import SYSTEM_PROMPT
from aime_rubrics import AIMEProcessRubric

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def load_environment(
    system_prompt: str = SYSTEM_PROMPT,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    judge_model: str = "anthropic/claude-haiku-4.5",
    judge_base_url: str = OPENROUTER_BASE_URL,
    answer_weight: float = 1.0,
    step_weight: float = 0.5,
) -> vf.SingleTurnEnv:
    """AIME process-reward evaluation environment.

    Train split: AIME 2024. Eval split: AIME 2026 (held out — released after
    most current models' training cutoffs).
    Reward = answer correctness + per-step validity from an OpenRouter-hosted
    judge model. Completion size and step count are reported as pure metrics
    (weight 0) so efficiency is visible without shaping the reward by default.
    """
    vf.ensure_keys(["OPENROUTER_API_KEY"])

    def build_train_dataset():
        ds = load_example_dataset("aime2024")
        if num_train_examples > 0:
            ds = ds.select(range(num_train_examples))
        return ds

    def build_eval_dataset():
        ds = load_example_dataset("aime2026")
        if num_eval_examples > 0:
            ds = ds.select(range(num_eval_examples))
        return ds

    judge_client = AsyncOpenAI(
        base_url=judge_base_url,
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    parser = AIMECoTParser()
    rubric = AIMEProcessRubric(
        parser=parser,
        judge_client=judge_client,
        judge_model=judge_model,
        answer_weight=answer_weight,
        step_weight=step_weight,
    )
    return vf.SingleTurnEnv(
        dataset=build_train_dataset,
        eval_dataset=build_eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
