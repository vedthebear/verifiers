import asyncio
import logging
from typing import Any

from openai import AsyncOpenAI

import verifiers as vf
from verifiers.types import Messages, State

from aime_parsers import AIMECoTParser
from aime_prompts import SEGMENTER_PROMPT, STEP_JUDGE_PROMPT

logger = logging.getLogger(__name__)


class StepSegmenter:
    """One LLM call per rollout returns a JSON list of discrete reasoning steps.

    Cheap model (e.g. gpt-4o-mini) is sufficient — segmentation is a
    structural task, not a math-judgement task. Verbatim step text is
    preserved so the downstream judge sees what the model wrote, not a
    paraphrase.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        prompt_template: str = SEGMENTER_PROMPT,
        sampling_args: dict[str, Any] | None = None,
    ):
        self.client = client
        self.model = model
        self.prompt_template = prompt_template
        self.sampling_args = sampling_args or {
            "max_tokens": 4096,
            "temperature": 0.0,
        }

    STEP_MARKER = "###STEP###"

    async def segment(self, problem: str, completion_text: str) -> list[str]:
        prompt = self.prompt_template.format(
            problem=problem, completion=completion_text
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **self.sampling_args,
            )
            content = (response.choices[0].message.content or "").strip()
            chunks = content.split(self.STEP_MARKER)
            return [c.strip() for c in chunks if c.strip()]
        except Exception as exc:
            logger.warning(
                "StepSegmenter failed (model=%s): %r", self.model, exc
            )
        return []


class StepJudge:
    """Per-step LLM-as-judge for chain-of-thought reasoning.

    One judge call per step (problem, prior accepted steps, current step) ->
    {valid, invalid, unclear}. Score is the fraction of "valid" verdicts over
    successfully judged steps; infra failures are excluded from the
    denominator. After ``ERROR_STREAK_THRESHOLD`` consecutive failures the
    circuit breaker trips and remaining calls short-circuit.
    """

    ERROR_SENTINEL = "__judge_error__"
    ERROR_STREAK_THRESHOLD = 3

    def __init__(
        self,
        judge_client: AsyncOpenAI,
        judge_model: str,
        judge_prompt: str = STEP_JUDGE_PROMPT,
        judge_sampling_args: dict[str, Any] | None = None,
    ):
        self.client = judge_client
        self.model = judge_model
        self.prompt_template = judge_prompt
        self.sampling_args = judge_sampling_args or {
            "max_tokens": 8,
            "temperature": 0.0,
        }
        # Single-threaded asyncio: int += on these is safe without a lock.
        self._error_streak = 0
        self._tripped = False

    async def _judge_step(
        self, problem: str, prior: list[str], step: str
    ) -> str:
        if self._tripped:
            return self.ERROR_SENTINEL
        prior_block = (
            "\n\n".join(prior) if prior else "(none — this is the first step)"
        )
        prompt = self.prompt_template.format(
            problem=problem, prior_steps=prior_block, step=step
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **self.sampling_args,
            )
        except Exception as exc:
            self._error_streak += 1
            if (
                not self._tripped
                and self._error_streak >= self.ERROR_STREAK_THRESHOLD
            ):
                self._tripped = True
                logger.error(
                    "StepJudge circuit breaker tripped after %d consecutive "
                    "judge failures (model=%s). Remaining judge calls will "
                    "short-circuit; step_validity will be 0 for the rest of "
                    "the run. Last error: %r",
                    self._error_streak,
                    self.model,
                    exc,
                )
            return self.ERROR_SENTINEL
        self._error_streak = 0
        return (response.choices[0].message.content or "").strip().lower()

    async def score(
        self, problem: str, steps: list[str], state: State | None = None
    ) -> float:
        if not steps:
            return 0.0
        verdicts = await asyncio.gather(
            *(
                self._judge_step(problem, steps[:i], steps[i])
                for i in range(len(steps))
            )
        )
        if state is not None:
            state["step_audit"] = [
                {"index": i, "step": s, "verdict": v}
                for i, (s, v) in enumerate(zip(steps, verdicts))
            ]
        judged = sum(1 for v in verdicts if v != self.ERROR_SENTINEL)
        if not judged:
            return 0.0
        valid = sum(1 for v in verdicts if v.startswith("valid"))
        return valid / judged


class AIMEProcessRubric(vf.MathRubric):
    """Compose answer correctness + per-step validity + efficiency metrics.

    Inherits ``correct_answer`` from MathRubric (math_verify in a process pool).
    Adds ``step_validity`` (LLM-segmented + LLM-judged with prior-only context)
    plus two pure metrics (``completion_chars``, ``step_count``).
    """

    def __init__(
        self,
        parser: AIMECoTParser,
        segmenter_client: AsyncOpenAI,
        segmenter_model: str,
        judge_client: AsyncOpenAI,
        judge_model: str,
        answer_weight: float = 1.0,
        step_weight: float = 0.5,
    ):
        super().__init__(parser=parser)
        # MathRubric.__init__ unconditionally registers ``correct_answer`` with
        # weight 1.0; rebind that weight here so callers can dial it.
        self.weights[0] = answer_weight
        self.segmenter = StepSegmenter(
            client=segmenter_client, model=segmenter_model
        )
        self.judge = StepJudge(
            judge_client=judge_client, judge_model=judge_model
        )
        self.add_class_object("segmenter", self.segmenter)
        self.add_class_object("judge", self.judge)
        self.add_reward_func(self.step_validity, weight=step_weight)
        self.add_metric(self.completion_chars)
        self.add_metric(self.step_count)

    async def _ensure_steps(
        self,
        parser: AIMECoTParser,
        prompt: Messages,
        completion: Messages,
        state: State | None,
    ) -> list[str]:
        cached = state.get("segmented_steps") if state else None
        if cached is not None:
            return list(cached)
        problem = parser.user_text(prompt)
        text = parser.completion_text(completion)
        steps = await self.segmenter.segment(problem, text)
        if state is not None:
            state["segmented_steps"] = steps
        return steps

    async def step_validity(
        self,
        parser: AIMECoTParser,
        prompt: Messages,
        completion: Messages,
        state: State,
        **kwargs,
    ) -> float:
        cached = state.get("step_validity_score") if state else None
        if cached is not None:
            return float(cached)
        problem = parser.user_text(prompt)
        steps = await self._ensure_steps(parser, prompt, completion, state)
        score = await self.judge.score(problem, steps, state=state)
        if state is not None:
            state["step_validity_score"] = score
        return score

    async def completion_chars(
        self, parser: AIMECoTParser, completion: Messages, **kwargs
    ) -> float:
        return float(len(parser.completion_text(completion)))

    async def step_count(
        self,
        parser: AIMECoTParser,
        prompt: Messages,
        completion: Messages,
        state: State,
        **kwargs,
    ) -> float:
        steps = await self._ensure_steps(parser, prompt, completion, state)
        return float(len(steps))
