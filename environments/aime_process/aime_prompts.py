SYSTEM_PROMPT = (
    "You are an expert competition mathematician solving AIME problems. "
    "Reason step by step. Separate each major step with a blank line. "
    "Place your final integer answer (an integer from 0 to 999) inside \\boxed{}."
)


STEP_JUDGE_PROMPT = """\
You are auditing a single step of a chain-of-thought solution to an AIME problem.

Problem:
{problem}

Prior steps (already accepted):
{prior_steps}

Step to evaluate:
{step}

Is this step mathematically valid and a reasonable continuation of the prior work?
- "valid" means: no arithmetic, algebraic, or logical errors; follows from prior steps.
- "invalid" means: contains an error or an unjustified leap.
- "unclear" means: too vague to evaluate (e.g., pure restatement, no math content).

Respond with exactly one word: valid, invalid, or unclear.
"""
