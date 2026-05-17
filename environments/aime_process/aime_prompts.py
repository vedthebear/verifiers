SYSTEM_PROMPT = (
    "You are an expert competition mathematician solving AIME problems. "
    "Reason step by step. Separate each major step with a blank line. "
    "Place your final integer answer (an integer from 0 to 999) inside \\boxed{}."
)


SEGMENTER_PROMPT = """\
You are analyzing a chain-of-thought solution to a competition math problem.
Identify the discrete reasoning steps. A "step" is a self-contained reasoning
move: a definition, a calculation, an algebraic manipulation, a substitution,
or a conclusion.

Rules:
- Output the VERBATIM text of each step — do not summarize or paraphrase.
- LaTeX, math notation, and backslashes should appear exactly as written.
- Cover the entire solution; do not skip any reasoning content.
- Do NOT include the final \\boxed{{}} answer line as a step.
- Section headers, pure restatements, and bare equation labels are not steps.

Problem:
{problem}

Solution:
{completion}

Output the steps separated by the marker `###STEP###` on its own line.
Do not include any preamble, numbering, or commentary — only the steps and
the markers between them.

Example output format:
###STEP###
Let p be Patrick's walking speed; then Tanya's speed is p + 2.
###STEP###
Patrick walks for T hours, Tanya for T-1 hours, Jose for T-2 hours.
###STEP###
...
"""


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
