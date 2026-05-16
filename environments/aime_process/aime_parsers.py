import re

import verifiers as vf
from verifiers.types import Messages
from verifiers.utils.data_utils import extract_boxed_answer

# Inner-content capture; matches even multi-line bodies.
_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
# Strict pure-LaTeX block: whole chunk is `\[ ... \]` or `$$ ... $$`.
_PURE_LATEX_RE = re.compile(r"^(\\\[.*\\\]|\$\$.*\$\$)$", flags=re.DOTALL)
# Chunks shorter than this fold into the previous step.
_MIN_STEP_CHARS = 30


class AIMECoTParser(vf.Parser):
    """Parser for AIME chain-of-thought completions.

    - `parse_answer` returns the final ``\\boxed{...}`` integer string.
    - `segment_steps` splits the reasoning body on blank lines into ordered
      steps, dropping the final boxed-answer line (which is graded by
      correctness, not by the step judge).
    """

    @staticmethod
    def _field(message: object, name: str) -> object | None:
        """Read a field from a message — works for both dicts and Pydantic models."""
        if isinstance(message, dict):
            return message.get(name)
        return getattr(message, name, None)

    def _last_of_role(self, messages: Messages, role: str) -> str:
        if isinstance(messages, list):
            for message in reversed(messages):
                if self._field(message, "role") == role:
                    return str(self._field(message, "content") or "")
            return ""
        return str(messages)

    def completion_text(self, completion: Messages) -> str:
        return self._last_of_role(completion, "assistant")

    def user_text(self, prompt: Messages) -> str:
        return self._last_of_role(prompt, "user")

    def parse_answer(self, completion: Messages) -> str | None:
        boxed = extract_boxed_answer(self.completion_text(completion), strict=True)
        return boxed or None

    def segment_steps(self, completion: Messages) -> list[str]:
        text = self.completion_text(completion)
        match = _THINK_PATTERN.search(text)
        body = match.group(1) if match else text
        chunks = [c.strip() for c in body.split("\n\n")]
        chunks = [c for c in chunks if c and "\\boxed{" not in c]
        steps: list[str] = []
        for chunk in chunks:
            if chunk.startswith("**") and chunk.endswith("**"):
                continue  # drop bold-wrapped section headers
            is_short = len(chunk) < _MIN_STEP_CHARS
            is_pure_latex = bool(_PURE_LATEX_RE.match(chunk))
            if steps and (is_short or is_pure_latex):
                steps[-1] = steps[-1] + "\n\n" + chunk
            elif is_short or is_pure_latex:
                continue  # drop short/pure-latex chunks with nothing to fold into
            else:
                steps.append(chunk)
        return steps
