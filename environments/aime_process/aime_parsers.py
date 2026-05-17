import verifiers as vf
from verifiers.types import Messages
from verifiers.utils.data_utils import extract_boxed_answer


class AIMECoTParser(vf.Parser):
    """Parser for AIME chain-of-thought completions.

    Step segmentation lives on the rubric (`StepSegmenter`), which uses an
    LLM call rather than heuristics — see ``aime_rubrics.py``. This parser
    only owns answer extraction and message-text plumbing.
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
