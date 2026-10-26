import ast
import json
import re


def parse_binary_answer(text: str) -> tuple[int, str]:
    """Parse a model response into (0/1, justification), tolerating minor format drift."""
    text = text.strip()
    candidates = [text]
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(candidate)
                if isinstance(obj, dict):
                    answer = str(obj.get("Answer", "")).strip().lower()
                    if answer in {"yes", "no"}:
                        return int(answer == "yes"), str(obj.get("Justification", "")).strip()
            except Exception:
                pass
    m = re.search(r"\b(yes|no)\b", text, flags=re.I)
    if not m:
        raise ValueError(f"Could not parse Yes/No response: {text[:200]!r}")
    return int(m.group(1).lower() == "yes"), ""
