import ast
import json
import re


def parse_binary_answer(text: str) -> tuple[int, str]:
    """Parse a model response using the behavior of the original experiment.

    Historical rule:
    - Answer == "yes" -> 1
    - Any other parsed Answer value -> 0
    - If the response cannot be parsed, -> 0

    This preserves the original experiment's behavior, where only an
    affirmative "yes" response was coded as 1.
    """
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
                    justification = str(
                        obj.get("Justification", "")
                    ).strip()

                    # Historical behavior:
                    # only "yes" counts as affirmative.
                    return int(answer == "yes"), justification

            except Exception:
                pass

    # Try to recover an Answer field from malformed output.
    match = re.search(
        r'["\']?Answer["\']?\s*:\s*["\']?([^"\',}\n]+)',
        text,
        flags=re.I,
    )

    if match:
        answer = match.group(1).strip().lower()
        return int(answer == "yes"), ""

    # Historical fallback: parsing failure was treated as No.
    return 0, ""