SYSTEM_PROMPT = (
    "Decide if the headline's frame corresponds to a given description.\n"
    "Give your answer in Python dictionary format with the following information: "
    "\"Answer\" and \"Justification\". Follow this format: "
    "{\"Answer\": \"Yes or No\", \"Justification\": \"Your justification\"}."
)


def make_messages(headline: str, description: str) -> list[dict[str, str]]:
    """Appendix A prompt structure."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f'Headline: "{headline}"\nDescription: {description}'},
    ]
