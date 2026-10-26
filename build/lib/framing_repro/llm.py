from __future__ import annotations

from pathlib import Path
import importlib.util
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from .aggregation import historical_model_frame_predictions
from .codebook import (
    ALL_BINARY_ITEMS,
    ALL_HISTORICAL_ITEMS,
    FULL_DESCRIPTIONS,
    FRAME_ORDER,
)
from .parsing import parse_binary_answer
from .prompts import make_messages

GENERATION_KWARGS = dict(
    do_sample=True,
    temperature=0.0001,
    top_p=0.01,
    max_new_tokens=50,
)


def _require_accelerate() -> None:
    if importlib.util.find_spec("accelerate") is None:
        raise RuntimeError(
            "LLM inference with device_map requires 'accelerate'. "
            "Install the LLM extras with: pip install -e \".[llm]\""
        )


def _build_generator(
    model_name: str,
    *,
    batch_size: int = 8,
    revision: str | None = None,
    historical_tokenizer: bool = False,
):
    """Create the HF text-generation pipeline used by the experiments.

    ``historical_tokenizer=True`` reproduces the tokenizer/terminator setup in
    the supplied original code as closely as possible: a separate tokenizer is
    assigned BOS/EOS/PAD IDs 1/2/0 for terminator construction, while the
    generation tokenizer pads with the model's EOS token. The original project
    did not record Hugging Face commit revisions, so ``revision`` is optional
    and can only be pinned for future reruns.
    """
    _require_accelerate()

    pipeline_tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=pipeline_tokenizer,
        batch_size=batch_size,
    )
    generator.model.eval()

    model_eos = generator.model.config.eos_token_id
    pad_id = model_eos[0] if isinstance(model_eos, list) else model_eos
    if pad_id is not None:
        generator.tokenizer.pad_token_id = pad_id

    if historical_tokenizer:
        terminator_tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        # These assignments are preserved from LLMs/gun_violence_framing_simple.py.
        terminator_tokenizer.bos_token_id = 1
        terminator_tokenizer.eos_token_id = 2
        terminator_tokenizer.pad_token_id = 0
        terminators = [
            terminator_tokenizer.eos_token_id,
            terminator_tokenizer.convert_tokens_to_ids("```"),
        ]
    else:
        # The paper-facing runner uses the model-native EOS token. The extra
        # code-fence terminator from the historical source is preserved only
        # in historical mode because some tokenizers map "```" to an unknown
        # token ID.
        terminators = generator.tokenizer.eos_token_id

    return generator, terminators


def generate_annotations(
    model_name: str,
    headlines: list[str],
    descriptions: list[str],
    *,
    batch_size: int = 8,
    revision: str | None = None,
    historical_tokenizer: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate binary answers and justifications for headline-description pairs."""
    generator, terminators = _build_generator(
        model_name,
        batch_size=batch_size,
        revision=revision,
        historical_tokenizer=historical_tokenizer,
    )

    raw_rows: list[dict] = []
    justification_rows: list[dict] = []
    for headline in headlines:
        messages = [make_messages(headline, d) for d in descriptions]
        generated = generator(messages, eos_token_id=terminators, **GENERATION_KWARGS)
        raw_row = {"headline": headline}
        justification_row = {"headline": headline}
        for i, result in enumerate(generated, 1):
            # Chat-mode text-generation returns the conversation with the newly
            # generated assistant message as its final element.
            text = result[0]["generated_text"]
            if isinstance(text, list):
                content = text[-1]["content"]
            else:
                content = str(text)
            value, justification = parse_binary_answer(content)
            raw_row[f"S{i}"] = value
            justification_row[f"S{i}"] = justification
        raw_rows.append(raw_row)
        justification_rows.append(justification_row)

    return pd.DataFrame(raw_rows), pd.DataFrame(justification_rows)


def annotate(
    model_name: str,
    headlines: list[str],
    descriptions: list[str],
    output: str | Path,
    batch_size: int = 8,
    *,
    revision: str | None = None,
    historical_tokenizer: bool = False,
) -> None:
    """Run one model and save statement-level binary predictions.

    This function keeps the original simple file-output API. For an exact
    historical 99-statement run with both raw and frame-level outputs, use
    :func:`annotate_historical_99`.
    """
    raw, justifications = generate_annotations(
        model_name,
        headlines,
        descriptions,
        batch_size=batch_size,
        revision=revision,
        historical_tokenizer=historical_tokenizer,
    )
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.rename(columns={"headline": "news_title"}).to_csv(out, sep=";", index=False)
    just_path = out.with_name(out.stem + "_justifications" + out.suffix)
    justifications.rename(columns={"headline": "news_title"}).to_csv(just_path, sep=";", index=False)


def annotate_historical_99(
    model_name: str,
    headlines: list[str],
    output_dir: str | Path,
    batch_size: int = 8,
    *,
    revision: str | None = None,
) -> None:
    """Rerun the historical 99-statement experiment.

    Outputs mirror the useful original artifacts: ``raw_results.csv`` and
    ``framing_results.csv``. Justifications, which were not retained by the
    original raw matrix, are saved separately for transparency.
    """
    raw, justifications = generate_annotations(
        model_name,
        headlines,
        list(ALL_HISTORICAL_ITEMS),
        batch_size=batch_size,
        revision=revision,
        historical_tokenizer=True,
    )

    frame_rows = []
    for _, row in raw.iterrows():
        values = {item: int(row[f"S{i}"]) for i, item in enumerate(ALL_HISTORICAL_ITEMS, 1)}
        pred = historical_model_frame_predictions(values, threshold=0.5)
        frame_rows.append({"news_title": row["headline"], **pred})

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(outdir / "raw_results.csv", sep=";", index=False)
    pd.DataFrame(frame_rows).to_csv(outdir / "framing_results.csv", sep=";", index=False)
    justifications.to_csv(outdir / "justifications.csv", sep=";", index=False)


def binary_descriptions() -> list[str]:
    return list(ALL_BINARY_ITEMS)


def historical_binary_descriptions() -> list[str]:
    return list(ALL_HISTORICAL_ITEMS)


def full_codebook_descriptions() -> list[str]:
    return [FULL_DESCRIPTIONS[f] for f in FRAME_ORDER]
