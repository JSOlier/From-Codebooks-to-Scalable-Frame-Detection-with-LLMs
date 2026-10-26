from __future__ import annotations

from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer

from .aggregation import historical_frame_predictions
from .codebook import ALL_BINARY_ITEMS, ALL_HISTORICAL_ITEMS


def _similarity_matrix(model: SentenceTransformer, headlines: list[str], statements: list[str]):
    h = model.encode(headlines, convert_to_tensor=True)
    s = model.encode(statements, convert_to_tensor=True)
    return model.similarity(h, s).cpu().numpy()


def annotate_similarity(
    model_name: str,
    headlines: list[str],
    output: str | Path,
    threshold: float = 0.5,
) -> None:
    """Published sentence-similarity baseline using 91 frame items and threshold 0.5."""
    model = SentenceTransformer(model_name)
    similarities = _similarity_matrix(model, headlines, list(ALL_BINARY_ITEMS))
    rows = []
    for headline, scores in zip(headlines, similarities):
        row = {"news_title": headline}
        for i, score in enumerate(scores, 1):
            row[f"S{i}"] = int(float(score) > threshold)
            row[f"S{i}_similarity"] = float(score)
        rows.append(row)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, sep=";", index=False)


def annotate_similarity_historical(
    model_name: str,
    headlines: list[str],
    output_dir: str | Path,
) -> float:
    """Reproduce the supplied historical similarity implementation.

    The original source used all 99 statements and a data-dependent threshold
    equal to the global similarity mean plus one standard deviation. This is
    intentionally separate from the paper-facing fixed-0.5 implementation.

    Returns the calculated threshold.
    """
    model = SentenceTransformer(model_name)
    statements = list(ALL_HISTORICAL_ITEMS)
    similarities = _similarity_matrix(model, headlines, statements)
    threshold = float(similarities.mean() + similarities.std())

    raw_rows = []
    frame_rows = []
    for headline, scores in zip(headlines, similarities):
        raw = {"headline": headline}
        values = {}
        for i, (statement, score) in enumerate(zip(statements, scores), 1):
            score = float(score)
            raw[f"S{i}"] = int(score >= threshold)
            values[statement] = score
        raw_rows.append(raw)
        frame_rows.append({"news_title": headline, **historical_frame_predictions(values, threshold=threshold)})

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(raw_rows).to_csv(outdir / "raw_results.csv", sep=";", index=False)
    pd.DataFrame(frame_rows).to_csv(outdir / "framing_results.csv", sep=";", index=False)
    (outdir / "threshold.txt").write_text(f"{threshold:.12g}\n", encoding="utf-8")
    return threshold
