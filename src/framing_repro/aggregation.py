from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

from .codebook import (
    FRAME_ORDER,
    FRAME_TO_ID,
    ITEM_TO_FRAME,
    HISTORICAL_FRAME_ORDER,
    ALL_HISTORICAL_ITEMS,
    HISTORICAL_ITEM_TO_FRAME,
)


def rank_frames(binary_values: dict[str, int]) -> list[int]:
    """Rank the nine published frames by affirmative indicator count.

    Ties are stable in the published frame order. This is equivalent to the
    insertion-order behavior of the original Python implementation for the
    frame-specific items.
    """
    counts = Counter({frame: 0 for frame in FRAME_ORDER})
    for item, value in binary_values.items():
        if int(value) == 1:
            counts[ITEM_TO_FRAME[item]] += 1
    ranked = sorted(FRAME_ORDER, key=lambda f: (-counts[f], FRAME_ORDER.index(f)))
    positive = [FRAME_TO_ID[f] for f in ranked if counts[f] > 0]
    return positive + [99] * (len(FRAME_ORDER) - len(positive))


def historical_frame_predictions(values: dict[str, float], threshold: float = 0.5) -> dict[str, int]:
    """Reproduce the original 99-statement ``process_answer`` frame mapping.

    The first item is relevance (Q1), the next seven items distinguish
    episodic/thematic focus (Q2), and the remaining 91 items contribute counts
    to the nine GVFC frames. Q3_1..Q3_4 are the four highest positive frame
    counts. The published evaluation uses Q3_1 and Q3_2.
    """
    frame_to_index = {frame: idx for idx, frame in enumerate(HISTORICAL_FRAME_ORDER)}
    q2_counts: dict[int, int] = defaultdict(int)
    q3_counts: dict[int, int] = defaultdict(int)
    out: dict[str, int] = {"Q1": 99, "Q2": 99}

    for item in ALL_HISTORICAL_ITEMS:
        if item not in values:
            continue
        if float(values[item]) < threshold:
            continue
        frame = HISTORICAL_ITEM_TO_FRAME[item]
        frame_idx = frame_to_index[frame]
        if frame_idx == 0:
            out["Q1"] = 1
        elif frame_idx < 3:
            q2_counts[frame_idx] += 1
        else:
            # Rights -> 1, ..., Economic -> 9, exactly as in the original code.
            q3_counts[frame_idx - 2] += 1

    if q2_counts:
        out["Q2"] = sorted(q2_counts.items(), key=lambda x: x[1], reverse=True)[0][0]

    top_four = sorted(q3_counts.items(), key=lambda x: x[1], reverse=True)[:4]
    for i in range(4):
        out[f"Q3_{i + 1}"] = top_four[i][0] if i < len(top_four) and top_four[i][1] >= 1 else 99
    return out



def historical_model_frame_predictions(values: dict[str, float], threshold: float = 0.5) -> dict[str, object]:
    """Reproduce ``GunViolenceAnnotationHandler.process_annotations``.

    This is the model-level mapping used when the historical 99-statement raw
    files were first converted to ``framing_results.csv``. It differs slightly
    from the later Agreement mapping: it keeps only the top two Q3 frames and
    uses the original Q2 counter/override logic. Q3_3 and Q3_4 were left empty
    (NaN) in the original model-level CSVs.
    """
    frame_to_index = {frame: idx for idx, frame in enumerate(HISTORICAL_FRAME_ORDER)}
    frame_indices: dict[str, object] = {}
    frame_counter: dict[str, int] = {}
    q3_dict: dict[int, int] = {}

    for item in ALL_HISTORICAL_ITEMS:
        if item not in values:
            continue
        frame = HISTORICAL_ITEM_TO_FRAME[item]
        frame_idx = frame_to_index[frame]
        if float(values[item]) >= threshold:
            if frame_idx == 0:
                frame_indices["Q1"] = 1
            elif frame_idx < 3:
                if "Q2" not in frame_indices or (
                    frame_idx == 2
                    and (frame not in frame_counter or frame_counter[frame] > frame_counter.get(HISTORICAL_FRAME_ORDER[1], 0))
                ):
                    frame_indices["Q2"] = frame_idx
                    if frame not in frame_counter:
                        frame_counter[frame] = 1
            else:
                q3_idx = frame_idx - 2
                q3_dict[q3_idx] = q3_dict.get(q3_idx, 0) + 1
            frame_counter[frame] = frame_counter.get(frame, 0) + 1
        else:
            frame_counter[frame] = frame_counter.get(frame, 0)

    top_two = sorted(q3_dict.items(), key=lambda item: item[1], reverse=True)[:2]
    frame_indices["Q3_1"] = top_two[0][0] if len(top_two) > 0 and top_two[0][1] > 0 else 99
    frame_indices["Q3_2"] = top_two[1][0] if len(top_two) > 1 and top_two[1][1] > 0 else 99
    frame_indices.setdefault("Q1", 99)
    frame_indices.setdefault("Q2", 99)
    # The historical DataFrame schema contained these columns, but the model
    # handler never populated them.
    frame_indices["Q3_3"] = float("nan")
    frame_indices["Q3_4"] = float("nan")
    return frame_indices

def majority_vote(raw_files: list[str | Path], threshold: int = 2) -> pd.DataFrame:
    """Combine model-level binary outputs with a direct >= ``threshold`` vote.

    This is the publication-style majority operation. For exact recreation of
    the original Agreement directory, use :func:`historical_agreement`.
    """
    frames = [pd.read_csv(p, sep=";") for p in raw_files]
    headline_col = "news_title" if "news_title" in frames[0].columns else "headline"
    base = frames[0][[headline_col]].copy().rename(columns={headline_col: "news_title"})
    item_cols = [
        c for c in frames[0].columns
        if c.startswith("S") and not c.endswith("_justification") and c[1:].isdigit()
    ]
    for df in frames[1:]:
        this_col = "news_title" if "news_title" in df.columns else "headline"
        if not base["news_title"].reset_index(drop=True).equals(df[this_col].reset_index(drop=True)):
            raise ValueError("Raw files are not aligned by headline")
    for col in item_cols:
        base[col] = sum(df[col].astype(int) for df in frames).ge(threshold).astype(int)
    return base


def historical_agreement(raw_files: list[str | Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recreate the original four-model Agreement procedure for 99 statements.

    This intentionally preserves two historical details:

    * A statement is excluded when *any* model has zero variance for it.
    * Agreement decisions use mean >= 2 / number_of_models (0.5 for four
      models), whereas the archived ``raw_results.csv`` stores pandas-rounded
      means, matching the original implementation.

    Returns
    -------
    raw_agreement:
        Historical Agreement/raw_results.csv equivalent.
    framing_results:
        Historical Agreement/framing_results.csv equivalent.
    id_to_statement:
        Mapping of retained S-columns to statement text.
    """
    models = [pd.read_csv(p, sep=";") for p in raw_files]
    if not models:
        raise ValueError("At least one raw model file is required")

    def hcol(df: pd.DataFrame) -> str:
        if "headline" in df.columns:
            return "headline"
        if "news_title" in df.columns:
            return "news_title"
        raise ValueError("Raw result file has no headline column")

    headlines = models[0][hcol(models[0])].reset_index(drop=True)
    for df in models[1:]:
        if not headlines.equals(df[hcol(df)].reset_index(drop=True)):
            raise ValueError("Raw files are not aligned by headline")

    # The old implementation retained a statement only when every annotator's
    # column had non-zero standard deviation.
    retained: list[tuple[str, str]] = []
    means: dict[str, pd.Series] = {}
    for i, statement in enumerate(ALL_HISTORICAL_ITEMS, 1):
        col = f"S{i}"
        if not all(col in df.columns for df in models):
            continue
        if all(float(df[col].astype(float).std()) > 0 for df in models):
            retained.append((col, statement))
            means[col] = pd.concat([df[col].astype(float).reset_index(drop=True) for df in models], axis=1).mean(axis=1)

    mean_df = pd.DataFrame({"headline": headlines})
    for col, _ in retained:
        mean_df[col] = means[col]

    # Framing decisions are calculated from unrounded averages at >= 0.5.
    threshold = 2 / len(models)
    framing_rows = []
    for row_idx, headline in enumerate(headlines):
        values = {
            statement: float(mean_df.at[row_idx, col])
            for col, statement in retained
        }
        pred = historical_frame_predictions(values, threshold=threshold)
        pred = {"news_title": headline, **pred}
        framing_rows.append(pred)
    framing = pd.DataFrame(framing_rows, columns=["news_title", "Q1", "Q2", "Q3_1", "Q3_2", "Q3_3", "Q3_4"])

    # pandas.DataFrame.round() is intentionally used because that is what the
    # original code wrote to Agreement/raw_results.csv.
    raw = mean_df.round()
    mapping = pd.DataFrame({"ID": [c for c, _ in retained], "Statement": [s for _, s in retained]})
    return raw, framing, mapping
