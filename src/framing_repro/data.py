from pathlib import Path
import pandas as pd


def load_gvfc(
    path: str | Path,
    relevant_only: bool = True,
    *,
    historical: bool = False,
) -> pd.DataFrame:
    """Load the semicolon-delimited GVFC annotations.

    Parameters
    ----------
    path:
        Path to ``GVFC_headlines_and_annotations.csv``.
    relevant_only:
        Retain only rows with ``Q3 Theme1 < 99``. This is the frame-detection
        subset evaluated in the article.
    historical:
        Reproduce the row-selection behavior of the original experiment code:
        remove *all* occurrences of duplicated ``news_title`` values using
        ``drop_duplicates(..., keep=False)`` before selecting relevant rows.
        With the dataset snapshot archived in this repository this yields 1,293
        relevant headlines, matching the archived model outputs.

    Notes
    -----
    The paper states that 1,300 relevant headlines were used. The supplied
    dataset snapshot contains 1,301 relevant rows before historical duplicate
    removal, while the original loader yields 1,293. This discrepancy is
    documented in ``REPRODUCIBILITY_NOTES.md`` rather than silently reconciled.
    """
    df = pd.read_csv(path, sep=";")
    required = {"news_title", "Q3 Theme1", "Q3 Theme2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if historical:
        df = df.drop_duplicates(subset=["news_title"], keep=False)
    if relevant_only:
        df = df.loc[df["Q3 Theme1"] < 99].copy()
    return df.reset_index(drop=True)
