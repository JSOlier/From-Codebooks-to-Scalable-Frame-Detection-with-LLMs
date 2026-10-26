from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from .aggregation import rank_frames
from .codebook import ALL_BINARY_ITEMS


def add_ranked_predictions(raw: pd.DataFrame, mode: str = "binary") -> pd.DataFrame:
    """Convert agreed item decisions into top-two frame predictions."""
    out = raw[["news_title"]].copy()
    if mode == "binary":
        cols = [f"S{i}" for i in range(1, len(ALL_BINARY_ITEMS) + 1)]
        ranked = raw.apply(lambda r: rank_frames(dict(zip(ALL_BINARY_ITEMS, [r[c] for c in cols]))), axis=1)
    elif mode == "descriptions":
        cols = [f"S{i}" for i in range(1, 10)]
        def direct_rank(r):
            positive = [i for i, c in enumerate(cols, 1) if int(r[c]) == 1]
            return positive + [99] * (9 - len(positive))
        ranked = raw.apply(direct_rank, axis=1)
    else:
        raise ValueError("mode must be 'binary' or 'descriptions'")
    out["pred_1"] = [x[0] for x in ranked]
    out["pred_2"] = [x[1] for x in ranked]
    return out


def align_primary(gold_a: int, gold_b: int, pred_1: int, pred_2: int) -> tuple[int, int]:
    """Align top-two predictions to GVFC A/B labels as described in the Evaluation section."""
    if pred_1 == gold_a:
        return pred_1, pred_2
    if pred_2 == gold_a:
        return pred_2, pred_1
    if gold_b < 99 and pred_1 == gold_b:
        return pred_2, pred_1
    return pred_1, pred_2


def bootstrap_kappa(y_true, y_pred, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed); n=len(y_true); vals=[]
    for _ in range(n_boot):
        idx=rng.integers(0,n,n); vals.append(cohen_kappa_score(np.asarray(y_true)[idx], np.asarray(y_pred)[idx]))
    return tuple(np.quantile(vals,[0.025,0.975]))


def evaluate(predictions: pd.DataFrame, gold: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = predictions.merge(gold[["news_title","Q3 Theme1","Q3 Theme2"]], on="news_title", how="inner")
    aligned = df.apply(lambda r: align_primary(int(r["Q3 Theme1"]), int(r["Q3 Theme2"]), int(r["pred_1"]), int(r["pred_2"])), axis=1)
    df["aligned_pred_1"]=[x[0] for x in aligned]; df["aligned_pred_2"]=[x[1] for x in aligned]
    y=df["Q3 Theme1"].astype(int); p=df["aligned_pred_1"].astype(int)
    report=classification_report(y,p,labels=list(range(1,10)),output_dict=True,zero_division=0)
    metrics={"n":len(df),"accuracy":accuracy_score(y,p),"cohen_kappa":cohen_kappa_score(y,p),"kappa_ci95":bootstrap_kappa(y.to_numpy(),p.to_numpy()),"weighted_f1":report["weighted avg"]["f1-score"],"per_class":report}
    return df, metrics
