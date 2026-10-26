from pathlib import Path
import pandas as pd

from framing_repro.codebook import ALL_BINARY_ITEMS, ALL_HISTORICAL_ITEMS, FRAME_ORDER
from framing_repro.aggregation import rank_frames, historical_agreement, historical_model_frame_predictions
from framing_repro.data import load_gvfc
from framing_repro.evaluate import align_primary, evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_indicator_counts():
    assert len(ALL_BINARY_ITEMS) == 91
    assert len(ALL_HISTORICAL_ITEMS) == 99
    assert len(FRAME_ORDER) == 9


def test_rank():
    values = {x: 0 for x in ALL_BINARY_ITEMS}
    values[ALL_BINARY_ITEMS[0]] = 1
    assert rank_frames(values)[0] == 1


def test_alignment():
    assert align_primary(3, 7, 7, 3) == (3, 7)


def test_historical_dataset_selection_matches_archive():
    df = load_gvfc(ROOT / "data" / "GVFC_headlines_and_annotations.csv", historical=True)
    assert len(df) == 1293


def test_historical_agreement_reconstructs_archived_files():
    base = ROOT / "archived_results" / "binary_99_statements"
    model_root = base / "Gun_Violence"
    files = [
        model_root / "Meta-Llama-3-8B-Instruct" / "raw_results.csv",
        model_root / "Mistral-7B-Instruct-v0.2" / "raw_results.csv",
        model_root / "Phi-3.5-mini-instruct" / "raw_results.csv",
        model_root / "Qwen1.5-7B-Chat" / "raw_results.csv",
    ]
    raw, framing, mapping = historical_agreement(files)
    archived_raw = pd.read_csv(base / "Agreement" / "raw_results.csv", sep=";")
    archived_framing = pd.read_csv(base / "Agreement" / "framing_results.csv", sep=";")

    pd.testing.assert_frame_equal(raw, archived_raw, check_dtype=False)
    pd.testing.assert_frame_equal(framing, archived_framing, check_dtype=False)
    assert len(mapping) == 90  # 9 zero-variance statements were historically excluded.


def test_historical_model_mapping_reconstructs_archived_phi():
    base = ROOT / "archived_results" / "binary_99_statements" / "Gun_Violence" / "Phi-3.5-mini-instruct"
    raw = pd.read_csv(base / "raw_results.csv", sep=";")
    archived = pd.read_csv(base / "framing_results.csv", sep=";")
    rows = []
    for _, row in raw.iterrows():
        values = {item: int(row[f"S{i}"]) for i, item in enumerate(ALL_HISTORICAL_ITEMS, 1)}
        rows.append({"news_title": row["headline"], **historical_model_frame_predictions(values)})
    rebuilt = pd.DataFrame(rows)[archived.columns]
    pd.testing.assert_frame_equal(rebuilt, archived, check_dtype=False)


def test_archived_agreement_reproduces_main_metrics():
    base = ROOT / "archived_results" / "binary_99_statements" / "Agreement" / "framing_results.csv"
    pred = pd.read_csv(base, sep=";").rename(columns={"Q3_1": "pred_1", "Q3_2": "pred_2"})
    pred = pred[["news_title", "pred_1", "pred_2"]].fillna(99)
    gold = load_gvfc(ROOT / "data" / "GVFC_headlines_and_annotations.csv", historical=True)
    _, metrics = evaluate(pred, gold)
    assert metrics["n"] == 1293
    assert abs(metrics["accuracy"] - 0.8120649651972158) < 1e-12
    assert abs(metrics["weighted_f1"] - 0.8065987180306272) < 1e-12
    assert abs(metrics["cohen_kappa"] - 0.7703990699013561) < 1e-12
