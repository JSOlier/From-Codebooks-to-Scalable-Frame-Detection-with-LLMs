"""Generate the paper-facing result tables from completed experiment outputs.

Expected directory layout is produced by ``scripts/run_all_experiments.py``.
The script does not run any models; it only aggregates and evaluates existing
CSV outputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from framing_repro.aggregation import majority_vote
from framing_repro.data import load_gvfc
from framing_repro.evaluate import add_ranked_predictions, evaluate

FRAME_NAMES = [
    "Gun Rights",
    "Gun Control",
    "Politics",
    "Mental Health",
    "Public/School Safety",
    "Race/Ethnicity",
    "Public Opinion",
    "Social/Cultural Issues",
    "Economic Consequences",
]

# Values reported by Liu et al. (2019) and reproduced in Table 3 of the paper.
BERT_BENCHMARK_RECALL = [0.65, 0.84, 0.89, 0.78, 0.78, 0.92, 0.86, 0.58, 0.80]
BERT_BENCHMARK_ACCURACY = 0.84

LLM_MODELS = {
    "Llama": "meta-llama_Meta-Llama-3-8B-Instruct",
    "Mistral": "mistralai_Mistral-7B-Instruct-v0.2",
    "Phi": "microsoft_Phi-3.5-mini-instruct",
    "Qwen": "Qwen_Qwen1.5-7B-Chat",
}
SIM_MODELS = {
    "RoBERTa": "sentence-transformers_all-distilroberta-v1",
    "MiniLM": "sentence-transformers_all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers_all-mpnet-base-v2",
    "multi-QA": "sentence-transformers_multi-qa-mpnet-base-dot-v1",
}


def _metric_for_raw(raw_path: Path, gold: pd.DataFrame, mode: str) -> dict:
    raw = pd.read_csv(raw_path, sep=";")
    pred = add_ranked_predictions(raw, mode)
    _, metrics = evaluate(pred, gold)
    return metrics


def _agreed_metric(raw_paths: list[Path], gold: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, dict]:
    agreed = majority_vote(raw_paths, threshold=2)
    pred = add_ranked_predictions(agreed, mode)
    aligned, metrics = evaluate(pred, gold)
    return aligned, metrics


def _class_value(metrics: dict, class_id: int, key: str) -> float:
    return float(metrics["per_class"][str(class_id)][key])


def _f1_table(metrics_by_name: dict[str, dict], column_order: list[str]) -> pd.DataFrame:
    rows = []
    for cid, frame in enumerate(FRAME_NAMES, 1):
        row = {"Class": frame}
        for name in column_order:
            row[name] = _class_value(metrics_by_name[name], cid, "f1-score")
        rows.append(row)
    weighted = {"Class": "Weighted average F1"}
    accuracy = {"Class": "Accuracy"}
    for name in column_order:
        weighted[name] = float(metrics_by_name[name]["weighted_f1"])
        accuracy[name] = float(metrics_by_name[name]["accuracy"])
    rows.extend([weighted, accuracy])
    return pd.DataFrame(rows)


def generate(root: Path, data_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    gold = load_gvfc(data_path, historical=True)

    # Binary questions -----------------------------------------------------
    binary_paths = {name: root / "binary" / slug / "raw.csv" for name, slug in LLM_MODELS.items()}
    binary_metrics = {name: _metric_for_raw(path, gold, "binary") for name, path in binary_paths.items()}
    binary_aligned, binary_agreed = _agreed_metric(list(binary_paths.values()), gold, "binary")
    binary_metrics = {"Agreed": binary_agreed, **binary_metrics}
    table4 = _f1_table(binary_metrics, ["Agreed", "Llama", "Mistral", "Phi", "Qwen"])
    table4.to_csv(output_dir / "table4_binary_f1.csv", index=False)
    binary_aligned.to_csv(output_dir / "binary_agreed_aligned.csv", index=False)

    # Full codebook descriptions -----------------------------------------
    desc_paths = {name: root / "descriptions" / slug / "raw.csv" for name, slug in LLM_MODELS.items()}
    desc_metrics = {name: _metric_for_raw(path, gold, "descriptions") for name, path in desc_paths.items()}
    _, desc_agreed = _agreed_metric(list(desc_paths.values()), gold, "descriptions")
    desc_metrics = {"Agreed": desc_agreed, **desc_metrics}
    table5 = _f1_table(desc_metrics, ["Agreed", "Llama", "Mistral", "Phi", "Qwen"])
    table5.to_csv(output_dir / "table5_descriptions_f1.csv", index=False)

    # Sentence similarity -------------------------------------------------
    sim_paths = {name: root / "similarity" / slug / "raw.csv" for name, slug in SIM_MODELS.items()}
    sim_metrics = {name: _metric_for_raw(path, gold, "binary") for name, path in sim_paths.items()}
    _, sim_agreed = _agreed_metric(list(sim_paths.values()), gold, "binary")
    sim_metrics = {"Agreed": sim_agreed, **sim_metrics}
    table6 = _f1_table(sim_metrics, ["Agreed", "RoBERTa", "MiniLM", "MPNet", "multi-QA"])
    table6.to_csv(output_dir / "table6_similarity_f1.csv", index=False)

    # Table 3: class-level accuracy in the paper corresponds to the
    # class-specific recall values used in the original evaluation outputs.
    table3_rows = []
    for cid, frame in enumerate(FRAME_NAMES, 1):
        table3_rows.append({
            "Class": frame,
            "BERT model (Liu et al., 2019)": BERT_BENCHMARK_RECALL[cid - 1],
            "Binary (Agreed)": _class_value(binary_agreed, cid, "recall"),
            "Descriptions (Agreed)": _class_value(desc_agreed, cid, "recall"),
            "Similarity (Agreed)": _class_value(sim_agreed, cid, "recall"),
        })
    table3_rows.append({
        "Class": "Overall accuracy",
        "BERT model (Liu et al., 2019)": BERT_BENCHMARK_ACCURACY,
        "Binary (Agreed)": binary_agreed["accuracy"],
        "Descriptions (Agreed)": desc_agreed["accuracy"],
        "Similarity (Agreed)": sim_agreed["accuracy"],
    })
    table3_rows.append({
        "Class": "Cohen's Kappa",
        "BERT model (Liu et al., 2019)": None,
        "Binary (Agreed)": binary_agreed["cohen_kappa"],
        "Descriptions (Agreed)": desc_agreed["cohen_kappa"],
        "Similarity (Agreed)": sim_agreed["cohen_kappa"],
    })
    table3_rows.append({
        "Class": "95% CI Cohen's Kappa",
        "BERT model (Liu et al., 2019)": None,
        "Binary (Agreed)": str(binary_agreed["kappa_ci95"]),
        "Descriptions (Agreed)": str(desc_agreed["kappa_ci95"]),
        "Similarity (Agreed)": str(sim_agreed["kappa_ci95"]),
    })
    pd.DataFrame(table3_rows).to_csv(output_dir / "table3_comparison.csv", index=False)

    # Appendix C: precision, recall and F1 for agreed binary predictions.
    appendix_rows = []
    for cid, frame in enumerate(FRAME_NAMES, 1):
        pc = binary_agreed["per_class"][str(cid)]
        appendix_rows.append({"Class": frame, "Precision": pc["precision"], "Recall": pc["recall"], "F1": pc["f1-score"]})
    for label, key in [("Macro Average", "macro avg"), ("Weighted Average", "weighted avg")]:
        pc = binary_agreed["per_class"][key]
        appendix_rows.append({"Class": label, "Precision": pc["precision"], "Recall": pc["recall"], "F1": pc["f1-score"]})
    pd.DataFrame(appendix_rows).to_csv(output_dir / "appendix_c_precision_recall_f1.csv", index=False)

    # Machine-readable summary.
    summary = {
        "binary": {k: v for k, v in binary_agreed.items() if k != "per_class"},
        "descriptions": {k: v for k, v in desc_agreed.items() if k != "per_class"},
        "similarity": {k: v for k, v in sim_agreed.items() if k != "per_class"},
    }
    with (output_dir / "summary_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote Tables 3-6 and Appendix C to: {output_dir}")
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    generate(a.results_root, a.data, a.output_dir)


if __name__ == "__main__":
    main()
