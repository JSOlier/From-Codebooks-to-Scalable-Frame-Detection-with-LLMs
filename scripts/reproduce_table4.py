"""Recalculate Table 4 metrics from the archived final binary-question predictions."""
from pathlib import Path
import json
import pandas as pd
from framing_repro.data import load_gvfc
from framing_repro.evaluate import evaluate
from framing_repro.codebook import ID_TO_FRAME

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archived_results" / "binary_99_statements"
DATA = ROOT / "data" / "GVFC_headlines_and_annotations.csv"
MODELS = {
    "Agreed": ARCHIVE / "Agreement" / "framing_results.csv",
    "Llama": ARCHIVE / "Gun_Violence" / "Meta-Llama-3-8B-Instruct" / "framing_results.csv",
    "Mistral": ARCHIVE / "Gun_Violence" / "Mistral-7B-Instruct-v0.2" / "framing_results.csv",
    "Phi": ARCHIVE / "Gun_Violence" / "Phi-3.5-mini-instruct" / "framing_results.csv",
    "Qwen": ARCHIVE / "Gun_Violence" / "Qwen1.5-7B-Chat" / "framing_results.csv",
}

def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    return df.rename(columns={"Q3_1": "pred_1", "Q3_2": "pred_2"})[["news_title", "pred_1", "pred_2"]].fillna(99)

def main():
    gold = load_gvfc(DATA, historical=True)
    results = {}
    per_class = {}
    for name, path in MODELS.items():
        _, metrics = evaluate(load_predictions(path), gold)
        results[name] = {
            "n": metrics["n"], "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"], "cohen_kappa": metrics["cohen_kappa"],
            "kappa_ci95": list(metrics["kappa_ci95"]),
        }
        per_class[name] = {ID_TO_FRAME[i]: metrics["per_class"][str(i)]["f1-score"] for i in range(1, 10)}
    table = pd.DataFrame(per_class)
    table.loc["Weighted average F1"] = [results[c]["weighted_f1"] for c in table.columns]
    table.loc["Accuracy"] = [results[c]["accuracy"] for c in table.columns]
    outdir = ROOT / "outputs"; outdir.mkdir(exist_ok=True)
    table.to_csv(outdir / "table4_reproduced.csv")
    (outdir / "table4_metrics.json").write_text(json.dumps(results, indent=2))
    print(table.round(2).to_string())
    print("\nAgreed exact metrics:")
    print(json.dumps(results["Agreed"], indent=2))

if __name__ == "__main__":
    main()
