import argparse
import json
from framing_repro.aggregation import majority_vote
from framing_repro.data import load_gvfc
from framing_repro.evaluate import add_ranked_predictions, evaluate

p = argparse.ArgumentParser()
p.add_argument("--raw", nargs="+", required=True)
p.add_argument("--data", required=True)
p.add_argument("--output-prefix", required=True)
p.add_argument("--threshold", type=int, default=2)
p.add_argument("--mode", choices=["binary", "descriptions"], default="binary")
p.add_argument("--all-relevant", action="store_true", help="Do not apply the original duplicate-title removal")
a = p.parse_args()

raw = majority_vote(a.raw, a.threshold)
raw.to_csv(a.output_prefix + "_agreed_raw.csv", sep=";", index=False)
pred = add_ranked_predictions(raw, a.mode)
pred.to_csv(a.output_prefix + "_predictions.csv", sep=";", index=False)
gold = load_gvfc(a.data, historical=not a.all_relevant)
aligned, metrics = evaluate(pred, gold)
aligned.to_csv(a.output_prefix + "_aligned.csv", sep=";", index=False)
with open(a.output_prefix + "_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))
