"""Run the dynamic-threshold, 99-statement similarity code preserved in the original project."""
import argparse
from framing_repro.data import load_gvfc
from framing_repro.similarity import annotate_similarity_historical

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--model", required=True)
p.add_argument("--output-dir", required=True)
p.add_argument("--limit", type=int)
a = p.parse_args()

df = load_gvfc(a.data, historical=True)
if a.limit:
    df = df.head(a.limit)
th = annotate_similarity_historical(a.model, df.news_title.tolist(), a.output_dir)
print(f"Historical dynamic threshold: {th:.6f}")
