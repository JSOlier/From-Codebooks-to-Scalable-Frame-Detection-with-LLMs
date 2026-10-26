"""Run the paper-facing fixed-threshold sentence-similarity baseline."""
import argparse
from framing_repro.data import load_gvfc
from framing_repro.similarity import annotate_similarity

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
p.add_argument("--threshold", type=float, default=0.5)
p.add_argument("--limit", type=int)
p.add_argument("--all-relevant", action="store_true")
a = p.parse_args()

df = load_gvfc(a.data, historical=not a.all_relevant)
if a.limit:
    df = df.head(a.limit)
annotate_similarity(a.model, df.news_title.tolist(), a.output, a.threshold)
