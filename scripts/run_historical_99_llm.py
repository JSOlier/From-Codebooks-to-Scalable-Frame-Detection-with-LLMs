"""Rerun the original 99-statement LLM experiment as closely as preserved source permits."""
import argparse
from framing_repro.data import load_gvfc
from framing_repro.llm import annotate_historical_99

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--model", required=True)
p.add_argument("--output-dir", required=True)
p.add_argument("--limit", type=int)
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--revision", help="Optional HF revision. Historical revisions were not recorded in the original source.")
a = p.parse_args()

df = load_gvfc(a.data, historical=True)
if a.limit:
    df = df.head(a.limit)
annotate_historical_99(
    a.model,
    df.news_title.tolist(),
    a.output_dir,
    batch_size=a.batch_size,
    revision=a.revision,
)
