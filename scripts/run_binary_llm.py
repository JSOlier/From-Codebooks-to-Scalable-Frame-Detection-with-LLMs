"""Run the paper-facing 91-indicator LLM experiment."""
import argparse
from framing_repro.data import load_gvfc
from framing_repro.llm import annotate, binary_descriptions

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
p.add_argument("--limit", type=int)
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--revision", help="Optional Hugging Face model revision/commit to pin a rerun")
p.add_argument("--all-relevant", action="store_true", help="Use all relevant rows instead of the historical duplicate-removal subset")
a = p.parse_args()

df = load_gvfc(a.data, historical=not a.all_relevant)
if a.limit:
    df = df.head(a.limit)
annotate(
    a.model,
    df.news_title.tolist(),
    binary_descriptions(),
    a.output,
    batch_size=a.batch_size,
    revision=a.revision,
)
