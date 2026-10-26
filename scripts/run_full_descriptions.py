"""Run the one-full-description-per-frame LLM comparison."""
import argparse
from framing_repro.data import load_gvfc
from framing_repro.llm import annotate, full_codebook_descriptions

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
p.add_argument("--limit", type=int)
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--revision")
p.add_argument("--all-relevant", action="store_true")
a = p.parse_args()

df = load_gvfc(a.data, historical=not a.all_relevant)
if a.limit:
    df = df.head(a.limit)
annotate(
    a.model,
    df.news_title.tolist(),
    full_codebook_descriptions(),
    a.output,
    batch_size=a.batch_size,
    revision=a.revision,
)
