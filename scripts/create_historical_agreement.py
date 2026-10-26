"""Recreate the original Agreement folder from four historical raw result files."""
import argparse
from pathlib import Path
from framing_repro.aggregation import historical_agreement

p = argparse.ArgumentParser()
p.add_argument("--raw", nargs="+", required=True, help="Model raw_results.csv files, in any model order")
p.add_argument("--output-dir", required=True)
a = p.parse_args()

raw, framing, mapping = historical_agreement(a.raw)
out = Path(a.output_dir)
out.mkdir(parents=True, exist_ok=True)
raw.to_csv(out / "raw_results.csv", sep=";", index=False)
framing.to_csv(out / "framing_results.csv", sep=";", index=False)
mapping.set_index("ID").to_csv(out / "ID_to_statement.csv", sep=";")
print(f"Retained {len(mapping)} of 99 statements; wrote {len(framing)} headline predictions to {out}")
