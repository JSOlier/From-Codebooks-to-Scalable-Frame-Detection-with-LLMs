"""Run all paper-facing experiments and regenerate Tables 3-6 in one command.

This orchestrator deliberately launches each model in a separate Python
process. That releases GPU memory between models and makes the workflow more
robust on workstations with a single GPU.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

LLM_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "microsoft/Phi-3.5-mini-instruct",
    "Qwen/Qwen1.5-7B-Chat",
]
SIMILARITY_MODELS = [
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
]


def slug(model: str) -> str:
    return model.replace("/", "_")


def run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("\n$ " + " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(
        description="Run binary-question, full-description, and sentence-similarity experiments, then regenerate Tables 3-6."
    )
    p.add_argument("--data", type=Path, default=project_root / "data" / "GVFC_headlines_and_annotations.csv")
    p.add_argument("--output-dir", type=Path, default=project_root / "outputs" / "full_experiment")
    p.add_argument("--batch-size", type=int, default=8, help="LLM text-generation batch size")
    p.add_argument("--limit", type=int, help="Run only the first N historical headlines (smoke testing; tables will not match the paper)")
    p.add_argument("--skip-existing", action="store_true", help="Reuse an existing raw.csv instead of rerunning that model")
    p.add_argument("--dry-run", action="store_true", help="Print the commands without executing them")
    p.add_argument("--all-relevant", action="store_true", help="Use all relevant rows instead of the historical 1,293-headline subset")
    p.add_argument("--similarity-threshold", type=float, default=0.5)
    a = p.parse_args()

    scripts = project_root / "scripts"
    out = a.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Record exactly what this orchestrated run requested.
    manifest = {
        "data": str(a.data.resolve()),
        "llm_models": LLM_MODELS,
        "similarity_models": SIMILARITY_MODELS,
        "batch_size": a.batch_size,
        "limit": a.limit,
        "historical_dataset_selection": not a.all_relevant,
        "similarity_threshold": a.similarity_threshold,
        "note": "Exact historical Hugging Face revision hashes were not recorded in the original project.",
    }
    if not a.dry_run:
        (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    common = ["--data", str(a.data.resolve())]
    if a.limit is not None:
        common += ["--limit", str(a.limit)]
    if a.all_relevant:
        common += ["--all-relevant"]

    # 1) Binary indicator questions (91 published frame-specific items).
    for model in LLM_MODELS:
        raw = out / "binary" / slug(model) / "raw.csv"
        if a.skip_existing and raw.exists():
            print(f"Skipping existing {raw}")
            continue
        raw.parent.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, str(scripts / "run_binary_llm.py"),
            *common,
            "--model", model,
            "--output", str(raw),
            "--batch-size", str(a.batch_size),
        ], dry_run=a.dry_run)

    # 2) Full codebook descriptions (one question per each of nine frames).
    for model in LLM_MODELS:
        raw = out / "descriptions" / slug(model) / "raw.csv"
        if a.skip_existing and raw.exists():
            print(f"Skipping existing {raw}")
            continue
        raw.parent.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, str(scripts / "run_full_descriptions.py"),
            *common,
            "--model", model,
            "--output", str(raw),
            "--batch-size", str(a.batch_size),
        ], dry_run=a.dry_run)

    # 3) Sentence-similarity models, fixed threshold 0.5 as described in paper.
    for model in SIMILARITY_MODELS:
        raw = out / "similarity" / slug(model) / "raw.csv"
        if a.skip_existing and raw.exists():
            print(f"Skipping existing {raw}")
            continue
        raw.parent.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, str(scripts / "run_similarity.py"),
            *common,
            "--model", model,
            "--output", str(raw),
            "--threshold", str(a.similarity_threshold),
        ], dry_run=a.dry_run)

    # 4) Generate all paper tables from the freshly created outputs.
    run([
        sys.executable, str(scripts / "generate_all_tables.py"),
        "--results-root", str(out),
        "--data", str(a.data.resolve()),
        "--output-dir", str(out / "tables"),
    ], dry_run=a.dry_run)

    if not a.dry_run:
        print("\nComplete.")
        print(f"Results: {out}")
        print(f"Tables:  {out / 'tables'}")


if __name__ == "__main__":
    main()
