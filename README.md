# From Codebooks to Scalable Frame Detection with LLMs

Reproducibility code and archived experimental outputs accompanying:

**Olier, J. S. (2026). “From Codebooks to Scalable Frame Detection with LLMs: A Zero-Shot Approach via Binary Questions.” _Computational Communication Research.**

The project implements the three approaches reported in the article:

1. binary indicator questions derived from simplified codebook statements and evaluated with open-source LLMs;
2. one full codebook frame description per frame, evaluated with the same LLMs; and
3. a sentence-similarity baseline.

It also contains the historical final predictions used to reproduce the main binary-question results without rerunning the LLMs.

## Quick reproduction of the main published result

This is the recommended first step. It does **not** require a GPU or model downloads.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
python scripts\reproduce_table4.py
```

If PowerShell blocks environment activation, run once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python scripts/reproduce_table4.py
```

The archived agreed predictions produce approximately:

```text
N                 1293
Accuracy          0.8121
Weighted F1       0.8066
Cohen's kappa     0.7704
95% CI kappa      [0.7446, 0.7969]
```

Rounded to two decimals, the main metrics reproduce the article's reported **0.81 accuracy, 0.81 weighted F1, and 0.77 Cohen's kappa**. The Social/Cultural class is 0.55 under the current `scikit-learn` calculation, whereas the published Table 4 reports 0.56; this small historical reporting difference is documented rather than forced in code.

## One-command full rerun and table regeneration

After installing all optional dependencies, the complete **paper-facing** workflow can be launched with one command:

### Windows PowerShell

```powershell
pip install -e ".[all]"
python scripts\run_all_experiments.py
```

This sequentially runs the four LLMs for the 91 binary indicators, the four LLMs for the full-description comparison, the four sentence-similarity models at the published 0.5 threshold, and then regenerates Tables 3–6 plus Appendix C under:

```text
outputs/full_experiment/tables/
```

The generated files are:

```text
table3_comparison.csv
table4_binary_f1.csv
table5_descriptions_f1.csv
table6_similarity_f1.csv
appendix_c_precision_recall_f1.csv
summary_metrics.json
```

A full rerun is computationally expensive. To resume an interrupted run without repeating model outputs that already exist, use:

```powershell
python scripts\run_all_experiments.py --skip-existing
```

For a quick smoke test of the workflow, use a small subset (the resulting metrics are **not** comparable to the paper):

```powershell
python scripts\run_all_experiments.py --limit 10
```

To inspect the commands without running any models:

```powershell
python scripts\run_all_experiments.py --dry-run
```

Fresh model inference is not guaranteed to reproduce every historical prediction bit-for-bit because the original Hugging Face commit revisions were not recorded and generation used `do_sample=True` (with a near-deterministic temperature). The archived-output path remains the authoritative way to verify the historical main result.

## Two rerun paths

The repository deliberately distinguishes two related but non-identical workflows.

### A. Historical reconstruction

The supplied original research code actually ran **99 statements**: 8 preliminary relevance/episodic/thematic statements followed by the 91 frame-specific indicators. It also removed every duplicated headline string with `keep=False`, yielding the 1,293 titles found in the archived outputs.

To rerun that execution structure:

```powershell
pip install -e ".[llm]"
python scripts\run_historical_99_llm.py `
  --data data\GVFC_headlines_and_annotations.csv `
  --model microsoft/Phi-3.5-mini-instruct `
  --output-dir outputs\historical_phi
```

Repeat for the other three LLMs. Then reconstruct the historical Agreement directory:

```powershell
python scripts\create_historical_agreement.py `
  --raw outputs\historical_llama\raw_results.csv outputs\historical_mistral\raw_results.csv outputs\historical_phi\raw_results.csv outputs\historical_qwen\raw_results.csv `
  --output-dir outputs\historical_agreement
```

The agreement implementation has been regression-tested against the archived results: it recreates the archived `Agreement/raw_results.csv` and `Agreement/framing_results.csv` exactly from the four model-level raw CSV files. Nine zero-variance statement columns are removed by the historical agreement procedure, leaving 90 statement columns in the archived agreement matrix.

### B. Method as described in the final article

The final paper describes the frame-detection stage as **91 frame-specific binary questions**. To run that cleaner paper-facing implementation:

```powershell
pip install -e ".[llm]"
python scripts\run_binary_llm.py `
  --data data\GVFC_headlines_and_annotations.csv `
  --model microsoft/Phi-3.5-mini-instruct `
  --output outputs\phi_binary.csv
```

By default the rerun scripts still use the historical 1,293-title selection so comparisons with the archived run are meaningful. Add `--all-relevant` if you intentionally want every row currently marked relevant in the supplied GVFC snapshot.

After running four models:

```powershell
python scripts\aggregate_and_evaluate.py `
  --data data\GVFC_headlines_and_annotations.csv `
  --raw outputs\phi_binary.csv outputs\llama_binary.csv outputs\mistral_binary.csv outputs\qwen_binary.csv `
  --output-prefix outputs\binary
```

## Installation

Python 3.10 or newer is required. Dependencies are declared in `pyproject.toml`; there is intentionally no duplicate `requirements.txt`.

For archived-result reproduction and evaluation:

```powershell
pip install -e .
```

For LLM inference:

```powershell
pip install -e ".[llm]"
```

This installs PyTorch, Transformers, **Accelerate**, Hugging Face Hub, and tqdm. `accelerate` is required because the LLM runner uses `device_map="auto"`.

For sentence similarity:

```powershell
pip install -e ".[similarity]"
```

For everything:

```powershell
pip install -e ".[all]"
```

For tests:

```powershell
pip install -e ".[dev]"
pytest
```

## Models

The model repositories used in the original source are recorded in `MODEL_MANIFEST.json`. The original project did **not** record Hugging Face commit/revision hashes, so exact historical checkpoint revisions cannot be recovered from the supplied materials. Fresh inference scripts accept `--revision` so future runs can be pinned.

LLMs:

- `microsoft/Phi-3.5-mini-instruct`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.2`
- `Qwen/Qwen1.5-7B-Chat`

For gated checkpoints, authenticate with Hugging Face outside the source code. No token is stored in this repository.

## Full-description comparison

The paper-facing implementation uses one description for each of the nine GVFC frames:

```powershell
python scripts\run_full_descriptions.py `
  --data data\GVFC_headlines_and_annotations.csv `
  --model microsoft/Phi-3.5-mini-instruct `
  --output outputs\phi_descriptions.csv
```

Run all four LLMs and then use `aggregate_and_evaluate.py --mode descriptions`.

## Sentence-similarity baseline

The paper reports a fixed cosine-similarity threshold of 0.5. That implementation is:

```powershell
pip install -e ".[similarity]"
python scripts\run_similarity.py `
  --data data\GVFC_headlines_and_annotations.csv `
  --model all-mpnet-base-v2 `
  --output outputs\mpnet_similarity.csv
```

The supplied historical source code instead used all 99 statements and selected a dataset-dependent threshold equal to `mean(similarity) + std(similarity)`. That historical implementation is preserved separately rather than silently mixed with the published method:

```powershell
python scripts\run_similarity_historical.py `
  --data data\GVFC_headlines_and_annotations.csv `
  --model all-mpnet-base-v2 `
  --output-dir outputs\similarity_historical_mpnet
```

## Tokenizer and generation settings

The paper-facing and historical LLM runners use the published generation parameters:

```text
do_sample=True
temperature=0.0001
top_p=0.01
max_new_tokens=50
```

`run_historical_99_llm.py` additionally preserves the unusual BOS/EOS/PAD and terminator behavior present in the supplied original source as closely as possible. The normal 91-question runner uses the model's native tokenizer IDs while padding with the model EOS token.

## Repository layout

```text
src/framing_repro/          reusable Python implementation
scripts/                    experiment, orchestration, table generation, historical reconstruction, and evaluation entry points
data/                       GVFC working snapshot + provenance note
archived_results/           historical outputs for the final binary experiment
tests/                      integrity + archived-output regression tests
outputs/                    generated files; empty in the archival release
MODEL_MANIFEST.json         model names and revision-status record
LICENSE                     MIT license for repository software
CITATION.cff                citation metadata
.zenodo.json                Zenodo metadata template
ZENODO_UPLOAD.md            deposit checklist
THIRD_PARTY_NOTICES.md      third-party data/model licensing notes
REPRODUCIBILITY_NOTES.md    audit against the supplied original code and final article
```

## Important sample-count note

The supplied GVFC snapshot contains 1,301 rows with `Q3 Theme1 < 99`, while the article states 1,300 relevant headlines. The original loader then removed **all occurrences** of duplicate `news_title` values, producing 1,293 relevant titles. Those 1,293 titles match the archived model outputs and are therefore used by the historical reconstruction path. The discrepancy is documented rather than silently corrected.

## Licensing and third-party materials

Repository software is released under the **MIT License**. This does not relicense GVFC, model checkpoints, or other third-party materials. See `THIRD_PARTY_NOTICES.md` before public deposition, especially regarding redistribution of the included GVFC snapshot.

## Citation

Use `CITATION.cff` for machine-readable citation metadata. When Zenodo publishes this software archive, also cite the Zenodo software DOI assigned to that release.
