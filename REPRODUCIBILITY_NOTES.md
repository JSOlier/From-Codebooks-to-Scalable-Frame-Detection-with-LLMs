# Reproducibility audit

This release was reconstructed by comparing the supplied research code, archived result CSVs, and the final 2026 article. Components unrelated to the publication (migration/Mexico-Colombia experiments, PCA/factor/item-selection exploration, IDE metadata, caches, and obsolete tests) were excluded.

## What is verified

### Published Table 4 metrics

`python scripts/reproduce_table4.py` evaluates the archived historical frame predictions and yields:

```text
N                 1293
Accuracy          0.8120649652
Weighted F1       0.8065987180
Cohen's kappa     0.7703990699
95% CI kappa      [0.7445509443, 0.7968956939]
```

These round to the paper's 0.81 accuracy, 0.81 weighted F1, and 0.77 kappa. The Social/Cultural F1 evaluates to 0.55 with the current stack versus 0.56 printed in Table 4; the code does not alter calculations to force the printed value.

### Historical agreement reconstruction

The original agreement code was recovered and reimplemented in `framing_repro.aggregation.historical_agreement`. Regression tests confirm that, given the four archived 99-statement model `raw_results.csv` files, it recreates both:

- `archived_results/binary_99_statements/Agreement/raw_results.csv`
- `archived_results/binary_99_statements/Agreement/framing_results.csv`

exactly (ignoring CSV dtype inference differences).

The original agreement procedure removed a statement when any annotator/model column had zero variance. Nine statement columns were therefore excluded, leaving 90 S-columns in the historical Agreement raw matrix. Agreement frame decisions were calculated from the unrounded four-model mean at a threshold of 2/4 = 0.5; the written Agreement raw matrix used pandas rounding, as in the original source.

## Historical execution vs. final-paper method

The supplied original experiment code and final paper are not identical in every implementation detail. This repository preserves both paths explicitly.

### Binary LLM method

- **Historical source/run:** 99 statements = 1 Related + 1 Episodic + 6 Thematic + 91 frame-specific items. `scripts/run_historical_99_llm.py` reproduces this structure and the old tokenizer/terminator configuration as closely as the preserved source permits.
- **Final article:** describes 91 frame-specific binary questions. `scripts/run_binary_llm.py` implements this paper-facing form directly.

Both use the Appendix-A-style headline/description prompt and the published generation settings (`do_sample=True`, `temperature=0.0001`, `top_p=0.01`, `max_new_tokens=50`).

### Dataset selection

The original loader executed:

```python
df = df.drop_duplicates(subset=['news_title'], keep=False)
df = df[df['Q3 Theme1'] < 99]
```

With the archived dataset snapshot, this yields **1,293** relevant titles and matches the archived outputs. The same snapshot has **1,301** relevant rows before duplicate removal, whereas the paper states **1,300**. The historical rerun path therefore uses 1,293 titles by default. `--all-relevant` is available on paper-facing scripts when the current snapshot should be used without historical duplicate removal.

### Full-description method

The final repository exposes the paper-facing nine-frame implementation: one full description for each GVFC frame. Earlier working code also contained preliminary Related/Episodic/Thematic categories. Those preliminary categories are not part of the paper's nine-frame comparison.

### Sentence similarity

- **Final article:** fixed cosine threshold = 0.5. This is `scripts/run_similarity.py`.
- **Supplied historical source:** all 99 statements with a threshold computed as global `mean + std`. This is preserved separately as `scripts/run_similarity_historical.py`.

The discrepancy is documented rather than choosing one implementation silently.

## Tokenizer/model loading

The original LLM code used `torch.bfloat16`, batch size 8, model placement through `device_map`, EOS-based padding, and a separate tokenizer whose BOS/EOS/PAD IDs were explicitly set to 1/2/0 before terminator construction. `run_historical_99_llm.py` preserves these behaviors as closely as possible. The cleaner paper-facing runner retains EOS padding but uses model-native tokenizer IDs.

`accelerate` is an explicit `llm` dependency because `device_map="auto"` requires it.

## Model revisions

The preserved code records model repository names but no Hugging Face revision/commit hashes. `MODEL_MANIFEST.json` records this fact. Fresh scripts accept `--revision` so newly archived reruns can be pinned, but the historical revisions cannot be honestly reconstructed from the supplied materials.

## Credentials

The original source contained a credential-looking Hugging Face token. It is not present in this release. Any real token exposed in the historical source should be revoked/rotated before that source or its Git history is published.

## Third-party materials

The MIT license applies only to the repository software. It does not relicense GVFC, Hugging Face checkpoints, or other third-party materials. See `THIRD_PARTY_NOTICES.md` and `data/README.md`.
