# Archived experimental outputs

`binary_99_statements/` contains the historical outputs supplied by the author for the final binary-question experiment.

The historical run used 99 statements: 8 preliminary relevance/episodic/thematic items followed by the 91 GVFC frame-specific indicators reported in Appendix B of the article.

- `Gun_Violence/<model>/raw_results.csv`: historical model-level 99-statement binary outputs.
- `Gun_Violence/<model>/framing_results.csv`: historical model-level frame predictions.
- `Agreement/raw_results.csv`: historical four-model agreement matrix after zero-variance statement removal.
- `Agreement/framing_results.csv`: historical agreed frame predictions used for the main evaluation.
- `Agreement/ID_to_statement.csv`: the 90 retained statement columns.

The historical Agreement files are now **fully reconstructable** from the four model-level raw files using:

```bash
python scripts/create_historical_agreement.py --raw <four raw_results.csv paths> --output-dir outputs/agreement
```

The regression test `test_historical_agreement_reconstructs_archived_files` verifies exact reconstruction of the archived raw and frame-prediction files.

To reproduce Table 4 metrics without LLM inference:

```bash
python scripts/reproduce_table4.py
```
