# Data

`GVFC_headlines_and_annotations.csv` is the working copy present in the research archive supplied for this project. It originates from the Gun Violence Frame Corpus (GVFC) introduced by Liu et al. (2019), DOI `10.18653/v1/K19-1047`.

Important audit notes:

- this supplied CSV contains **1,301** rows with `Q3 Theme1 < 99`, whereas the final article states that 1,300 relevant headlines were used;
- the archived final prediction files match **1,293** titles during evaluation;
- the original working loader removed all occurrences of duplicated headline strings (`keep=False`), producing **1,293** relevant titles in this snapshot;
- `load_gvfc(..., historical=True)` reproduces that historical selection explicitly, while `historical=False` retains all currently relevant rows.

Before public redistribution, review `THIRD_PARTY_NOTICES.md` and confirm that redistribution of this GVFC snapshot is permitted under the dataset's current terms. If it is not, remove the CSV from the Zenodo deposit and replace it with retrieval instructions; users will then need to obtain GVFC separately before running the reproduction script.
