# NMDID cause-of-death: metadata baseline

Capstone CSE 499A. Leakage-audited, metadata-only study of cause of death using
the New Mexico Decedent Image Database (NMDID).

This is the metadata arm of a planned multimodal study (PMCT image embeddings +
metadata). The PMCT images are currently unavailable, so we start with metadata only.

## Data (not in this repo)
The data are NOT included, and must never be committed or uploaded.
Get the 3 NMDID carts (2960, 2966, 2975) through your own NMDID access, and follow
the NMDID data-use terms. Do not upload data to Drive or Colab until the terms are checked.

Place them like this:

```
data/raw/NMDID-cart-2960-meta/   (8 CSV files each)
data/raw/NMDID-cart-2966-meta/
data/raw/NMDID-cart-2975-meta/
```

## How to run
1. Create a virtual environment with Python 3.11:
   `py -3.11 -m venv .venv`
2. Activate it: `.\.venv\Scripts\Activate.ps1`
3. Install packages: `pip install -r requirements.txt`
4. Register the kernel:
   `python -m ipykernel install --user --name nmdid --display-name "Python (nmdid)"`
5. Open `notebooks/01_schema_audit.ipynb` with the "Python (nmdid)" kernel.

Before committing a notebook, clear all outputs (they can contain data rows).

## Project layout
- `notebooks/`: analysis notebooks
- `src/`: reusable code (empty for now)
- `data/raw/`: raw carts (git-ignored, never edit)
- `data/processed/`: generated tables (git-ignored)
- `outputs/`: figures and tables (git-ignored)

## Status

### Done (Steps 0-2)
- **Setup:** Python 3.11 venv, requirements, git.
- **Schema audit:** all 3 carts loaded (8 tables each); sizes, missing values and value lists checked.
- **De-duplication:** 1,500 rows are 1,457 unique people. 43 people appear in two carts with identical data; one row per `id` is kept.
- **Master table:** one row per person, 63 columns. Long tables were turned into features that keep 3 states: present, recorded none, not recorded.
- **Exclusions (flagged, not deleted):** 2 people (blank label; unknown sex with missing age).

### Key findings so far
- Carts are cause-selected and capped, so class balance is artificial. Predicted probabilities are not real-world risk.
- `manner_of_death` leaks the label and is excluded from modelling.
- Documentation depth differs by cart (for example, diagnosis records: 26% in the injury cart vs 58% in the cardiac cart). "Has record" flags are shortcuts and will be tested with and without.
- Several columns are almost empty (for example `bone_density`, `years_in_us`).
- Single source (one medical examiner system), so generalisation is limited.

### Next
3. Label map and leakage table (frozen before modelling)
4. EDA and association statistics (FDR-corrected)
5. Leakage-audited baselines (grouped CV; LR, LightGBM, RF)
6. TabPFN, conformal prediction, calibration (Colab, after data-terms check)