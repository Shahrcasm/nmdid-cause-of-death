# Lifestyle and cause of death in medicolegal deaths (NMDID metadata)

CSE 499A Capstone, North South University.

Does knowing how a person lived (drinking, smoking, drug use, illnesses, marital
status, body size) help estimate how they died? For which kinds of death does it
help, and for which doesn't it? We use metadata from the New Mexico Decedent Image
Database (NMDID). CT imaging may be added in a later project.

## Research questions
- **RQ1:** How much does lifestyle (and medical history) add beyond age and sex?
- **RQ2:** Which lifestyle factors matter, and do they differ by sex and age?
- **RQ3:** Is the signal real lifestyle information or a recording artifact
  (which fields happen to be blank)?

Three levels: (1) natural cardiac vs drug/alcohol vs injury; (2) within injury:
suicide vs homicide vs accident; (3) negative control: homicide gunshot vs stab.

## Data (not in this repo)
NMDID data are shared under NMDID data-use terms and must never be committed or
uploaded. Get the 3 carts through your own NMDID access and place them as:

```
data/raw/cart2960/   (8 CSV files)
data/raw/cart2966/
data/raw/cart2975/
```

## Setup (Windows, Python 3.11)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Layout
- `src/`: pipeline scripts (`config.py`, `01_build_dataset.py`, ...), run in order
- `data/raw/`: raw carts (git-ignored, never edit)
- `data/processed/`: generated tables (git-ignored)
- `results/tables/`, `results/figures/`: aggregate summary outputs only

## Status
- Phase 0 (repository cleanup for the new direction): done
- Phase 1 (data pipeline and first Level-1 results): next

The previous direction (multimodal benchmark) is preserved in the git tag
`old-direction-backup`.
