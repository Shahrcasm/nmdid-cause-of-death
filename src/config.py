import os
from pathlib import Path

# Folders, found relative to this file, so it works on any computer.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

CARTS = ["cart2960", "cart2966", "cart2975"]
KEY = "deidentified_record_number"   # links all tables to one person
SEED = 42                             # same random choices every run

# Set NMDID_FAST=1 for a quick smoke test of the whole pipeline (1 CV repeat,
# 50 bootstrap draws). Published results use 5 repeats and 1,000 draws.
FAST = os.environ.get("NMDID_FAST") == "1"
N_REPEATS = 1 if FAST else 5
N_BOOT = 50 if FAST else 1000

# Cause-of-death groups, exact strings.
CARDIAC = {"Cardiac arrhythmia",
           "Hypertension (hypertensive cardiovascular disease)"}
DRUG = {"Substance intoxication (drug, poison, alcohol, etc.)",
        "Ethanol (alcohol) intoxication"}
INJURY = {"Gunshot wound of", "Stab wound (slash, penetrating cut)", "Hanging",
          "Multiple injuries (fractures, lacerations to internal organs)",
          "Head and neck injuries (cervical, cranio-, cerebral)"}

# Never use as model inputs (post-mortem, administrative, or leakage columns).
FORBIDDEN_INPUTS = [
    "id", KEY, "source_cart",
    "primary_cause_of_death", "manner_of_death",
    "identification_method_code", "environment_cadaver_condition_code",
    "death_city", "death_county", "decomposition_score", "settings",
    "time_delay_after_death", "death_year", "zip_code",
    "gender_code", "birthweight_category", "bone_density",
]

# ---- Columns stored in features.csv that are NEVER part of the main input sets M0-M2 ----
# Medications are partly scene evidence ("taken at death or found in the residence"),
# so they enter only the extended-record analysis of the accuracy ceiling.
EXTENDED_INPUTS = ["n_medications", "any_medication", "med_none_recorded"]
# Race and ethnicity are used to check that performance is similar across groups,
# and in the accuracy-ceiling analysis. They are not inputs of the main models.
DEMOGRAPHIC_EXTRA = ["race_native_american", "race_black", "race_other", "ethnicity_hispanic"]
# Marker of a family (next-of-kin) interview: used in the data audit only.
AUDIT_ONLY = ["nok_interview"]
# The decomposition-blanked cadaver BMI depends on decomposition_score (a blocked
# post-mortem variable), so it is a sensitivity analysis only.
SENSITIVITY_ONLY = ["cadaver_bmi_decomp_blanked", "cadaver_bmi_decomp_blanked_missing"]
NON_MODEL_COLUMNS = EXTENDED_INPUTS + DEMOGRAPHIC_EXTRA + AUDIT_ONLY + SENSITIVITY_ONLY

if __name__ == "__main__":
    print("Raw folder exists:", RAW_DIR.exists())
    for cart in CARTS:
        print(cart, "CSV files:", len(list((RAW_DIR / cart).glob("*.csv"))))
    print("Forbidden columns:", len(FORBIDDEN_INPUTS))
    print("Fast mode:", FAST, "| CV repeats:", N_REPEATS, "| bootstrap draws:", N_BOOT)
