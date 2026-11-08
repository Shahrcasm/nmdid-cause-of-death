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

if __name__ == "__main__":
    print("Raw folder exists:", RAW_DIR.exists())
    for cart in CARTS:
        print(cart, "CSV files:", len(list((RAW_DIR / cart).glob("*.csv"))))
    print("Forbidden columns:", len(FORBIDDEN_INPUTS))