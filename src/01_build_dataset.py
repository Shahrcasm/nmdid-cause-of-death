import pandas as pd
import config

# The 8 tables that every cart contains.
TABLES = ["decedents", "env-history", "medical-history", "substance-usages",
          "medications", "ancestry", "distinctions", "chromosomal-abnormalities"]

# Exact answer texts in NMDID metadata tables.
DRINK_ITEMS = ["High risk", "Low risk", "Never drank", "Previous High risk",
               "Drinker, current status unknown", "Unknown if ever drank"]

TOB_TYPE_ITEMS = ["smoke tobacco", "chew tobacco", "Never used tobacco",
                  "tobacco user, type unknown", "Unknown if ever used tobacco"]

TOB_USE_ITEMS = ["Heavy tobacco user", "Current everyday tobacco user",
                 "Current some day tobacco user", "light tobacco user",
                 "Former tobacco user", "tobacco user, current status unknown"]

DRUG_CURRENT = ["Cannabinoids", "Opiods", "Cocaine", "Other Stimulants",
                "Sedative hypnotics/depressants", "Hallucinogens",
                "Other psychoactive substances", "Drug user, type unknown"]

MARITAL_ITEMS = ["Married- Spouse in household", "Married-spouse not in household",
                 "Living as married", "Divorced", "Separated", "Widowed",
                 "Never married"]

EDU_ORDER = {
    "8th grade/less": 1,
    "9th-12th, no diploma": 2,
    "High school graduate or GED diploma": 3,
    "Some college credit but no degree": 4,
    "Associate degree": 5,
    "Bachelor's degree": 6,
    "Master's degree": 7,
    "Doctorate or professional degree": 8,
}

SES_ORDER = {
    "Lower class": 1,
    "Lower middle class": 2,
    "Middle class": 3,
    "Upper middle class": 4,
    "Upper class": 5,
}

DIET_ITEMS = ["Normal", "Convenience food", "Weight reduction", "Overeating", "Vegetarian"]
LIFT_ITEMS = ["No", "Yes"]
FIREARM_OCC_ITEMS = [
    "Armed Forces Occupations, Other Ranks",
    "Non-commissioned Armed Forces Officers",
    "Protective Services Workers",
]

DX_ITEMS = [
    "Hypertension", "Mental illness", "Diabetes Type II", "Diabetes Type I",
    "Coronary artery disease", "Hyperlipedimia", "COPD", "Asthma",
    "Hepatitis C", "Cirrhosis of the liver", "Stroke", "Chronic heart failure",
    "Non-epileptic seizures", "Epilepsy", "Schizophrenia", "Dementia",
    "Myocardial infarction", "HIV/AIDS",
]

MED_OTHER_TYPES = ["Broken Bone", "Facial Trauma", "Surgery", "Implant"]

NOK_ENV_TYPES = [
    "Educational Level", "Socio Economic Status - Adult",
    "Socio Economic Status - Child", "Dietary Pattern",
    "Activities", "Strenuous Lifting",
]

# Keywords used to read the "manner of death" text.
UNCLEAR_WORDS = ["undetermined", "-unknown", "- unknown", "slashed ___",
                 "blunt trauma/multiple", "skeletal", "explosion"]
SUICIDE_WORDS = ["self with", "suicide", "jumped from", "drowned self"]
HOMICIDE_WORDS = ["by assailant", "assaulted with", "homicide"]


def clean_text(value):
    """Remove extra spaces. Example: 'Previous High risk ' -> 'Previous High risk'."""
    if pd.isna(value):
        return value
    words = str(value).split()
    cleaned = " ".join(words)
    if cleaned == "":
        return pd.NA
    return cleaned


def load_table(name):
    """Read one table from all 3 carts and stack them into one table."""
    parts = []
    for cart in config.CARTS:
        file_path = config.RAW_DIR / cart / f"{name}.csv"
        df = pd.read_csv(file_path, dtype=str, keep_default_na=False, na_values=[""])
        for column in df.columns:
            df[column] = df[column].map(clean_text)
        df["source_cart"] = cart
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def load_all():
    """Load all tables and keep each person only once."""
    tables = {}
    for name in TABLES:
        tables[name] = load_table(name)

    decedents = tables["decedents"]
    print("Decedent rows across 3 carts:", len(decedents))

    home_cart = decedents.drop_duplicates(subset=config.KEY, keep="first")
    home_cart = home_cart[[config.KEY, "source_cart"]]

    for name in TABLES:
        tables[name] = tables[name].merge(home_cart, on=[config.KEY, "source_cart"])

    print("Unique decedents:", tables["decedents"][config.KEY].nunique(),
          "| rows:", len(tables["decedents"]))
    return tables


def contains_any(text, words):
    """True if the text contains at least one of the words."""
    for word in words:
        if word in text:
            return True
    return False


def level1(cause, manner):
    """Kind of death."""
    if pd.isna(cause):
        return "other"
    if cause in config.CARDIAC:
        if not pd.isna(manner) and manner == "Natural":
            return "natural_cardiac"
        return "cardiac_conflict"
    if cause in config.DRUG:
        return "drug_alcohol"
    if cause in config.INJURY:
        return "injury"
    return "other"


def intent(manner):
    """Who caused the death, read from the manner text."""
    if pd.isna(manner):
        return "unclear"
    text = str(manner).strip().lower()
    if text == "natural":
        return "natural"
    if contains_any(text, UNCLEAR_WORDS):
        return "unclear"
    if text in ["gunshot wound", "nan", "other"]:
        return "unclear"
    if contains_any(text, SUICIDE_WORDS):
        return "suicide"
    if contains_any(text, HOMICIDE_WORDS):
        return "homicide"
    return "accident"


def build_cohort(decedents):
    """Add the two labels to every person, then keep adults only."""
    cohort = decedents.copy()
    cohort["age_years"] = pd.to_numeric(cohort["age_years"], errors="coerce")

    level1_labels = []
    intent_labels = []
    for index, row in cohort.iterrows():
        level1_labels.append(level1(row["primary_cause_of_death"], row["manner_of_death"]))
        intent_labels.append(intent(row["manner_of_death"]))
    cohort["level1"] = level1_labels
    cohort["intent"] = intent_labels

    under_18 = int((cohort["age_years"] < 18).sum())
    missing_age = int(cohort["age_years"].isna().sum())
    print("Under 18:", under_18, "| missing age:", missing_age)

    cohort = cohort[cohort["age_years"] >= 18].copy()
    print("Adults:", len(cohort))
    return cohort


def make_column_name(prefix, item):
    """Example: ('drink_', 'High risk') -> 'drink_high_risk'."""
    name = item.lower()
    for character in [" ", ",", "/", "-", "(", ")", "'"]:
        name = name.replace(character, "_")
    while "__" in name:
        name = name.replace("__", "_")
    return prefix + name.strip("_")


def add_yes_no_columns(features, rows, value_column, items, prefix):
    """Add one 0/1 column per item: 1 = the person has this answer."""
    for item in items:
        people_with_item = rows[rows[value_column] == item][config.KEY]
        column_name = make_column_name(prefix, item)
        features[column_name] = features[config.KEY].isin(people_with_item).astype(int)


def has_real_answer(features, rows, value_column, unknown_values):
    """True if the person has at least one informative answer."""
    is_real = rows[value_column].notna() & ~rows[value_column].isin(unknown_values)
    people_with_answer = rows[is_real][config.KEY]
    return features[config.KEY].isin(people_with_answer)


def build_base(cohort):
    """Base block: age_years and male."""
    features = pd.DataFrame({config.KEY: cohort[config.KEY].values})
    features["age_years"] = cohort["age_years"].values
    features["male"] = (cohort["sex_code"] == "Male").astype(int).values
    return features


def build_lifestyle(tables, people):
    """Drinking, tobacco and drug-use columns, one row per person."""
    features = pd.DataFrame({config.KEY: people})
    env = tables["env-history"]
    substances = tables["substance-usages"]
    item = "environmental_factor_item"
    drug = "substance_usage_type"

    drinking = env[env["environmental_factor_type"] == "Drinking Status"]
    tobacco_type = env[env["environmental_factor_type"] == "Tobacco Type"]
    tobacco_usage = env[env["environmental_factor_type"] == "Tobacco Usage"]
    tobacco_all = pd.concat([tobacco_type, tobacco_usage])

    add_yes_no_columns(features, drinking, item, DRINK_ITEMS, "drink_")
    add_yes_no_columns(features, tobacco_type, item, TOB_TYPE_ITEMS, "tobtype_")
    add_yes_no_columns(features, tobacco_usage, item, TOB_USE_ITEMS, "tobuse_")
    add_yes_no_columns(features, substances, drug, DRUG_CURRENT, "drug_")

    is_former = substances[drug].str.startswith("Former", na=False)
    former_users = substances[is_former][config.KEY]
    features["drug_former"] = features[config.KEY].isin(former_users).astype(int)

    drink_known = has_real_answer(features, drinking, item, ["Unknown if ever drank"])
    tob_known = has_real_answer(features, tobacco_all, item, ["Unknown if ever used tobacco"])
    drug_known = has_real_answer(features, substances, drug, ["Unknown if used drugs"])
    features["drink_missing"] = (~drink_known).astype(int)
    features["tob_missing"] = (~tob_known).astype(int)
    features["drug_missing"] = (~drug_known).astype(int)
    return features


def compute_bmi(weight_series, height_cm_series):
    """Calculate BMI = weight_kg / (height_m ^ 2), blanking impossible adult values."""
    weight = pd.to_numeric(weight_series, errors="coerce")
    height_cm = pd.to_numeric(height_cm_series, errors="coerce")
    height_cm = height_cm.where((height_cm >= 120) & (height_cm <= 220))
    bmi = weight / ((height_cm / 100.0) ** 2)
    bmi = bmi.where((bmi >= 12) & (bmi <= 70))
    return bmi


def build_marital_and_bmi(cohort):
    """Marital status and body size (BMI) columns from decedents table."""
    features = pd.DataFrame({config.KEY: cohort[config.KEY].values})

    add_yes_no_columns(features, cohort, "marital_status_code", MARITAL_ITEMS, "marital_")
    marital_known = has_real_answer(
        features, cohort, "marital_status_code", ["Blank but applicable", "Unknown"]
    )
    features["marital_missing"] = (~marital_known).astype(int)

    cad_bmi = compute_bmi(cohort["cadaver_weight"], cohort["cadaver_length"])
    bad_decomp = cohort["decomposition_score"].isin(
        ["moderate decomposition", "advanced decomposition"]
    )
    cad_bmi = cad_bmi.where(~bad_decomp)
    features["cadaver_bmi"] = cad_bmi.values
    features["cadaver_bmi_missing"] = features["cadaver_bmi"].isna().astype(int)

    liv_bmi = compute_bmi(cohort["living_weight"], cohort["living_height"])
    features["living_bmi"] = liv_bmi.values
    features["living_bmi_missing"] = features["living_bmi"].isna().astype(int)
    return features


def map_ordinal_column(features, rows, order_map, col_name):
    """Turn ordered text answers into numbers (1, 2, 3...) and add a _missing flag."""
    item_col = "environmental_factor_item"
    mapped = rows.copy()
    mapped["score"] = mapped[item_col].map(order_map)
    per_person = mapped.dropna(subset=["score"]).groupby(config.KEY)["score"].max()
    features[col_name] = features[config.KEY].map(per_person)
    features[col_name + "_missing"] = features[col_name].isna().astype(int)


def build_extended_lifestyle(tables, people):
    """Education, SES, Diet, Strenuous Lifting, and Firearm Occupations."""
    features = pd.DataFrame({config.KEY: people})
    env = tables["env-history"]
    item = "environmental_factor_item"

    edu = env[env["environmental_factor_type"] == "Educational Level"]
    ses_adult = env[env["environmental_factor_type"] == "Socio Economic Status - Adult"]
    ses_child = env[env["environmental_factor_type"] == "Socio Economic Status - Child"]
    diet = env[env["environmental_factor_type"] == "Dietary Pattern"]
    lift = env[env["environmental_factor_type"] == "Strenuous Lifting"]
    occ = env[env["environmental_factor_type"] == "Occupations"]

    map_ordinal_column(features, edu, EDU_ORDER, "edu_level")
    map_ordinal_column(features, ses_adult, SES_ORDER, "ses_adult")
    map_ordinal_column(features, ses_child, SES_ORDER, "ses_child")

    add_yes_no_columns(features, diet, item, DIET_ITEMS, "diet_")
    diet_known = has_real_answer(features, diet, item, ["Unknown"])
    features["diet_missing"] = (~diet_known).astype(int)

    add_yes_no_columns(features, lift, item, LIFT_ITEMS, "lift_")
    lift_known = has_real_answer(features, lift, item, ["Unknown"])
    features["lift_missing"] = (~lift_known).astype(int)

    firearm_occ_people = occ[occ[item].isin(FIREARM_OCC_ITEMS)][config.KEY]
    features["occ_firearm_access"] = features[config.KEY].isin(firearm_occ_people).astype(int)
    occ_known = has_real_answer(features, occ, item, ["Unknown"])
    features["occ_missing"] = (~occ_known).astype(int)
    return features


def build_medical_history(tables, people):
    """Diagnoses and lifetime injury/surgery history."""
    features = pd.DataFrame({config.KEY: people})
    med = tables["medical-history"]
    detail = "medical_history_detail"

    dx = med[med["medical_history_type"] == "Diagnosis"]
    add_yes_no_columns(features, dx, detail, DX_ITEMS, "dx_")
    dx_known = has_real_answer(features, dx, detail, ["Unknown"])
    features["dx_missing"] = (~dx_known).astype(int)

    for hist_type in MED_OTHER_TYPES:
        sub = med[med["medical_history_type"] == hist_type]
        short = make_column_name("medhist_", hist_type)
        pos_rows = sub[sub[detail].notna() & ~sub[detail].isin(["None", "No", "Absent", "Unknown"])]
        features[short + "_any"] = features[config.KEY].isin(pos_rows[config.KEY]).astype(int)
        known = has_real_answer(features, sub, detail, ["Unknown"])
        features[short + "_missing"] = (~known).astype(int)
    return features


def build_rq3_extras(tables, cohort):
    """Medications and Family-interview marker for RQ3 analysis."""
    features = pd.DataFrame({config.KEY: cohort[config.KEY].values})

    # Medications block.
    meds = tables["medications"]
    real_meds = meds[meds["drug_number"].notna() & (meds["drug_number"] != "None")]
    med_counts = real_meds.groupby(config.KEY).size()
    features["n_medications"] = features[config.KEY].map(med_counts).fillna(0).astype(int)
    features["any_medication"] = (features["n_medications"] > 0).astype(int)
    features["med_missing"] = (features["any_medication"] == 0).astype(int)

    # Family-interview (NOK) marker.
    env = tables["env-history"]
    nok_env_rows = env[env["environmental_factor_type"].isin(NOK_ENV_TYPES)]
    nok_env_rows = nok_env_rows[nok_env_rows["environmental_factor_item"].notna()]
    has_nok_env = features[config.KEY].isin(nok_env_rows[config.KEY])
    has_gender = cohort["gender_code"].notna().values
    has_birthwt = cohort["birthweight_category"].notna().values
    features["nok_interview"] = (has_nok_env | has_gender | has_birthwt).astype(int)
    return features


if __name__ == "__main__":
    tables = load_all()
    cohort = build_cohort(tables["decedents"])

    people = cohort[config.KEY].values
    base = build_base(cohort)
    life = build_lifestyle(tables, people)
    mbmi = build_marital_and_bmi(cohort)
    ext = build_extended_lifestyle(tables, people)
    medh = build_medical_history(tables, people)
    rq3 = build_rq3_extras(tables, cohort)

    # Combine all blocks into one wide features table (1 row per adult decedent).
    features = base.merge(life, on=config.KEY)
    features = features.merge(mbmi, on=config.KEY)
    features = features.merge(ext, on=config.KEY)
    features = features.merge(medh, on=config.KEY)
    features = features.merge(rq3, on=config.KEY)

    # Verify no forbidden input column leaked into features (except the join key).
    leaked = [col for col in features.columns if col in config.FORBIDDEN_INPUTS and col != config.KEY]
    print("Leaked forbidden columns in features:", leaked)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(config.PROCESSED_DIR / "cohort.csv", index=False)
    features.to_csv(config.PROCESSED_DIR / "features.csv", index=False)

    print("Saved cohort.csv:", cohort.shape)
    print("Saved features.csv:", features.shape)

    # Check RQ3 markers on the Level-1 cohort (n=1380).
    l1_ids = cohort[cohort["level1"].isin(["natural_cardiac", "drug_alcohol", "injury"])][config.KEY]
    f_l1 = features[features[config.KEY].isin(l1_ids)]
    print(f"Medications present (Level-1 cohort): {100 * f_l1['any_medication'].mean():.1f}%")
    print(f"NOK interview marker (Level-1 cohort): {100 * f_l1['nok_interview'].mean():.1f}%")