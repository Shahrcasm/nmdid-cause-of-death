"""Interactive Streamlit Demo UI for CSE 499A Capstone Project:
Lifestyle and Cause of Death in Medicolegal Deaths (NMDID Metadata).

Run from repository root:
    streamlit run app/app.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "demo_models.joblib"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

st.set_page_config(
    page_title="NMDID Lifestyle & Cause of Death Demo",
    page_icon="🧬",
    layout="wide",
)


class AggregateLogRegModel:
    """Lightweight predictor reconstructed from aggregate CSV weights in results/tables/."""
    def __init__(self, df_sub, classes):
        self.cols = df_sub["feature"].tolist()
        self.medians = df_sub["impute_median"].values.astype(float)
        self.means = df_sub["scale_mean"].values.astype(float)
        self.stds = df_sub["scale_std"].values.astype(float)
        self.coefs = np.column_stack([df_sub[f"coef_{c}"].values.astype(float) for c in classes])
        self.intercepts = np.array([float(df_sub[f"intercept_{c}"].iloc[0]) for c in classes])

    def predict_proba(self, X_df):
        x = X_df[self.cols].values.astype(float)
        nan_mask = np.isnan(x)
        if nan_mask.any():
            x = np.where(nan_mask, self.medians, x)
        z = (x - self.means) / np.where(self.stds == 0, 1.0, self.stds)
        logits = z @ self.coefs + self.intercepts
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        return exp_l / exp_l.sum(axis=1, keepdims=True)


@st.cache_resource
def load_bundle():
    if MODEL_PATH.exists():
        b = joblib.load(MODEL_PATH)
        b["mode"] = "HistGradientBoosting (local model)"
        return b
    w_path = TABLES_DIR / "demo_logreg_weights.csv"
    if not w_path.exists():
        return None
    w_df = pd.read_csv(w_path)
    l1_classes = ["natural_cardiac", "drug_alcohol", "injury"]
    l2_classes = ["accident", "homicide", "suicide"]
    m0_df = w_df[w_df["model_id"] == "level1_M0"]
    m2_l1_df = w_df[w_df["model_id"] == "level1_M2"]
    m2_l2_df = w_df[w_df["model_id"] == "level2_M2"]
    cad_med = float(m2_l1_df.loc[m2_l1_df["feature"] == "cadaver_bmi", "impute_median"].iloc[0])
    liv_med = float(m2_l1_df.loc[m2_l1_df["feature"] == "living_bmi", "impute_median"].iloc[0])
    return {
        "level1_M0": AggregateLogRegModel(m0_df, l1_classes),
        "level1_M2": AggregateLogRegModel(m2_l1_df, l1_classes),
        "level2_M2": AggregateLogRegModel(m2_l2_df, l2_classes),
        "columns_M0": m0_df["feature"].tolist(),
        "columns_M2": m2_l1_df["feature"].tolist(),
        "level1_classes": l1_classes,
        "level2_classes": l2_classes,
        "n_level1": 1380,
        "n_level2": 487,
        "median_cadaver_bmi": cad_med,
        "median_living_bmi": liv_med,
        "mode": "LogisticRegression (aggregate cloud weights)",
    }


def load_table(name):
    p = TABLES_DIR / name
    return pd.read_csv(p) if p.exists() else None


def show_fig(name, caption):
    p = FIGURES_DIR / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


def build_feature_row(cols_m2, inputs):
    """Build a 1-row DataFrame matching M2_med_history columns exactly."""
    row = {c: 0 for c in cols_m2}
    row["age_years"] = float(inputs["age_years"])
    row["male"] = 1 if inputs["sex"] == "Male" else 0

    # Drinking
    drink_map = {
        "High risk": "drink_high_risk",
        "Low risk": "drink_low_risk",
        "Never drank": "drink_never_drank",
        "Previous High risk": "drink_previous_high_risk",
        "Drinker, status unknown": "drink_drinker_current_status_unknown",
    }
    if inputs["drinking"] == "Unknown / Not recorded":
        row["drink_missing"] = 1
    else:
        row[drink_map[inputs["drinking"]]] = 1

    # Tobacco
    tob_map = {
        "Smoke tobacco": "tobtype_smoke_tobacco",
        "Chew tobacco": "tobtype_chew_tobacco",
        "Never used tobacco": "tobtype_never_used_tobacco",
        "Tobacco user, type unknown": "tobtype_tobacco_user_type_unknown",
    }
    if inputs["tobacco"] == "Unknown / Not recorded":
        row["tob_missing"] = 1
    else:
        row[tob_map[inputs["tobacco"]]] = 1
        if inputs["heavy_tobacco"]:
            row["tobuse_heavy_tobacco_user"] = 1

    # Drug use
    drug_map = {
        "Opioids": "drug_opiods",
        "Cannabinoids": "drug_cannabinoids",
        "Cocaine": "drug_cocaine",
        "Other Stimulants": "drug_other_stimulants",
        "Sedative hypnotics/depressants": "drug_sedative_hypnotics_depressants",
    }
    if not inputs["drugs"] and not inputs["drug_former"]:
        row["drug_missing"] = 1
    else:
        for d in inputs["drugs"]:
            row[drug_map[d]] = 1
        if inputs["drug_former"]:
            row["drug_former"] = 1

    # Marital status
    marital_map = {
        "Married (spouse in household)": "marital_married_spouse_in_household",
        "Living as married": "marital_living_as_married",
        "Divorced": "marital_divorced",
        "Separated": "marital_separated",
        "Widowed": "marital_widowed",
        "Never married": "marital_never_married",
    }
    if inputs["marital"] == "Unknown / Not recorded":
        row["marital_missing"] = 1
    else:
        row[marital_map[inputs["marital"]]] = 1

    # Anthropometrics
    if inputs["cadaver_bmi_known"]:
        row["cadaver_bmi"] = float(inputs["cadaver_bmi"])
        row["cadaver_bmi_missing"] = 0
    else:
        row["cadaver_bmi"] = np.nan
        row["cadaver_bmi_missing"] = 1

    if inputs["living_bmi_known"]:
        row["living_bmi"] = float(inputs["living_bmi"])
        row["living_bmi_missing"] = 0
    else:
        row["living_bmi"] = np.nan
        row["living_bmi_missing"] = 1

    # Extended NOK lifestyle defaults (marked missing unless provided)
    for col in ["edu_level", "ses_adult", "ses_child"]:
        row[col] = np.nan
        row[col + "_missing"] = 1
    row["diet_missing"] = 1
    row["lift_missing"] = 1
    if inputs["firearm_occ"]:
        row["occ_firearm_access"] = 1
        row["occ_missing"] = 0
    else:
        row["occ_missing"] = 1

    # Diagnoses
    dx_map = {
        "Hypertension": "dx_hypertension",
        "Mental illness": "dx_mental_illness",
        "Diabetes Type II": "dx_diabetes_type_ii",
        "Coronary artery disease": "dx_coronary_artery_disease",
        "Hyperlipidemia": "dx_hyperlipedimia",
        "COPD": "dx_copd",
        "Asthma": "dx_asthma",
        "Cirrhosis of the liver": "dx_cirrhosis_of_the_liver",
    }
    if not inputs["diagnoses"]:
        row["dx_missing"] = 1
    else:
        for dx in inputs["diagnoses"]:
            row[dx_map[dx]] = 1

    # Lifetime medical history (mark missing unless checked)
    for mcol in ["broken_bone", "facial_trauma", "surgery", "implant"]:
        if mcol in inputs["medhist"]:
            row[f"medhist_{mcol}_any"] = 1
            row[f"medhist_{mcol}_missing"] = 0
        else:
            row[f"medhist_{mcol}_missing"] = 1

    return pd.DataFrame([row])[cols_m2]


PRESETS = {
    "Custom": None,
    "Cardiac Profile (Age 62, Hypertension, CAD, Recorded)": {
        "age_years": 62, "sex": "Male", "drinking": "Low risk",
        "tobacco": "Smoke tobacco", "heavy_tobacco": True,
        "drugs": [], "drug_former": False,
        "marital": "Married (spouse in household)",
        "cadaver_bmi_known": True, "cadaver_bmi": 31.5,
        "living_bmi_known": False, "living_bmi": 27.0,
        "firearm_occ": False,
        "diagnoses": ["Hypertension", "Coronary artery disease", "Hyperlipidemia"],
        "medhist": ["surgery"],
    },
    "Drug / Alcohol Profile (Age 41, High-Risk Drinking, Opioids)": {
        "age_years": 41, "sex": "Male", "drinking": "High risk",
        "tobacco": "Smoke tobacco", "heavy_tobacco": False,
        "drugs": ["Opioids"], "drug_former": False,
        "marital": "Never married",
        "cadaver_bmi_known": True, "cadaver_bmi": 25.0,
        "living_bmi_known": False, "living_bmi": 25.0,
        "firearm_occ": False,
        "diagnoses": ["Mental illness"],
        "medhist": [],
    },
    "Injury Profile (Age 29, Sparse / Blank Metadata)": {
        "age_years": 29, "sex": "Male", "drinking": "Unknown / Not recorded",
        "tobacco": "Unknown / Not recorded", "heavy_tobacco": False,
        "drugs": [], "drug_former": False,
        "marital": "Unknown / Not recorded",
        "cadaver_bmi_known": True, "cadaver_bmi": 24.5,
        "living_bmi_known": False, "living_bmi": 24.5,
        "firearm_occ": False,
        "diagnoses": [],
        "medhist": [],
    },
}


# ---------------- UI Header ----------------
st.title("Lifestyle & Cause of Death in Medicolegal Deaths (NMDID)")
st.caption(
    "CSE 499A Capstone Demo · North South University · "
    "Research demonstration of predictive signal vs. investigation-driven recording artifacts."
)

bundle = load_bundle()

tab_sim, tab_results = st.tabs(["Interactive Case Simulator", "Research Findings & Audit Dashboard"])

with tab_sim:
    if bundle is None:
        st.error(
            "Trained demo models not found at `models/demo_models.joblib`. "
            "Run `python src/09_train_demo_model.py` first."
        )
    else:
        st.info(
            "**Important:** Predictions below come from `HistGradientBoosting` trained on NMDID metadata "
            f"(Level 1 `n={bundle['n_level1']}`, Level 2 `n={bundle['n_level2']}`). Notice how changing fields between "
            "**Unknown / Not recorded** and recorded answers shifts probabilities—illustrating both lifestyle signal "
            "and recording-pattern artifacts (RQ3)."
        )

        preset_name = st.selectbox("Load an illustrative case preset (or choose Custom):", list(PRESETS.keys()))
        p = PRESETS[preset_name]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("1. Demographics & Body Size")
            age_years = st.slider("Age (years)", 18, 95, p["age_years"] if p else 45)
            sex = st.radio("Sex", ["Male", "Female"], index=0 if (not p or p["sex"] == "Male") else 1, horizontal=True)
            cad_known = st.checkbox("Cadaver BMI recorded (autopsy weight and length)", value=p["cadaver_bmi_known"] if p else True)
            cad_bmi = st.number_input("Cadaver BMI (kg/m²)", 12.0, 70.0, float(p["cadaver_bmi"]) if p else round(bundle["median_cadaver_bmi"], 1), step=0.5, disabled=not cad_known)
            liv_known = st.checkbox("Living BMI recorded (NOK interview)", value=p["living_bmi_known"] if p else False)
            liv_bmi = st.number_input("Living BMI (kg/m²)", 12.0, 70.0, float(p["living_bmi"]) if p else round(bundle["median_living_bmi"], 1), step=0.5, disabled=not liv_known)

        with col2:
            st.subheader("2. Lifestyle & Substance Use")
            drink_opts = ["Unknown / Not recorded", "High risk", "Low risk", "Never drank", "Previous High risk", "Drinker, status unknown"]
            drinking = st.selectbox("Drinking Status", drink_opts, index=drink_opts.index(p["drinking"]) if p else 0)
            tob_opts = ["Unknown / Not recorded", "Smoke tobacco", "Chew tobacco", "Never used tobacco", "Tobacco user, type unknown"]
            tobacco = st.selectbox("Tobacco Type", tob_opts, index=tob_opts.index(p["tobacco"]) if p else 0)
            heavy_tob = st.checkbox("Heavy tobacco user", value=p["heavy_tobacco"] if p else False, disabled=(tobacco == "Unknown / Not recorded"))
            drugs = st.multiselect("Active Substance Use", ["Opioids", "Cannabinoids", "Cocaine", "Other Stimulants", "Sedative hypnotics/depressants"], default=p["drugs"] if p else [])
            drug_former = st.checkbox("Former drug user", value=p["drug_former"] if p else False)
            mar_opts = ["Unknown / Not recorded", "Married (spouse in household)", "Living as married", "Divorced", "Separated", "Widowed", "Never married"]
            marital = st.selectbox("Marital Status", mar_opts, index=mar_opts.index(p["marital"]) if p else 0)
            firearm_occ = st.checkbox("Protective Services / Armed Forces occupation", value=p["firearm_occ"] if p else False)

        with col3:
            st.subheader("3. Medical History")
            diagnoses = st.multiselect(
                "Recorded Clinical Diagnoses",
                ["Hypertension", "Mental illness", "Diabetes Type II", "Coronary artery disease", "Hyperlipidemia", "COPD", "Asthma", "Cirrhosis of the liver"],
                default=p["diagnoses"] if p else [],
            )
            medhist = st.multiselect(
                "Lifetime Injury / Procedure History",
                ["broken_bone", "facial_trauma", "surgery", "implant"],
                default=p["medhist"] if p else [],
            )

        inputs = {
            "age_years": age_years, "sex": sex, "drinking": drinking,
            "tobacco": tobacco, "heavy_tobacco": heavy_tob,
            "drugs": drugs, "drug_former": drug_former, "marital": marital,
            "cadaver_bmi_known": cad_known, "cadaver_bmi": cad_bmi,
            "living_bmi_known": liv_known, "living_bmi": liv_bmi,
            "firearm_occ": firearm_occ, "diagnoses": diagnoses, "medhist": medhist,
        }
        X_m2 = build_feature_row(bundle["columns_M2"], inputs)
        X_m0 = X_m2[bundle["columns_M0"]]

        p1_m0 = bundle["level1_M0"].predict_proba(X_m0)[0]
        p1_m2 = bundle["level1_M2"].predict_proba(X_m2)[0]
        p2_m2 = bundle["level2_M2"].predict_proba(X_m2)[0]

        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.subheader("Level 1: Broad Cause of Death")
            df_l1 = pd.DataFrame({
                "Class": ["Natural Cardiac", "Drug / Alcohol", "Injury"],
                "M0 (Age + Sex only)": np.round(p1_m0, 3),
                "M2 (+ Lifestyle & Med History)": np.round(p1_m2, 3),
                "Shift (M2 - M0)": np.round(p1_m2 - p1_m0, 3),
            })
            st.dataframe(df_l1, hide_index=True, width="stretch")
            st.bar_chart(df_l1.set_index("Class")[["M0 (Age + Sex only)", "M2 (+ Lifestyle & Med History)"]])

        with r2:
            st.subheader("Level 2: Injury Intent (Conditional on Injury)")
            df_l2 = pd.DataFrame({
                "Intent": ["Accident", "Homicide", "Suicide"],
                "M2 Probability": np.round(p2_m2, 3),
            })
            st.dataframe(df_l2, hide_index=True, width="stretch")
            st.bar_chart(df_l2.set_index("Intent")[["M2 Probability"]])
            _m = load_table("rq1_metrics.csv")
            if _m is not None:
                _l2 = _m[(_m["level"] == "Level2") & (_m["feature_set"] == "M2_med_history")]
                _rng = f"{_l2['macro_auc'].min():.2f}–{_l2['macro_auc'].max():.2f}"
                st.caption(f"Note: Level 2 discrimination is weak (cross-validated M2 macro AUC {_rng} across models); "
                           "mental illness in suicides is the main recorded signal.")

with tab_results:
    sub1, sub2, sub3, sub4, sub5 = st.tabs([
        "Data Audit & Leakage", "RQ1: Nested Gain", "RQ2: Factors & SHAP",
        "RQ3: Recording Artifact", "Level 3 & Robustness"
    ])

    with sub1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Cohort Flow & Table 1")
            st.dataframe(load_table("cohort_flow.csv"), hide_index=True, width="stretch")
            st.dataframe(load_table("table1_by_class.csv"), hide_index=True, width="stretch")
            st.subheader("Excluded Leakage Columns (Cramér's V)")
            st.dataframe(load_table("leakage_check.csv"), hide_index=True, width="stretch")
        with c2:
            show_fig("coverage_heatmap.png", "Metadata Field Coverage by Level-1 Class (%)")
            st.dataframe(load_table("missingness_tests.csv"), hide_index=True, width="stretch")
            st.subheader("Cadaver BMI Flag Audit")
            st.caption("Blanking BMI for decomposition copies a forbidden post-mortem variable into the missing flag; "
                       "the main analysis uses the raw-measurement version.")
            st.dataframe(load_table("bmi_flag_audit.csv"), hide_index=True, width="stretch")
            st.subheader("Cart Invariance (Cross-Cart Duplicate Records)")
            st.dataframe(load_table("cart_invariance_check.csv"), hide_index=True, width="stretch")

    with sub2:
        st.subheader("RQ1: 5×5 Repeated Stratified CV Metrics & Gains")
        st.dataframe(load_table("rq1_metrics.csv"), hide_index=True, width="stretch")
        st.dataframe(load_table("rq1_gains.csv"), hide_index=True, width="stretch")
        st.subheader("Reverse Entry Order (medical history first)")
        st.dataframe(load_table("rq1_reverse_order.csv"), hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            show_fig("rq1_calibration_level1.png", "Level 1 Calibration Curves (HistGB)")
        with c2:
            show_fig("rq1_calibration_level2.png", "Level 2 Calibration Curves (HistGB)")

    with sub3:
        st.subheader("RQ2: Univariate BH-FDR Tests & TreeSHAP")
        c1, c2 = st.columns(2)
        with c1:
            show_fig("rq2_shap_summary.png", "Top 15 Predictors by Mean |SHAP| on held-out folds (Level 1, Random Forest M2)")
        with c2:
            show_fig("rq2_rf_importance_by_sex.png", "Sex-Stratified Out-of-Fold Random Forest Permutation Importance")
        st.subheader("Age-Band Stratified Performance")
        st.dataframe(load_table("rq2_age_bands.csv"), hide_index=True, width="stretch")
        st.subheader("Age- and Sex-Adjusted Associations (multinomial LR test, BH-FDR)")
        st.dataframe(load_table("rq2_adjusted_level1.csv"), hide_index=True, width="stretch")
        st.subheader("Top Level-1 Univariate Associations (BH-FDR)")
        df_u1 = load_table("rq2_univariate_level1.csv")
        if df_u1 is not None:
            st.dataframe(df_u1.head(20), hide_index=True, width="stretch")

    with sub4:
        st.subheader("RQ3: Biological Signal vs. Recording Artifact")
        c1, c2 = st.columns(2)
        with c1:
            show_fig("rq3_encoding_regimes.png", "Blanks Only vs. Recorded Values Only vs. Full Features")
        with c2:
            show_fig("rq3_subgroups.png", "Complete Cases, NOK Interview Subgroups, and Cardiac-Conflict Sensitivity")
        st.subheader("Share of the Gain: Recording Flags vs Recorded Values")
        st.dataframe(load_table("rq3_gain_shares.csv"), hide_index=True, width="stretch")
        st.dataframe(load_table("rq3_artifact.csv"), hide_index=True, width="stretch")

    with sub5:
        st.subheader("Level 3 Negative Control, Permutation Check & Robustness")
        show_fig("level3_and_robustness.png", "Level 3 Homicide Weapon Negative Control & Level 1 Robustness")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Level 3 Control (Gunshot vs. Stab; 30 gunshot deaths, low power)**")
            st.dataframe(load_table("level3_negative_control.csv"), hide_index=True, width="stretch")
            st.write("**Permuted-Label Control (5 Shuffles)**")
            st.dataframe(load_table("permutation_check.csv"), hide_index=True, width="stretch")
        with c2:
            st.write("**Robustness Checks (Level 1, HistGB)**")
            st.dataframe(load_table("robustness.csv"), hide_index=True, width="stretch")

