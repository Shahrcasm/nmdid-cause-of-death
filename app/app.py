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
        _acc = load_table("accuracy_metrics.csv")
        _key = "LogReg" if bundle["mode"].startswith("Logistic") else "HistGB"
        _acc_txt = "not available"
        if _acc is not None:
            _r = _acc[(_acc["level"] == "Level1") & (_acc["model"] == _key) & (_acc["feature_set"] == "M2_med_history")]
            if len(_r):
                _acc_txt = f"{100 * float(_r['accuracy'].iloc[0]):.1f}%"
        _compare = ("Set a field to **Unknown** and compare how the standard model (which reads a blank as 'the investigator "
                    "recorded nothing') and the withholding-trained model (which reads it as 'not available to us') respond."
                    if "level1_withheld" in bundle else
                    "The withholding-trained model and the shortlists need the local model file and are not shown in this deployment.")
        st.info(
            f"Research demonstration, not a forensic tool. Models: {bundle['mode']} trained on NMDID records "
            f"(broad cause n={bundle['n_level1']}, intent n={bundle['n_level2']}). Held-out accuracy of this model for the "
            f"broad cause: {_acc_txt} (three causes). {_compare}"
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
        has_withheld = "level1_withheld" in bundle
        if has_withheld:
            X_w = X_m2.copy()
            unavailable = []
            for sec_name, flags in bundle["sections"].items():
                blank = all(X_w.at[0, f] == 1 for f in flags)
                X_w["withheld_" + sec_name] = 1 if blank else 0
                if blank:
                    unavailable.append(bundle["section_labels"][sec_name])
                    for f in flags:
                        X_w.at[0, f] = 0
            p1_w = bundle["level1_withheld"].predict_proba(X_w[bundle["columns_withheld"]])[0]

        n_blank = None
        if "sections" in bundle:
            n_blank = sum(all(X_m2.at[0, f] == 1 for f in flags) for flags in bundle["sections"].values())
        band = next((b for b, lo, hi in bundle.get("bands", []) if n_blank is not None and lo <= n_blank <= hi), None)

        def shortlist(p, model_key, target=90, band=None):
            thr = None
            if band is not None:
                thr = bundle.get("shortlist_thresholds_band", {}).get(model_key, {}).get(target, {}).get(band)
            if thr is None:
                thr = bundle.get("shortlist_thresholds", {}).get(model_key, {}).get(target)
            if thr is None:
                return "not available"
            names = ["Natural cardiac", "Drug / alcohol", "Injury"]
            keep = [n for n, v in zip(names, p) if v >= 1 - thr]
            return ", ".join(keep) if keep else "none (all probabilities low)"

        st.divider()
        r1, r2 = st.columns([3, 2])
        with r1:
            st.subheader("Broad cause of death")
            table = {"Cause": ["Natural cardiac", "Drug / alcohol", "Injury"],
                     "Age + sex only": np.round(p1_m0, 3),
                     "Standard model (blank = not recorded)": np.round(p1_m2, 3)}
            if has_withheld:
                table["Withholding-trained (blank = not available)"] = np.round(p1_w, 3)
            df_l1 = pd.DataFrame(table)
            st.dataframe(df_l1, hide_index=True, width="stretch")
            st.bar_chart(df_l1.set_index("Cause"))
            st.markdown(f"**90% shortlist, standard model:** {shortlist(p1_m2, 'standard', band=band)}")
            if band is not None:
                st.caption(f"Shortlist threshold adjusted for {n_blank} of 8 sections left blank.")
            if has_withheld:
                st.markdown(f"**90% shortlist, withholding-trained model:** {shortlist(p1_w, 'withheld')}")
                st.caption("Sections treated as not available: " + (", ".join(unavailable) if unavailable else "none"))
            if has_withheld and "donors" in bundle:
                def _entropy(p):
                    q = np.clip(p, 1e-12, 1.0)
                    return -(q * np.log(q)).sum(axis=-1)
                h_now, ranking = float(_entropy(p1_w)), []
                for sec_name, cols in bundle["section_columns"].items():
                    if X_w.at[0, "withheld_" + sec_name] != 1:
                        continue
                    exp_h = 0.0
                    for ci, c in enumerate(bundle["level1_classes"]):
                        D = bundle["donors"][c]
                        block = pd.concat([X_w[bundle["columns_withheld"]]] * len(D), ignore_index=True)
                        block[cols] = D[cols].values
                        block["withheld_" + sec_name] = 0
                        q = bundle["level1_withheld"].predict_proba(block[bundle["columns_withheld"]])
                        exp_h += p1_w[ci] * float(_entropy(q).mean())
                    ranking.append((bundle["section_labels"][sec_name], 100 * (h_now - exp_h) / h_now if h_now > 0 else 0.0))
                useful = sorted([t for t in ranking if t[1] >= 1.0], key=lambda t: -t[1])
                if useful:
                    st.markdown("**Which question to ask next** (withholding-trained model)")
                    st.dataframe(pd.DataFrame([{"Ask about": a, "Expected reduction in uncertainty": f"{g:.0f}%"} for a, g in useful[:3]]),
                                 hide_index=True, width="stretch")
                elif ranking:
                    st.caption("Which question to ask next: no missing section is expected to reduce the model's "
                               "uncertainty by 1% or more for this record.")
            else:
                _fc = load_table("questions_first_choice.csv")
                if _fc is not None:
                    st.caption("Sections the model most often asks about first (held-out results): "
                               + ", ".join(_fc["section"].head(3)) + ".")
            st.caption("A 90% shortlist contains the true cause for about 90% of held-out people (split conformal "
                       "prediction; threshold from results/tables/reliability_conformal.csv, applied approximately here).")
        with r2:
            st.subheader("Intent, if the death was an injury")
            df_l2 = pd.DataFrame({"Intent": ["Accident", "Homicide", "Suicide"], "Probability": np.round(p2_m2, 3)})
            st.dataframe(df_l2, hide_index=True, width="stretch")
            st.bar_chart(df_l2.set_index("Intent"))
            _m = load_table("accuracy_metrics.csv")
            if _m is not None:
                _l2 = _m[(_m["level"] == "Level2") & (_m["feature_set"] == "M2_med_history")]
                st.caption(f"Intent is hard to estimate from these records: held-out accuracy "
                           f"{100 * _l2['accuracy'].min():.0f}% to {100 * _l2['accuracy'].max():.0f}% across models.")

with tab_results:
    sub1, sub2, sub3, sub4, sub5, sub6 = st.tabs([
        "Data audit", "Accuracy", "Missing information", "Reliability", "Explanation", "Controls"])

    with sub1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Cohort flow and characteristics")
            st.dataframe(load_table("cohort_flow.csv"), hide_index=True, width="stretch")
            st.dataframe(load_table("paper_table1_cohort.csv"), hide_index=True, width="stretch")
            st.subheader("Blocked columns (Cramér's V with the broad cause)")
            st.dataframe(load_table("leakage_check.csv"), hide_index=True, width="stretch")
        with c2:
            show_fig("coverage_heatmap.png", "Which fields investigators record, by cause of death (%)")
            st.subheader("Cadaver BMI flag audit")
            st.dataframe(load_table("bmi_flag_audit.csv"), hide_index=True, width="stretch")
            st.subheader("Cross-cart duplicate records")
            st.dataframe(load_table("cart_invariance_check.csv"), hide_index=True, width="stretch")

    with sub2:
        show_fig("accuracy_by_inputs.png", "Accuracy and macro AUC by inputs and model (held-out people)")
        show_fig("validation.png", "Validation on later years and on held-out regions")
        st.subheader("Accuracy by model and inputs")
        st.dataframe(load_table("paper_table2_accuracy.csv"), hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Per cause**")
            st.dataframe(load_table("accuracy_per_class.csv"), hide_index=True, width="stretch")
            st.write("**Confusion matrix (gradient boosting, all inputs)**")
            st.dataframe(load_table("confusion_matrix.csv"), hide_index=True, width="stretch")
        with c2:
            st.write("**Subgroups**")
            st.dataframe(load_table("accuracy_subgroups.csv"), hide_index=True, width="stretch")

    with sub3:
        c1, c2 = st.columns(2)
        with c1:
            show_fig("missing_information_natural.png", "Records as they are, and one section hidden at a time")
        with c2:
            show_fig("missing_information_withheld.png", "Same people, less information")
        st.subheader("Missing fields, training share, ground truth, test share and test accuracy")
        st.dataframe(load_table("paper_table3_missing_information.csv"), hide_index=True, width="stretch")
        st.subheader("Sections hidden at test time")
        st.dataframe(load_table("paper_table3b_hidden_sections.csv"), hide_index=True, width="stretch")
        show_fig("missing_question_order.png", "Which question to ask next: accuracy as sections are added one at a time")
        st.dataframe(load_table("questions_first_choice.csv"), hide_index=True, width="stretch")
        st.subheader("Most common exact patterns of blank sections")
        st.dataframe(load_table("missing_natural_patterns.csv"), hide_index=True, width="stretch")

    with sub4:
        c1, c2 = st.columns(2)
        with c1:
            show_fig("reliability_calibration.png", "Calibration of the predicted probabilities")
        with c2:
            show_fig("reliability_shortlists.png", "Answering only when confident, and 90% shortlists")
        st.dataframe(load_table("reliability_selective.csv"), hide_index=True, width="stretch")
        st.dataframe(load_table("reliability_conformal.csv"), hide_index=True, width="stretch")

    with sub5:
        c1, c2 = st.columns(2)
        with c1:
            show_fig("explanation_content_vs_blanks.png", "Which sections are filled in versus what they say")
        with c2:
            show_fig("explanation_shap.png", "Inputs that move the estimate most (held-out SHAP)")
        st.dataframe(load_table("explanation_gain_shares.csv"), hide_index=True, width="stretch")
        st.dataframe(load_table("explanation_regimes.csv"), hide_index=True, width="stretch")
        st.subheader("Age- and sex-adjusted associations (broad cause)")
        st.dataframe(load_table("factors_adjusted_level1.csv"), hide_index=True, width="stretch")

    with sub6:
        show_fig("controls.png", "Negative control, permutation test and robustness checks")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Negative control: stab vs gunshot (small sample)**")
            st.dataframe(load_table("controls_level3.csv"), hide_index=True, width="stretch")
            st.write("**Permutation test**")
            st.dataframe(load_table("controls_permutation.csv"), hide_index=True, width="stretch")
        with c2:
            st.write("**Robustness (broad cause, gradient boosting)**")
            st.dataframe(load_table("controls_robustness.csv"), hide_index=True, width="stretch")
            st.write("**Leakage demonstration (why the cart variable is blocked)**")
            st.dataframe(load_table("controls_leakage_demo.csv"), hide_index=True, width="stretch")
