"""Streamlit demonstration: estimating the broad cause of death from incomplete records (NMDID).

Run from the repository root:
    streamlit run app/app.py

Local mode uses the models in models/demo_models.joblib (git-ignored, derived from
restricted data). Without that file, as on Streamlit Community Cloud, the app rebuilds
logistic regressions from the aggregate weights in results/tables/. No individual-level
data is used by the deployed app.
"""
import importlib
import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import model_utils as mu  # noqa: E402  (record sections, shared with the research pipeline)
build = importlib.import_module("01_build_dataset")  # answer codes, shared with the research pipeline

MODEL_PATH = ROOT / "models" / "demo_models.joblib"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
CAUSES = mu.LEVEL1_CLASSES
CAUSE_NAMES = ["Natural cardiac", "Drug or alcohol", "Injury"]
INTENT_NAMES = ["Accident", "Homicide", "Suicide"]
BANDS = [("none", 0, 0), ("1 to 3", 1, 3), ("4 or 5", 4, 5), ("6", 6, 6), ("7 or 8", 7, 8)]
UNKNOWN = "Unknown / not recorded"
NOT_RECORDED, RECORDED = "Not recorded", "Recorded"

st.set_page_config(page_title="Cause of Death from Incomplete Records", page_icon="🩺", layout="wide")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class AggregateLogRegModel:
    """Logistic regression rebuilt from aggregate weights (median imputation, scaling, coefficients)."""

    def __init__(self, df_sub, classes):
        self.cols = df_sub["feature"].tolist()
        self.medians = df_sub["impute_median"].values.astype(float)
        self.means = df_sub["scale_mean"].values.astype(float)
        self.stds = df_sub["scale_std"].values.astype(float)
        self.coefs = np.column_stack([df_sub[f"coef_{c}"].values.astype(float) for c in classes])
        self.intercepts = np.array([float(df_sub[f"intercept_{c}"].iloc[0]) for c in classes])

    def predict_proba(self, X_df):
        x = X_df[self.cols].values.astype(float)
        x = np.where(np.isnan(x), self.medians, x)
        z = (x - self.means) / np.where(self.stds == 0, 1.0, self.stds)
        logits = z @ self.coefs + self.intercepts
        logits = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)


def load_table(name):
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None


@st.cache_resource
def load_bundle():
    acc = load_table("accuracy_metrics.csv")

    def heldout_accuracy(model):
        if acc is None:
            return None
        r = acc[(acc["level"] == "Level1") & (acc["model"] == model) & (acc["feature_set"] == "M2_med_history")]
        return float(r["accuracy"].iloc[0]) if len(r) else None

    if MODEL_PATH.exists():
        b = joblib.load(MODEL_PATH)
        return {"mode": "local", "label": "gradient boosting", "accuracy": heldout_accuracy("HistGB"),
                "base": b["level1_M0"], "standard": b["level1_M2"], "withheld": b["level1_withheld"], "intent": b["level2_M2"],
                "cols0": b["columns_M0"], "cols2": b["columns_M2"], "colsw": b["columns_withheld"],
                "thr": b.get("shortlist_thresholds", {}), "thr_band": b.get("shortlist_thresholds_band", {}),
                "donors": b.get("donors"), "n1": b["n_level1"], "n2": b["n_level2"]}
    w = load_table("demo_logreg_weights.csv")
    if w is None:
        return None
    part = {m: w[w["model_id"] == m] for m in w["model_id"].unique()}
    thr, thr_band = {}, {}
    t = load_table("demo_shortlist_thresholds.csv")
    if t is not None:
        for _, r in t.iterrows():
            target = int(r["target_coverage_pct"])
            if r["method"] == "marginal":
                thr.setdefault(r["model"], {})[target] = float(r["threshold"])
            else:
                thr_band.setdefault(r["model"], {}).setdefault(target, {})[r["band"]] = float(r["threshold"])
    has_w = "level1_withheld" in part
    return {"mode": "deployed", "label": "logistic regression", "accuracy": heldout_accuracy("LogReg"),
            "base": AggregateLogRegModel(part["level1_M0"], CAUSES),
            "standard": AggregateLogRegModel(part["level1_M2"], CAUSES),
            "withheld": AggregateLogRegModel(part["level1_withheld"], CAUSES) if has_w else None,
            "intent": AggregateLogRegModel(part["level2_M2"], mu.LEVEL2_CLASSES),
            "cols0": part["level1_M0"]["feature"].tolist(), "cols2": part["level1_M2"]["feature"].tolist(),
            "colsw": part["level1_withheld"]["feature"].tolist() if has_w else None,
            "thr": thr, "thr_band": thr_band, "donors": None, "n1": 1380, "n2": 487}


def band_of(n_blank):
    return next(b for b, lo, hi in BANDS if lo <= n_blank <= hi)


def shortlist(bundle, p, model_key, n_blank=None, target=90):
    """Causes whose probability passes the conformal threshold (adjusted for blank sections when available)."""
    thr = None
    if n_blank is not None:
        thr = bundle["thr_band"].get(model_key, {}).get(target, {}).get(band_of(n_blank))
    if thr is None:
        thr = bundle["thr"].get(model_key, {}).get(target)
    if thr is None:
        return None
    return [CAUSE_NAMES[i] for i in range(3) if p[i] >= 1 - thr]


def entropy(p):
    q = np.clip(p, 1e-12, 1.0)
    return -(q * np.log(q)).sum(axis=-1)


# ---------------------------------------------------------------------------
# Record builder: inputs -> the 87 model inputs used in the research pipeline
# ---------------------------------------------------------------------------
DRINK = {"High risk": "drink_high_risk", "Low risk": "drink_low_risk", "Never drank": "drink_never_drank",
         "Previously high risk": "drink_previous_high_risk", "Drinker, current status unknown": "drink_drinker_current_status_unknown"}
TOBACCO = {"Smokes tobacco": "tobtype_smoke_tobacco", "Chews tobacco": "tobtype_chew_tobacco",
           "Never used tobacco": "tobtype_never_used_tobacco", "Tobacco user, type unknown": "tobtype_tobacco_user_type_unknown"}
DRUGS = {"Opioids": "drug_opiods", "Cannabinoids": "drug_cannabinoids", "Cocaine": "drug_cocaine",
         "Other stimulants": "drug_other_stimulants", "Sedatives or hypnotics": "drug_sedative_hypnotics_depressants"}
MARITAL = {"Married, spouse in household": "marital_married_spouse_in_household", "Living as married": "marital_living_as_married",
           "Divorced": "marital_divorced", "Separated": "marital_separated", "Widowed": "marital_widowed",
           "Never married": "marital_never_married"}
DIAGNOSES = {"Hypertension": "dx_hypertension", "Coronary artery disease": "dx_coronary_artery_disease",
             "Mental illness": "dx_mental_illness", "Type 2 diabetes": "dx_diabetes_type_ii", "High cholesterol": "dx_hyperlipedimia",
             "COPD": "dx_copd", "Asthma": "dx_asthma", "Cirrhosis of the liver": "dx_cirrhosis_of_the_liver"}
INJURIES = {"Broken bone": "broken_bone", "Facial trauma": "facial_trauma", "Surgery": "surgery", "Implant": "implant"}
OCCUPATION = {"Protective services or armed forces": 1, "Other occupation": 0}


def build_record(cols2, d):
    row = {c: 0.0 for c in cols2}
    row["age_years"], row["male"] = float(d["age"]), 1.0 if d["sex"] == "Male" else 0.0
    if d["drinking"] == UNKNOWN:
        row["drink_missing"] = 1
    else:
        row[DRINK[d["drinking"]]] = 1
    if d["tobacco"] == UNKNOWN:
        row["tob_missing"] = 1
    else:
        row[TOBACCO[d["tobacco"]]] = 1
        row["tobuse_heavy_tobacco_user"] = 1 if d["heavy_tobacco"] else 0
    if d["drug_status"] == NOT_RECORDED:
        row["drug_missing"] = 1
    else:
        for x in d["drugs"]:
            row[DRUGS[x]] = 1
        row["drug_former"] = 1 if d["drug_former"] else 0
    if d["marital"] == UNKNOWN:
        row["marital_missing"] = 1
    else:
        row[MARITAL[d["marital"]]] = 1
    for key, col in [("cadaver", "cadaver_bmi"), ("living", "living_bmi")]:
        known = d[f"{key}_known"]
        row[col], row[col + "_missing"] = (float(d[f"{key}_bmi"]), 0) if known else (np.nan, 1)
    for key, col, order in [("education", "edu_level", build.EDU_ORDER), ("ses", "ses_adult", build.SES_ORDER)]:
        row[col], row[col + "_missing"] = (np.nan, 1) if d[key] == UNKNOWN else (float(order[d[key]]), 0)
    row["ses_child"], row["ses_child_missing"] = np.nan, 1
    if d["diet"] == UNKNOWN:
        row["diet_missing"] = 1
    else:
        row["diet_" + d["diet"].lower().replace(" ", "_")] = 1
    if d["lifting"] == UNKNOWN:
        row["lift_missing"] = 1
    else:
        row["lift_" + d["lifting"].lower()] = 1
    if d["occupation"] == UNKNOWN:
        row["occ_missing"] = 1
    else:
        row["occ_firearm_access"] = OCCUPATION[d["occupation"]]
    if d["dx_status"] == NOT_RECORDED:
        row["dx_missing"] = 1
    else:
        for x in d["diagnoses"]:
            row[DIAGNOSES[x]] = 1
    for label, code in INJURIES.items():
        if d["inj_status"] == NOT_RECORDED:
            row[f"medhist_{code}_missing"] = 1
        else:
            row[f"medhist_{code}_any"] = 1 if label in d["injuries"] else 0
    return pd.DataFrame([row])[cols2]


BASE = {"age": 45, "sex": "Male", "drinking": UNKNOWN, "tobacco": UNKNOWN, "heavy_tobacco": False,
        "drug_status": NOT_RECORDED, "drugs": [], "drug_former": False, "marital": UNKNOWN,
        "cadaver_known": True, "cadaver_bmi": 26.0, "living_known": False, "living_bmi": 26.0,
        "education": UNKNOWN, "ses": UNKNOWN, "diet": UNKNOWN, "lifting": UNKNOWN, "occupation": UNKNOWN,
        "dx_status": NOT_RECORDED, "diagnoses": [], "inj_status": NOT_RECORDED, "injuries": []}
PRESETS = {
    "Custom": dict(BASE),
    "Heart-disease profile (62, hypertension, coronary disease)": dict(
        BASE, age=62, drinking="Low risk", tobacco="Smokes tobacco", heavy_tobacco=True, marital="Married, spouse in household",
        cadaver_bmi=31.5, dx_status=RECORDED, diagnoses=["Hypertension", "Coronary artery disease", "High cholesterol"],
        inj_status=RECORDED, injuries=["Surgery"]),
    "Drug or alcohol profile (41, high-risk drinking, opioids)": dict(
        BASE, age=41, drinking="High risk", tobacco="Smokes tobacco", drug_status=RECORDED, drugs=["Opioids"],
        marital="Never married", cadaver_bmi=25.0, dx_status=RECORDED, diagnoses=["Mental illness"]),
    "Sparse record (29, almost nothing recorded)": dict(BASE, age=29, cadaver_bmi=24.5),
}


# ---------------------------------------------------------------------------
# Dashboard helpers
# ---------------------------------------------------------------------------
VALUE_NAMES = {"natural_cardiac": "Natural cardiac", "drug_alcohol": "Drug or alcohol", "injury": "Injury",
               "M0_base": "Age + sex", "M1_lifestyle": "+ lifestyle", "M2_med_history": "+ medical history",
               "HistGB": "Gradient boosting", "RF": "Random forest", "LogReg": "Logistic regression",
               "Level1": "Broad cause", "Level2": "Intent", "standard": "Standard model",
               "withheld": "Withholding-trained", "withheld_recal": "Withholding-trained, recalibrated",
               "marginal": "One threshold", "missingness-conditional": "Adjusted for blank sections",
               "class-conditional": "One threshold per cause", "guided": "Model chooses", "fixed": "Fixed order",
               "random": "Random order", "not_recorded": "Hidden = left blank", "imputed": "Hidden = typical values",
               "multiple_imputation": "Hidden = multiple imputation", "withheld_model": "Withholding-trained model",
               "reduced_model": "Reduced model (reference)"}


def tidy(df, columns=None, rename=None):
    d = (df if columns is None else df[columns]).copy()
    for c in d.columns:
        if not pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]):
            d[c] = d[c].replace(VALUE_NAMES)
    rename = rename or {}
    d.columns = [rename.get(c, c.replace("_", " ").replace(" pct", " (%)").capitalize()) for c in d.columns]
    return d


def table(name, columns=None, rename=None, query=None):
    df = load_table(name)
    if df is None:
        st.caption(f"{name} not found.")
        return
    if query is not None:
        df = df.query(query)
    st.dataframe(tidy(df, columns, rename), hide_index=True, width="stretch")


def full_table(name, label):
    with st.expander(f"Show full table: {label}"):
        table(name)


def figure(name, caption):
    p = FIGURES / name
    if p.exists():
        st.image(str(p), caption=caption, width="stretch")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("Estimating the Cause of Death from Incomplete Records")
st.caption("Lifestyle and cause of death in medicolegal deaths (NMDID metadata) · CSE 499A Capstone, North South University · "
           "research demonstration, not a forensic tool")
st.markdown("**Research question:** from a person's recorded lifestyle and medical history, how well can we estimate the broad "
            "cause of death, and how much does the estimate suffer when parts of the record are missing?")

bundle = load_bundle()
if bundle is None:
    st.error("No models found. Run `src/09_train_demo_model.py`, or make sure `results/tables/demo_logreg_weights.csv` exists.")
    st.stop()
SEC = mu.Sections(bundle["cols2"])
SECTION_NAMES = [mu.SECTION_LABELS[s] for s in SEC.names]

tab_sim, tab_cmp, tab_res = st.tabs(["Case simulator", "Compare missing fields", "Results dashboard"])

# ---------------- Case simulator ----------------
with tab_sim:
    acc_txt = f"{100 * bundle['accuracy']:.1f}%" if bundle["accuracy"] else "not available"
    mode_txt = ("Running the gradient-boosting models trained on all NMDID records (local mode)." if bundle["mode"] == "local"
                else "Running logistic regressions rebuilt from the published summary weights (online mode).")
    st.info(f"{mode_txt} Held-out accuracy of the standard model for the broad cause: {acc_txt} (three causes). "
            "Set any section to unknown to compare the **standard model**, which reads a blank as \"the investigator recorded "
            "nothing\", with the **withholding-trained model**, which reads it as \"not available to us\".")
    p = PRESETS[st.selectbox("Start from an example, or choose Custom", list(PRESETS))]

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Person and body size")
        age = st.slider("Age (years)", 18, 95, int(p["age"]))
        sex = st.radio("Sex", ["Male", "Female"], index=0 if p["sex"] == "Male" else 1, horizontal=True)
        cadaver_known = st.checkbox("Body size measured at autopsy", value=p["cadaver_known"])
        cadaver_bmi = st.number_input("Body mass index at autopsy (kg/m²)", 12.0, 70.0, float(p["cadaver_bmi"]), step=0.5,
                                      disabled=not cadaver_known)
        living_known = st.checkbox("Body size reported by the family", value=p["living_known"])
        living_bmi = st.number_input("Body mass index reported by the family (kg/m²)", 12.0, 70.0, float(p["living_bmi"]),
                                     step=0.5, disabled=not living_known)
        st.subheader("Education, work and diet")
        education = st.selectbox("Education", [UNKNOWN] + list(build.EDU_ORDER),
                                 index=([UNKNOWN] + list(build.EDU_ORDER)).index(p["education"]))
        ses = st.selectbox("Socioeconomic status (adult)", [UNKNOWN] + list(build.SES_ORDER),
                           index=([UNKNOWN] + list(build.SES_ORDER)).index(p["ses"]))
        diet = st.selectbox("Diet", [UNKNOWN] + build.DIET_ITEMS, index=([UNKNOWN] + build.DIET_ITEMS).index(p["diet"]))
        lifting = st.selectbox("Heavy lifting at work", [UNKNOWN, "Yes", "No"], index=[UNKNOWN, "Yes", "No"].index(p["lifting"]))
        occupation = st.selectbox("Occupation", [UNKNOWN] + list(OCCUPATION), index=([UNKNOWN] + list(OCCUPATION)).index(p["occupation"]))
    with c2:
        st.subheader("Lifestyle")
        drinking = st.selectbox("Drinking", [UNKNOWN] + list(DRINK), index=([UNKNOWN] + list(DRINK)).index(p["drinking"]))
        tobacco = st.selectbox("Tobacco", [UNKNOWN] + list(TOBACCO), index=([UNKNOWN] + list(TOBACCO)).index(p["tobacco"]))
        heavy_tobacco = st.checkbox("Heavy tobacco user", value=p["heavy_tobacco"], disabled=tobacco == UNKNOWN)
        drug_status = st.radio("Drug use", [NOT_RECORDED, RECORDED], index=[NOT_RECORDED, RECORDED].index(p["drug_status"]),
                               horizontal=True)
        drugs = st.multiselect("Current drug use (leave empty for none)", list(DRUGS), default=p["drugs"],
                               disabled=drug_status == NOT_RECORDED)
        drug_former = st.checkbox("Former drug user", value=p["drug_former"], disabled=drug_status == NOT_RECORDED)
        marital = st.selectbox("Marital status", [UNKNOWN] + list(MARITAL), index=([UNKNOWN] + list(MARITAL)).index(p["marital"]))
    with c3:
        st.subheader("Medical history")
        dx_status = st.radio("Diagnoses", [NOT_RECORDED, RECORDED], index=[NOT_RECORDED, RECORDED].index(p["dx_status"]),
                             horizontal=True)
        diagnoses = st.multiselect("Diagnoses on record (leave empty for none)", list(DIAGNOSES), default=p["diagnoses"],
                                   disabled=dx_status == NOT_RECORDED)
        inj_status = st.radio("Past injuries and surgery", [NOT_RECORDED, RECORDED],
                              index=[NOT_RECORDED, RECORDED].index(p["inj_status"]), horizontal=True)
        injuries = st.multiselect("Past injuries and surgery on record (leave empty for none)", list(INJURIES),
                                  default=p["injuries"], disabled=inj_status == NOT_RECORDED)

    person = dict(age=age, sex=sex, drinking=drinking, tobacco=tobacco, heavy_tobacco=heavy_tobacco, drug_status=drug_status,
                  drugs=drugs, drug_former=drug_former, marital=marital, cadaver_known=cadaver_known, cadaver_bmi=cadaver_bmi,
                  living_known=living_known, living_bmi=living_bmi, education=education, ses=ses, diet=diet,
                  lifting=lifting, occupation=occupation, dx_status=dx_status, diagnoses=diagnoses,
                  inj_status=inj_status, injuries=injuries)
    X = build_record(bundle["cols2"], person)
    blank = SEC.not_recorded(X)[0]
    n_blank = int(blank.sum())
    recorded = [mu.SECTION_LABELS[s] for s, b in zip(SEC.names, blank) if not b]
    st.markdown(f"**Sections recorded: {8 - n_blank} of 8.** " +
                ("Recorded: " + ", ".join(recorded) + "." if recorded else "Nothing beyond age and sex is recorded."))

    p_base = bundle["base"].predict_proba(X[bundle["cols0"]])[0]
    p_std = bundle["standard"].predict_proba(X)[0]
    Xw = SEC.as_withheld(X, SEC.not_recorded(X)) if bundle["withheld"] is not None else None
    p_w = bundle["withheld"].predict_proba(Xw[bundle["colsw"]])[0] if Xw is not None else None

    st.divider()
    r1, r2 = st.columns([3, 2])
    with r1:
        st.subheader("Broad cause of death")
        probs = {"Cause": CAUSE_NAMES, "Age and sex only": np.round(p_base, 3), "Standard model": np.round(p_std, 3)}
        if p_w is not None:
            probs["Withholding-trained model"] = np.round(p_w, 3)
        df_p = pd.DataFrame(probs)
        st.dataframe(df_p, hide_index=True, width="stretch")
        st.bar_chart(df_p.set_index("Cause"))
        s_std = shortlist(bundle, p_std, "standard", n_blank)
        st.markdown(f"**90% shortlist, standard model:** {', '.join(s_std) if s_std else 'not available'} "
                    f"(threshold adjusted for {n_blank} blank sections)")
        if p_w is not None:
            s_w = shortlist(bundle, p_w, "withheld")
            st.markdown(f"**90% shortlist, withholding-trained model:** {', '.join(s_w) if s_w else 'not available'}")
        st.caption("A 90% shortlist contains the true cause for about 90% of held-out people (split conformal prediction). "
                   "Thresholds come from held-out predictions and are applied here to a hypothetical person, so they are approximate.")
    with r2:
        st.subheader("Which question to ask next")
        if bundle["donors"] is not None and p_w is not None:
            h_now, ranking = float(entropy(p_w)), []
            for j, s in enumerate(SEC.names):
                if Xw.at[0, f"withheld_{s}"] != 1:
                    continue
                cols = SEC.section_columns(s)
                expected = 0.0
                for ci, c in enumerate(CAUSES):
                    D = bundle["donors"][c]
                    block = pd.concat([Xw] * len(D), ignore_index=True)
                    block[cols] = D[cols].values
                    block[f"withheld_{s}"] = 0
                    expected += p_w[ci] * float(entropy(bundle["withheld"].predict_proba(block[bundle["colsw"]])).mean())
                ranking.append((mu.SECTION_LABELS[s], 100 * (h_now - expected) / h_now if h_now > 0 else 0.0))
            useful = sorted([t for t in ranking if t[1] >= 1.0], key=lambda t: -t[1])
            if useful:
                st.dataframe(pd.DataFrame([{"Ask about": a, "Expected reduction in uncertainty": f"{g:.0f}%"} for a, g in useful[:3]]),
                             hide_index=True, width="stretch")
            else:
                st.caption("No missing section is expected to reduce the model's uncertainty by 1% or more for this record.")
        else:
            fc = load_table("questions_first_choice.csv")
            if fc is not None:
                st.markdown("Across held-out people, the model most often asks first about: **" +
                            ", ".join(fc["section"].head(3)) + "**.")
            st.caption("A personal ranking needs the local model file, which is not published.")
        st.subheader("Intent, if the death was an injury")
        p_int = bundle["intent"].predict_proba(X)[0]
        st.dataframe(pd.DataFrame({"Intent": INTENT_NAMES, "Probability": np.round(p_int, 3)}), hide_index=True, width="stretch")
        st.caption("Intent is hard to estimate from these records: held-out accuracy is about 46% across models.")

# ---------------- Compare missing fields ----------------
with tab_cmp:
    st.markdown("Take the person from the **Case simulator** and remove whole sections, as if they had never been collected. "
                "Each version is the **same person with less information**, so differences come only from the missing sections.")
    options = recorded
    if not options:
        st.warning("Record at least one section in the Case simulator first.")
    else:
        default_a = [s for s in ["Drinking", "Drug use", "Tobacco"] if s in options][:2] or options[:1]
        default_b = [s for s in ["Diagnoses", "Past injuries and surgery"] if s in options] or options[-1:]
        ca, cb = st.columns(2)
        hide_a = ca.multiselect("Version A: sections missing", options, default=default_a)
        hide_b = cb.multiselect("Version B: sections missing", options, default=default_b)
        rows, prob_rows = [], []
        for name, hidden_labels in [("Record as entered", []), ("Version A", hide_a), ("Version B", hide_b)]:
            hidden = [s for s in SEC.names if mu.SECTION_LABELS[s] in hidden_labels]
            m = SEC.mask(1, hidden)
            Xs = SEC.as_not_recorded(X, m)
            ps = bundle["standard"].predict_proba(Xs)[0]
            nb = int(SEC.not_recorded(Xs).sum())
            row = {"Version": name, "Sections missing": ", ".join(hidden_labels) if hidden_labels else "none added",
                   "Sections recorded": f"{8 - nb} of 8",
                   "Standard model": f"{CAUSE_NAMES[int(np.argmax(ps))]} ({100 * ps.max():.0f}%)",
                   "Standard model, 90% shortlist": ", ".join(shortlist(bundle, ps, "standard", nb) or ["not available"])}
            prob_rows.append({"Version": name, "Model": "Standard", **{c: round(float(v), 3) for c, v in zip(CAUSE_NAMES, ps)}})
            if bundle["withheld"] is not None:
                Xw2 = SEC.as_withheld(X, SEC.not_recorded(X) | m)
                pw = bundle["withheld"].predict_proba(Xw2[bundle["colsw"]])[0]
                row["Withholding-trained model"] = f"{CAUSE_NAMES[int(np.argmax(pw))]} ({100 * pw.max():.0f}%)"
                row["Withholding-trained, 90% shortlist"] = ", ".join(shortlist(bundle, pw, "withheld") or ["not available"])
                prob_rows.append({"Version": name, "Model": "Withholding-trained",
                                  **{c: round(float(v), 3) for c, v in zip(CAUSE_NAMES, pw)}})
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown("**Probabilities for each cause**")
        st.dataframe(pd.DataFrame(prob_rows), hide_index=True, width="stretch")
        st.caption("The standard model reads a removed section as one the investigator left blank; the withholding-trained "
                   "model reads it as not available. In held-out tests with every section removed, the withholding-trained "
                   "model kept 52.8% accuracy against 42.9% for the standard model.")

# ---------------- Results dashboard ----------------
with tab_res:
    t_over, t_audit, t_acc, t_miss, t_rel, t_expl, t_ctrl = st.tabs(
        ["Overview", "Data audit", "Accuracy", "Missing information", "Reliability", "Explanation", "Controls"])

    with t_over:
        acc = load_table("accuracy_metrics.csv")
        scen = load_table("missing_withheld_scenarios.csv")
        conf = load_table("reliability_conformal.csv")
        vt = load_table("validation_temporal.csv")
        if all(x is not None for x in (acc, scen, conf, vt)):
            a = acc[(acc["level"] == "Level1") & (acc["model"] == "HistGB")].set_index("feature_set")
            s = scen.set_index(["hidden_at_test_time", "handling"])
            every = "Everything except age and sex"
            c90 = conf[(conf["model"] == "standard") & (conf["method"] == "marginal") & (conf["group_type"] == "all people") &
                       (conf["target_coverage_pct"] == 90)].iloc[0]
            v = vt.set_index("inputs").loc["all inputs (standard model)"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Correct cause, all inputs", f"{100 * a.loc['M2_med_history', 'accuracy']:.1f}%",
                      f"{100 * (a.loc['M2_med_history', 'accuracy'] - a.loc['M0_base', 'accuracy']):+.1f} points vs age and sex")
            m2.metric("Every section missing, withholding-trained", f"{100 * s.loc[(every, 'withheld_model'), 'accuracy']:.1f}%",
                      f"{100 * (s.loc[(every, 'withheld_model'), 'accuracy'] - s.loc[(every, 'not_recorded'), 'accuracy']):+.1f} points vs reading gaps as blanks")
            m3.metric("90% shortlist coverage", f"{c90['coverage_pct']:.1f}%", f"average length {c90['mean_shortlist_size']:.2f} of 3",
                      delta_color="off")
            m4.metric("Later years (2015 on)", f"{100 * v['accuracy']:.1f}%", "trained on deaths up to 2014", delta_color="off")
        figure("missing_information_withheld.png", "Same people, less information: accuracy as sections are hidden at random")
        st.markdown("**Summary of reliability, explanation, validation and controls**")
        table("paper_table4_reliability_explanation_controls.csv")

    with t_audit:
        st.markdown("Who is in the study, which fields investigators record for each cause, and why some columns are never used.")
        c1, c2 = st.columns(2)
        with c1:
            table("cohort_flow.csv", rename={"step": "Step", "count": "People"})
            table("paper_table1_cohort.csv")
        with c2:
            figure("coverage_heatmap.png", "Which fields investigators record, by cause of death (%)")
            table("leakage_check.csv", rename={"forbidden_column": "Blocked column", "n_categories": "Categories",
                                               "cramers_v_with_level1": "Cramér's V with the broad cause"})
        full_table("bmi_flag_audit.csv", "cadaver body-size flag audit")
        full_table("cart_invariance_check.csv", "people found in two data carts")
        full_table("intent_parsing_audit.csv", "how manner-of-death text was classified")

    with t_acc:
        st.markdown("How often the models name the correct broad cause, with which inputs, and whether this holds for later "
                    "years and other regions.")
        c1, c2 = st.columns(2)
        with c1:
            figure("accuracy_by_inputs.png", "Accuracy and macro AUC by inputs and model")
        with c2:
            figure("validation.png", "Later years and held-out regions")
        table("paper_table2_accuracy.csv")
        full_table("accuracy_per_class.csv", "results for each cause")
        full_table("confusion_matrix.csv", "confusion matrix")
        full_table("accuracy_subgroups.csv", "results by sex, age, race and ethnicity")
        full_table("validation_temporal.csv", "later years")
        full_table("validation_regional.csv", "held-out county groups")

    with t_miss:
        st.markdown("What happens to the estimate when sections of the record are missing, for records as they are and for "
                    "the same people with sections removed.")
        c1, c2 = st.columns(2)
        with c1:
            figure("missing_information_natural.png", "Records as they are, and one section removed at a time")
        with c2:
            figure("missing_information_withheld.png", "Same people, less information")
        st.markdown("**Accuracy by missing fields and true cause** (withholding-trained model)")
        table("paper_table3_missing_information.csv")
        figure("missing_question_order.png", "Which question to ask next")
        table("questions_first_choice.csv", rename={"asked_first_by_model_pct": "Asked first by the model (%)",
                                                    "first_in_fixed_order_pct_of_folds": "First in the fixed order (% of folds)"})
        full_table("paper_table3b_hidden_sections.csv", "every handling of hidden sections")
        full_table("paper_table3c_records_as_they_are.csv", "records as they are, by number of blank sections")
        full_table("missing_natural_patterns.csv", "most common patterns of blank sections")

    with t_rel:
        st.markdown("When a single estimate can be trusted: calibration, answering only when confident, and shortlists with "
                    "a stated coverage.")
        c1, c2 = st.columns(2)
        with c1:
            figure("reliability_calibration.png", "Calibration of the predicted probabilities")
        with c2:
            figure("reliability_shortlists.png", "Confident answers and 90% shortlists")
        st.markdown("**Shortlists, standard model, all people**")
        table("reliability_conformal.csv",
              columns=["method", "target_coverage_pct", "coverage_pct", "mean_shortlist_size", "single_cause_pct"],
              rename={"method": "Threshold", "target_coverage_pct": "Target (%)", "coverage_pct": "Coverage reached (%)",
                      "mean_shortlist_size": "Average length (of 3)", "single_cause_pct": "Single cause (%)"},
              query="model == 'standard' and group_type == 'all people'")
        st.markdown("**Answering only when confident, standard model**")
        table("reliability_selective.csv",
              columns=["min_top_probability", "answered_pct", "accuracy_answered", "accuracy_ci_low", "accuracy_ci_high"],
              rename={"min_top_probability": "Answer if top probability is at least", "answered_pct": "People answered (%)",
                      "accuracy_answered": "Accuracy", "accuracy_ci_low": "95% CI low", "accuracy_ci_high": "95% CI high"},
              query="model == 'standard'")
        full_table("reliability_conformal.csv", "every shortlist version and group")
        full_table("reliability_calibration.csv", "calibration by cause")

    with t_expl:
        st.markdown("Why the estimate survives missing information: how much comes from what the record says, and how much "
                    "from which sections investigators filled in.")
        c1, c2 = st.columns(2)
        with c1:
            figure("explanation_content_vs_blanks.png", "Which sections are filled in versus what they say")
        with c2:
            figure("explanation_shap.png", "Inputs that move the estimate most")
        table("explanation_gain_shares.csv",
              columns=["model", "block", "auc_share_blank_pattern_pct", "auc_share_content_pct"],
              rename={"model": "Model", "block": "Inputs", "auc_share_blank_pattern_pct": "Share from which sections are filled in (%)",
                      "auc_share_content_pct": "Share from what they say, at least (%)"},
              query="level == 'Level1'")
        full_table("explanation_regimes.csv", "every comparison, including complete cases")
        full_table("factors_adjusted_level1.csv", "factors after adjusting for age and sex")

    with t_ctrl:
        st.markdown("Checks that the results are honest: a negative control, shuffled labels, robustness checks, and a "
                    "demonstration of why some columns are blocked.")
        figure("controls.png", "Negative control, permutation test and robustness checks")
        table("controls_permutation.csv",
              columns=["level", "n_permutations", "observed_auc_1rep", "null_mean", "null_2.5pct", "null_97.5pct", "empirical_p"],
              rename={"level": "Outcome", "n_permutations": "Shuffles", "observed_auc_1rep": "AUC, real labels",
                      "null_mean": "AUC, shuffled (mean)", "null_2.5pct": "Shuffled, 2.5%", "null_97.5pct": "Shuffled, 97.5%",
                      "empirical_p": "p"})
        table("controls_leakage_demo.csv", columns=["inputs", "accuracy", "macro_auc"],
              rename={"inputs": "Inputs", "accuracy": "Accuracy", "macro_auc": "Macro AUC"})
        st.caption("The cart variable is blocked from every model: on its own it identifies the cause almost perfectly, "
                   "because each data cart was drawn for one cause.")
        full_table("controls_level3.csv", "negative control (stab versus gunshot)")
        full_table("controls_robustness.csv", "robustness checks")
        full_table("controls_withholding_design.csv", "withholding designs")
