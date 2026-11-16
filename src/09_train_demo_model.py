"""Train the final demo models used by the Streamlit app (app/app.py).

Fits HistGradientBoosting on ALL eligible people (no CV: the CV estimates of
performance are in results/tables). Saves to models/ (git-ignored because the
models are derived from restricted NMDID data).

  models/demo_models.joblib  ->  dict with:
      level1_M0, level1_M2  (natural_cardiac / drug_alcohol / injury)
      level2_M2             (accident / homicide / suicide, injury deaths only)
      columns_M0, columns_M2, class names, training sizes
"""
import warnings
warnings.filterwarnings("ignore")
import joblib
import pandas as pd
import config
import model_utils as mu

if __name__ == "__main__":
    cohort, features = mu.load_data()
    nested = mu.get_nested_sets(features)
    cols_m0, cols_m2 = nested["M0_base"], nested["M2_med_history"]

    l1 = cohort[cohort["level1"].isin(mu.LEVEL1_CLASSES)][[config.KEY, "level1"]].merge(
        features, on=config.KEY)
    l2 = cohort[(cohort["level1"] == "injury") & cohort["intent"].isin(mu.LEVEL2_CLASSES)][
        [config.KEY, "intent"]].merge(features, on=config.KEY)

    y1 = l1["level1"].map({c: i for i, c in enumerate(mu.LEVEL1_CLASSES)}).values
    y2 = l2["intent"].map({c: i for i, c in enumerate(mu.LEVEL2_CLASSES)}).values

    bundle = {
        "level1_M0": mu.make_model("HistGB", n_train=len(l1)).fit(l1[cols_m0], y1),
        "level1_M2": mu.make_model("HistGB", n_train=len(l1)).fit(l1[cols_m2], y1),
        "level2_M2": mu.make_model("HistGB", n_train=len(l2)).fit(l2[cols_m2], y2),
        "columns_M0": cols_m0,
        "columns_M2": cols_m2,
        "level1_classes": mu.LEVEL1_CLASSES,
        "level2_classes": mu.LEVEL2_CLASSES,
        "n_level1": len(l1),
        "n_level2": len(l2),
        # Median BMI values, used as defaults in the app form.
        "median_cadaver_bmi": float(l1["cadaver_bmi"].median()),
        "median_living_bmi": float(l1["living_bmi"].median()),
    }
    out_dir = config.PROJECT_ROOT / "models"
    out_dir.mkdir(exist_ok=True)
    joblib.dump(bundle, out_dir / "demo_models.joblib")

    # Also export aggregate linear (LogisticRegression) parameters to results/tables/
    # so the Streamlit Cloud / GitHub-only demo works without models/demo_models.joblib.
    lr_l1_m0 = mu.make_model("LogReg").fit(l1[cols_m0], y1)
    lr_l1_m2 = mu.make_model("LogReg").fit(l1[cols_m2], y1)
    lr_l2_m2 = mu.make_model("LogReg").fit(l2[cols_m2], y2)

    def pack_logreg(pipe, cols, name, classes):
        imp = pipe.named_steps["imputer"]
        scl = pipe.named_steps["scaler"]
        clf = pipe.named_steps["clf"]
        rows = []
        for idx, col in enumerate(cols):
            r = {
                "model_id": name,
                "feature": col,
                "impute_median": float(imp.statistics_[idx]),
                "scale_mean": float(scl.mean_[idx]),
                "scale_std": float(scl.scale_[idx]),
            }
            for c_i, c_name in enumerate(classes):
                r[f"coef_{c_name}"] = float(clf.coef_[c_i, idx])
                r[f"intercept_{c_name}"] = float(clf.intercept_[c_i])
            rows.append(r)
        return rows

    w_rows = (
        pack_logreg(lr_l1_m0, cols_m0, "level1_M0", mu.LEVEL1_CLASSES)
        + pack_logreg(lr_l1_m2, cols_m2, "level1_M2", mu.LEVEL1_CLASSES)
        + pack_logreg(lr_l2_m2, cols_m2, "level2_M2", mu.LEVEL2_CLASSES)
    )
    pd.DataFrame(w_rows).to_csv(config.TABLES_DIR / "demo_logreg_weights.csv", index=False)

    print("Saved models/demo_models.joblib and results/tables/demo_logreg_weights.csv")
    print("Level 1 training n:", len(l1), "| Level 2 training n:", len(l2))
    print("M2 columns:", len(cols_m2))

