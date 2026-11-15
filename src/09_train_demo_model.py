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
    print("Saved models/demo_models.joblib")
    print("Level 1 training n:", len(l1), "| Level 2 training n:", len(l2))
    print("M2 columns:", len(cols_m2))

