"""Step 6 - Reliability: when can a single estimate be trusted?

Uses the held-out probabilities cached by 04_missing_information.py
(standard model, withholding-trained model, age + sex model).

  A. Calibration: does a predicted 70% come true about 70% of the time?
     Reliability curves, expected calibration error and calibration slope.
  B. Answer only when confident: accuracy among people whose top probability
     passes a threshold, and the share of people who get an answer.
  C. Shortlists with a coverage target (split conformal prediction): a set of
     causes that contains the true cause for 90%, 95% or 99% of people, and how
     long the shortlist has to be, overall, by true cause and by how much of the
     record is blank. Three versions: marginal (one threshold), class-conditional
     (one per true cause) and missingness-conditional (one per band of blank
     sections, so the coverage target holds however incomplete the record is).

Outputs: tables/reliability_calibration.csv, reliability_calibration_bins.csv,
         reliability_selective.csv, reliability_conformal.csv,
         figures/reliability_calibration.png, figures/reliability_shortlists.png
"""
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
import model_utils as mu

C = mu.LEVEL1_CLASSES
MODELS = {"standard": "Standard model", "withheld": "Withholding-trained model",
          "withheld_recal": "Withholding-trained model, recalibrated"}
BANDS = [("none", 0, 0), ("1 to 3", 1, 3), ("4 or 5", 4, 5), ("6", 6, 6), ("7 or 8", 7, 8)]
THRESHOLDS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ALPHAS = [0.10, 0.05, 0.01]
N_SPLITS = 500
N_BINS = 10


def calibration(y, p, model):
    bins_rows, rows = [], []
    for i, c in enumerate(C):
        pi, yi = p[:, i], (y == i).astype(float)
        edges = np.quantile(pi, np.linspace(0, 1, N_BINS + 1))
        b = np.clip(np.searchsorted(edges[1:-1], pi, side="right"), 0, N_BINS - 1)
        ece = 0.0
        for k in range(N_BINS):
            m = b == k
            if not m.any():
                continue
            ece += m.mean() * abs(pi[m].mean() - yi[m].mean())
            bins_rows.append({"model": model, "class_name": c, "bin": k + 1, "n": int(m.sum()),
                              "mean_predicted": round(float(pi[m].mean()), 3), "observed_rate": round(float(yi[m].mean()), 3)})
        z = np.log(np.clip(pi, 1e-6, 1 - 1e-6) / np.clip(1 - pi, 1e-6, 1))
        fit = sm.Logit(yi, sm.add_constant(z)).fit(disp=0)
        rows.append({"model": model, "class_name": c, "mean_predicted": round(float(pi.mean()), 3),
                     "observed_rate": round(float(yi.mean()), 3), "expected_calibration_error": round(float(ece), 3),
                     "calibration_slope": round(float(fit.params[1]), 2),
                     "calibration_slope_ci_low": round(float(fit.conf_int()[1, 0]), 2),
                     "calibration_slope_ci_high": round(float(fit.conf_int()[1, 1]), 2),
                     "brier_one_class": round(float(np.mean((pi - yi) ** 2)), 3)})
    return rows, bins_rows


def band_of(n_blank):
    out = np.empty(len(n_blank), dtype=object)
    for name, lo, hi in BANDS:
        out[(n_blank >= lo) & (n_blank <= hi)] = name
    return out


def quantile_threshold(scores, alpha):
    """Conformal threshold: the ceil((m + 1)(1 - alpha))-th smallest of m scores.
    When that exceeds m, every cause is included (threshold 1)."""
    sc = np.sort(scores)
    k = int(np.ceil((len(sc) + 1) * (1 - alpha)))
    return float(sc[k - 1]) if k <= len(sc) else 1.0


def conformal(p, y, groups, alpha, method, rng, strata):
    """Repeated split conformal with the score 1 - p(true cause).
    Half of the people calibrate the threshold, the other half are evaluated."""
    n = len(y)
    stats = {}
    for _ in range(N_SPLITS):
        perm = rng.permutation(n)
        cal, ev = perm[: n // 2], perm[n // 2:]
        score = 1 - p[cal, y[cal]]
        sets = np.zeros((len(ev), p.shape[1]), dtype=bool)
        if method == "marginal":
            sets = p[ev] >= 1 - quantile_threshold(score, alpha)
        elif method == "class-conditional":       # one threshold per true cause
            for c in range(p.shape[1]):
                sets[:, c] = p[ev, c] >= 1 - quantile_threshold(score[y[cal] == c], alpha)
        else:                                     # missingness-conditional: one threshold per band
            for b in np.unique(strata):
                m = strata[ev] == b
                sets[m] = p[ev[m]] >= 1 - quantile_threshold(score[strata[cal] == b], alpha)
        hit = sets[np.arange(len(ev)), y[ev]]
        size = sets.sum(axis=1)
        for gtype, gvals in groups.items():
            for g in np.unique(gvals):
                m = gvals[ev] == g
                if not m.any():
                    continue
                s = stats.setdefault((gtype, g), {"cov": [], "size": [], "single": [], "empty": [], "full": []})
                s["cov"].append(hit[m].mean()); s["size"].append(size[m].mean())
                s["single"].append((size[m] == 1).mean()); s["empty"].append((size[m] == 0).mean())
                s["full"].append((size[m] == p.shape[1]).mean())
    return stats


def threshold_all(p, y, alpha):
    """Threshold from all the given people (used by the demo app for new inputs)."""
    return quantile_threshold(1 - p[np.arange(len(y)), y], alpha)


if __name__ == "__main__":
    t0 = time.time()
    cache = pd.read_csv(config.PROCESSED_DIR / "oof_level1.csv")
    y = cache["y"].values
    probs = {m: cache[[f"p_{m}_{c}" for c in C]].values for m in ["age_sex"] + list(MODELS)}
    n_blank = cache["n_sections_not_recorded"].values
    bands = band_of(n_blank)
    print("People:", len(y), "| check: standard accuracy", round(mu.accuracy(y, probs["standard"]), 3),
          "| withholding-trained accuracy", round(mu.accuracy(y, probs["withheld"]), 3))

    # ---------- A. Calibration ----------
    cal_rows, bin_rows = [], []
    for m in MODELS:
        r, b = calibration(y, probs[m], m)
        cal_rows += r
        bin_rows += b
    cal = pd.DataFrame(cal_rows)
    cal.to_csv(config.TABLES_DIR / "reliability_calibration.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(config.TABLES_DIR / "reliability_calibration_bins.csv", index=False)
    print("\n--- Calibration ---")
    print(cal.to_string(index=False))

    # ---------- B. Answer only when confident ----------
    rng = np.random.default_rng(config.SEED)
    rows = []
    for m in MODELS:
        p = probs[m]
        conf, ok = p.max(axis=1), p.argmax(axis=1) == y
        for thr in THRESHOLDS:
            sel = np.flatnonzero(conf >= thr)
            row = {"model": m, "min_top_probability": thr, "n_answered": len(sel),
                   "answered_pct": round(100 * len(sel) / len(y), 1)}
            if len(sel) >= 10:
                draws = ok[sel][rng.integers(0, len(sel), size=(config.N_BOOT, len(sel)))].mean(axis=1)
                row.update({"accuracy_answered": round(float(ok[sel].mean()), 3),
                            "accuracy_ci_low": round(float(np.percentile(draws, 2.5)), 3),
                            "accuracy_ci_high": round(float(np.percentile(draws, 97.5)), 3)})
            rows.append(row)
    sel_df = pd.DataFrame(rows)
    sel_df.to_csv(config.TABLES_DIR / "reliability_selective.csv", index=False)
    print("\n--- Answer only when confident ---")
    print(sel_df.to_string(index=False))

    # ---------- C. Shortlists with a coverage target ----------
    groups = {"all people": np.array(["all people"] * len(y), dtype=object),
              "true cause": np.array([C[i] for i in y], dtype=object),
              "sections not recorded": bands}
    rows = []
    for m in MODELS:
        for method in ["marginal", "class-conditional", "missingness-conditional"]:
            for alpha in ALPHAS:
                stats = conformal(probs[m], y, groups, alpha, method, np.random.default_rng(config.SEED), bands)
                for (gtype, g), s in stats.items():
                    thr = np.nan
                    if method == "marginal" and gtype == "all people":
                        thr = round(threshold_all(probs[m], y, alpha), 4)
                    elif method == "missingness-conditional" and gtype == "sections not recorded":
                        thr = round(threshold_all(probs[m][bands == g], y[bands == g], alpha), 4)
                    rows.append({"model": m, "method": method, "target_coverage_pct": round(100 * (1 - alpha)),
                                 "group_type": gtype, "group": g, "n_people": int((groups[gtype] == g).sum()),
                                 "coverage_pct": round(100 * float(np.mean(s["cov"])), 1),
                                 "coverage_sd_pct": round(100 * float(np.std(s["cov"], ddof=1)), 1),
                                 "mean_shortlist_size": round(float(np.mean(s["size"])), 2),
                                 "single_cause_pct": round(100 * float(np.mean(s["single"])), 1),
                                 "all_three_pct": round(100 * float(np.mean(s["full"])), 1),
                                 "empty_pct": round(100 * float(np.mean(s["empty"])), 1),
                                 "threshold_all_people": thr})
        mu.log(f"conformal {m} done", t0)
    conf_df = pd.DataFrame(rows)
    conf_df.to_csv(config.TABLES_DIR / "reliability_conformal.csv", index=False)
    print("\n--- Shortlists (all people) ---")
    print(conf_df[conf_df["group_type"] == "all people"].drop(columns=["group_type", "group"]).to_string(index=False))
    print("\n--- 90% shortlist by band of blank sections (standard model) ---")
    print(conf_df[(conf_df["model"] == "standard") & (conf_df["target_coverage_pct"] == 90) &
                  (conf_df["group_type"] == "sections not recorded")].to_string(index=False))

    # ---------- Figures ----------
    bins = pd.DataFrame(bin_rows)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, c in zip(axes, C):
        for m, color in [("standard", "#c0392b"), ("withheld", "#1f77b4"), ("withheld_recal", "#2ca02c")]:
            d = bins[(bins["model"] == m) & (bins["class_name"] == c)]
            ece = cal[(cal["model"] == m) & (cal["class_name"] == c)]["expected_calibration_error"].iloc[0]
            ax.plot(d["mean_predicted"], d["observed_rate"], marker="o", color=color, label=f"{MODELS[m]} (ECE {ece:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1)
        ax.set_title(mu.CLASS_LABELS[c], fontsize=10); ax.set_xlabel("Predicted probability")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(linestyle="--", alpha=0.5); ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Observed share")
    fig.suptitle("Calibration: does a predicted probability come true that often? (held-out people, deciles)", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "reliability_calibration.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for m, color in [("standard", "#c0392b"), ("withheld", "#1f77b4")]:
        d = sel_df[(sel_df["model"] == m) & sel_df["accuracy_answered"].notna()]
        axes[0].errorbar(d["answered_pct"], d["accuracy_answered"],
                         yerr=[d["accuracy_answered"] - d["accuracy_ci_low"], d["accuracy_ci_high"] - d["accuracy_answered"]],
                         marker="o", capsize=2, color=color, label=MODELS[m])
    axes[0].set_xlabel("People who get an answer (%)"); axes[0].set_ylabel("Accuracy among them")
    axes[0].invert_xaxis(); axes[0].grid(linestyle="--", alpha=0.5); axes[0].legend(fontsize=8)
    axes[0].set_title("Answer only when confident", fontsize=10)
    d = conf_df[(conf_df["model"] == "standard") & (conf_df["target_coverage_pct"] == 90) &
                (conf_df["group_type"] == "sections not recorded")]
    order = [b[0] for b in BANDS]
    for k, (meth, color, lab) in enumerate([("marginal", "#c0392b", "One threshold for everyone"),
                                            ("missingness-conditional", "#1f77b4", "Threshold adjusted for blank sections")]):
        sub = d[d["method"] == meth].set_index("group").reindex(order)
        xs = np.arange(len(order)) + (k - 0.5) * 0.38
        axes[1].bar(xs, sub["coverage_pct"], 0.38, color=color, label=lab)
        for x_, cov, size in zip(xs, sub["coverage_pct"], sub["mean_shortlist_size"]):
            axes[1].text(x_, cov + 0.3, f"{size:.2f}", ha="center", fontsize=7)
    axes[1].axhline(90, color="grey", linestyle="--", linewidth=1)
    axes[1].set_xticks(range(len(order))); axes[1].set_xticklabels(order)
    axes[1].set_xlabel("Sections the investigator left blank (of 8)"); axes[1].set_ylabel("True cause inside the shortlist (%)")
    axes[1].set_ylim(80, 100); axes[1].grid(axis="y", linestyle="--", alpha=0.5); axes[1].legend(fontsize=8, loc="lower right")
    axes[1].set_title("90% shortlists, standard model (labels: average length)", fontsize=10)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "reliability_shortlists.png", dpi=150)
    plt.close(fig)
    mu.log("06_reliability finished", t0)
