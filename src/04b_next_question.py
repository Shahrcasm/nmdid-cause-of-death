"""Step 4b - Which question to ask next?

For a record with sections not available, the withholding-trained model ranks
the missing sections by how much the answer is expected to reduce its
uncertainty about the cause of death (expected information gain). Possible
answers come from training people: for each cause, from people who died of that
cause, weighted by the model's current probability of that cause. The person's
true cause and true answers are never used to choose a question.

Evaluation (held-out people, the same 5 x 5 cross-validation): every test person
starts with only age and sex, then sections are revealed one at a time in three
orders:
  guided  the section with the largest expected gain for this person
  fixed   one order for everyone (largest average expected gain at the start)
  random  a random order per person
Accuracy and the probability of the true cause are recorded after each question.

Outputs: tables/questions_curve.csv, questions_comparison.csv, questions_first_choice.csv,
         figures/missing_question_order.png
"""
import time
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config
import model_utils as mu

K_DONORS = 10                      # possible answers per cause and section
POLICIES = ["guided", "fixed", "random"]
LABELS = {"guided": "Model chooses the next question for each person",
          "fixed": "One fixed order for everyone", "random": "Random order"}


def expected_entropy(model, sec, S, rows_i, j, donors, p_now):
    """Expected uncertainty after asking section j, for the people in rows_i."""
    idx, wpos = sec.section_positions(j)
    blocks = []
    for c in range(len(donors)):
        B = np.repeat(S[rows_i], K_DONORS, axis=0)
        B[:, idx] = np.tile(donors[c][:, idx], (len(rows_i), 1))
        B[:, wpos] = 0.0
        blocks.append(B)
    q = model.predict_proba(pd.DataFrame(np.vstack(blocks), columns=sec.all_columns))
    H = mu.entropy(q).reshape(len(donors), len(rows_i), K_DONORS).mean(axis=2)    # cause x person
    return (p_now[rows_i] * H.T).sum(axis=1)


def all_expected(model, sec, S, asked, donors):
    p_now = model.predict_proba(pd.DataFrame(S, columns=sec.all_columns))
    E = np.full(asked.shape, np.inf)
    for j in range(asked.shape[1]):
        rows_i = np.flatnonzero(~asked[:, j])
        if len(rows_i):
            E[rows_i, j] = expected_entropy(model, sec, S, rows_i, j, donors, p_now)
    return E, p_now


def reveal(sec, S, asked, X_plain, choice):
    for j in np.unique(choice):
        rows_i = np.flatnonzero(choice == j)
        idx, wpos = sec.section_positions(j)
        S[np.ix_(rows_i, idx)] = X_plain[np.ix_(rows_i, idx)]
        S[rows_i, wpos] = 0.0
        asked[rows_i, j] = True


if __name__ == "__main__":
    t0 = time.time()
    cohort, features = mu.load_data()
    m2 = mu.get_nested_sets(features)["M2_med_history"]
    l1, y = mu.level1_frame(cohort, features)
    C, n = mu.LEVEL1_CLASSES, len(y)
    sec = mu.Sections(m2)
    S_n = len(sec.names)
    X = l1[m2]
    splits, R = mu.cv_splits(X, y)
    P = {pol: np.zeros((S_n + 1, R, n, 3)) for pol in POLICIES}
    first = {pol: np.full((R, n), -1) for pol in POLICIES}
    fixed_orders = []

    for f, (rep, tr, te) in enumerate(splits):
        model = sec.fit_withholding_model(X.iloc[tr], y[tr], seed=config.SEED + f)   # same model as step 4
        rng = np.random.default_rng(config.SEED + 2000 + f)
        Xtr_plain = X.iloc[tr][sec.columns].values.astype(float)
        donors = [Xtr_plain[rng.choice(np.flatnonzero(y[tr] == c), K_DONORS, replace=False)] for c in range(len(C))]
        X_plain = X.iloc[te][sec.columns].values.astype(float)
        S0 = sec.as_withheld(X.iloc[te], sec.mask(len(te), sec.names)).values.astype(float)
        E0, p0 = all_expected(model, sec, S0, np.zeros((len(te), S_n), dtype=bool), donors)
        gain0 = (mu.entropy(p0)[:, None] - E0).mean(axis=0)
        fixed_order = list(np.argsort(-gain0))
        fixed_orders.append([sec.names[j] for j in fixed_order])
        perms = np.argsort(rng.random((len(te), S_n)), axis=1)
        for pol in POLICIES:
            S, asked = S0.copy(), np.zeros((len(te), S_n), dtype=bool)
            P[pol][0, rep, te] = p0
            for step in range(S_n):
                if pol == "guided":
                    E = E0 if step == 0 else all_expected(model, sec, S, asked, donors)[0]
                    choice = E.argmin(axis=1)
                elif pol == "fixed":
                    choice = np.full(len(te), fixed_order[step])
                else:
                    choice = perms[:, step]
                if step == 0:
                    first[pol][rep, te] = choice
                reveal(sec, S, asked, X_plain, choice)
                P[pol][step + 1, rep, te] = model.predict_proba(pd.DataFrame(S, columns=sec.all_columns))
        if (f + 1) % 5 == 0:
            mu.log(f"repeat {rep + 1} of {R} done", t0)

    # ---------- Curve: accuracy after k questions (mean over single repeats, as in step 4) ----------
    rows = []
    for pol in POLICIES:
        for k in range(S_n + 1):
            accs = [mu.accuracy(y, P[pol][k, r]) for r in range(R)]
            ptrue = [float(P[pol][k, r][np.arange(n), y].mean()) for r in range(R)]
            rows.append({"order": pol, "description": LABELS[pol], "questions_asked": k,
                         "accuracy": round(float(np.mean(accs)), 3),
                         "accuracy_sd": round(float(np.std(accs, ddof=1)), 4) if R > 1 else 0.0,
                         "mean_probability_of_true_cause": round(float(np.mean(ptrue)), 3)})
    curve = pd.DataFrame(rows)
    curve.to_csv(config.TABLES_DIR / "questions_curve.csv", index=False)
    print("\n--- Accuracy after k questions ---")
    print(curve.pivot(index="questions_asked", columns="order", values="accuracy").to_string())

    # ---------- Paired comparison after 1, 2 and 3 questions (repeat-averaged probabilities) ----------
    cmp_rows = []
    for k in [1, 2, 3]:
        sets = {pol: P[pol][k].mean(axis=0) for pol in POLICIES}
        _, g = mu.bootstrap_compare(y, sets, pairs=[("guided_minus_fixed", "guided", "fixed"),
                                                    ("guided_minus_random", "guided", "random")])
        for r in g:
            cmp_rows.append({"questions_asked": k, "comparison": r["comparison"],
                             "accuracy_difference": r["accuracy_gain"], "ci_low": r["accuracy_ci_low"], "ci_high": r["accuracy_ci_high"],
                             "macro_auc_difference": r["macro_auc_gain"], "auc_ci_low": r["macro_auc_ci_low"], "auc_ci_high": r["macro_auc_ci_high"]})
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(config.TABLES_DIR / "questions_comparison.csv", index=False)
    print("\n--- Guided versus other orders ---")
    print(cmp_df.to_string(index=False))

    # ---------- Which question is asked first ----------
    rows = []
    fo = pd.Series([o[0] for o in fixed_orders]).value_counts()
    for j, s in enumerate(sec.names):
        row = {"section": mu.SECTION_LABELS[s],
               "asked_first_by_model_pct": round(100 * float((first["guided"] == j).mean()), 1),
               "first_in_fixed_order_pct_of_folds": round(100 * float(fo.get(s, 0)) / len(fixed_orders), 1)}
        for i, c in enumerate(C):
            row[f"asked_first_pct_true_{c}"] = round(100 * float((first["guided"][:, y == i] == j).mean()), 1)
        rows.append(row)
    fc = pd.DataFrame(rows).sort_values("asked_first_by_model_pct", ascending=False)
    fc.to_csv(config.TABLES_DIR / "questions_first_choice.csv", index=False)
    print("\n--- Question asked first ---")
    print(fc.to_string(index=False))

    # ---------- Figure ----------
    colors = {"guided": "#1f77b4", "fixed": "#2c3e50", "random": "#c0392b"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for pol in POLICIES:
        d = curve[curve["order"] == pol]
        axes[0].errorbar(d["questions_asked"], d["accuracy"], yerr=d["accuracy_sd"], marker="o", capsize=2, color=colors[pol], label=LABELS[pol])
        axes[1].plot(d["questions_asked"], d["mean_probability_of_true_cause"], marker="o", color=colors[pol], label=LABELS[pol])
    axes[0].set_ylabel("Accuracy"); axes[1].set_ylabel("Average probability given to the true cause")
    for ax in axes:
        ax.set_xlabel("Sections asked about (of 8), starting from age and sex only")
        ax.grid(linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=8)
    fig.suptitle("Which question to ask next: accuracy as sections are added one at a time (held-out people)", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "missing_question_order.png", dpi=150)
    plt.close(fig)
    mu.log("04b_next_question finished", t0)
