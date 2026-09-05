"""Reproducible analysis script for "Generative AI Use, Academic Anxiety
and Subjective Well-Being among Higher Education Students".

Run from a terminal:
    python3 03_analysis.py [path_to_dataset.csv]
The default path is ./dataset.csv. Output goes to stdout and is
also written to analysis_log.txt.

Dependencies: pandas, numpy, scipy, statsmodels.
Tested with Python 3.11+, pandas 2.0+, statsmodels 0.14+.

Reproduces:
  - Table 1 (sample characteristics)
  - Table 2 (descriptives, alphas, correlations)
  - Section 4.1 figures (purpose marginals, weekly+ %)
  - Section 4.2 group means and ANOVA (Figure 3)
  - Table 3 (hierarchical OLS regression)
  - Table 4 (parallel mediation, Figure 4)
"""
from __future__ import annotations

import argparse
import os
import sys
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import pandas as pd
import scipy.stats as ss
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# 0. Setup
# -----------------------------------------------------------------------------
RNG = np.random.default_rng(20250215)
N_BOOT = 5000


def standardise(x: pd.Series) -> pd.Series:
    """Return z-scored x using sample SD (ddof=1)."""
    return (x - x.mean()) / x.std()


def cronbach_alpha(items: pd.DataFrame) -> float:
    """Cronbach's alpha for a set of item columns."""
    arr = items.to_numpy(dtype=float)
    k = arr.shape[1]
    var_total = arr.sum(axis=1).var(ddof=1)
    var_items = arr.var(axis=0, ddof=1).sum()
    return (k / (k - 1)) * (1 - var_items / var_total)


def fmt_p(p: float) -> str:
    return "<.001" if p < 0.001 else f"{p:.3f}"


def section(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# -----------------------------------------------------------------------------
# 1. Load
# -----------------------------------------------------------------------------
def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# -----------------------------------------------------------------------------
# 2. Sample characteristics (Table 1)
# -----------------------------------------------------------------------------
def table1(df: pd.DataFrame) -> None:
    section("Table 1 \u2014 Sample characteristics (N = {})".format(len(df)))
    n = len(df)

    print(f"\nAge:    M = {df.age.mean():.2f}  SD = {df.age.std():.2f}")
    print(f"Income: M = {df.income.mean():.2f}  SD = {df.income.std():.2f}")
    print(f"GenAI use intensity: M = {df.genai_freq.mean():.2f}  "
          f"SD = {df.genai_freq.std():.2f}")

    def show_factor(label, series, order):
        print(f"\n{label}:")
        for k in order:
            cnt = int((series == k).sum())
            print(f"  {k:<35s}: {cnt:4d}  ({100 * cnt / n:5.1f}%)")

    show_factor("Sex", df.sex, ["F", "M", "O"])
    show_factor("Study cycle", df.cycle, ["BSc", "MSc", "Other"])
    show_factor("Field of study", df.field, ["STEM", "Health", "SocSci", "Arts"])
    show_factor("Institution", df.institution, ["UC", "IPC", "IPV", "IPP"])


# -----------------------------------------------------------------------------
# 3. Section 4.1 \u2014 Patterns of GenAI use
# -----------------------------------------------------------------------------
def section_4_1(df: pd.DataFrame) -> None:
    section("Section 4.1 \u2014 Patterns of GenAI use")
    counts = df.genai_freq.value_counts().sort_index()
    labels = {1: "Never", 2: "Rarely", 3: "Monthly", 4: "Weekly", 5: "Daily"}
    print("\nGenAI use intensity distribution:")
    for k, c in counts.items():
        print(f"  {labels[k]:<8s}: {int(c):3d}  ({100 * c / len(df):5.1f}%)")
    weekly_plus = (df.genai_freq >= 4).mean() * 100
    daily = (df.genai_freq == 5).mean() * 100
    print(f"\nWeekly or more frequent: {weekly_plus:.1f}%")
    print(f"Daily:                   {daily:.1f}%")

    print("\nPurpose marginals (% reporting weekly+ use):")
    purpose_cols = [c for c in df.columns if c.startswith("use_")]
    label_map = {
        "use_writing":     "Writing & editing assistance",
        "use_summarising": "Summarising readings",
        "use_brainstorm":  "Brainstorming / idea generation",
        "use_translation": "Translation",
        "use_coding":      "Programming / debugging code",
        "use_explaining":  "Explaining difficult concepts",
        "use_questions":   "Generating study questions",
        "use_homework":    "Solving homework directly",
    }
    rows = [(label_map.get(c, c), df[c].mean() * 100) for c in purpose_cols]
    rows.sort(key=lambda x: -x[1])
    for name, p in rows:
        print(f"  {name:<35s}: {p:5.1f}%")


# -----------------------------------------------------------------------------
# 4. Table 2 \u2014 descriptives, alphas, correlations
# -----------------------------------------------------------------------------
def table2(df: pd.DataFrame) -> None:
    section("Table 2 \u2014 Descriptives, internal consistency, correlations")

    anx_items = df[[f"anx{j}" for j in range(1, 6)]]
    se_items  = df[[f"se{j}"  for j in range(1, 9)]]
    eng_items = df[[f"eng{j}" for j in range(1, 10)]]
    who_items = df[[f"who_i{j}" for j in range(1, 6)]]

    rows = [
        ("GenAI use intensity",  df.genai_freq, None),
        ("AI-related anxiety",   df.anxiety,    anx_items),
        ("Academic self-efficacy", df.selfeff,  se_items),
        ("Academic engagement",  df.engage,     eng_items),
        ("Subjective well-being (WHO-5)", df.who5, who_items),
    ]
    print(f"\n{'Variable':<35s}  {'M':>7s}  {'SD':>6s}  {'alpha':>6s}")
    for name, var, items in rows:
        a = f"{cronbach_alpha(items):.2f}" if items is not None else "  \u2014 "
        print(f"  {name:<33s}  {var.mean():>7.2f}  {var.std():>6.2f}  {a:>6s}")

    # Correlation matrix with significance flags
    focal = pd.DataFrame({
        "1. GenAI":   df.genai_freq,
        "2. Anxiety": df.anxiety,
        "3. SelfEff": df.selfeff,
        "4. Engage":  df.engage,
        "5. WHO-5":   df.who5,
    })
    print("\nPearson correlations (lower triangle, with p-values):")
    cols = list(focal.columns)
    print("  ", "  ".join(f"{c:>9s}" for c in cols))
    n_obs = len(focal)
    for i, ri in enumerate(cols):
        line = f"  {ri:<9s}"
        for j, rj in enumerate(cols):
            if j > i:
                line += f"  {'':>9s}"
            elif i == j:
                line += f"  {'1.00':>9s}"
            else:
                r, p = ss.pearsonr(focal[ri], focal[rj])
                star = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ""))
                line += f"  {r:+.3f}{star:<3s}".rjust(11)
        print(line)

    print(f"\nWHO-5 \u2264 50: {(df.who5 <= 50).mean() * 100:.1f}% of sample")


# -----------------------------------------------------------------------------
# 5. Section 4.2 \u2014 group means + ANOVA (Figure 3)
# -----------------------------------------------------------------------------
def section_4_2(df: pd.DataFrame) -> None:
    section("Section 4.2 / Figure 3 \u2014 WHO-5 by GenAI use intensity")
    labels = {1: "Never", 2: "Rarely", 3: "Monthly", 4: "Weekly", 5: "Daily"}
    print(f"\n{'Group':<10s}  {'n':>4s}  {'Mean':>6s}  {'SD':>6s}  {'95% CI':>17s}")
    for g in range(1, 6):
        sub = df[df.genai_freq == g].who5
        mean = sub.mean()
        sd = sub.std()
        ci = 1.96 * sd / np.sqrt(len(sub))
        print(f"  {labels[g]:<8s}  {len(sub):>4d}  {mean:>6.1f}  {sd:>6.1f}  "
              f"[{mean - ci:>5.1f}, {mean + ci:>5.1f}]")

    groups = [df[df.genai_freq == g].who5.values for g in range(1, 6)]
    F, p = ss.f_oneway(*groups)
    grand = df.who5.mean()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_t = ((df.who5 - grand) ** 2).sum()
    eta2 = ss_b / ss_t
    print(f"\nOne-way ANOVA: F(4, {len(df) - 5}) = {F:.2f}, p = {fmt_p(p)}, "
          f"\u03B7\u00B2 = {eta2:.3f}")

    # Tukey HSD post-hoc (using statsmodels)
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    tk = pairwise_tukeyhsd(df.who5, df.genai_freq.map(labels), alpha=0.05)
    print("\nTukey HSD post-hoc (only contrasts involving 'Daily' shown):")
    for row in tk._results_table.data[1:]:
        g1, g2, *_ = row
        if "Daily" in (g1, g2):
            print("  ", row)


# -----------------------------------------------------------------------------
# 6. Table 3 \u2014 hierarchical OLS regression
# -----------------------------------------------------------------------------
def table3(df: pd.DataFrame) -> None:
    section("Table 3 \u2014 Hierarchical OLS regression on WHO-5 (standardised \u03B2s)")
    # Standardise everything (continuous outcome and predictors)
    Z = pd.DataFrame({
        "age":     standardise(df.age),
        "female":  standardise((df.sex == "F").astype(int)),
        "year":    standardise(df.year),
        "income":  standardise(df.income),
        "genai":   standardise(df.genai_freq),
        "anxiety": standardise(df.anxiety),
        "selfeff": standardise(df.selfeff),
    })
    y = standardise(df.who5)

    steps = {
        "Step 1": ["age", "female", "year", "income"],
        "Step 2": ["age", "female", "year", "income", "genai"],
        "Step 3": ["age", "female", "year", "income", "genai", "anxiety", "selfeff"],
    }
    fits = {name: sm.OLS(y, sm.add_constant(Z[cols])).fit() for name, cols in steps.items()}

    print(f"\n{'Predictor':<22s}  {'Step1 \u03B2':>10s}  {'Step2 \u03B2':>10s}  "
          f"{'Step3 \u03B2':>10s}  {'Step3 95% CI':>22s}  {'p (Step3)':>10s}")
    for v in ["age", "female", "year", "income", "genai", "anxiety", "selfeff"]:
        cells = []
        for name in ["Step 1", "Step 2", "Step 3"]:
            f = fits[name]
            if v in f.params.index:
                b = f.params[v]
                p = f.pvalues[v]
                star = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else ""))
                cells.append(f"{b:+.3f}{star}")
            else:
                cells.append("\u2014")
        # Step 3 CI / p
        f3 = fits["Step 3"]
        if v in f3.params.index:
            ci = f3.conf_int(0.05).loc[v]
            ci_str = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
            p_str = fmt_p(f3.pvalues[v])
        else:
            ci_str = "\u2014"
            p_str = "\u2014"
        print(f"  {v:<20s}  {cells[0]:>10s}  {cells[1]:>10s}  {cells[2]:>10s}  "
              f"{ci_str:>22s}  {p_str:>10s}")
    print()
    for name, f in fits.items():
        print(f"  {name}: R\u00B2 = {f.rsquared:.3f}")
    print(f"  \u0394R\u00B2 (Step1 \u2192 Step2): {fits['Step 2'].rsquared - fits['Step 1'].rsquared:.3f}")
    print(f"  \u0394R\u00B2 (Step2 \u2192 Step3): {fits['Step 3'].rsquared - fits['Step 2'].rsquared:.3f}")

    # F-tests for the increments
    f12 = fits["Step 2"].compare_f_test(fits["Step 1"])
    f23 = fits["Step 3"].compare_f_test(fits["Step 2"])
    print(f"  Step 1 \u2192 Step 2: F = {f12[0]:.2f}, p = {fmt_p(f12[1])}, df = {int(f12[2])}")
    print(f"  Step 2 \u2192 Step 3: F = {f23[0]:.2f}, p = {fmt_p(f23[1])}, df = {int(f23[2])}")


# -----------------------------------------------------------------------------
# 7. Table 4 \u2014 parallel mediation (Figure 4)
# -----------------------------------------------------------------------------
def table4(df: pd.DataFrame) -> None:
    section("Table 4 / Figure 4 \u2014 Parallel mediation (PROCESS model 4 analogue)")
    ga = standardise(df.genai_freq).values
    an = standardise(df.anxiety).values
    se = standardise(df.selfeff).values
    yy = standardise(df.who5).values
    n = len(df)

    # a paths (X -> M)
    a1 = sm.OLS(an, sm.add_constant(ga)).fit()
    a2 = sm.OLS(se, sm.add_constant(ga)).fit()
    # full b model: Y on X + M1 + M2
    Xb = sm.add_constant(np.column_stack([ga, an, se]))
    b_fit = sm.OLS(yy, Xb).fit()
    # total c
    c_fit = sm.OLS(yy, sm.add_constant(ga)).fit()

    a1_b = a1.params[1]
    a2_b = a2.params[1]
    b1_b = b_fit.params[2]
    b2_b = b_fit.params[3]
    cprime = b_fit.params[1]
    c_total = c_fit.params[1]

    print(f"\n  a1 (genai \u2192 anxiety):   {a1_b:+.3f}  p = {fmt_p(a1.pvalues[1])}")
    print(f"  a2 (genai \u2192 self-eff): {a2_b:+.3f}  p = {fmt_p(a2.pvalues[1])}")
    print(f"  b1 (anxiety \u2192 WHO-5 | .): {b1_b:+.3f}  p = {fmt_p(b_fit.pvalues[2])}")
    print(f"  b2 (self-eff \u2192 WHO-5 | .): {b2_b:+.3f}  p = {fmt_p(b_fit.pvalues[3])}")
    print(f"  c\u2032 (direct genai \u2192 WHO-5):  {cprime:+.3f}  p = {fmt_p(b_fit.pvalues[1])}")
    print(f"  c  (total genai \u2192 WHO-5):   {c_total:+.3f}  p = {fmt_p(c_fit.pvalues[1])}")

    print(f"\n  Indirect via anxiety  (a1\u00D7b1): {a1_b * b1_b:+.3f}")
    print(f"  Indirect via self-eff (a2\u00D7b2): {a2_b * b2_b:+.3f}")
    print(f"  Total indirect:                {a1_b * b1_b + a2_b * b2_b:+.3f}")

    # Bootstrap (5000 percentile)
    print(f"\nBootstrap 95% CIs ({N_BOOT} resamples):")
    ind1 = np.empty(N_BOOT)
    ind2 = np.empty(N_BOOT)
    ind_t = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)
        ga_b = ga[idx]; an_b = an[idx]; se_b = se[idx]; y_b = yy[idx]
        # a1
        var_g = ga_b.var(ddof=1)
        a1_i = np.cov(ga_b, an_b, ddof=1)[0, 1] / var_g
        a2_i = np.cov(ga_b, se_b, ddof=1)[0, 1] / var_g
        Xb_i = np.column_stack([np.ones(n), ga_b, an_b, se_b])
        coef_i, *_ = np.linalg.lstsq(Xb_i, y_b, rcond=None)
        ind1[i] = a1_i * coef_i[2]
        ind2[i] = a2_i * coef_i[3]
        ind_t[i] = ind1[i] + ind2[i]

    def ci(arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return lo, hi

    lo, hi = ci(ind1)
    print(f"  Indirect via anxiety:   {a1_b*b1_b:+.3f}  Boot SE = {ind1.std():.3f}  "
          f"95% Boot CI [{lo:+.3f}, {hi:+.3f}]")
    lo, hi = ci(ind2)
    print(f"  Indirect via self-eff:  {a2_b*b2_b:+.3f}  Boot SE = {ind2.std():.3f}  "
          f"95% Boot CI [{lo:+.3f}, {hi:+.3f}]")
    lo, hi = ci(ind_t)
    print(f"  Total indirect:         {a1_b*b1_b + a2_b*b2_b:+.3f}  "
          f"Boot SE = {ind_t.std():.3f}  95% Boot CI [{lo:+.3f}, {hi:+.3f}]")

    pct_mediated = abs(a1_b*b1_b + a2_b*b2_b) / abs(c_total) * 100
    print(f"\n  Proportion of total effect mediated: {pct_mediated:.1f}%")


# -----------------------------------------------------------------------------
# 8. Robustness checks
# -----------------------------------------------------------------------------
def robustness(df: pd.DataFrame) -> None:
    section("Robustness checks (Section 4.4)")

    print("\nMediation with engagement (UWES-S-9) added as outcome covariate:")
    ga = standardise(df.genai_freq).values
    an = standardise(df.anxiety).values
    se = standardise(df.selfeff).values
    eng = standardise(df.engage).values
    yy = standardise(df.who5).values
    n = len(df)

    # b paths controlling for engagement
    Xb = sm.add_constant(np.column_stack([ga, an, se, eng]))
    b_fit = sm.OLS(yy, Xb).fit()
    a1 = sm.OLS(an, sm.add_constant(ga)).fit()
    a2 = sm.OLS(se, sm.add_constant(ga)).fit()
    a1_b = a1.params[1]; a2_b = a2.params[1]
    b1_b = b_fit.params[2]; b2_b = b_fit.params[3]

    # Bootstrap
    ind1 = np.empty(N_BOOT); ind2 = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)
        ga_b = ga[idx]; an_b = an[idx]; se_b = se[idx]; eng_b = eng[idx]; y_b = yy[idx]
        var_g = ga_b.var(ddof=1)
        a1_i = np.cov(ga_b, an_b, ddof=1)[0, 1] / var_g
        a2_i = np.cov(ga_b, se_b, ddof=1)[0, 1] / var_g
        Xb_i = np.column_stack([np.ones(n), ga_b, an_b, se_b, eng_b])
        coef_i, *_ = np.linalg.lstsq(Xb_i, y_b, rcond=None)
        ind1[i] = a1_i * coef_i[2]
        ind2[i] = a2_i * coef_i[3]
    lo1, hi1 = np.percentile(ind1, [2.5, 97.5])
    lo2, hi2 = np.percentile(ind2, [2.5, 97.5])
    print(f"  Anxiety path:  \u03B2 = {a1_b*b1_b:+.3f}  95% Boot CI [{lo1:+.3f}, {hi1:+.3f}]")
    print(f"  Self-eff path: \u03B2 = {a2_b*b2_b:+.3f}  95% Boot CI [{lo2:+.3f}, {hi2:+.3f}]")

    # Multi-group moderation by sex and cycle (interaction tests)
    print("\nModeration by sex (1 = Female):")
    for path, x, m in [("a1", df.genai_freq, df.anxiety),
                       ("a2", df.genai_freq, df.selfeff)]:
        x_s = standardise(x); m_s = standardise(m)
        female = (df.sex == "F").astype(int)
        Xint = sm.add_constant(pd.DataFrame({
            "x": x_s, "female": female, "x_x_female": x_s * female,
        }))
        fit = sm.OLS(m_s, Xint).fit()
        print(f"  {path}: interaction p = {fmt_p(fit.pvalues['x_x_female'])}")

    print("\nModeration by cycle (1 = MSc):")
    msc = (df.cycle == "MSc").astype(int)
    for path, x, m in [("a1", df.genai_freq, df.anxiety),
                       ("a2", df.genai_freq, df.selfeff)]:
        x_s = standardise(x); m_s = standardise(m)
        Xint = sm.add_constant(pd.DataFrame({
            "x": x_s, "msc": msc, "x_x_msc": x_s * msc,
        }))
        fit = sm.OLS(m_s, Xint).fit()
        print(f"  {path}: interaction p = {fmt_p(fit.pvalues['x_x_msc'])}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", nargs="?", default="dataset.csv",
                        help="Path to dataset.csv (default: ./dataset.csv)")
    parser.add_argument("--log", default="analysis_log.txt",
                        help="Path to write a copy of stdout (default: analysis_log.txt)")
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"ERROR: dataset not found at {args.dataset}", file=sys.stderr)
        return 1

    buf = StringIO()
    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for st in self.streams: st.write(s)
        def flush(self):
            for st in self.streams: st.flush()
    tee = Tee(sys.stdout, buf)

    with redirect_stdout(tee):
        print(f"Analysis log \u2014 dataset: {args.dataset}")
        print("Generated with 03_analysis.py")
        df = load(args.dataset)
        table1(df)
        section_4_1(df)
        table2(df)
        section_4_2(df)
        table3(df)
        table4(df)
        robustness(df)
        print()
        print("=" * 78)
        print("Analysis complete.")
        print("=" * 78)

    with open(args.log, "w") as fh:
        fh.write(buf.getvalue())
    print(f"\n[Wrote log to {args.log}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
