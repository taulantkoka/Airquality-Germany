"""
============================================================================
  Structural Break Analysis — Did the Umweltzone Work?
============================================================================

  Uses annual balance data (2000–2024) to test for structural breaks.

  Methods:
    1. Chow test at the known Umweltzone break (2015)
    2. Regression discontinuity visualization
    3. Segmented regression (before/after with separate slopes)
    4. Scan for unknown break dates (sup-F test)

  Run: python structural_breaks.py
============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from data_fetcher import HessenAirAPI

sns.set_style("whitegrid")


KNOWN_BREAKS = {
    "Umweltzone": 2015,
    "Diesel ban": 2019,
}


def fetch_annual_data(city="Darmstadt"):
    """Get annual means for NO2 and PM10."""
    api = HessenAirAPI()
    frames = []
    for poll in ["NO2", "PM10"]:
        df = api.get_annual_balances(
            pollutant=poll, city=city,
            start_year=2000, end_year=2024,
        )
        if not df.empty:
            df["pollutant"] = poll
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    # Average across stations per year
    return (df.groupby(["pollutant", "year"])["annual_mean"]
            .mean().reset_index())


def chow_test(y, X, break_idx):
    """
    Chow test for structural break at a given index.
    H0: no structural break (same coefficients before and after).

    Returns F-statistic and p-value.
    """
    n = len(y)
    k = X.shape[1]

    # Full model
    model_full = sm.OLS(y, X).fit()
    rss_full = np.sum(model_full.resid ** 2)

    # Split models
    y1, X1 = y[:break_idx], X[:break_idx]
    y2, X2 = y[break_idx:], X[break_idx:]

    if len(y1) <= k or len(y2) <= k:
        return np.nan, np.nan

    model1 = sm.OLS(y1, X1).fit()
    model2 = sm.OLS(y2, X2).fit()
    rss_split = np.sum(model1.resid ** 2) + np.sum(model2.resid ** 2)

    # F-statistic
    F = ((rss_full - rss_split) / k) / (rss_split / (n - 2 * k))
    p = 1 - stats.f.cdf(F, k, n - 2 * k)

    return F, p


def analyse_structural_breaks(annual_df):
    """Run structural break analysis for each pollutant."""

    print("\n" + "=" * 70)
    print("  STRUCTURAL BREAK ANALYSIS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for row_idx, poll in enumerate(["NO2", "PM10"]):
        sub = annual_df[annual_df["pollutant"] == poll].sort_values("year")
        if sub.empty or len(sub) < 8:
            print(f"\n  {poll}: Insufficient data ({len(sub)} years)")
            continue

        years = sub["year"].values
        values = sub["annual_mean"].values

        print(f"\n  --- {poll} ---")
        print(f"  Years: {years[0]}–{years[-1]} ({len(years)} observations)")

        # ---- Left panel: Segmented regression at Umweltzone (2015) ----
        ax = axes[row_idx, 0]

        # Build segmented regression
        # y = α + β₁·t + β₂·D_post + β₃·(t × D_post) + ε
        # where D_post = 1 if year >= break_year
        break_year = KNOWN_BREAKS["Umweltzone"]
        t = years - years[0]  # Time trend from 0
        d_post = (years >= break_year).astype(int)
        t_post = t * d_post  # Interaction: allows slope change

        X = np.column_stack([np.ones(len(t)), t, d_post, t_post])
        model = sm.OLS(values, X).fit()

        # Predict for both periods
        t_pred = np.arange(t[0], t[-1] + 1)
        years_pred = t_pred + years[0]

        for period, color, label in [("pre", "#e74c3c", "Pre-Umweltzone"),
                                      ("post", "#2ecc71", "Post-Umweltzone")]:
            if period == "pre":
                mask = years_pred < break_year
            else:
                mask = years_pred >= break_year

            t_p = t_pred[mask]
            d_p = (years_pred[mask] >= break_year).astype(float)
            t_p_interaction = t_p * d_p
            # Manually build [const, t, d_post, t*d_post] — sm.add_constant
            # fails when columns are all-zero (pre-period)
            ones = np.ones(len(t_p))
            X_p = np.column_stack([ones, t_p, d_p, t_p_interaction])
            y_pred = model.predict(X_p)

            ax.plot(years_pred[mask], y_pred, color=color, linewidth=2.5,
                    label=f"{label} trend")

        ax.scatter(years, values, color='navy', s=40, zorder=5)
        ax.axvline(break_year, color='grey', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'Umweltzone ({break_year})')

        # EU limit
        if poll == "NO2":
            ax.axhline(40, color='crimson', linestyle=':', alpha=0.5,
                       label='EU limit')

        # Chow test
        break_idx = np.searchsorted(years, break_year)
        X_chow = sm.add_constant(t.reshape(-1, 1))
        F, p = chow_test(values, X_chow, break_idx)

        ax.set_title(f"{poll} — Segmented Regression at Umweltzone\n"
                     f"Chow F={F:.2f}, p={p:.4f} "
                     f"{'(significant break!)' if p < 0.05 else '(no significant break)'}",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual Mean (µg/m³)")
        ax.legend(fontsize=8, loc='upper right')

        # Print coefficients
        print(f"\n  Segmented regression at {break_year}:")
        print(f"    Pre-trend slope:  {model.params[1]:.2f} µg/m³ per year")
        post_slope = model.params[1] + model.params[3]
        print(f"    Post-trend slope: {post_slope:.2f} µg/m³ per year")
        print(f"    Level shift:      {model.params[2]:+.2f} µg/m³")
        print(f"    Chow test: F={F:.2f}, p={p:.4f}")

        # ---- Right panel: Scan for unknown breaks (sup-F) ----
        ax = axes[row_idx, 1]

        trim = max(4, len(years) // 5)  # 20% trimming
        f_stats = []
        scan_years = []

        for i in range(trim, len(years) - trim):
            F_i, _ = chow_test(values, X_chow, i)
            f_stats.append(F_i)
            scan_years.append(years[i])

        if f_stats:
            ax.plot(scan_years, f_stats, color='navy', linewidth=2)
            ax.fill_between(scan_years, f_stats, alpha=0.15, color='navy')

            # Mark known breaks
            for name, by in KNOWN_BREAKS.items():
                if by in scan_years:
                    idx = scan_years.index(by)
                    ax.axvline(by, color='red', linestyle='--', alpha=0.7)
                    ax.annotate(f"{name}\n({by})",
                               xy=(by, f_stats[idx]),
                               xytext=(by + 1, max(f_stats) * 0.9),
                               arrowprops=dict(arrowstyle='->', color='red'),
                               fontsize=8, color='red')

            # Mark the maximum F
            best_idx = np.argmax(f_stats)
            best_year = scan_years[best_idx]
            ax.scatter([best_year], [f_stats[best_idx]], color='red',
                       s=80, zorder=5)

            # Critical value line (approximate 5% for Chow with k=2)
            cv_5pct = stats.f.ppf(0.95, 2, len(years) - 4)
            ax.axhline(cv_5pct, color='crimson', linestyle=':',
                       alpha=0.5, label=f'5% critical value ({cv_5pct:.1f})')

            print(f"\n  Sup-F scan:")
            print(f"    Strongest break: {best_year} "
                  f"(F={f_stats[best_idx]:.2f})")
            print(f"    5% critical value: {cv_5pct:.1f}")

        ax.set_title(f"{poll} — Break Date Scan (F-statistics)\n"
                     f"Peak at {best_year if f_stats else '?'}",
                     fontsize=11, fontweight='bold')
        ax.set_xlabel("Candidate Break Year")
        ax.set_ylabel("Chow F-statistic")
        ax.legend(fontsize=8)

    fig.suptitle("Structural Break Analysis — Annual Pollution Trends (2000–2024)",
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig("11_structural_breaks.png", dpi=200, bbox_inches='tight')
    print(f"\n  Saved: 11_structural_breaks.png")
    plt.close(fig)


def plot_counterfactual(annual_df):
    """
    What would pollution look like without the Umweltzone?
    Extrapolate the pre-break trend forward.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, poll in enumerate(["NO2", "PM10"]):
        ax = axes[i]
        sub = annual_df[annual_df["pollutant"] == poll].sort_values("year")
        if sub.empty:
            continue

        years = sub["year"].values
        values = sub["annual_mean"].values
        break_year = KNOWN_BREAKS["Umweltzone"]

        # Fit pre-break trend
        pre_mask = years < break_year
        if pre_mask.sum() < 3:
            continue

        t_pre = years[pre_mask] - years[0]
        X_pre = sm.add_constant(t_pre)
        model_pre = sm.OLS(values[pre_mask], X_pre).fit()

        # Extrapolate to full range
        t_all = years - years[0]
        X_all = sm.add_constant(t_all)
        counterfactual = model_pre.predict(X_all)

        # Plot
        ax.scatter(years, values, color='navy', s=40, zorder=5,
                   label='Observed')
        ax.plot(years, counterfactual, color='#e74c3c', linestyle='--',
                linewidth=2, label='Counterfactual (pre-trend extrapolated)')
        ax.axvline(break_year, color='grey', linestyle='--', alpha=0.5)
        ax.axvspan(break_year, years[-1], alpha=0.05, color='green')

        # Shade the "saved" pollution
        post_mask = years >= break_year
        if post_mask.any():
            ax.fill_between(years[post_mask],
                            values[post_mask],
                            counterfactual[post_mask],
                            alpha=0.2, color='green',
                            label='Reduction vs counterfactual')

            # Calculate cumulative "savings"
            savings = counterfactual[post_mask] - values[post_mask]
            total_saving = savings.mean()
            print(f"\n  {poll} counterfactual:")
            print(f"    Mean annual saving: {total_saving:.1f} µg/m³")
            print(f"    2024 observed:      {values[-1]:.1f} µg/m³")
            print(f"    2024 counterfactual: {counterfactual[-1]:.1f} µg/m³")

        ax.set_xlabel("Year")
        ax.set_ylabel("Annual Mean (µg/m³)")
        ax.set_title(f"{poll} — Counterfactual Analysis", fontweight='bold')
        ax.legend(fontsize=8)

        if poll == "NO2":
            ax.axhline(40, color='crimson', linestyle=':', alpha=0.3)

    fig.suptitle("What If There Were No Interventions?\n"
                 "Pre-2015 trend extrapolated vs actual observations",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig("12_counterfactual.png", dpi=200, bbox_inches='tight')
    print(f"\n  Saved: 12_counterfactual.png")
    plt.close(fig)


# ======================================================================
#  MAIN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  STRUCTURAL BREAK ANALYSIS")
    print("=" * 70)

    annual_df = fetch_annual_data(city="Darmstadt")
    if annual_df.empty:
        raise SystemExit("No annual data.")

    print(f"\n  Data: {len(annual_df)} observations")
    print(annual_df.groupby("pollutant")["year"].agg(["min", "max", "count"]))

    analyse_structural_breaks(annual_df)
    plot_counterfactual(annual_df)

    print("\n" + "=" * 70)
    print("  DONE:")
    print("    11_structural_breaks.png   — Chow tests + break scan")
    print("    12_counterfactual.png      — What-if without interventions")
    print("=" * 70)