"""
============================================================================
  Hessen Air Quality — Exploratory Analysis
============================================================================

  1. Multi-pollutant trends   (NO2 + PM10 from annual balances)
  2. Seasonal patterns        (NO2 + PM10 + PM2.5 + O3 from hourly data)
  3. Traffic vs background    (road proximity effect)
  4. Cross-city comparison    (Darmstadt / Frankfurt / Kassel / Wiesbaden)

  Note: O3 and PM2.5 are excluded from annual balance plots because:
    - O3 annual balances return exceedance day counts, not mean concentrations
    - PM2.5 monitoring started late (~2020) at Darmstadt urban stations
  Both work correctly from hourly data (seasonal plots).

  Run:  python 01_exploration.py
============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from data_fetcher import HessenAirAPI

sns.set_style("whitegrid")
sns.set_palette("colorblind")

API = HessenAirAPI()

SEASONAL_YEARS = 2
CITIES = ["Darmstadt", "Frankfurt", "Kassel", "Wiesbaden"]

# Annual trends: only NO2 + PM10 (reliable from /annualbalances)
ANNUAL_POLLUTANTS = {
    "NO2":  {"id": 5, "unit": "µg/m³", "eu_limit": 40, "color": "#e74c3c"},
    "PM10": {"id": 1, "unit": "µg/m³", "eu_limit": 40, "color": "#3498db"},
}

# Seasonal/hourly: all four (reliable from /measures)
HOURLY_POLLUTANTS = {
    "NO2":   {"id": 5, "unit": "µg/m³", "color": "#e74c3c"},
    "PM10":  {"id": 1, "unit": "µg/m³", "color": "#3498db"},
    "PM2.5": {"id": 9, "unit": "µg/m³", "color": "#2ecc71"},
    "O3":    {"id": 3, "unit": "µg/m³", "color": "#f39c12"},
}


# ======================================================================
#  DATA FETCHING
# ======================================================================

def fetch_annual_data():
    """Fetch annual balances for NO2 + PM10 × all cities."""
    print("\n" + "=" * 70)
    print("  FETCHING ANNUAL DATA (NO2 + PM10, 2000–2024)")
    print("=" * 70)

    frames = []
    for poll_name in ANNUAL_POLLUTANTS:
        for city in CITIES:
            print(f"\n  {poll_name} × {city}")
            df = API.get_annual_balances(
                pollutant=poll_name, city=city,
                start_year=2000, end_year=2024,
            )
            if not df.empty:
                df["pollutant"] = poll_name
                df["city"] = city
                frames.append(df)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def fetch_hourly_data(city):
    """Fetch recent hourly data for all pollutants (seasonal analysis)."""
    frames = []
    for poll_name in HOURLY_POLLUTANTS:
        print(f"\n  Hourly {poll_name} × {city}")
        df = API.get_api_data(
            pollutant=poll_name, city=city,
            days=365 * SEASONAL_YEARS,
            scope="1SMW",
            chunk_days=180,
        )
        if not df.empty:
            df["pollutant"] = poll_name
            df["city"] = city
            frames.append(df)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# ======================================================================
#  1. MULTI-POLLUTANT TRENDS (NO2 + PM10 only)
# ======================================================================

def analyse_multi_pollutant(df_annual):
    """Long-term trends for NO2 and PM10 at Darmstadt."""
    print("\n\n" + "=" * 70)
    print("  1. MULTI-POLLUTANT TRENDS — Darmstadt (NO2 + PM10)")
    print("=" * 70)

    da = df_annual[df_annual["city"] == "Darmstadt"].copy()
    if da.empty:
        print("  No Darmstadt data.")
        return

    trend = (da.groupby(["pollutant", "year"])["annual_mean"]
             .mean().reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, (poll_name, meta) in enumerate(ANNUAL_POLLUTANTS.items()):
        ax = axes[i]
        subset = trend[trend["pollutant"] == poll_name]

        if subset.empty:
            ax.set_title(f"{poll_name} — no data")
            continue

        ax.fill_between(subset["year"], subset["annual_mean"],
                        alpha=0.2, color=meta["color"])
        ax.plot(subset["year"], subset["annual_mean"],
                color=meta["color"], linewidth=2.5, marker='o', markersize=4)

        ax.axhline(y=meta["eu_limit"], color='crimson',
                    linestyle='--', linewidth=1, alpha=0.7,
                    label=f'EU limit ({meta["eu_limit"]} {meta["unit"]})')
        ax.legend(fontsize=9)

        ax.set_title(f"{poll_name}", fontsize=14, fontweight='bold')
        ax.set_ylabel(meta["unit"])
        ax.set_xlabel("Year")

        # Print stats
        if not subset.empty:
            first = subset.iloc[0]
            last = subset.iloc[-1]
            change = ((last["annual_mean"] - first["annual_mean"])
                      / first["annual_mean"] * 100)
            print(f"  {poll_name}: {first['year']:.0f} → {last['year']:.0f}  "
                  f"{first['annual_mean']:.1f} → {last['annual_mean']:.1f} "
                  f"({change:+.0f}%)")

    fig.suptitle("Darmstadt — Long-Term Air Quality Trends (Annual Means)",
                 fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig("01_multi_pollutant_trends.png", dpi=200, bbox_inches='tight')
    print("  Saved: 01_multi_pollutant_trends.png")
    plt.close(fig)

    # Correlation
    pivot = trend.pivot(index="year", columns="pollutant",
                        values="annual_mean")
    if pivot.shape[1] >= 2:
        corr = pivot.corr().round(2)
        print(f"\n  NO2–PM10 correlation: {corr.iloc[0,1]}")


# ======================================================================
#  2. SEASONAL PATTERNS (all 4 pollutants from hourly data)
# ======================================================================

def analyse_seasonal(df_hourly):
    """Hourly, weekday, and monthly patterns."""
    print("\n\n" + "=" * 70)
    print("  2. SEASONAL PATTERNS — Darmstadt")
    print("=" * 70)

    if df_hourly.empty:
        print("  No hourly data.")
        return

    df = df_hourly.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["weekday"] >= 5

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    # --- A) Hourly profile ---
    ax = axes[0, 0]
    for poll_name, meta in HOURLY_POLLUTANTS.items():
        sub = df[df["pollutant"] == poll_name]
        if sub.empty:
            continue
        hourly = sub.groupby("hour")["value"].mean()
        ax.plot(hourly.index, hourly.values, color=meta["color"],
                linewidth=2, label=poll_name)
    ax.set_title("Average Hourly Profile", fontsize=13, fontweight='bold')
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("µg/m³")
    ax.set_xticks(range(0, 24, 3))
    ax.legend()

    # --- B) Weekday vs Weekend (NO2) ---
    ax = axes[0, 1]
    no2 = df[df["pollutant"] == "NO2"]
    if not no2.empty:
        for label, grp in no2.groupby("is_weekend"):
            name = "Weekend" if label else "Weekday"
            hourly = grp.groupby("hour")["value"].mean()
            ax.plot(hourly.index, hourly.values, linewidth=2, label=name)
        ax.set_title("NO2: Weekday vs Weekend", fontsize=13, fontweight='bold')
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("µg/m³")
        ax.set_xticks(range(0, 24, 3))
        ax.legend()

        wkday_mean = no2[~no2["is_weekend"]]["value"].mean()
        wkend_mean = no2[no2["is_weekend"]]["value"].mean()
        diff = (wkend_mean - wkday_mean) / wkday_mean * 100
        print(f"  NO2 weekday mean: {wkday_mean:.1f}, "
              f"weekend: {wkend_mean:.1f} ({diff:+.0f}%)")

    # --- C) Monthly pattern ---
    ax = axes[1, 0]
    month_labels = ["J", "F", "M", "A", "M", "J",
                     "J", "A", "S", "O", "N", "D"]
    for poll_name, meta in HOURLY_POLLUTANTS.items():
        sub = df[df["pollutant"] == poll_name]
        if sub.empty:
            continue
        monthly = sub.groupby("month")["value"].mean()
        ax.plot(monthly.index, monthly.values, color=meta["color"],
                linewidth=2, marker='o', markersize=4, label=poll_name)
    ax.set_title("Monthly Profile", fontsize=13, fontweight='bold')
    ax.set_xlabel("Month")
    ax.set_ylabel("µg/m³")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.legend()

    # --- D) Day-of-week ---
    ax = axes[1, 1]
    day_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for poll_name, meta in HOURLY_POLLUTANTS.items():
        sub = df[df["pollutant"] == poll_name]
        if sub.empty:
            continue
        daily = sub.groupby("weekday")["value"].mean()
        ax.plot(daily.index, daily.values, color=meta["color"],
                linewidth=2, marker='o', markersize=5, label=poll_name)
    ax.set_title("Day-of-Week Profile", fontsize=13, fontweight='bold')
    ax.set_xlabel("Day")
    ax.set_ylabel("µg/m³")
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_labels)
    ax.legend()

    fig.suptitle(f"Darmstadt — Seasonal Patterns (last {SEASONAL_YEARS} years)",
                 fontsize=16, y=1.01)
    fig.tight_layout()
    fig.savefig("02_seasonal_patterns.png", dpi=200, bbox_inches='tight')
    print("  Saved: 02_seasonal_patterns.png")
    plt.close(fig)

    # Rush hour stats
    if not no2.empty:
        rush_am = no2[(no2["hour"] >= 7) & (no2["hour"] <= 9)]
        rush_pm = no2[(no2["hour"] >= 16) & (no2["hour"] <= 19)]
        midday = no2[(no2["hour"] >= 11) & (no2["hour"] <= 14)]
        night = no2[(no2["hour"] >= 1) & (no2["hour"] <= 5)]
        print(f"\n  NO2 by period:")
        print(f"    Morning rush (7–9):   {rush_am['value'].mean():.1f} µg/m³")
        print(f"    Midday (11–14):        {midday['value'].mean():.1f} µg/m³")
        print(f"    Evening rush (16–19): {rush_pm['value'].mean():.1f} µg/m³")
        print(f"    Night (1–5):           {night['value'].mean():.1f} µg/m³")


# ======================================================================
#  3. TRAFFIC vs BACKGROUND
# ======================================================================

def analyse_traffic_vs_background(df_annual, df_hourly):
    """Compare traffic stations with urban background."""
    print("\n\n" + "=" * 70)
    print("  3. TRAFFIC vs BACKGROUND")
    print("=" * 70)

    stations = API.stations
    traffic = {c for c, info in stations.items()
               if "traffic" in info.get("type", "")}
    background = {c for c, info in stations.items()
                  if "background" in info.get("type", "")}

    da = df_annual[df_annual["pollutant"] == "NO2"].copy()
    if da.empty:
        print("  No NO2 annual data.")
        return

    da["station_type"] = da["station_code"].apply(
        lambda c: "Traffic" if c in traffic
        else ("Background" if c in background else "Other")
    )
    da = da[da["station_type"].isin(["Traffic", "Background"])]

    type_trend = (da.groupby(["station_type", "year"])["annual_mean"]
                  .mean().reset_index())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: annual trend
    for stype, grp in type_trend.groupby("station_type"):
        color = "#e74c3c" if stype == "Traffic" else "#3498db"
        ax1.plot(grp["year"], grp["annual_mean"],
                 linewidth=2.5, marker='o', markersize=4,
                 color=color, label=stype)
    ax1.axhline(y=40, color='grey', linestyle='--', alpha=0.5,
                 label='EU limit')
    ax1.set_title("NO2 Annual Means: Traffic vs Background",
                  fontweight='bold')
    ax1.set_ylabel("µg/m³")
    ax1.set_xlabel("Year")
    ax1.legend()

    # Print gap
    pivot = type_trend.pivot(index="year", columns="station_type",
                             values="annual_mean")
    if "Traffic" in pivot and "Background" in pivot:
        pivot["gap"] = pivot["Traffic"] - pivot["Background"]
        recent = pivot.dropna().tail(5)
        early = pivot.dropna().head(5)
        print(f"  NO2 Traffic–Background gap:")
        print(f"    Early ({int(early.index[0])}–{int(early.index[-1])}): "
              f"{early['gap'].mean():.1f} µg/m³")
        print(f"    Recent ({int(recent.index[0])}–{int(recent.index[-1])}): "
              f"{recent['gap'].mean():.1f} µg/m³")

    # Right: hourly profile by station type
    if not df_hourly.empty:
        no2h = df_hourly[df_hourly["pollutant"] == "NO2"].copy()
        no2h["station_type"] = no2h["station_code"].apply(
            lambda c: "Traffic" if c in traffic
            else ("Background" if c in background else "Other")
        )
        no2h = no2h[no2h["station_type"].isin(["Traffic", "Background"])]
        no2h["hour"] = no2h["timestamp"].dt.hour

        for stype, grp in no2h.groupby("station_type"):
            color = "#e74c3c" if stype == "Traffic" else "#3498db"
            hourly = grp.groupby("hour")["value"].mean()
            ax2.plot(hourly.index, hourly.values,
                     linewidth=2.5, color=color, label=stype)
        ax2.set_title("NO2 Hourly Profile: Traffic vs Background",
                      fontweight='bold')
        ax2.set_xlabel("Hour of Day")
        ax2.set_ylabel("µg/m³")
        ax2.set_xticks(range(0, 24, 3))
        ax2.legend()

    fig.tight_layout()
    fig.savefig("03_traffic_vs_background.png", dpi=200)
    print("  Saved: 03_traffic_vs_background.png")
    plt.close(fig)


# ======================================================================
#  4. CROSS-CITY COMPARISON (NO2 + PM10 only)
# ======================================================================

def analyse_cross_city(df_annual):
    """Compare NO2 and PM10 trends across Hessen cities."""
    print("\n\n" + "=" * 70)
    print("  4. CROSS-CITY COMPARISON")
    print("=" * 70)

    city_colors = {
        "Darmstadt": "#e74c3c", "Frankfurt": "#3498db",
        "Kassel": "#2ecc71", "Wiesbaden": "#9b59b6",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, (poll_name, meta) in enumerate(ANNUAL_POLLUTANTS.items()):
        ax = axes[i]
        sub = df_annual[df_annual["pollutant"] == poll_name]

        for city in CITIES:
            city_data = sub[sub["city"] == city]
            if city_data.empty:
                continue
            trend = (city_data.groupby("year")["annual_mean"]
                     .mean().reset_index())
            ax.plot(trend["year"], trend["annual_mean"],
                    color=city_colors.get(city, "grey"),
                    linewidth=2, marker='o', markersize=3, label=city)

        ax.axhline(y=meta["eu_limit"], color='grey',
                    linestyle='--', linewidth=1, alpha=0.5)

        ax.set_title(f"{poll_name}", fontsize=14, fontweight='bold')
        ax.set_ylabel(meta["unit"])
        ax.set_xlabel("Year")
        ax.legend(fontsize=9)

    fig.suptitle("Cross-City Comparison — Annual Means (NO2 + PM10)",
                 fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig("04_cross_city_comparison.png", dpi=200, bbox_inches='tight')
    print("  Saved: 04_cross_city_comparison.png")
    plt.close(fig)

    # Ranking table
    recent = df_annual[df_annual["year"] >= 2020]
    if not recent.empty:
        ranking = (recent.groupby(["city", "pollutant"])["annual_mean"]
                   .mean().round(1).unstack("pollutant"))
        print(f"\n  Average annual means 2020–2024:")
        print(ranking.to_string())


# ======================================================================
#  SUMMARY
# ======================================================================

def print_summary(df_annual, df_hourly):
    print("\n\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    if not df_annual.empty:
        no2 = df_annual[df_annual["pollutant"] == "NO2"]
        no2_da = no2[no2["city"] == "Darmstadt"]

        if not no2_da.empty:
            latest = no2_da[no2_da["year"] == no2_da["year"].max()]
            earliest = no2_da[no2_da["year"] == no2_da["year"].min()]

            print(f"\n  Darmstadt NO2:")
            print(f"    Earliest: {int(earliest['year'].iloc[0])} — "
                  f"{earliest['annual_mean'].mean():.1f} µg/m³")
            print(f"    Latest:   {int(latest['year'].iloc[0])} — "
                  f"{latest['annual_mean'].mean():.1f} µg/m³")

            yearly_avg = no2_da.groupby("year")["annual_mean"].mean()
            exceedances = yearly_avg[yearly_avg > 40]
            print(f"    Years above EU limit: "
                  f"{len(exceedances)} of {len(yearly_avg)}")

    if not df_hourly.empty:
        no2h = df_hourly[df_hourly["pollutant"] == "NO2"]
        if not no2h.empty:
            print(f"\n  Hourly NO2 (last {SEASONAL_YEARS} years):")
            print(f"    Max:    {no2h['value'].max():.0f} µg/m³")
            print(f"    Mean:   {no2h['value'].mean():.1f} µg/m³")
            print(f"    Median: {no2h['value'].median():.1f} µg/m³")
            print(f"    P95:    {no2h['value'].quantile(0.95):.0f} µg/m³")


# ======================================================================
#  MAIN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  HESSEN AIR QUALITY — EXPLORATORY ANALYSIS")
    print("=" * 70)

    df_annual = fetch_annual_data()

    print("\n" + "=" * 70)
    print("  FETCHING HOURLY DATA — Darmstadt")
    print("=" * 70)
    df_hourly_da = fetch_hourly_data("Darmstadt")

    if not df_annual.empty:
        analyse_multi_pollutant(df_annual)
        analyse_traffic_vs_background(df_annual, df_hourly_da)
        analyse_cross_city(df_annual)

    if not df_hourly_da.empty:
        analyse_seasonal(df_hourly_da)

    print_summary(df_annual, df_hourly_da)

    print("\n" + "=" * 70)
    print("  DONE:")
    print("    01_multi_pollutant_trends  — NO2 + PM10 annual (2000–2024)")
    print("    02_seasonal_patterns       — All 4 pollutants hourly")
    print("    03_traffic_vs_background   — NO2 roadside vs residential")
    print("    04_cross_city_comparison   — Hessen cities NO2 + PM10")
    print("=" * 70)