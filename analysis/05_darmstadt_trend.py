import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from data_fetcher import HessenAirAPI


def run_full_trend(pollutant="NO2", city="Darmstadt",
                   start_year=2000):
    """
    Full historical trend — fully automatic.

    Data sources (stitched together):
      1. /annualbalances  →  yearly means, start_year–2015
      2. /measures (1SMW) →  hourly data 2016–today, resampled to daily
    """
    api = HessenAirAPI()
    p = pollutant.upper()

    # ----- Phase 1: Annual summaries (pre-2016) -----
    print(f"\n{'='*60}")
    print(f"Phase 1: Annual balances {start_year}–2015")
    print(f"{'='*60}")

    df_annual = api.get_annual_balances(
        pollutant=p, city=city,
        start_year=start_year, end_year=2015,
    )

    annual_rows = []
    if not df_annual.empty:
        for _, row in df_annual.iterrows():
            annual_rows.append({
                "station": row["station_name"],
                "station_code": row["station_code"],
                "timestamp": pd.Timestamp(year=int(row["year"]),
                                          month=7, day=1),
                "value": row["annual_mean"],
                "source": "annual",
            })
        print(f"  → {len(annual_rows)} annual data points")
    else:
        print("  → No annual data found (Phase 2 will still run)")

    # ----- Phase 2: Hourly data (2016 → now), resampled to daily -----
    print(f"\n{'='*60}")
    print(f"Phase 2: Hourly data 2016 → today (resampled to daily)")
    print(f"{'='*60}")

    df_hourly = api.get_api_data(
        pollutant=p, city=city,
        start_date="2016-01-01",
        scope="1SMW",
        chunk_days=365,
    )

    daily_rows = []
    if not df_hourly.empty:
        for station, grp in df_hourly.groupby("station"):
            code = grp["station_code"].iloc[0]
            daily = (grp.set_index("timestamp")["value"]
                     .resample("D").mean().dropna())
            for ts, val in daily.items():
                daily_rows.append({
                    "station": station,
                    "station_code": code,
                    "timestamp": ts,
                    "value": val,
                    "source": "daily",
                })
        print(f"  → {len(daily_rows)} daily data points "
              f"(from {len(df_hourly)} hourly)")

    # ----- Combine -----
    all_rows = annual_rows + daily_rows
    if not all_rows:
        print("\nNo data at all. Run debug_api.py to check endpoints.")
        return

    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    print(f"\n{'='*60}")
    print(f"Combined: {len(df)} records, "
          f"{df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"Stations: {df['station'].unique().tolist()}")
    print(f"{'='*60}")

    _plot_per_station(df, p, city)
    _plot_combined(df, p, city)


def _plot_per_station(df, pollutant, city):
    """One line per station: yearly means over the full range."""
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.set_style("whitegrid")

    for station, grp in df.groupby("station"):
        ts = grp.set_index("timestamp")["value"]
        yearly = ts.resample("YE").mean().dropna()
        ax.plot(yearly.index, yearly.values,
                linewidth=2.5, marker='o', markersize=4, label=station)

    if pollutant == "NO2":
        ax.axhline(y=40, color='crimson', linestyle='--',
                    linewidth=1.5, label='EU Limit (40 µg/m³)')

    first = df["timestamp"].min().year
    last = df["timestamp"].max().year
    ax.set_title(f"{city} {pollutant} — {first}–{last} "
                 f"Annual Trend by Station", fontsize=16)
    ax.set_ylabel("Annual Mean (µg/m³)")
    ax.set_xlabel("Year")
    ax.legend(frameon=True, loc='upper right', fontsize=9)
    fig.tight_layout()

    outfile = f"{city.lower()}_{pollutant}_station_trend.png"
    fig.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")
    plt.close(fig)


def _plot_combined(df, pollutant, city):
    """Combined: monthly + yearly averages across all stations."""
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.set_style("whitegrid")

    ts = df.set_index("timestamp")["value"]
    monthly = ts.resample("ME").mean().dropna()
    yearly = ts.resample("YE").mean().dropna()

    ax.fill_between(monthly.index, monthly.values,
                    alpha=0.15, color='steelblue')
    ax.plot(monthly.index, monthly.values,
            alpha=0.4, color='steelblue', linewidth=0.8,
            label="Monthly mean")
    ax.plot(yearly.index, yearly.values,
            color='navy', linewidth=3, marker='o', markersize=5,
            label="Yearly mean")

    if pollutant == "NO2":
        ax.axhline(y=40, color='crimson', linestyle='--',
                    linewidth=1.5, label='EU Limit (40 µg/m³)')

    first = df["timestamp"].min().year
    last = df["timestamp"].max().year
    ax.set_title(f"{city} {pollutant} — {first}–{last} "
                 f"Combined Trend", fontsize=16)
    ax.set_ylabel("Concentration (µg/m³)")
    ax.set_xlabel("Year")
    ax.legend(frameon=True, loc='upper right')
    fig.tight_layout()

    outfile = f"{city.lower()}_{pollutant}_combined_trend.png"
    fig.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")
    plt.close(fig)


def run_recent_comparison(pollutant="NO2", city="Darmstadt", days=90):
    """Short-term hourly comparison across stations."""
    api = HessenAirAPI()
    df = api.get_api_data(pollutant=pollutant, city=city, days=days)

    if df.empty:
        print("Recent analysis aborted: no data.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='timestamp', y='value', hue='station', ax=ax)

    p = pollutant.upper()
    ax.set_title(f"{city} {p}: Last {days} Days", fontsize=14)
    ax.set_ylabel("µg/m³")
    plt.xticks(rotation=45)
    fig.tight_layout()

    outfile = f"{city.lower()}_recent_{p}_comparison.png"
    fig.savefig(outfile, dpi=300)
    print(f"Saved: {outfile}")
    plt.close(fig)


if __name__ == "__main__":
    run_full_trend(pollutant="NO2", city="Darmstadt", start_year=2000)