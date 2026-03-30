"""
============================================================================
  Panel DiD Analysis — Diesel Ban Effect With Full Controls
============================================================================

  Two complementary approaches:

  Option A (Robustness): Per-city ARX model → weather-corrected residuals
    → compare residual trends across ban/no-ban cities

  Option B (Main): Pooled panel regression with city fixed effects
    NO2_ct = α_c + β·weather_ct + γ·season_t + φ·AR_lags
           + δ·(ban_city × post_t) + ε_ct
    δ = causal DiD estimate of diesel ban effect

  Both use Newey-West HAC SEs + Benjamini-Yekutieli.

  Data cached to panel_data.csv after first fetch (~20 min).
  Re-runs load from cache automatically.

  Run: python panel_analysis.py
============================================================================
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from data_fetcher import HessenAirAPI
from dwd_weather import fetch_dwd_weather

sns.set_style("whitegrid")

# ======================================================================
#  CONFIG
# ======================================================================

CACHE_FILE = Path(__file__).parent.parent / "data" / "panel_data.csv"
STATIONS_CACHE = Path(__file__).parent.parent / "data" / "all_stations.json"
REGISTRY_FILE = Path(__file__).parent.parent / "data" / "germany_stations.json"

# Diesel ban treatment periods (only cities that actually had bans)
DIESEL_BAN = {
    "Darmstadt":  {"start": "2019-06-01", "end": None},
    "Stuttgart":  {"start": "2019-01-01", "end": None},
    "München":    {"start": "2023-02-01", "end": None},
    "Hamburg":    {"start": "2018-06-01", "end": "2023-09-13"},
    "Berlin":     {"start": "2019-11-01", "end": "2022-01-01"},
}

N_FOURIER = 3
AR_LAGS = 2
NW_MAXLAGS = 15


# ======================================================================
#  DATA FETCHING
# ======================================================================

def _get_ban_info(city):
    """Check if a city has/had a diesel ban. Returns dict or None."""
    for ban_city, info in DIESEL_BAN.items():
        if ban_city.lower() in city.lower() or city.lower() in ban_city.lower():
            return info
    return None


def load_city_stations():
    """
    Load city list + coordinates. Uses annual_all_cities.csv (from script 06)
    to determine which cities to include. Falls back to germany_stations.json.
    Returns dict: {city_name: (lat, lon)}, {city_name: [station_codes]}
    """
    ANNUAL_CACHE = Path(__file__).parent.parent / "data" / "annual_all_cities.csv"

    # Determine which cities to include
    target_cities = None
    if ANNUAL_CACHE.exists():
        annual = pd.read_csv(ANNUAL_CACHE)
        target_cities = set(annual["city"].unique())
        print(f"  Using {len(target_cities)} cities from annual panel cache")

    # Try the full national station cache from script 06
    if STATIONS_CACHE.exists():
        print(f"  Loading station list from {STATIONS_CACHE}")
        with open(STATIONS_CACHE) as f:
            all_stations = json.load(f)

        city_coords = {}
        city_codes = {}
        for sid, info in all_stations.items():
            city = info.get("city", "").strip()
            if not city:
                continue
            # Filter to target cities if available
            if target_cities and city not in target_cities:
                continue
            lat, lon = info.get("lat"), info.get("lon")
            if lat and lon and city not in city_coords:
                city_coords[city] = (float(lat), float(lon))
            if city not in city_codes:
                city_codes[city] = []
            code = info.get("code", "")
            if code:
                city_codes[city].append(code)

        print(f"  Filtered to {len(city_coords)} cities with coordinates")
        return city_coords, city_codes

    # Fallback: germany_stations.json (12 cities)
    print(f"  all_stations.json not found, falling back to {REGISTRY_FILE}")
    print(f"  (Run script 06 first for full national coverage)")
    with open(REGISTRY_FILE, encoding="utf-8") as f:
        data = json.load(f)
    registry = {k: v for k, v in data.items() if not k.startswith("_")}

    # Hardcoded coords for the 12 original cities
    fallback_coords = {
        "Darmstadt":(49.87,8.65),"Frankfurt":(50.11,8.68),
        "Wiesbaden":(50.08,8.24),"Kassel":(51.31,9.48),
        "Stuttgart":(48.78,9.18),"München":(48.14,11.58),
        "Hamburg":(53.55,9.99),"Berlin":(52.52,13.41),
        "Köln":(50.94,6.96),"Düsseldorf":(51.23,6.77),
        "Essen":(51.46,7.01),"Hannover":(52.38,9.73),
    }

    city_codes = {}
    for code, info in registry.items():
        city = info["city"]
        if city not in city_codes:
            city_codes[city] = []
        city_codes[city].append(code)

    city_coords = {c: fallback_coords[c] for c in city_codes if c in fallback_coords}
    return city_coords, city_codes


def build_panel():
    """
    Build the full panel dataset: daily NO2 + weather for each city.
    Uses all_stations.json (from script 06) for national coverage.
    Caches to CSV after first fetch.
    """
    if CACHE_FILE.exists():
        print(f"\n  Loading cached panel data from {CACHE_FILE}")
        df = pd.read_csv(CACHE_FILE, parse_dates=["date"])
        cached_cities = sorted(df['city'].unique())
        print(f"  {len(df)} rows, {len(cached_cities)} cities")

        # If we have all_stations.json and cache has far fewer cities, re-fetch
        if STATIONS_CACHE.exists() and len(cached_cities) < 15:
            print(f"  WARNING: Cache has only {len(cached_cities)} cities "
                  f"but full station list available. Re-fetching...")
            CACHE_FILE.unlink()
        else:
            return df

    print("\n" + "=" * 70)
    print("  BUILDING PANEL DATASET (first run — will take a while)")
    print("=" * 70)

    city_coords, city_codes = load_city_stations()
    print(f"  {len(city_coords)} cities with coordinates")
    print(f"  {sum(len(v) for v in city_codes.values())} total station codes")

    # We'll query the UBA API directly by station code
    api = HessenAirAPI.__new__(HessenAirAPI)
    api.BASE_URL = "https://luftdaten.umweltbundesamt.de/api-proxy"
    api.COMPONENTS = {"NO2": 5}
    api.SCOPES = {"1SMW": 2}

    all_frames = []
    failed_cities = []

    for i, (city, codes) in enumerate(sorted(city_codes.items())):
        if city not in city_coords:
            continue

        print(f"\n  [{i+1}/{len(city_codes)}] {city} "
              f"({len(codes)} station(s))...")

        # Try fetching NO2 for this city using its station codes
        city_records = []
        for code in codes[:3]:  # Try up to 3 stations per city
            try:
                params = {
                    "date_from": "2016-01-01",
                    "date_to": pd.Timestamp.now().strftime("%Y-%m-%d"),
                    "time_from": 1, "time_to": 24,
                    "station": code,
                    "component": 5,  # NO2
                    "scope": 2,      # 1SMW (hourly)
                }

                # Fetch in yearly chunks to avoid timeouts
                from datetime import datetime, timedelta
                dt_start = datetime(2016, 1, 1)
                dt_end = datetime.now()
                cursor = dt_start
                chunk_records = []

                while cursor < dt_end:
                    chunk_end = min(cursor + timedelta(days=365), dt_end)
                    p = {
                        "date_from": cursor.strftime("%Y-%m-%d"),
                        "date_to": chunk_end.strftime("%Y-%m-%d"),
                        "time_from": 1, "time_to": 24,
                        "station": code, "component": 5, "scope": 2,
                    }
                    res = api._get_json(f"{api.BASE_URL}/measures/json",
                                         params=p)
                    data = res.get("data", {})
                    if isinstance(data, dict):
                        for nid, timestamps in data.items():
                            if isinstance(timestamps, dict):
                                for ts, arr in timestamps.items():
                                    if (isinstance(arr, (list,tuple))
                                            and len(arr) > 2
                                            and arr[2] is not None):
                                        try:
                                            chunk_records.append({
                                                "timestamp": ts,
                                                "value": float(arr[2]),
                                            })
                                        except (ValueError, TypeError):
                                            pass
                    cursor = chunk_end + timedelta(days=1)

                if chunk_records:
                    city_records.extend(chunk_records)
                    print(f"    {code}: {len(chunk_records)} hourly records")
                    break  # Got data from this station, no need to try more

            except Exception as e:
                print(f"    {code}: failed ({e})")

        if not city_records:
            print(f"    No NO2 data, skipping.")
            failed_cities.append(city)
            continue

        # Aggregate to daily
        df_hr = pd.DataFrame(city_records)
        df_hr["timestamp"] = pd.to_datetime(df_hr["timestamp"],
                                             errors="coerce")
        df_hr = df_hr.dropna(subset=["timestamp", "value"])
        daily = (df_hr.groupby(df_hr["timestamp"].dt.date)["value"]
                 .mean().reset_index())
        daily.columns = ["date", "no2"]
        daily["date"] = pd.to_datetime(daily["date"])
        daily["city"] = city

        # Fetch weather
        lat, lon = city_coords[city]
        print(f"    Fetching weather ({lat:.2f}, {lon:.2f})...")
        try:
            weather = fetch_dwd_weather(
                lat=lat, lon=lon,
                start_date="2016-01-01",
                chunk_months=6,
            )
            if not weather.empty:
                daily = daily.merge(weather, on="date", how="left")
        except Exception as e:
            print(f"    Weather failed: {e}")

        all_frames.append(daily)
        print(f"    -> {len(daily)} daily records")

    if not all_frames:
        raise SystemExit("No data for any city.")

    if failed_cities:
        print(f"\n  Failed cities ({len(failed_cities)}): {failed_cities}")

    panel = pd.concat(all_frames, ignore_index=True)
    panel = panel.sort_values(["city", "date"]).reset_index(drop=True)

    panel.to_csv(CACHE_FILE, index=False)
    print(f"\n  Cached to {CACHE_FILE}: {len(panel)} rows, "
          f"{panel['city'].nunique()} cities")
    return panel


def add_features(df):
    """Add all regression features to the panel."""
    # AR lags (per city)
    for lag in range(1, AR_LAGS + 1):
        df[f"no2_lag{lag}"] = df.groupby("city")["no2"].shift(lag)

    # Weather lags (per city)
    for var in ["temp_mean", "wind_speed", "precipitation", "humidity"]:
        if var not in df.columns:
            continue
        for lag in range(1, 3):
            df[f"{var}_lag{lag}"] = df.groupby("city")[var].shift(lag)

    # Fourier seasonality
    doy = df["date"].dt.dayofyear
    for k in range(1, N_FOURIER + 1):
        df[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * doy / 365.25)
        df[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * doy / 365.25)

    # Temporal
    df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
    df["trend"] = (df["date"] - df["date"].min()).dt.days / 365.25

    # Treatment variable: ban_active
    df["ban_active"] = 0
    for city in df["city"].unique():
        info = _get_ban_info(city)
        if not info:
            continue
        start = pd.Timestamp(info["start"])
        end = pd.Timestamp(info["end"]) if info.get("end") else df["date"].max()
        mask = (df["city"] == city) & (df["date"] >= start) & (df["date"] <= end)
        df.loc[mask, "ban_active"] = 1

    # Group labels
    def _group(city):
        info = _get_ban_info(city)
        if not info:
            return "No ban"
        if info.get("end"):
            return "Ban lifted"
        return "Diesel ban"
    df["group"] = df["city"].map(_group)

    return df


# ======================================================================
#  OPTION A: PER-CITY ARX → WEATHER-CORRECTED RESIDUALS
# ======================================================================

def per_city_arx(df):
    """
    Fit ARX model per city using ONLY weather + temporal predictors
    (no intervention dummies). The residuals represent
    "weather-corrected NO2" — pollution unexplained by weather/season.
    Compare these across ban/no-ban groups.
    """
    print("\n" + "=" * 70)
    print("  OPTION A: PER-CITY ARX MODELS")
    print("=" * 70)

    weather_cols = [c for c in df.columns
                    if any(c.startswith(w) for w in
                           ["temp_mean", "wind_speed", "precipitation", "humidity"])
                    and c in df.columns]
    fourier_cols = [c for c in df.columns if c.startswith("fourier_")]
    ar_cols = [f"no2_lag{i}" for i in range(1, AR_LAGS + 1)]
    temporal_cols = ["is_weekend", "trend"]

    predictors = ar_cols + weather_cols + fourier_cols + temporal_cols
    predictors = [c for c in predictors if c in df.columns]

    city_results = {}
    residual_frames = []

    for city in sorted(df["city"].unique()):
        cdf = df[df["city"] == city].copy()
        cols = ["date", "no2"] + predictors
        work = cdf[cols].dropna()

        if len(work) < 100:
            print(f"  {city}: too few observations ({len(work)}), skipping")
            continue

        # Drop zero-variance
        active = [c for c in predictors if work[c].std() > 0]
        y = work["no2"].values
        X = sm.add_constant(work[active].values.astype(float))

        try:
            model = sm.OLS(y, X).fit(cov_type='HAC',
                                      cov_kwds={'maxlags': NW_MAXLAGS})
            city_results[city] = {
                "r2": model.rsquared,
                "n": len(work),
            }

            work_out = work[["date"]].copy()
            work_out["city"] = city
            work_out["residual"] = model.resid
            work_out["no2"] = y
            work_out["fitted"] = model.fittedvalues
            residual_frames.append(work_out)

            print(f"  {city:<15} R²={model.rsquared:.3f}  n={len(work)}")
        except Exception as e:
            print(f"  {city}: model failed — {e}")

    residuals = pd.concat(residual_frames, ignore_index=True)
    def _group(city):
        info = _get_ban_info(city)
        if not info: return "No ban"
        if info.get("end"): return "Ban lifted"
        return "Diesel ban"
    residuals["group"] = residuals["city"].map(_group)

    return residuals, city_results


def plot_residual_comparison(residuals):
    """Compare weather-corrected residuals across groups."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    group_colors = {
        "Diesel ban": "#e74c3c",
        "No ban": "#3498db",
        "Ban lifted": "#f39c12",
    }

    # --- Panel 1: Monthly residual averages by group ---
    ax = axes[0]
    monthly = (residuals.groupby([pd.Grouper(key="date", freq="ME"), "group"])
               ["residual"].mean().reset_index())

    for group, color in group_colors.items():
        sub = monthly[monthly["group"] == group]
        if sub.empty:
            continue
        ax.plot(sub["date"], sub["residual"], color=color,
                linewidth=1.5, alpha=0.8, label=group)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(pd.Timestamp("2019-06-01"), color='grey', linestyle='--',
               alpha=0.5, label="Diesel bans start (2019)")
    ax.set_title("Weather-Corrected NO2 Residuals by Group (monthly avg)\n"
                 "Negative = cleaner than weather predicts, "
                 "Positive = dirtier",
                 fontsize=13, fontweight='bold')
    ax.set_ylabel("Residual (µg/m³)")
    ax.legend(fontsize=9)

    # --- Panel 2: Per-city residuals ---
    ax = axes[1]
    monthly_city = (residuals.groupby(
        [pd.Grouper(key="date", freq="3ME"), "city", "group"])
        ["residual"].mean().reset_index())

    for city in sorted(monthly_city["city"].unique()):
        sub = monthly_city[monthly_city["city"] == city]
        group = sub["group"].iloc[0]
        color = group_colors.get(group, "grey")

        ax.plot(sub["date"], sub["residual"], color=color,
                linewidth=1, alpha=0.5)
        if not sub.empty:
            last = sub.iloc[-1]
            ax.text(last["date"] + pd.Timedelta(days=30), last["residual"],
                    city, fontsize=7, color=color, va='center')

    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(pd.Timestamp("2019-06-01"), color='grey', linestyle='--',
               alpha=0.5)

    patches = [mpatches.Patch(color=c, label=g) for g, c in group_colors.items()]
    ax.legend(handles=patches, fontsize=9, loc='upper right')
    ax.set_title("Per-City Weather-Corrected Residuals (quarterly avg)",
                 fontsize=13, fontweight='bold')
    ax.set_ylabel("Residual (µg/m³)")
    ax.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig("16_residual_comparison.png", dpi=200, bbox_inches='tight')
    print(f"\n  Saved: 16_residual_comparison.png")
    plt.close(fig)


# ======================================================================
#  OPTION B: POOLED PANEL DiD WITH CITY FIXED EFFECTS
# ======================================================================

def panel_did(df):
    """
    Pooled panel regression:
      NO2_ct = α_c + β·weather_ct + γ·season_t + φ·AR_lags
             + δ·ban_active_ct + ε_ct

    City fixed effects absorb all time-invariant city differences.
    δ is the DiD treatment effect of having an active diesel ban.
    """
    print("\n" + "=" * 70)
    print("  OPTION B: POOLED PANEL DiD MODEL")
    print("=" * 70)

    # Predictors
    weather_cols = sorted([c for c in df.columns
                           if any(c.startswith(w) for w in
                                  ["temp_mean", "wind_speed",
                                   "precipitation", "humidity"])])
    fourier_cols = sorted([c for c in df.columns if c.startswith("fourier_")])
    ar_cols = [f"no2_lag{i}" for i in range(1, AR_LAGS + 1)]
    temporal_cols = ["is_weekend", "trend"]
    treatment_col = ["ban_active"]

    predictors = ar_cols + weather_cols + fourier_cols + temporal_cols + treatment_col
    predictors = [c for c in predictors if c in df.columns]

    # City fixed effects (dummy encoding, drop one for identification)
    cities = sorted(df["city"].unique())
    ref_city = cities[0]  # Reference category
    for city in cities[1:]:
        df[f"fe_{city}"] = (df["city"] == city).astype(int)
    fe_cols = [f"fe_{c}" for c in cities[1:]]

    all_cols = predictors + fe_cols
    cols_needed = ["date", "city", "no2"] + all_cols
    work = df[cols_needed].dropna(subset=["no2"]).copy()

    # Fill weather NaNs with city-level medians
    for c in weather_cols:
        if c in work.columns:
            work[c] = work.groupby("city")[c].transform(
                lambda x: x.fillna(x.median()))

    work = work.dropna(subset=["no2"] + ar_cols)

    # Drop zero-variance
    active = [c for c in all_cols if work[c].std() > 0]
    y = work["no2"].values
    X = sm.add_constant(work[active].values.astype(float))

    print(f"  Observations: {len(y)}")
    print(f"  Cities: {len(cities)} (ref = {ref_city})")
    print(f"  Predictors: {len(active)} (+ constant)")
    print(f"  Treatment variable: ban_active "
          f"({work['ban_active'].sum()} treated obs)")

    model = sm.OLS(y, X).fit(cov_type='HAC',
                              cov_kwds={'maxlags': NW_MAXLAGS})

    # BY correction
    pvals = model.pvalues[1:]
    coefs = model.params[1:]
    tvals = model.tvalues[1:]
    reject, pvals_by, _, _ = multipletests(pvals, alpha=0.05, method='fdr_by')

    results = pd.DataFrame({
        "predictor": active,
        "coefficient": coefs,
        "t_stat": tvals,
        "p_raw": pvals,
        "p_BY": pvals_by,
        "significant": reject,
    })

    print(f"\n  R²:     {model.rsquared:.4f}")
    print(f"  Adj R²: {model.rsquared_adj:.4f}")

    # The key result
    ban_row = results[results["predictor"] == "ban_active"]
    if not ban_row.empty:
        r = ban_row.iloc[0]
        print(f"\n  *** DIESEL BAN DiD EFFECT ***")
        print(f"  Coefficient: {r['coefficient']:+.3f} µg/m³")
        print(f"  t-stat:      {r['t_stat']:.3f}")
        print(f"  p (raw):     {r['p_raw']:.6f}")
        print(f"  p (BY):      {r['p_BY']:.6f}")
        print(f"  Significant: {'YES ★' if r['significant'] else 'NO'}")

    return results, model, work


def plot_panel_results(results, model):
    """Plot the key panel model results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: non-FE coefficients ---
    # Filter out city fixed effects for readability
    display = results[~results["predictor"].str.startswith("fe_")].copy()
    display = display.sort_values("coefficient")

    colors = []
    for _, row in display.iterrows():
        if row["predictor"] == "ban_active":
            colors.append("#e74c3c" if row["significant"] else "#ffcccc")
        elif row["significant"]:
            colors.append("#3498db")
        else:
            colors.append("#cccccc")

    y_pos = range(len(display))
    ax1.barh(y_pos, display["coefficient"], color=colors,
             edgecolor='white', linewidth=0.5)

    labels = [p.replace("_", " ").replace("fourier ", "F")
              + (" ★" if s else "")
              for p, s in zip(display["predictor"], display["significant"])]
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.axvline(0, color='black', linewidth=0.8)
    ax1.set_xlabel("Coefficient (µg/m³)")
    ax1.set_title("Panel Model Coefficients\n"
                   "(red = diesel ban, blue = other significant)",
                   fontweight='bold')
    ax1.invert_yaxis()

    # --- Right: City fixed effects ---
    fe_rows = results[results["predictor"].str.startswith("fe_")].copy()
    if not fe_rows.empty:
        fe_rows["city"] = fe_rows["predictor"].str.replace("fe_", "")
        fe_rows = fe_rows.sort_values("coefficient")

        ban_colors = []
        for _, row in fe_rows.iterrows():
            city = row["city"]
            info = _get_ban_info(city)
            if info and not info.get("end"):
                ban_colors.append("#e74c3c")
            elif info and info.get("end"):
                ban_colors.append("#f39c12")
            else:
                ban_colors.append("#3498db")

        y_pos = range(len(fe_rows))
        ax2.barh(y_pos, fe_rows["coefficient"], color=ban_colors,
                 edgecolor='white')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(fe_rows["city"], fontsize=9)
        ax2.axvline(0, color='black', linewidth=0.8)
        ax2.set_xlabel("Fixed Effect (µg/m³ vs reference city)")
        ax2.set_title("City Fixed Effects\n"
                       "(baseline pollution level relative to reference)",
                       fontweight='bold')

        patches = [
            mpatches.Patch(color="#e74c3c", label="Diesel ban"),
            mpatches.Patch(color="#f39c12", label="Ban lifted"),
            mpatches.Patch(color="#3498db", label="No ban"),
        ]
        ax2.legend(handles=patches, fontsize=8, loc='lower right')
        ax2.invert_yaxis()

    fig.suptitle(f"Pooled Panel DiD — R² = {model.rsquared:.3f}\n"
                 f"(Newey-West HAC SEs, BY-corrected α=0.05)",
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig("17_panel_did_results.png", dpi=200, bbox_inches='tight')
    print(f"  Saved: 17_panel_did_results.png")
    plt.close(fig)



# ======================================================================
#  MAIN
# ======================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  PANEL DiD ANALYSIS — DIESEL BAN EFFECT")
    print("=" * 70)

    # Build / load panel
    panel = build_panel()
    panel = add_features(panel)

    print(f"\n  Panel: {len(panel)} obs, "
          f"{panel['city'].nunique()} cities, "
          f"{panel['date'].min().date()} → {panel['date'].max().date()}")

    # Option A: per-city ARX
    residuals, city_r2 = per_city_arx(panel)
    plot_residual_comparison(residuals)

    # Option B: pooled panel DiD
    panel_results, panel_model, _ = panel_did(panel)
    plot_panel_results(panel_results, panel_model)

    # Print summary table
    print("\n" + "=" * 70)
    print("  PER-CITY MODEL FIT (Option A)")
    print("=" * 70)
    for city, info in sorted(city_r2.items()):
        bi = _get_ban_info(city)
        if bi and not bi.get("end"):
            label = "BAN"
        elif bi and bi.get("end"):
            label = "LFT"
        else:
            label = "   "
        print(f"  {label}  {city:<20}  R²={info['r2']:.3f}  n={info['n']}")

    print("\n" + "=" * 70)
    print("  DONE:")
    print("    16_residual_comparison.png  — Per-city weather-corrected residuals")
    print("    17_panel_did_results.png    — Panel DiD coefficients + city FEs")
    print("=" * 70)