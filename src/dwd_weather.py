"""
Fetch daily weather data from DWD via the Bright Sky API.
https://brightsky.dev — free, no auth, JSON.

Provides: temperature, wind speed, precipitation, humidity, sunshine.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from time import sleep


BRIGHTSKY_URL = "https://api.brightsky.dev/weather"

# Darmstadt approximate coordinates
DARMSTADT_LAT = 49.8728
DARMSTADT_LON = 8.6512


def fetch_dwd_weather(lat=DARMSTADT_LAT, lon=DARMSTADT_LON,
                      start_date="2016-01-01", end_date=None,
                      chunk_months=3):
    """
    Fetch hourly DWD weather, resample to daily.

    Args:
        lat, lon:      Coordinates (default: Darmstadt)
        start_date:    Start date string YYYY-MM-DD
        end_date:      End date (default: today)
        chunk_months:  Months per API request (brightsky limits range)

    Returns:
        DataFrame with daily columns:
            date, temp_mean, temp_min, temp_max,
            wind_speed, precipitation, humidity, sunshine_hours
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    dt_start = datetime.strptime(start_date, "%Y-%m-%d")
    dt_end = datetime.strptime(end_date, "%Y-%m-%d")

    # Build chunks
    chunks = []
    cursor = dt_start
    while cursor < dt_end:
        chunk_end = min(cursor + timedelta(days=chunk_months * 30), dt_end)
        chunks.append((cursor.strftime("%Y-%m-%d"),
                       chunk_end.strftime("%Y-%m-%d")))
        cursor = chunk_end + timedelta(days=1)

    print(f"  Fetching DWD weather: {start_date} → {end_date} "
          f"({len(chunks)} chunks)")

    all_records = []
    for i, (c_start, c_end) in enumerate(chunks):
        try:
            r = requests.get(BRIGHTSKY_URL, params={
                "lat": lat, "lon": lon,
                "date": c_start,
                "last_date": c_end,
            }, timeout=30)
            r.raise_for_status()
            data = r.json()

            for entry in data.get("weather", []):
                all_records.append({
                    "timestamp": entry.get("timestamp"),
                    "temperature": entry.get("temperature"),
                    "wind_speed": entry.get("wind_speed"),
                    "precipitation": entry.get("precipitation"),
                    "humidity": entry.get("relative_humidity"),
                    "sunshine": entry.get("sunshine"),  # minutes
                    "pressure": entry.get("pressure_msl"),
                    "cloud_cover": entry.get("cloud_cover"),
                })

            if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                print(f"    chunk {i+1}/{len(chunks)} — "
                      f"{len(all_records)} hourly records")

            sleep(0.3)  # Be polite to the API

        except Exception as e:
            print(f"    chunk {i+1} failed: {e}")

    if not all_records:
        print("  No weather data returned.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["date"] = df["timestamp"].dt.date

    # Resample to daily
    daily = df.groupby("date").agg(
        temp_mean=("temperature", "mean"),
        temp_min=("temperature", "min"),
        temp_max=("temperature", "max"),
        wind_speed=("wind_speed", "mean"),
        precipitation=("precipitation", "sum"),
        humidity=("humidity", "mean"),
        sunshine_hours=("sunshine", lambda x: x.sum() / 60),  # min → hours
        pressure=("pressure", "mean"),
        cloud_cover=("cloud_cover", "mean"),
    ).reset_index()

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    print(f"  Weather: {len(daily)} daily records, "
          f"{daily['date'].min().date()} → {daily['date'].max().date()}")

    return daily


if __name__ == "__main__":
    # Quick test
    df = fetch_dwd_weather(start_date="2025-01-01", end_date="2025-01-10")
    print(df)