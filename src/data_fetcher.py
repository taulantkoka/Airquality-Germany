import json
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from time import sleep


class HessenAirAPI:
    """
    Client for the UBA Luftdaten API v4.

    Key discovery: the API uses internal numeric station IDs (e.g. 668),
    not the DEHE codes. We probe /measures with each station code to
    learn the numeric ID mapping, then use those IDs to filter
    /annualbalances rows.
    """

    BASE_URL = "https://luftdaten.umweltbundesamt.de/api-proxy"
    API_START = "2016-01-01"

    COMPONENTS = {
        "PM10": 1, "CO": 2, "O3": 3, "SO2": 4, "NO2": 5,
        "PB": 6, "BAP": 7, "C6H6": 8, "PM2.5": 9,
        "AS": 10, "CD": 11, "NI": 12,
    }

    SCOPES = {
        "1TMW": 1, "1SMW": 2, "1SMW_MAX": 3,
        "8SMW": 4, "8SMW_MAX": 5, "1TMWGL": 6,
    }

    def __init__(self, stations_file=None):
        if stations_file is None:
            stations_file = Path(__file__).parent / "hessen_stations.json"
        self.stations = self._load_stations(stations_file)
        # Populated lazily: {numeric_id_str: station_code}
        self._id_to_code = {}

    # ------------------------------------------------------------------
    #  Station lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _load_stations(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return {k: v for k, v in data.items() if not k.startswith("_")}
        except FileNotFoundError:
            print(f"ERROR: Station file not found: {path}")
            return {}

    def find_stations(self, query):
        q = query.lower()
        return {
            code: info for code, info in self.stations.items()
            if q in info["city"].lower()
            or q in info["name"].lower()
            or q in code.lower()
        }

    def list_cities(self):
        return sorted({info["city"] for info in self.stations.values()})

    # ------------------------------------------------------------------
    #  Discover numeric station IDs
    # ------------------------------------------------------------------

    def _discover_numeric_ids(self, station_codes, comp_id=5):
        """
        Probe /measures with 1 day of data per station code to learn
        the API's internal numeric ID for each station.

        The response keys data as {numeric_id: {timestamp: [...]}}.
        We query by code and read back whatever numeric key appears.
        """
        discovered = {}
        probe_date = "2025-01-15"  # A recent date likely to have data

        for code in station_codes:
            if code in [v for v in self._id_to_code.values()]:
                # Already known
                for nid, c in self._id_to_code.items():
                    if c == code:
                        discovered[code] = nid
                        break
                continue

            try:
                res = self._get_json(
                    f"{self.BASE_URL}/measures/json",
                    params={
                        "date_from": probe_date, "date_to": probe_date,
                        "time_from": 12, "time_to": 13,
                        "station": code, "component": comp_id, "scope": 2,
                    },
                )
                data = res.get("data", {})
                if isinstance(data, dict) and data:
                    numeric_id = list(data.keys())[0]
                    discovered[code] = str(numeric_id)
                    self._id_to_code[str(numeric_id)] = code
                    print(f"    {code} → numeric ID {numeric_id}")
            except Exception as e:
                print(f"    {code} probe failed: {e}")

        return discovered

    # ------------------------------------------------------------------
    #  HTTP helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_json(url, params=None, retries=3, timeout=30):
        for attempt in range(retries):
            try:
                r = requests.get(url, params=params, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except requests.RequestException as e:
                if attempt < retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"    Retry {attempt+1}/{retries} in {wait}s — {e}")
                    sleep(wait)
                else:
                    raise
        return {}

    # ------------------------------------------------------------------
    #  API: /annualbalances — yearly summaries (~2000 onwards)
    # ------------------------------------------------------------------

    def get_annual_balances(self, pollutant="NO2", city="Darmstadt",
                           start_year=2000, end_year=None):
        """
        Fetch yearly summary statistics from /annualbalances.

        Response format (from debug):
          indices: ["station id", "component id", "year", "value",
                    "transgression type id"]
          data: list of rows, e.g. ["668", "17", "0"]
                (shorter than indices — station_id, value, transgression)

        We first discover numeric IDs for our stations, then filter.
        """
        comp_id = self.COMPONENTS.get(pollutant.upper())
        if comp_id is None:
            print(f"Unknown pollutant '{pollutant}'.")
            return pd.DataFrame()

        matches = self.find_stations(city)
        if not matches:
            print(f"No stations match '{city}'.")
            return pd.DataFrame()

        if end_year is None:
            end_year = datetime.now().year - 1

        # Step 1: discover numeric IDs for our stations
        print(f"  Discovering numeric station IDs …")
        code_to_numid = self._discover_numeric_ids(
            list(matches.keys()), comp_id=comp_id
        )

        if not code_to_numid:
            print("  Could not discover any numeric IDs.")
            return pd.DataFrame()

        # Reverse map: numeric_id → station_code
        numid_to_code = {nid: code for code, nid in code_to_numid.items()}
        target_numids = set(numid_to_code.keys())

        print(f"  Fetching annual balances {start_year}–{end_year} "
              f"for {pollutant} …")

        all_records = []
        for year in range(start_year, end_year + 1):
            try:
                res = self._get_json(
                    f"{self.BASE_URL}/annualbalances/json",
                    params={"component": comp_id, "year": year, "lang": "en"},
                )
                data = res.get("data", [])
                if not isinstance(data, list):
                    continue

                for row in data:
                    if not isinstance(row, (list, tuple)) or len(row) < 2:
                        continue

                    # Row format: [station_id, value, transgression_count]
                    row_id = str(row[0]).strip()
                    if row_id not in target_numids:
                        continue

                    try:
                        annual_mean = float(row[1])
                    except (ValueError, TypeError, IndexError):
                        continue

                    code = numid_to_code[row_id]
                    all_records.append({
                        "station_code": code,
                        "station_name": matches[code]["name"],
                        "year": year,
                        "annual_mean": annual_mean,
                    })

                if year % 5 == 0 or year == end_year:
                    print(f"    {year} — {len(all_records)} records so far")

            except Exception as e:
                print(f"    {year} failed: {e}")

        df = pd.DataFrame(all_records)
        if not df.empty:
            df = df.sort_values(["station_code", "year"]).reset_index(drop=True)
            print(f"  Annual balances: {len(df)} station-year records, "
                  f"{df['year'].min()}–{df['year'].max()}")
        else:
            print("  No annual balance data found for your stations.")
        return df

    # ------------------------------------------------------------------
    #  API: /measures — hourly data (2016+)
    # ------------------------------------------------------------------

    def get_api_data(self, pollutant="NO2", city="Darmstadt",
                     days=None, start_date=None, end_date=None,
                     scope="1SMW", chunk_days=90):
        comp_id = self.COMPONENTS.get(pollutant.upper())
        if comp_id is None:
            print(f"Unknown pollutant '{pollutant}'.")
            return pd.DataFrame()

        scope_id = self.SCOPES.get(scope)
        if scope_id is None:
            print(f"Unknown scope '{scope}'.")
            return pd.DataFrame()

        matches = self.find_stations(city)
        if not matches:
            print(f"No stations match '{city}'.")
            return pd.DataFrame()

        dt_end = (datetime.strptime(end_date, "%Y-%m-%d")
                  if end_date else datetime.now())
        if start_date:
            dt_start = datetime.strptime(start_date, "%Y-%m-%d")
        elif days:
            dt_start = dt_end - timedelta(days=days)
        else:
            dt_start = datetime.strptime(self.API_START, "%Y-%m-%d")

        api_floor = datetime.strptime(self.API_START, "%Y-%m-%d")
        if dt_start < api_floor:
            dt_start = api_floor

        total_days = (dt_end - dt_start).days
        print(f"  Date range: {dt_start.date()} → {dt_end.date()} "
              f"({total_days} days)")

        chunks = []
        cursor = dt_start
        while cursor < dt_end:
            chunk_end = min(cursor + timedelta(days=chunk_days), dt_end)
            chunks.append((cursor.strftime("%Y-%m-%d"),
                           chunk_end.strftime("%Y-%m-%d")))
            cursor = chunk_end + timedelta(days=1)

        all_records = []
        for code, info in matches.items():
            print(f"  {info['name']} [{code}] — {len(chunks)} chunk(s)")
            for i, (c_start, c_end) in enumerate(chunks):
                params = {
                    "date_from": c_start, "date_to": c_end,
                    "time_from": 1, "time_to": 24,
                    "station": code, "component": comp_id, "scope": scope_id,
                }
                try:
                    res = self._get_json(
                        f"{self.BASE_URL}/measures/json", params=params
                    )
                    data = res.get("data", {})
                    if data and isinstance(data, dict):
                        # Data keyed by numeric station ID
                        for numeric_id, timestamps in data.items():
                            if isinstance(timestamps, dict):
                                for ts, arr in timestamps.items():
                                    v = _extract(arr)
                                    if v is not None:
                                        all_records.append({
                                            "station": info["name"],
                                            "station_code": code,
                                            "timestamp": ts,
                                            "value": v,
                                        })
                        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                            print(f"    chunk {i+1}/{len(chunks)} — "
                                  f"{len(all_records)} total so far")
                except Exception as e:
                    print(f"    chunk {i+1} failed: {e}")

        df = pd.DataFrame(all_records)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp", "value"])
            df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)
            print(f"  Done: {len(df)} records, "
                  f"{df['station'].nunique()} station(s)")
        else:
            print("  No measurement data returned.")
        return df

    # ------------------------------------------------------------------
    #  CSV: load historical HLNUG exports
    # ------------------------------------------------------------------

    def load_historical_csv(self, file_path, pollutant_keyword="no2"):
        print(f"Reading CSV: {file_path} …")
        try:
            df = pd.read_csv(
                file_path, sep=';', decimal=',',
                skiprows=8, encoding='latin1', low_memory=False,
            )
            df.columns = [c.split(' ')[0].strip().lower() for c in df.columns]

            date_col = 'datum' if 'datum' in df.columns else df.columns[0]
            time_col = next(
                (c for c in ['uhrzeit', 'zeit'] if c in df.columns), None
            )
            if time_col:
                df['timestamp'] = pd.to_datetime(
                    df[date_col] + ' ' + df[time_col],
                    dayfirst=True, errors='coerce',
                )
            else:
                df['timestamp'] = pd.to_datetime(
                    df[date_col], dayfirst=True, errors='coerce',
                )

            poll_cols = [c for c in df.columns
                         if pollutant_keyword.lower() in c]
            if not poll_cols:
                raise ValueError(
                    f"'{pollutant_keyword}' not found in {list(df.columns)}"
                )

            df = df.rename(columns={poll_cols[0]: 'value'})
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna(subset=['timestamp', 'value'])

            print(f"  Loaded {len(df)} records.")
            return df[['timestamp', 'value']].sort_values('timestamp')

        except Exception as e:
            print(f"CSV error: {e}")
            return pd.DataFrame()


def _extract(arr):
    """Value is at index 2: [comp_id, scope_id, value, date_end, status]"""
    if isinstance(arr, (list, tuple)) and len(arr) > 2 and arr[2] is not None:
        try:
            return float(arr[2])
        except (ValueError, TypeError):
            pass
    return None