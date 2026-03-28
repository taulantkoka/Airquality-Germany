"""
Diagnostic — run first to verify API responses.

    python debug_api.py
"""
import requests, json
from datetime import datetime, timedelta

BASE = "https://luftdaten.umweltbundesamt.de/api-proxy"


def test_measures():
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")

    print("=" * 60)
    print(f"1) /measures — DEHE040, NO2 hourly, {start} → {end}")
    print("=" * 60)

    params = {
        "date_from": start, "date_to": end,
        "time_from": 1, "time_to": 24,
        "station": "DEHE040", "component": 5, "scope": 2,
    }
    try:
        r = requests.get(f"{BASE}/measures/json", params=params, timeout=30)
        print(f"HTTP {r.status_code}")
        data = r.json()
        print(f"Keys: {list(data.keys())}")
        content = data.get("data", {})
        print(f"'data' type: {type(content).__name__}, entries: {len(content)}\n")

        if isinstance(content, dict):
            for i, (k, v) in enumerate(content.items()):
                print(f"  {k}  →  {json.dumps(v, ensure_ascii=False)[:150]}")
                if i >= 4:
                    break
        elif isinstance(content, list):
            for i, row in enumerate(content[:5]):
                print(f"  [{i}]  →  {json.dumps(row, ensure_ascii=False)[:150]}")
    except Exception as e:
        print(f"Failed: {e}")


def test_annual_balances():
    print("\n" + "=" * 60)
    print("2) /annualbalances — NO2, year 2010")
    print("=" * 60)

    params = {"component": 5, "year": 2010, "lang": "en"}
    try:
        r = requests.get(f"{BASE}/annualbalances/json",
                         params=params, timeout=30)
        print(f"HTTP {r.status_code}")
        data = r.json()
        print(f"Keys: {list(data.keys())}")

        if "indices" in data:
            print(f"Indices: {json.dumps(data['indices'], ensure_ascii=False)}")

        content = data.get("data", [])
        ctype = type(content).__name__
        clen = len(content)
        print(f"'data' type: {ctype}, entries: {clen}")

        # Handle list (tabular) format
        if isinstance(content, list):
            # Find rows containing DEHE
            hessen_rows = [
                row for row in content
                if isinstance(row, (list, tuple))
                and any("DEHE" in str(x) for x in row)
            ]
            print(f"\nHessen (DEHE) rows: {len(hessen_rows)}")
            for i, row in enumerate(hessen_rows[:8]):
                print(f"  {json.dumps(row, ensure_ascii=False)[:200]}")

            if not hessen_rows:
                print("\nNo DEHE found — showing first 5 rows:")
                for i, row in enumerate(content[:5]):
                    print(f"  {json.dumps(row, ensure_ascii=False)[:200]}")

        # Handle dict format
        elif isinstance(content, dict):
            hessen = {k: v for k, v in content.items()
                      if "DEHE" in str(k) or "DEHE" in str(v)}
            print(f"\nHessen entries: {len(hessen)}")
            for i, (k, v) in enumerate(list(hessen.items())[:8]):
                print(f"  {k} → {json.dumps(v, ensure_ascii=False)[:200]}")

    except Exception as e:
        print(f"Failed: {e}")


if __name__ == "__main__":
    test_measures()
    test_annual_balances()
    print("\n" + "=" * 60)
    print("Done. Share output if parsing needs adjustment.")
    print("=" * 60)