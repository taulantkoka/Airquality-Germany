# Airquality-Germany

**Do diesel bans actually work? A data-driven analysis of 25 years of German air quality resulting from an afternoons worth of free time.**

An open-source, fully reproducible study using public APIs from the German Federal Environment Agency (UBA), DWD weather service, and city-level open data.

## Key Findings

- **Diesel ban reduced NO2 by ~5 µg/m³/day** in Darmstadt (p < 0.001, surviving Benjamini-Yekutieli FDR correction)
- **Cross-city panel confirms: −4 µg/m³** treatment effect comparing ban cities vs control cities with city-specific weather controls
- **2019 was the strongest structural break** in 25 years of NO2 data — stronger than the 2015 Umweltzone
- **PM10 is largely unaffected** by diesel bans (driven by weather + Sahara dust, not traffic)
- **9-Euro-Ticket produced a measurable NO2 reduction** (−1.4 µg/m³)

## Quick Start

```bash
pip install -r requirements.txt
python run_analysis.py          # Full pipeline (~30 min)
python run_analysis.py --quick  # Skip cross-city panel
```

## Project Structure

```
src/              Data fetchers (UBA API, DWD weather)
data/             Station registries, cached data
analysis/         Analysis scripts (01–05) and
figures/          Output plots (PDF/SVG)
report/           Full report (report.md)
```

## Read the Full Report

See [`report/report.md`](report/report.md) for the complete analysis with methodology, results, diagnostics, and limitations.

## Data Sources

All data fetched live from public APIs — no downloads needed.

| Source | What |
|--------|------|
| [UBA Luftdaten](https://luftdaten.umweltbundesamt.de) | NO2, PM10, PM2.5, O3 (hourly + annual) |
| [Bright Sky](https://brightsky.dev) (DWD) | Temperature, wind, rain, humidity |
| [Darmstadt Open Data](https://opendata.darmstadt.de) | Traffic counts (future integration) |

*Quellenangabe: „Umweltbundesamt mit Daten der Messnetze der Länder und des Bundes"*
