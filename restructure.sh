#!/bin/bash
# ============================================================================
#  Restructure Airquality-Hessen project
#  Run from the project root: bash restructure.sh
# ============================================================================

set -e

echo "Restructuring Airquality-Hessen project..."

# Create directories
mkdir -p src analysis data figures report

# Move data layer
mv -f data_fetcher.py src/ 2>/dev/null || true
mv -f dwd_weather.py src/ 2>/dev/null || true
mv -f debug_api.py src/ 2>/dev/null || true
touch src/__init__.py

# Move station registries
mv -f hessen_stations.json data/ 2>/dev/null || true
mv -f germany_stations.json data/ 2>/dev/null || true
mv -f panel_data.csv data/ 2>/dev/null || true

# Move analysis scripts
mv -f hessen_exploration.py analysis/01_exploration.py 2>/dev/null || true
mv -f regression_v2.py analysis/02_arx_model.py 2>/dev/null || true
mv -f structural_breaks.py analysis/03_structural_breaks.py 2>/dev/null || true
mv -f panel_analysis.py analysis/04_panel_did.py 2>/dev/null || true
mv -f darmstadt_analysis.py analysis/05_darmstadt_trend.py 2>/dev/null || true

# Move report
mv -f report.md report/ 2>/dev/null || true
mv -f project_architecture.md report/ 2>/dev/null || true

# Move existing figures to figures/
mv -f *.png figures/ 2>/dev/null || true

# Remove deprecated files
rm -f regression_analysis.py 2>/dev/null || true
rm -f hessen_extrapolation.py 2>/dev/null || true
rm -f cross_state_analysis.py 2>/dev/null || true

echo ""
echo "New structure:"
echo "  src/              ← Data fetchers (API, weather)"
echo "  data/             ← Station registries, cached data"
echo "  analysis/         ← Analysis scripts (01–05)"
echo "  figures/          ← Output plots (PDF)"
echo "  report/           ← Report markdown"
echo ""
echo "Next: python run_analysis.py"
