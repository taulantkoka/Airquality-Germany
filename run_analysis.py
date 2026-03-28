#!/usr/bin/env python3
"""
============================================================================
  Airquality-Hessen — Master Analysis Runner
============================================================================

  Runs the full analysis pipeline and saves all figures as PDF
  to the figures/ directory.

  Usage:
    python run_analysis.py              # Run everything
    python run_analysis.py --quick      # Skip panel (saves ~20 min)
    python run_analysis.py --only 02    # Run only ARX model

  Project structure:
    src/         → Data fetchers (UBA API, DWD weather)
    data/        → Station registries, cached panel data
    analysis/    → Analysis scripts (01–05)
    figures/     → Output (PDF)
    report/      → Report markdown
============================================================================
"""

import sys
import os
import argparse
from pathlib import Path

# ---- Path setup ----
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

# Ensure figures/ exists
(ROOT / "figures").mkdir(exist_ok=True)

# ---- Monkey-patch matplotlib to save PDFs to figures/ ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_original_savefig = plt.Figure.savefig

def _patched_savefig(self, fname, *args, **kwargs):
    """Redirect all saves to figures/ as BOTH PDF and SVG."""
    fname = str(fname)
    base = os.path.splitext(os.path.basename(fname))[0]
    
    # Define paths
    pdf_path = ROOT / "figures" / f"{base}.pdf"
    svg_path = ROOT / "figures" / f"{base}.svg"
    
    kwargs.setdefault("bbox_inches", "tight")
    
    # Save the PDF (for your high-depth vector needs/LaTeX)
    _original_savefig(self, str(pdf_path), *args, **kwargs)
    
    # Save the SVG (for your Markdown report/Web)
    _original_savefig(self, str(svg_path), format='svg', **kwargs)
    
    print(f"  → Saved PDF: {pdf_path.name}")
    print(f"  → Saved SVG: {svg_path.name}")

plt.Figure.savefig = _patched_savefig


# ---- Fix data_fetcher station file paths ----
import data_fetcher
data_fetcher_original_init = data_fetcher.HessenAirAPI.__init__

def _patched_init(self, stations_file=None):
    if stations_file is None:
        # Try data/ directory first, then current directory
        for candidate in [
            ROOT / "data" / "hessen_stations.json",
            ROOT / "hessen_stations.json",
            Path("hessen_stations.json"),
        ]:
            if candidate.exists():
                stations_file = candidate
                break
    data_fetcher_original_init(self, stations_file)

data_fetcher.HessenAirAPI.__init__ = _patched_init

# Also patch panel_data.csv cache path
import importlib

# ---- Analysis modules ----
ANALYSES = {
    "01": ("Exploratory plots",         "analysis/01_exploration.py"),
    "02": ("ARX driver model",          "analysis/02_arx_model.py"),
    "03": ("Structural breaks",         "analysis/03_structural_breaks.py"),
    "04": ("Panel DiD (cross-city)",    "analysis/04_panel_did.py"),
    "05": ("Darmstadt 25-year trend",   "analysis/05_darmstadt_trend.py"),
}


def run_script(path):
    """Execute an analysis script."""
    full_path = ROOT / path
    if not full_path.exists():
        print(f"  SKIP: {path} not found")
        return False

    print(f"\n{'='*70}")
    print(f"  Running: {path}")
    print(f"{'='*70}")

    # Execute in a clean namespace but with our patched imports
    script_dir = str(full_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        exec(open(full_path).read(), {"__name__": "__main__"})
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Airquality-Hessen analysis")
    parser.add_argument("--quick", action="store_true",
                        help="Skip slow panel analysis (04)")
    parser.add_argument("--only", type=str, default=None,
                        help="Run only specific analysis (e.g. '02')")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  AIRQUALITY-HESSEN — FULL ANALYSIS PIPELINE")
    print("=" * 70)
    print(f"  Project root: {ROOT}")
    print(f"  Figures dir:  {ROOT / 'figures'}")

    results = {}

    for key, (name, path) in ANALYSES.items():
        if args.only and args.only != key:
            continue
        if args.quick and key == "04":
            print(f"\n  SKIP: {name} (--quick mode)")
            continue

        success = run_script(path)
        results[key] = (name, success)

    # Summary
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    for key, (name, success) in results.items():
        status = "✓" if success else "✗"
        print(f"  {status}  {key}: {name}")

    # List generated figures
    figs = sorted((ROOT / "figures").glob("*.png"))
    if figs:
        print(f"\n  Generated {len(figs)} figures in figures/:")
        for f in figs:
            print(f"    {f.name}")

    print(f"\n  Report: report/report.md")
    print("=" * 70)


if __name__ == "__main__":
    main()
