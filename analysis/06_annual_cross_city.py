"""
============================================================================
  Germany-Wide Annual Analysis: Umweltzone Staggered DiD
============================================================================

  1. Discovers ALL UBA stations via /stations/json endpoint
  2. Filters to traffic + urban background in Grossstaedte
  3. Fetches NO2 annual balances (2000-2024) for all discovered stations
  4. Matches cities to Umweltzone introduction dates
  5. Fetches annual weather (temperature + wind) per city
  6. Runs staggered DiD: NO2 ~ city_FE + trend + weather + uz_active

  Outputs:
    19_annual_all_cities        - Raw NO2 trajectories by UZ group
    20_annual_weather_corrected - Weather-corrected by group
    21_annual_break_scan        - Sup-F scan per city grid
    22_staggered_did            - Umweltzone staggered DiD results
    23_regression_discontinuity - RD/IV around EU limit threshold

  Run: python 06_annual_cross_city.py
============================================================================
"""

import json, requests, pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from pathlib import Path
from time import sleep

sns.set_style("whitegrid")

CACHE_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR.mkdir(exist_ok=True)
STATIONS_CACHE = CACHE_DIR / "all_stations.json"
ANNUAL_CACHE   = CACHE_DIR / "annual_all_cities.csv"
BASE_URL = "https://luftdaten.umweltbundesamt.de/api-proxy"

# Umweltzone: (start_year, end_year). end_year=None means still active.
UMWELTZONE = {
    # Early wave (2008)
    "Berlin":(2008,None),"Köln":(2008,None),"Stuttgart":(2008,None),
    "Mannheim":(2008,None),"Ludwigsburg":(2008,None),
    "Bochum":(2008,None),"Dortmund":(2008,None),"Duisburg":(2008,None),
    "Essen":(2008,None),"Gelsenkirchen":(2008,None),"Oberhausen":(2008,None),
    "Mülheim an der Ruhr":(2008,None),
    "München":(2008,None),"Frankfurt am Main":(2008,None),
    # Lifted early-wave
    "Hannover":(2008,2023),"Reutlingen":(2008,2024),"Tübingen":(2008,2024),
    "Schwäbisch Gmünd":(2008,2024),
    # 2009
    "Augsburg":(2009,None),"Bremen":(2009,None),"Düsseldorf":(2009,None),
    "Karlsruhe":(2009,2023),"Heidelberg":(2009,2023),"Heilbronn":(2009,2024),
    # 2010
    "Osnabrück":(2010,None),"Freiburg im Breisgau":(2010,2024),
    "Wuppertal":(2010,None),"Münster":(2010,None),"Bonn":(2010,None),
    "Aachen":(2010,None),
    # 2011+
    "Krefeld":(2011,None),"Leipzig":(2011,None),"Magdeburg":(2011,None),
    "Hagen":(2012,None),
    "Wiesbaden":(2013,None),"Offenbach am Main":(2013,None),
    "Darmstadt":(2015,None),"Marburg":(2015,None),
    "Limburg an der Lahn":(2018,None),"Regensburg":(2018,None),
    "Kassel":(2018,None),"Siegen":(2018,None),
    "Halle (Saale)":(2011,None),
    # Lifted (Thueringen 2021, Rheinland-Pfalz 2025)
    "Erfurt":(2012,2021),"Mainz":(2013,2025),
    # Never had one - Schleswig-Holstein (no UZ statewide)
    "Hamburg":None,"Kiel":None,"Lübeck":None,"Flensburg":None,
    # Never had one - Mecklenburg-Vorpommern (no UZ statewide)
    "Rostock":None,"Schwerin":None,
    # Never had one - Brandenburg (no UZ statewide)
    "Potsdam":None,"Cottbus":None,"Brandenburg an der Havel":None,
    # Never had one - Sachsen (Leipzig has one, but Dresden doesn't)
    "Dresden":None,"Chemnitz":None,
    # Never had one - Saarland (no UZ statewide)
    "Saarbrücken":None,
    # Never had one - individual cities that avoided UZ
    "Braunschweig":None,"Bielefeld":None,"Nürnberg":None,
    "Würzburg":None,"Göttingen":None,"Oldenburg":None,
    "Wolfsburg":None,"Hildesheim":None,"Salzgitter":None,
    "Jena":None,"Gera":None,
    "Kaiserslautern":None,"Trier":None,"Koblenz":None,
    "Lüneburg":None,"Celle":None,"Paderborn":None,
    "Gütersloh":None,"Moers":None,"Solingen":None,
    "Worms":None,"Bremerhaven":None,
}

DIESEL_BAN = {
    "Darmstadt":{"start":2019,"end":None},"Stuttgart":{"start":2019,"end":None},
    "München":{"start":2023,"end":None},"Hamburg":{"start":2018,"end":2023},
    "Berlin":{"start":2019,"end":2022},
}

CITY_COORDS = {}

GROUP_COLORS = {
    "Early UZ (2008)":"#2ecc71","Mid UZ (2009-10)":"#3498db",
    "Late UZ (2011+)":"#e74c3c","No Umweltzone":"#95a5a6",
}

def api_get(endpoint, params=None, retries=3):
    url = f"{BASE_URL}/{endpoint}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt < retries - 1: sleep(2**(attempt+1))
            else: print(f"    API error: {e}"); return {}

def _match_uz(city):
    """Return (start, end) tuple or None for a city."""
    if not city: return None
    if city in UMWELTZONE: return UMWELTZONE[city]
    cl = city.lower()
    for k,v in UMWELTZONE.items():
        if k.lower() in cl or cl in k.lower(): return v
    return None

def _uz_start(city):
    """Return Umweltzone start year or None."""
    uz = _match_uz(city)
    if uz is None: return None
    if isinstance(uz, tuple): return uz[0]
    return None

def _uz_active(city, year):
    """Check if Umweltzone was active for city in given year."""
    uz = _match_uz(city)
    if uz is None: return False
    if isinstance(uz, tuple):
        start, end = uz
        if end is None: return year >= start
        return start <= year <= end
    return False

def _ban_active(city, year):
    for bc, info in DIESEL_BAN.items():
        if bc.lower() in city.lower() or city.lower() in bc.lower():
            s,e = info["start"], info.get("end") or 9999
            if s <= year <= e: return True
    return False

def _classify(uz_start):
    if pd.isna(uz_start) or uz_start is None: return "No Umweltzone"
    if uz_start <= 2008: return "Early UZ (2008)"
    if uz_start <= 2010: return "Mid UZ (2009-10)"
    return "Late UZ (2011+)"

# ======================================================================
def discover_stations():
    if STATIONS_CACHE.exists():
        print(f"  Loading cached stations from {STATIONS_CACHE}")
        with open(STATIONS_CACHE) as f: return json.load(f)

    print("\n  Discovering all UBA stations...")
    res = api_get("stations/json", params={"lang":"en"})
    if not res: return {}

    indices = res.get("indices", [])
    data = res.get("data", {})
    print(f"  Indices: {indices}")
    print(f"  Total raw stations: {len(data)}")

    stations = {}
    for sid, vals in data.items():
        if not isinstance(vals, (list,tuple)): continue
        info = {indices[i]: vals[i] for i in range(min(len(indices),len(vals)))}
        # Field names vary by API version; try common variants
        code = str(info.get("station code", info.get("code", "")))
        name = str(info.get("station name", info.get("name", "")))
        city = str(info.get("city", info.get("station city", "")))
        setting = str(info.get("station setting", info.get("type", "")))
        lat = info.get("station latitude", info.get("latitude", None))
        lon = info.get("station longitude", info.get("longitude", None))
        try: lat = float(lat)
        except: lat = None
        try: lon = float(lon)
        except: lon = None
        stations[str(sid)] = {"code":code,"name":name,"city":city,
                               "setting":setting,"lat":lat,"lon":lon}

    print(f"  Parsed {len(stations)} stations")
    for sid,info in list(stations.items())[:3]:
        print(f"    {sid}: {info['code']} - {info['name']} ({info['city']})")

    with open(STATIONS_CACHE,'w') as f:
        json.dump(stations, f, ensure_ascii=False, indent=2)
    return stations

def _normalize_city(api_city):
    """
    Map API city names to our UMWELTZONE registry names.
    E.g. "Duisburg-Bruckhausen" -> "Duisburg",
         "Nürnberg, Stadtteil Mögeldorf" -> "Nürnberg",
         "Frankfurt am Main" -> "Frankfurt am Main"
    """
    if not api_city:
        return None
    api_city = api_city.strip()

    # Direct match
    if api_city in UMWELTZONE:
        return api_city

    # Check if any registry key is a prefix of the API name
    # (handles "Duisburg-Bruckhausen", "Köln-Weiden", etc.)
    api_lower = api_city.lower()
    best_match = None
    best_len = 0
    for uzc in UMWELTZONE:
        uzc_lower = uzc.lower()
        # API name starts with registry name (+ separator or end)
        if api_lower.startswith(uzc_lower):
            rest = api_lower[len(uzc_lower):]
            if rest == "" or rest[0] in "-,/ ":
                if len(uzc) > best_len:
                    best_match = uzc
                    best_len = len(uzc)
        # Registry name starts with API name
        if uzc_lower.startswith(api_lower):
            rest = uzc_lower[len(api_lower):]
            if rest == "" or rest[0] in "-,/ ":
                if len(uzc) > best_len:
                    best_match = uzc
                    best_len = len(uzc)
    return best_match


def filter_grossstaedte(stations):
    """Filter to cities in our UMWELTZONE registry only."""
    # Build mapping: API city name -> normalized registry name
    city_map = {}  # api_name -> registry_name
    for s in stations.values():
        c = s.get("city","").strip()
        if not c or c in city_map:
            continue
        norm = _normalize_city(c)
        if norm:
            city_map[c] = norm

    print(f"  Mapped {len(city_map)} API city names to "
          f"{len(set(city_map.values()))} registry cities")

    # Filter stations and collect one coordinate per REGISTRY city
    filtered = {}
    for sid, info in stations.items():
        c = info.get("city","").strip()
        if c in city_map:
            # Normalize the city name in the station info
            info = dict(info)
            info["city_original"] = c
            info["city"] = city_map[c]
            filtered[sid] = info

            reg_city = city_map[c]
            if reg_city not in CITY_COORDS and info.get("lat") and info.get("lon"):
                CITY_COORDS[reg_city] = (info["lat"], info["lon"])

    n_registry = len(set(city_map.values()))
    print(f"  {len(filtered)} stations across {n_registry} cities")
    return filtered

def fetch_annual_no2(station_map):
    if ANNUAL_CACHE.exists():
        print(f"  Loading cached annual data from {ANNUAL_CACHE}")
        df = pd.read_csv(ANNUAL_CACHE)
        n = df["city"].nunique()
        print(f"  {len(df)} rows, {n} cities")
        if n >= 10: return df
        print(f"  Too few cities, re-fetching...")
        ANNUAL_CACHE.unlink()

    print("\n  Fetching NO2 annual balances 2000-2024...")
    target = set(station_map.keys())
    records = []
    for year in range(2000,2025):
        res = api_get("annualbalances/json",
                      params={"component":5,"year":year,"lang":"en"})
        data = res.get("data",[])
        if not isinstance(data,list): continue
        yc = 0
        for row in data:
            if not isinstance(row,(list,tuple)) or len(row)<2: continue
            rid = str(row[0]).strip()
            if rid not in target: continue
            try: val = float(row[1])
            except: continue
            info = station_map[rid]
            records.append({"station_id":rid,"city":info["city"],
                            "year":year,"no2":val})
            yc += 1
        if year%5==0 or year==2024:
            print(f"    {year}: {yc} stations, {len(records)} total")

    df = pd.DataFrame(records)
    if df.empty: return df
    annual = df.groupby(["city","year"])["no2"].mean().reset_index()
    annual["uz_start"] = annual["city"].map(_uz_start)
    annual["has_uz"] = annual["uz_start"].notna()
    annual["uz_active"] = annual.apply(
        lambda r: _uz_active(r["city"], r["year"]), axis=1)
    annual["diesel_ban_active"] = annual.apply(
        lambda r: _ban_active(r["city"],r["year"]), axis=1)
    annual["group"] = annual["uz_start"].apply(_classify)

    print(f"  {len(annual)} city-years, {annual['city'].nunique()} cities")
    annual.to_csv(ANNUAL_CACHE, index=False)
    return annual

# ======================================================================
#  PLOTS
# ======================================================================
def plot_raw(df):
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(18,7))
    for city in sorted(df["city"].unique()):
        sub = df[df["city"]==city].sort_values("year")
        if len(sub)<5: continue
        g = sub["group"].iloc[0]
        ax1.plot(sub["year"],sub["no2"],color=GROUP_COLORS.get(g,"grey"),
                 linewidth=0.7,alpha=0.4)
    ax1.axhline(40,color='crimson',linestyle=':',alpha=0.3)
    patches = [mpatches.Patch(color=c,label=g) for g,c in GROUP_COLORS.items()]
    ax1.legend(handles=patches,fontsize=8,loc='upper right')
    ax1.set_title(f"Individual Cities (n={df['city'].nunique()})",fontweight='bold')
    ax1.set_ylabel("Annual NO2 (ug/m3)"); ax1.set_xlabel("Year")

    gavg = df.groupby(["group","year"])["no2"].mean().reset_index()
    for g,c in GROUP_COLORS.items():
        sub = gavg[gavg["group"]==g]
        if sub.empty: continue
        nc = df[df["group"]==g]["city"].nunique()
        ax2.plot(sub["year"],sub["no2"],color=c,linewidth=3,
                 marker='o',markersize=4,label=f"{g} (n={nc})")
    ax2.axhline(40,color='crimson',linestyle=':',alpha=0.3)
    ax2.legend(fontsize=8); ax2.set_title("Group Averages",fontweight='bold')
    ax2.set_ylabel("Annual NO2 (ug/m3)"); ax2.set_xlabel("Year")
    for ax in [ax1,ax2]:
        ax.axvline(2008,color='green',linestyle='--',alpha=0.2)
        ax.axvline(2019,color='red',linestyle='--',alpha=0.2)
    fig.suptitle("NO2 Annual Means Across German Cities (2000-2024)\n"
                 "Grouped by Umweltzone Introduction Year",fontsize=14,fontweight='bold')
    fig.tight_layout()
    fig.savefig("19_annual_all_cities.png",dpi=200,bbox_inches='tight')
    print(f"  Saved: 19_annual_all_cities.png"); plt.close(fig)

def plot_weather_corrected(df):
    hw = df.dropna(subset=["temp_annual","wind_annual"])
    if len(hw)<30: print("  Not enough weather data."); return df
    y = hw["no2"].values
    X = sm.add_constant(hw[["temp_annual","wind_annual"]].values)
    m = sm.OLS(y,X).fit()
    print(f"  Weather correction R2={m.rsquared:.3f}, "
          f"temp={m.params[1]:.2f}, wind={m.params[2]:.2f}")
    hw = hw.copy(); hw["no2_corr"] = m.resid

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,7))
    for ax,col,ttl in [(ax1,"no2","Raw"),(ax2,"no2_corr","Weather-Corrected")]:
        avg = hw.groupby(["group","year"])[col].mean().reset_index()
        for g,c in GROUP_COLORS.items():
            sub = avg[avg["group"]==g]
            if sub.empty: continue
            ax.plot(sub["year"],sub[col],color=c,linewidth=2.5,
                    marker='o',markersize=3,label=g)
        if col=="no2": ax.axhline(40,color='crimson',linestyle=':',alpha=0.3)
        else: ax.axhline(0,color='black',linewidth=0.5)
        ax.legend(fontsize=8); ax.set_title(ttl,fontweight='bold')
        ax.set_ylabel("ug/m3"); ax.set_xlabel("Year")
        for yr in [2008,2019]: ax.axvline(yr,color='grey',linestyle='--',alpha=0.2)
    fig.suptitle("Annual NO2: Raw vs Weather-Corrected",fontsize=14,fontweight='bold')
    fig.tight_layout()
    fig.savefig("20_annual_weather_corrected.png",dpi=200,bbox_inches='tight')
    print(f"  Saved: 20_annual_weather_corrected.png"); plt.close(fig)
    return hw

def chow_f(y,X,bi):
    n,k = len(y),X.shape[1]
    if bi<=k or (n-bi)<=k: return np.nan
    rf = np.sum(sm.OLS(y,X).fit().resid**2)
    r1 = np.sum(sm.OLS(y[:bi],X[:bi]).fit().resid**2)
    r2 = np.sum(sm.OLS(y[bi:],X[bi:]).fit().resid**2)
    return ((rf-r1-r2)/k)/((r1+r2)/(n-2*k))

def plot_breaks(df, max_cities=16):
    eligible = df.groupby("city")["year"].count()
    eligible = eligible[eligible>=12].index.tolist()
    sel = []
    for g in GROUP_COLORS:
        gc = [c for c in eligible if df[df["city"]==c]["group"].iloc[0]==g]
        sel.extend(gc[:max_cities//4])
    sel = sel[:max_cities]
    nc = 4; nr = int(np.ceil(len(sel)/nc))
    fig,axes = plt.subplots(nr,nc,figsize=(20,4*nr),squeeze=False)
    for i,city in enumerate(sel):
        ax = axes[i//nc][i%nc]
        sub = df[df["city"]==city].sort_values("year")
        yrs,vals = sub["year"].values, sub["no2"].values
        g = sub["group"].iloc[0]; col = GROUP_COLORS.get(g,"grey")
        t = yrs-yrs[0]; X = np.column_stack([np.ones(len(t)),t])
        trim = max(4,len(yrs)//5)
        fs,sy = [],[]
        for j in range(trim,len(yrs)-trim):
            F = chow_f(vals,X,j)
            if not np.isnan(F): fs.append(F); sy.append(yrs[j])
        if fs:
            ax.plot(sy,fs,color=col,linewidth=2)
            ax.fill_between(sy,fs,alpha=0.15,color=col)
            cv = stats.f.ppf(0.95,2,len(yrs)-4)
            ax.axhline(cv,color='crimson',linestyle=':',alpha=0.4)
            bi = np.argmax(fs)
            ax.scatter([sy[bi]],[fs[bi]],color='red',s=50,zorder=5)
            uz = _uz_start(city)
            if uz and uz in sy: ax.axvline(uz,color='green',linestyle='--',alpha=0.5)
            ax.set_title(f"{city}\npeak={sy[bi]}",fontsize=9,fontweight='bold',color=col)
        ax.tick_params(labelsize=7)
    for i in range(len(sel),nr*nc): axes[i//nc][i%nc].axis('off')
    fig.suptitle("Sup-F Break Scans: NO2 Annual Means\n"
                 "Green = UZ year. Red dot = strongest break.",
                 fontsize=13,fontweight='bold')
    fig.tight_layout()
    fig.savefig("21_annual_break_scan.png",dpi=200,bbox_inches='tight')
    print(f"  Saved: 21_annual_break_scan.png"); plt.close(fig)

def staggered_did(df):
    print("\n" + "="*70)
    print("  STAGGERED DiD")
    print("="*70)
    hw = df.dropna(subset=["temp_annual","wind_annual"]).copy()
    if len(hw)<50: print("  Not enough data."); return

    cities = sorted(hw["city"].unique())
    ref = cities[0]
    for c in cities[1:]: hw[f"fe_{c}"] = (hw["city"]==c).astype(int)
    fe = [f"fe_{c}" for c in cities[1:]]
    hw["trend"] = hw["year"]-2000
    hw["uz_int"] = hw["uz_active"].astype(int)
    hw["ban_int"] = hw["diesel_ban_active"].astype(int)

    pred = ["trend","temp_annual","wind_annual","uz_int","ban_int"]+fe
    w = hw.dropna(subset=["no2"]+pred)
    y = w["no2"].values
    X = sm.add_constant(w[pred].values.astype(float))
    m = sm.OLS(y,X).fit(cov_type='HC3')

    print(f"  N={len(y)}, Cities={len(cities)}, R2={m.rsquared:.3f}")
    print(f"  Trend:       {m.params[1]:+.3f}/yr (p={m.pvalues[1]:.4f})")
    print(f"  Temperature: {m.params[2]:+.3f} (p={m.pvalues[2]:.4f})")
    print(f"  Wind:        {m.params[3]:+.3f} (p={m.pvalues[3]:.4f})")
    print(f"  *** Umweltzone: {m.params[4]:+.3f} (p={m.pvalues[4]:.4f}) "
          f"{'SIG' if m.pvalues[4]<0.05 else 'ns'}")
    print(f"  *** Diesel ban: {m.params[5]:+.3f} (p={m.pvalues[5]:.4f}) "
          f"{'SIG' if m.pvalues[5]<0.05 else 'ns'}")

    # Plot
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(16,7))
    labels = ["Trend","Temperature","Wind speed","Umweltzone","Diesel ban"]
    coefs = [m.params[i+1] for i in range(5)]
    ses = [m.bse[i+1] for i in range(5)]
    pvals = [m.pvalues[i+1] for i in range(5)]
    colors = []
    for i,p in enumerate(pvals):
        if labels[i]=="Umweltzone": colors.append("#2ecc71" if p<0.05 else "#ccc")
        elif labels[i]=="Diesel ban": colors.append("#e74c3c" if p<0.05 else "#ccc")
        elif p<0.05: colors.append("#3498db")
        else: colors.append("#ccc")
    ax1.barh(range(5),coefs,xerr=[1.96*s for s in ses],color=colors,
             edgecolor='white',capsize=3)
    ax1.set_yticks(range(5))
    ax1.set_yticklabels([f"{l} {'*' if p<0.05 else ''}" for l,p in zip(labels,pvals)])
    ax1.axvline(0,color='black',linewidth=0.8)
    ax1.set_xlabel("Coefficient (ug/m3)")
    ax1.set_title(f"Key Coefficients\nR2={m.rsquared:.3f}, {len(cities)} cities",
                  fontweight='bold')
    ax1.invert_yaxis()

    yrs = np.arange(2000,2025)
    mt,mw = hw["temp_annual"].mean(), hw["wind_annual"].mean()
    base = m.params[0]+m.params[1]*(yrs-2000)+m.params[2]*mt+m.params[3]*mw
    wuz = base.copy(); wuz[yrs>=2008] += m.params[4]
    wboth = wuz.copy(); wboth[yrs>=2019] += m.params[5]
    ax2.plot(yrs,base,color='#95a5a6',linewidth=2,linestyle='--',label='No interventions')
    ax2.plot(yrs,wuz,color='#2ecc71',linewidth=2,
             label=f'+ Umweltzone ({m.params[4]:+.1f})')
    ax2.plot(yrs,wboth,color='#e74c3c',linewidth=2,
             label=f'+ Diesel ban ({m.params[5]:+.1f})')
    ax2.axhline(40,color='crimson',linestyle=':',alpha=0.3)
    ax2.fill_between(yrs,base,wboth,alpha=0.1,color='green')
    ax2.legend(fontsize=9)
    ax2.set_title("Predicted Trajectory (typical city, mean weather)",fontweight='bold')
    ax2.set_ylabel("Predicted NO2 (ug/m3)"); ax2.set_xlabel("Year")
    fig.suptitle("Staggered DiD: Umweltzone + Diesel Ban\n"
                 "(Annual data, city FE, HC3 robust SEs)",fontsize=13,fontweight='bold')
    fig.tight_layout()
    fig.savefig("22_staggered_did.png",dpi=200,bbox_inches='tight')
    print(f"  Saved: 22_staggered_did.png"); plt.close(fig)


# ======================================================================
#  PLOT 23: REGRESSION DISCONTINUITY AROUND EU LIMIT
# ======================================================================

def regression_discontinuity(df):
    """
    RD design: cities just above the EU NO2 limit (40 ug/m3) were
    pressured/sued into implementing Umweltzonen. Cities just below
    were not. Compare NO2 *changes* for cities near the threshold.

    Running variable: pre-treatment NO2 (max annual mean before 2008)
    Cutoff: 40 ug/m3
    Treatment: got an Umweltzone
    Outcome: change in NO2 from pre to post period
    """
    print("\n" + "="*70)
    print("  REGRESSION DISCONTINUITY AROUND EU LIMIT")
    print("="*70)

    # Pre-period: average NO2 before first Umweltzonen (2005-2007)
    pre = df[df["year"].between(2005, 2007)].groupby("city").agg(
        pre_no2=("no2","mean"),
        has_uz=("has_uz","first"),
        uz_start=("uz_start","first"),
    ).reset_index()

    # Post-period: average NO2 in recent years (2020-2024)
    post = df[df["year"].between(2020, 2024)].groupby("city")["no2"].mean()
    post.name = "post_no2"

    rd = pre.merge(post, on="city", how="inner")
    rd["change"] = rd["post_no2"] - rd["pre_no2"]
    rd["pct_change"] = rd["change"] / rd["pre_no2"] * 100
    rd["distance_from_limit"] = rd["pre_no2"] - 40  # running variable
    rd["got_uz"] = rd["has_uz"].astype(int)

    if len(rd) < 10:
        print("  Not enough cities for RD.")
        return

    print(f"  Cities: {len(rd)}")
    print(f"  Cities above 40: {(rd['pre_no2'] > 40).sum()}")
    print(f"  Cities below 40: {(rd['pre_no2'] <= 40).sum()}")
    print(f"  Cities with UZ: {rd['got_uz'].sum()}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # --- Panel 1: First stage: pre-NO2 predicts UZ adoption ---
    ax = axes[0]
    for _, row in rd.iterrows():
        color = "#2ecc71" if row["got_uz"] else "#95a5a6"
        marker = 'o' if row["got_uz"] else 'x'
        ax.scatter(row["pre_no2"], row["got_uz"], color=color,
                   s=40, marker=marker, alpha=0.7)

    ax.axvline(40, color='crimson', linestyle='--', linewidth=2,
               label="EU limit (40 ug/m3)")

    # Logistic-style curve for visual
    from scipy.special import expit
    x_range = np.linspace(rd["pre_no2"].min() - 2, rd["pre_no2"].max() + 2, 200)

    # Fit simple logistic regression
    try:
        from statsmodels.discrete.discrete_model import Logit
        y_first = rd["got_uz"].values
        X_first = sm.add_constant(rd["pre_no2"].values)
        logit_model = Logit(y_first, X_first).fit(disp=0)
        X_pred = sm.add_constant(x_range)
        prob_pred = logit_model.predict(X_pred)
        ax.plot(x_range, prob_pred, color='navy', linewidth=2,
                label="P(Umweltzone | pre-NO2)")
        print(f"\n  First stage (logistic):")
        print(f"    Pre-NO2 coefficient: {logit_model.params[1]:.3f} "
              f"(p={logit_model.pvalues[1]:.4f})")
    except Exception as e:
        print(f"  Logistic fit failed: {e}")

    ax.set_xlabel("Pre-treatment NO2 (2005-2007 avg, ug/m3)")
    ax.set_ylabel("Got Umweltzone (0/1)")
    ax.set_title("First Stage: Pre-NO2 Predicts\nUmweltzone Adoption",
                 fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(-0.1, 1.1)

    # --- Panel 2: RD plot - change in NO2 vs distance from cutoff ---
    ax = axes[1]
    bandwidth = 15  # ug/m3 around cutoff
    rd_bw = rd[rd["distance_from_limit"].between(-bandwidth, bandwidth)]

    for _, row in rd_bw.iterrows():
        color = "#2ecc71" if row["got_uz"] else "#95a5a6"
        ax.scatter(row["distance_from_limit"], row["change"],
                   color=color, s=50, alpha=0.7, edgecolors='white')

    # Fit separate local linear regressions on each side
    below = rd_bw[rd_bw["distance_from_limit"] <= 0]
    above = rd_bw[rd_bw["distance_from_limit"] > 0]

    for sub, color, label in [(below, "#95a5a6", "Below limit"),
                               (above, "#2ecc71", "Above limit")]:
        if len(sub) >= 3:
            X_loc = sm.add_constant(sub["distance_from_limit"].values)
            y_loc = sub["change"].values
            m_loc = sm.OLS(y_loc, X_loc).fit()
            x_pred = np.linspace(sub["distance_from_limit"].min(),
                                  sub["distance_from_limit"].max(), 50)
            y_pred = m_loc.predict(sm.add_constant(x_pred))
            ax.plot(x_pred, y_pred, color=color, linewidth=2.5)

    ax.axvline(0, color='crimson', linestyle='--', linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Distance from EU limit (ug/m3)")
    ax.set_ylabel("NO2 change: 2005-07 to 2020-24 (ug/m3)")
    ax.set_title(f"RD Plot: NO2 Change vs Distance\nfrom 40 ug/m3 Threshold",
                 fontweight='bold')

    # --- Panel 3: Fuzzy RD / IV estimate ---
    ax = axes[2]

    # IV: instrument = 1(pre_NO2 > 40), endogenous = got_uz, outcome = change
    rd["above_limit"] = (rd["pre_no2"] > 40).astype(int)

    # Reduced form: above_limit -> change
    X_rf = sm.add_constant(rd[["above_limit", "pre_no2"]].values)
    m_rf = sm.OLS(rd["change"].values, X_rf).fit(cov_type='HC3')

    # First stage: above_limit -> got_uz
    m_fs = sm.OLS(rd["got_uz"].values, X_rf).fit(cov_type='HC3')

    # Wald / IV estimate
    if m_fs.params[1] != 0:
        iv_estimate = m_rf.params[1] / m_fs.params[1]
        # Delta method SE (approximate)
        iv_se = abs(m_rf.bse[1] / m_fs.params[1])
    else:
        iv_estimate = np.nan
        iv_se = np.nan

    print(f"\n  Fuzzy RD / IV Estimates:")
    print(f"    First stage (above_limit -> got_uz): "
          f"{m_fs.params[1]:+.3f} (p={m_fs.pvalues[1]:.4f})")
    print(f"    Reduced form (above_limit -> NO2 change): "
          f"{m_rf.params[1]:+.3f} (p={m_rf.pvalues[1]:.4f})")
    print(f"    IV estimate (Wald): {iv_estimate:+.2f} ug/m3 "
          f"(SE={iv_se:.2f})")

    # Quick OLS estimate for comparison (same as staggered_did but simpler)
    ols_coef = 0.0
    ols_se = 0.0
    try:
        rd["change_uz"] = rd["change"] * rd["got_uz"]
        X_ols = sm.add_constant(rd[["got_uz", "pre_no2"]].values)
        m_ols = sm.OLS(rd["change"].values, X_ols).fit(cov_type='HC3')
        ols_coef = m_ols.params[1]
        ols_se = m_ols.bse[1]
        print(f"    OLS (UZ -> change, controlling pre-NO2): "
              f"{ols_coef:+.2f} (p={m_ols.pvalues[1]:.4f})")
    except:
        pass

    # Visualize the three estimates
    estimates = [
        ("OLS\n(UZ -> change, control pre-NO2)",
         ols_coef, ols_se, "#ccc"),
        ("Reduced Form\n(above limit -> change)",
         m_rf.params[1], m_rf.bse[1], "#3498db"),
        ("IV / Fuzzy RD\n(Wald estimate)",
         iv_estimate, iv_se, "#2ecc71"),
    ]

    y_pos = range(len(estimates))
    for i, (label, coef, se, color) in enumerate(estimates):
        ax.barh(i, coef, color=color, edgecolor='white', height=0.6)
        if se > 0:
            ax.errorbar(coef, i, xerr=1.96*se, color='black',
                        capsize=4, linewidth=1.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([e[0] for e in estimates], fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Umweltzone Effect on NO2 (ug/m3)")
    ax.set_title("Comparing Estimates:\nOLS vs Reduced Form vs IV",
                 fontweight='bold')
    ax.invert_yaxis()

    fig.suptitle("Regression Discontinuity: EU Limit as Instrument\n"
                 "Cities above 40 ug/m3 were pressured into Umweltzonen",
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig("23_regression_discontinuity.png", dpi=200, bbox_inches='tight')
    print(f"\n  Saved: 23_regression_discontinuity.png")
    plt.close(fig)

    return iv_estimate, iv_se

# ======================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("  GERMANY-WIDE ANNUAL ANALYSIS: UMWELTZONE STAGGERED DiD")
    print("="*70)

    stations = discover_stations()
    if not stations: raise SystemExit("No stations found.")
    filtered = filter_grossstaedte(stations)
    df = fetch_annual_no2(filtered)
    if df.empty: raise SystemExit("No annual data.")

    # Weather - only for cities actually in the annual panel
    if "temp_annual" not in df.columns:
        from dwd_weather import fetch_dwd_weather

        # Get unique cities from the annual data (not from CITY_COORDS which
        # includes station sub-names like "Duisburg-Bruckhausen")
        panel_cities = sorted(df["city"].unique())
        print(f"\n  Fetching weather for {len(panel_cities)} panel cities "
              f"(2010-2025, Bright Sky)")

        # Build one coordinate per panel city
        city_weather_coords = {}
        for city in panel_cities:
            if city in CITY_COORDS:
                city_weather_coords[city] = CITY_COORDS[city]
            else:
                # Try partial match from CITY_COORDS keys
                for cc_name, coords in CITY_COORDS.items():
                    if city.lower() in cc_name.lower() or cc_name.lower() in city.lower():
                        city_weather_coords[city] = coords
                        break

        print(f"  Found coordinates for {len(city_weather_coords)} of "
              f"{len(panel_cities)} cities")

        recs = []
        for i, (city, (lat, lon)) in enumerate(sorted(city_weather_coords.items())):
            print(f"  [{i+1}/{len(city_weather_coords)}] Weather: {city}...")
            try:
                w = fetch_dwd_weather(lat=lat, lon=lon,
                                      start_date="2010-01-01",
                                      chunk_months=12)
                if not w.empty:
                    w["year"] = w["date"].dt.year
                    yw = w.groupby("year").agg(
                        temp_annual=("temp_mean","mean"),
                        wind_annual=("wind_speed","mean")).reset_index()
                    yw["city"] = city
                    recs.append(yw)
            except Exception as e:
                print(f"    Failed: {e}")

        if recs:
            wdf = pd.concat(recs, ignore_index=True)
            df = df.merge(wdf, on=["city","year"], how="left")
            df.to_csv(ANNUAL_CACHE, index=False)
            print(f"  Weather merged for {wdf['city'].nunique()} cities")

    plot_raw(df)
    plot_weather_corrected(df)
    plot_breaks(df)
    staggered_did(df)
    regression_discontinuity(df)

    print("\n" + "="*70)
    print(f"  Cities: {df['city'].nunique()}")
    for g in GROUP_COLORS:
        n = df[df['group']==g]['city'].nunique()
        print(f"    {g}: {n}")
    print("  Figures: 19-23")
    print("="*70)