#!/usr/bin/env python
# coding: utf-8
"""
Aggregate MODIS MCD64A1 per-buffer CSV outputs into monthly master tables (one CSV per year),
including:
- pixel-weighted median Burn DOY (approx) using tile-level medians weighted by Burned_Pixels
- pixel-weighted median uncertainty (approx)
- min/max + ranges
- land-only denominator and % land burned

Notes:
- Exact pixel-median DOY would require per-pixel DOY distribution (or DOY histograms). This script uses a
  principled approximation: weighted median of tile-level medians by burned pixel counts.
"""

import os, glob, re, math, argparse
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime, timedelta

# Shapely compatibility
try:
    from shapely.validation import make_valid
except Exception:
    def make_valid(g):
        try:
            return g.buffer(0)
        except Exception:
            return g

# Prefer shapely 2.x union_all if available
try:
    from shapely import union_all as shapely_union_all  # shapely>=2
except Exception:
    shapely_union_all = None

try:
    from shapely.ops import unary_union  # shapely<=1.x path (may still exist)
except Exception:
    unary_union = None

try:
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
except Exception:
    Polygon = MultiPolygon = GeometryCollection = None


# ------------------
# Defaults (edit if you want hard defaults; CLI can override output-root/outdir/years/months)
# ------------------
OUTPUT_ROOT_DEFAULT = r"A:\Project BFA\output_csvs"

LAND_PATH_DEFAULT  = r"A:\Project BFA\Shapefiles\Land_cover\ne_10m_land.shp"
LAKES_PATH_DEFAULT = r"A:\Project BFA\Shapefiles\Land_cover\ne_10m_lakes.shp"

REGION_FILES_DEFAULT = {
    'Europe':       (r"A:\Project BFA\Shapefiles\buffers_per_region\buffers_Europe_3035.shp", 3035),
    'Siberia':      (r"A:\Project BFA\Shapefiles\buffers_per_region\buffers_Siberia_3576.shp", 3576),
    'NorthAmerica': (r"A:\Project BFA\Shapefiles\buffers_per_region\buffers_NorthAmerica_5070.shp", 5070),
}

SKIP_NAME_PATTERNS = re.compile(r"(summary|summaries|land_area|chronology|master)", re.IGNORECASE)

SCALE_MULT = {
    'original':    1.00,
    'buffer_10pct':1.10,
    'buffer_20pct':1.20,
    'buffer_50pct':1.50,
    'buffer_100pct':2.00
}
def pretty_scale(code: str) -> str:
    return f"{code} (×{SCALE_MULT.get(code,1.0):.2f})"

USE_COLS = {
    "Region":"str","AreaID":"str","Buffer_Size":"float64","Buffer_Scale":"str",
    "Year":"object","Month":"object","Tile":"str",
    "Burned_Pixels":"float64","Burned_Area_km2":"float64",
    "Burn_Date_Min":"float64","Burn_Date_Max":"float64",
    "Uncertainty_Min":"float64","Uncertainty_Max":"float64",
    "Burn_Date_Median_DOY":"float64","Uncertainty_Median":"float64",
}


# ------------------
# CLI
# ------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Aggregate per-buffer fire CSVs into yearly monthly master tables (+ land % burned)."
    )
    p.add_argument("--output-root", default=OUTPUT_ROOT_DEFAULT,
                   help="Root containing per-buffer CSV outputs.")
    p.add_argument("--outdir", default=None,
                   help="Directory to write summaries. Default: <output-root>/summaries")
    p.add_argument("--years", nargs="+", type=int, default=None,
                   help="Filter years, e.g. 2020 2021 2022")
    p.add_argument("--months", nargs="+", type=int, default=None,
                   help="Filter months (integers), e.g. 3 4 5 6 7 8 9 10")

    p.add_argument("--land-path", default=LAND_PATH_DEFAULT,
                   help="Natural Earth land shapefile path.")
    p.add_argument("--lakes-path", default=LAKES_PATH_DEFAULT,
                   help="Natural Earth lakes shapefile path.")

    # If you ever want to restrict which regions are used for land area calc:
    p.add_argument("--regions", nargs="+", default=None,
                   help="Regions to include for land-area denominator, e.g. Europe Siberia NorthAmerica. Default: all.")

    # Optional knobs for geometry robustness/speed
    p.add_argument("--union-grid-size", type=float, default=100.0,
                   help="grid_size used in shapely union_all (when available). Larger can reduce topology errors.")
    p.add_argument("--simplify-m", type=float, default=0.0,
                   help="Optional simplify tolerance in *meters* in the target CRS before union (0 = off).")

    return p.parse_args()


# ------------------
# Helpers
# ------------------
def read_csv_robust(path: str) -> pd.DataFrame | None:
    for enc in ("utf-8","utf-8-sig","latin-1"):
        try:
            df = pd.read_csv(
                path,
                dtype=USE_COLS,
                usecols=lambda c: c in USE_COLS,
                encoding=enc,
                low_memory=False
            )
            df["__src"] = path
            return df
        except Exception:
            continue
    print(f"⚠️  Skipping unreadable CSV: {path}")
    return None

def coerce_month(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x,(int,np.integer,float)) and not np.isnan(x):
        return int(x)
    s = str(x).strip()
    if "." in s:
        try:
            return int(s.split(".")[0])
        except Exception:
            return np.nan
    month_map = {m.lower():i for i,m in enumerate(
        ["January","February","March","April","May","June","July","August","September","October","November","December"],1)}
    if s.lower() in month_map:
        return month_map[s.lower()]
    try:
        return int(s)
    except Exception:
        return np.nan

def doy_to_date(year, doy):
    if pd.isna(doy):
        return None
    try:
        return (datetime(int(year),1,1) + timedelta(days=int(float(doy))-1)).strftime("%Y-%m-%d")
    except Exception:
        return None

def month_name_from_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d").strftime("%B") if isinstance(s,str) else None
    except Exception:
        return None

def pick_name_field(gdf: gpd.GeoDataFrame):
    for c in ['AreaID','areaid','Name','name','Site','site','Label','label']:
        if c in gdf.columns:
            return c
    return None

def weighted_median(values, weights):
    """Weighted median of values with weights. Returns NaN if no valid."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v = v[mask]
    w = w[mask]
    if v.size == 0:
        return np.nan
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    cum = np.cumsum(w)
    cutoff = 0.5 * w.sum()
    return float(v[np.searchsorted(cum, cutoff)])

def _polygon_only(geom):
    """Keep only polygonal components; drop lines/points/empties."""
    if geom is None:
        return None
    try:
        if geom.is_empty:
            return None
        t = geom.geom_type
        if t in ("Polygon", "MultiPolygon"):
            return geom
        if t == "GeometryCollection":
            polys = []
            for g in getattr(geom, "geoms", []):
                if g is not None and (not g.is_empty) and g.geom_type in ("Polygon", "MultiPolygon"):
                    polys.append(g)
            if not polys:
                return None
            if len(polys) == 1:
                return polys[0]
            # dissolve the polygon pieces later; for now wrap as a MultiPolygon-ish collection
            try:
                return MultiPolygon([p for p in polys if getattr(p, "geom_type", "") == "Polygon"])
            except Exception:
                # fallback: return first polygonal geom
                return polys[0]
        return None
    except Exception:
        return None

def _clean_geom_series_to_polys(gser: gpd.GeoSeries, simplify_tol: float = 0.0) -> list:
    """
    Defensive cleaning for messy Natural Earth geometries:
    make_valid -> polygon_only -> buffer(0) -> drop empties -> optional simplify
    Returns a Python list of geometries safe(r) to union.
    """
    if gser is None or len(gser) == 0:
        return []
    out = []
    for g in gser:
        if g is None:
            continue
        try:
            g = make_valid(g)
            g = _polygon_only(g)
            if g is None:
                continue
            # buffer(0) as additional repair
            try:
                g = g.buffer(0)
            except Exception:
                pass
            if g is None or getattr(g, "is_empty", True):
                continue
            if simplify_tol and simplify_tol > 0:
                try:
                    g = g.simplify(simplify_tol)
                except Exception:
                    pass
            if g is None or getattr(g, "is_empty", True):
                continue
            out.append(g)
        except Exception:
            continue
    return out


# ------------------
# Land-only denominator machinery (PATCHED)
# ------------------
_land_union_cache = {}       # crs_str -> land_only_geom
_region_buffer_cache = {}    # region -> (gdf_equal_area, name_col)
_region_areaid_index = {}    # region -> {AreaID-or-BufferX: row}

def land_union_in_crs(land_path: str, lakes_path: str, epsg_or_crs,
                      union_grid_size: float = 100.0, simplify_m: float = 0.0):
    """
    Build 'land-only' geometry in target CRS, robust to invalid geometries.

    Key change vs your previous version:
    - CLEAN geometries before union (make_valid + polygon_only + buffer(0))
    - Use shapely.union_all with grid_size when available (helps topology)
    - Retry union with fallback strategies instead of crashing
    """
    key = (str(epsg_or_crs), float(union_grid_size), float(simplify_m))
    if key in _land_union_cache:
        return _land_union_cache[key]

    land  = gpd.read_file(land_path)
    lakes = gpd.read_file(lakes_path)

    land  = land.to_crs(epsg_or_crs)
    lakes = lakes.to_crs(epsg_or_crs)

    land_geoms  = _clean_geom_series_to_polys(land.geometry,  simplify_tol=simplify_m)
    lakes_geoms = _clean_geom_series_to_polys(lakes.geometry, simplify_tol=simplify_m)

    # --- UNION (robust) ---
    def _do_union(geoms, grid_size):
        if not geoms:
            return None
        if shapely_union_all is not None:
            return shapely_union_all(geoms, grid_size=grid_size)
        # shapely<2 fallback
        if unary_union is not None:
            return unary_union(geoms)
        # geopandas fallback (may still call shapely union underneath)
        try:
            return gpd.GeoSeries(geoms).unary_union
        except Exception:
            return None

    # try union with requested grid_size, then with None/0, then with a larger grid_size
    land_u = None
    lakes_u = None
    for gs in (union_grid_size, 0.0, max(500.0, union_grid_size)):
        try:
            land_u = _do_union(land_geoms, gs)
            lakes_u = _do_union(lakes_geoms, gs)
            break
        except Exception:
            land_u = None
            lakes_u = None

    # final fallbacks if union still fails
    if land_u is None:
        # last resort: take first geometry (better than crashing)
        land_u = land_geoms[0] if land_geoms else None
    if lakes_u is None:
        lakes_u = lakes_geoms[0] if lakes_geoms else None

    land_u  = make_valid(land_u)  if land_u  is not None else None
    lakes_u = make_valid(lakes_u) if lakes_u is not None else None

    if land_u is None:
        _land_union_cache[key] = None
        return None

    try:
        land_only = land_u.difference(lakes_u) if lakes_u is not None else land_u
    except Exception:
        land_only = land_u

    land_only = make_valid(land_only)
    _land_union_cache[key] = land_only
    return land_only

def load_region_buffers(region: str, region_files: dict):
    if region in _region_buffer_cache:
        return _region_buffer_cache[region], _region_areaid_index[region]

    shp, epsg = region_files[region]
    gdf = gpd.read_file(shp)
    if gdf.crs is None:
        gdf.set_crs(epsg, inplace=True)
    else:
        gdf = gdf.to_crs(epsg)

    name_col = pick_name_field(gdf)

    idx_map = {}
    for idx, row in gdf.iterrows():
        if name_col and pd.notna(row[name_col]):
            idx_map[str(row[name_col])] = row
        idx_map[f"Buffer{int(idx)}"] = row

    _region_buffer_cache[region] = (gdf, name_col)
    _region_areaid_index[region] = idx_map
    return (gdf, name_col), idx_map

def scale_buffer_geom(base_row, scale_code: str):
    factor = SCALE_MULT.get(scale_code, 1.0)
    base_area = float(base_row.get("Buffer_km2", np.nan))
    if not np.isfinite(base_area):
        return make_valid(base_row.geometry)

    r = math.sqrt(base_area * 1e6 / math.pi)   # meters
    grow = r * (factor - 1.0)

    geom0 = make_valid(base_row.geometry)
    try:
        grown = geom0.buffer(grow)
        if scale_code != "original":
            grown = make_valid(grown.union(geom0))
        return grown
    except Exception:
        return geom0

_land_area_cache = {}  # (region, areaid, scale_code) -> land_km2

def land_area_km2_for(region: str, areaid: str, scale_code: str,
                      region_files: dict, land_path: str, lakes_path: str,
                      union_grid_size: float = 100.0, simplify_m: float = 0.0):
    key = (region, areaid, scale_code, float(union_grid_size), float(simplify_m))
    if key in _land_area_cache:
        return _land_area_cache[key]

    (gdf, name_col), idx_map = load_region_buffers(region, region_files)

    base_row = None
    if areaid in idx_map:
        base_row = idx_map[areaid]
    else:
        # loose match
        a = str(areaid).strip().lower()
        for k, v in idx_map.items():
            if str(k).strip().lower() == a:
                base_row = v
                break

    if base_row is None:
        _land_area_cache[key] = np.nan
        return np.nan

    scaled_geom = scale_buffer_geom(base_row, scale_code)

    land_only = land_union_in_crs(
        land_path, lakes_path, gdf.crs,
        union_grid_size=union_grid_size,
        simplify_m=simplify_m
    )
    if land_only is None:
        _land_area_cache[key] = np.nan
        return np.nan

    try:
        inter = make_valid(scaled_geom).intersection(make_valid(land_only))
        area_km2 = inter.area / 1e6
    except Exception:
        area_km2 = np.nan

    _land_area_cache[key] = area_km2
    return area_km2


# ------------------
# Main aggregation
# ------------------
def main():
    args = parse_args()

    output_root = args.output_root
    outdir = args.outdir or os.path.join(output_root, "summaries")
    os.makedirs(outdir, exist_ok=True)

    years_filter  = args.years
    months_filter = args.months

    land_path  = args.land_path
    lakes_path = args.lakes_path

    union_grid_size = float(args.union_grid_size)
    simplify_m = float(args.simplify_m)

    # Region files (you can later make CLI overrides if you want)
    region_files = REGION_FILES_DEFAULT.copy()
    if args.regions is not None:
        region_files = {k:v for k,v in region_files.items() if k in set(args.regions)}

    # 1) load per-buffer CSVs
    all_csvs = glob.glob(os.path.join(output_root, "**", "*.csv"), recursive=True)
    per_buffer_csvs = [p for p in all_csvs if not SKIP_NAME_PATTERNS.search(os.path.basename(p))]
    if not per_buffer_csvs:
        raise SystemExit(f"No per-buffer CSVs found under {output_root}")

    frames = []
    for p in per_buffer_csvs:
        df = read_csv_robust(p)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        raise SystemExit("No readable per-buffer CSVs")

    chron = pd.concat(frames, ignore_index=True)

    # 2) clean types
    chron["Year"]  = pd.to_numeric(chron["Year"], errors="coerce").astype("Int64")
    chron["Month"] = chron["Month"].map(coerce_month).astype("Int64")

    chron["Burned_Pixels"]   = pd.to_numeric(chron["Burned_Pixels"], errors="coerce").fillna(0).astype(int)
    chron["Burned_Area_km2"] = pd.to_numeric(chron["Burned_Area_km2"], errors="coerce").fillna(0.0)

    for c in ["Burn_Date_Min","Burn_Date_Max","Uncertainty_Min","Uncertainty_Max",
              "Buffer_Size","Burn_Date_Median_DOY","Uncertainty_Median"]:
        if c in chron.columns:
            chron[c] = pd.to_numeric(chron[c], errors="coerce")

    if years_filter is not None:
        chron = chron[chron["Year"].isin(years_filter)]
    if months_filter is not None:
        chron = chron[chron["Month"].isin(months_filter)]

    chron = chron.dropna(subset=["Region","AreaID","Buffer_Scale","Year","Month","Buffer_Size"])
    if chron.empty:
        raise SystemExit("After filtering/cleaning, no rows remain.")

    # 3) enhancements
    chron["Month_Name"] = chron["Month"].map({i:m for i,m in enumerate(
        ["January","February","March","April","May","June","July","August","September","October","November","December"],1)})

    chron["Scale_Pretty"] = chron["Buffer_Scale"].map(pretty_scale)
    chron["Scale_Mult"]   = chron["Buffer_Scale"].map(SCALE_MULT).fillna(1.0)
    chron["Effective_Buffer_Area_km2"] = chron["Buffer_Size"] * (chron["Scale_Mult"] ** 2)

    # Drop scaled rows whose effective area equals an original size (±0.01 km²)
    base_sizes = chron.loc[chron["Buffer_Scale"].str.lower()=="original","Buffer_Size"].dropna().unique()
    if base_sizes.size == 0:
        base_sizes = chron["Buffer_Size"].dropna().unique()
    base_sizes_set = set(np.round(base_sizes.astype(float), 6))

    def is_dup(row, tol=0.01):
        if str(row["Buffer_Scale"]).lower()=="original":
            return False
        eff = float(row["Effective_Buffer_Area_km2"])
        return any(abs(eff - bs) <= tol for bs in base_sizes_set)

    chron = chron[~chron.apply(is_dup, axis=1)]

    # Ensure row-wise medians exist (fallback midpoint if missing)
    burns = chron["Burned_Pixels"] > 0

    if "Burn_Date_Median_DOY" not in chron.columns:
        chron["Burn_Date_Median_DOY"] = np.nan
    if "Uncertainty_Median" not in chron.columns:
        chron["Uncertainty_Median"] = np.nan

    # Only fill where burns > 0 and median is NaN
    fill_bd = burns & chron["Burn_Date_Median_DOY"].isna()
    chron.loc[fill_bd, "Burn_Date_Median_DOY"] = 0.5*(chron.loc[fill_bd, "Burn_Date_Min"] + chron.loc[fill_bd, "Burn_Date_Max"])

    fill_un = burns & chron["Uncertainty_Median"].isna() & chron["Uncertainty_Min"].notna() & chron["Uncertainty_Max"].notna()
    chron.loc[fill_un, "Uncertainty_Median"] = 0.5*(chron.loc[fill_un, "Uncertainty_Min"] + chron.loc[fill_un, "Uncertainty_Max"])

    # 4) aggregate monthly sums/min/max (NO medians here)
    group_keys = ["Region","AreaID","Buffer_Scale","Scale_Pretty","Buffer_Size",
                  "Effective_Buffer_Area_km2","Year","Month","Month_Name"]

    monthly = (chron.groupby(group_keys, as_index=False)
               .agg(Burned_Area_km2=("Burned_Area_km2","sum"),
                    Burned_Pixels=("Burned_Pixels","sum"),
                    Burn_Date_Min=("Burn_Date_Min","min"),
                    Burn_Date_Max=("Burn_Date_Max","max"),
                    Uncertainty_Min=("Uncertainty_Min","min"),
                    Uncertainty_Max=("Uncertainty_Max","max")))

    # 5) pixel-weighted medians from tile rows
    #    (weighted median of tile-level medians, weights=Burned_Pixels)
    def wm_series(g):
        g = g[g["Burned_Pixels"] > 0]
        return pd.Series({
            "Burn_Date_Median_DOY": weighted_median(g["Burn_Date_Median_DOY"].values, g["Burned_Pixels"].values),
            "Uncertainty_Median":   weighted_median(g["Uncertainty_Median"].values,   g["Burned_Pixels"].values),
        })

    # avoid pandas FutureWarning by grouping only on keys (apply returns series)
    med = (chron.groupby(group_keys, as_index=False)
           .apply(wm_series, include_groups=False)
           .reset_index(drop=True))

    monthly = monthly.merge(med, on=group_keys, how="left")

    # ranges
    monthly["Burn_DOY_Range"]      = monthly["Burn_Date_Max"] - monthly["Burn_Date_Min"]
    monthly["Uncertainty_Range"]   = monthly["Uncertainty_Max"] - monthly["Uncertainty_Min"]

    # calendar strings
    monthly["Burn_Date_Min_Date"]    = [doy_to_date(y, d) for y,d in zip(monthly["Year"], monthly["Burn_Date_Min"])]
    monthly["Burn_Date_Max_Date"]    = [doy_to_date(y, d) for y,d in zip(monthly["Year"], monthly["Burn_Date_Max"])]
    monthly["Burn_Date_Median_Date"] = [doy_to_date(y, d) for y,d in zip(monthly["Year"], monthly["Burn_Date_Median_DOY"])]

    # 6) land-only denominator + % burned
    uniq = monthly[["Region","AreaID","Buffer_Scale"]].drop_duplicates()

    land_rows = []
    for _, r in uniq.iterrows():
        reg = r["Region"]
        if reg not in region_files:
            land_rows.append((reg, r["AreaID"], r["Buffer_Scale"], np.nan))
            continue
        la = land_area_km2_for(
            reg, str(r["AreaID"]), str(r["Buffer_Scale"]),
            region_files=region_files, land_path=land_path, lakes_path=lakes_path,
            union_grid_size=union_grid_size, simplify_m=simplify_m
        )
        land_rows.append((reg, r["AreaID"], r["Buffer_Scale"], la))

    land_df = pd.DataFrame(land_rows, columns=["Region","AreaID","Buffer_Scale","Land_Area_km2"])
    monthly = monthly.merge(land_df, on=["Region","AreaID","Buffer_Scale"], how="left")

    monthly["Pct_Land_Burned"] = np.where(
        monthly["Land_Area_km2"] > 0,
        100.0 * (monthly["Burned_Area_km2"] / monthly["Land_Area_km2"]),
        np.nan
    )

    # 7) write ONE CSV per year
    out_paths = []
    for yr, sub in monthly.groupby("Year"):
        fout = os.path.join(outdir, f"buffer_monthly_{int(yr)}_master.csv")

        cols = [
            # identity
            "Region","AreaID","Buffer_Scale","Scale_Pretty",
            "Buffer_Size","Effective_Buffer_Area_km2",
            # land denom + % (keep together)
            "Land_Area_km2","Pct_Land_Burned",
            # time
            "Year","Month","Month_Name",
            # burned amount
            "Burned_Area_km2","Burned_Pixels",
            # burn DOY stats
            "Burn_Date_Min","Burn_Date_Min_Date",
            "Burn_Date_Median_DOY","Burn_Date_Median_Date",
            "Burn_Date_Max","Burn_Date_Max_Date",
            "Burn_DOY_Range",
            # uncertainty stats
            "Uncertainty_Min","Uncertainty_Median","Uncertainty_Max","Uncertainty_Range",
        ]

        sub = sub[cols].sort_values(["Region","AreaID","Month","Buffer_Size","Buffer_Scale"])
        sub.to_csv(fout, index=False)
        print("Saved yearly file →", fout)
        out_paths.append(fout)

    if not out_paths:
        print("No yearly outputs were written (check filters).")
    else:
        print("\nFiles written:")
        for p in out_paths:
            print("  →", p)

    # preview
    print("\nPreview (first 10 rows):")
    print(monthly.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
