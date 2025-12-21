#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Aggregate per-buffer-per-tile CSVs into ONE master CSV per year, including:
- Land_Area_km2 (land-only within buffer at that scale)
- Pct_Land_Burned = 100 * Burned_Area_km2 / Land_Area_km2
- Burn_Date_Min/Max + Median DOY, with calendar date + median month name
- Uncertainty min/max + median

Notes:
- Median DOY at monthly level is computed as nanmedian across contributing tiles
  (not pixel-weighted global median). If you later want pixel-weighted median, we can do that too.
"""

import os
import glob
import re
import math
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

# Shapely validation
try:
    from shapely import make_valid, union_all
except Exception:  # pragma: no cover
    make_valid = None
    union_all = None

try:
    from shapely.validation import make_valid as make_valid_v1
except Exception:  # pragma: no cover
    make_valid_v1 = None

# Shapely 1.x union fallback
try:
    from shapely.ops import unary_union
except Exception:  # pragma: no cover
    unary_union = None


def _make_valid(g):
    if g is None:
        return g
    if make_valid is not None:
        try:
            return make_valid(g)
        except Exception:
            pass
    if make_valid_v1 is not None:
        try:
            return make_valid_v1(g)
        except Exception:
            pass
    try:
        return g.buffer(0)
    except Exception:
        return g


def doy_to_date(year, doy):
    if pd.isna(doy):
        return None
    try:
        return (datetime(int(year), 1, 1) + timedelta(days=int(float(doy)) - 1)).strftime("%Y-%m-%d")
    except Exception:
        return None


def month_name_from_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d").strftime("%B") if isinstance(s, str) else None
    except Exception:
        return None


def coerce_month(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, np.integer, float)) and not np.isnan(x):
        return int(x)
    s = str(x).strip()
    if "." in s:
        try:
            return int(s.split(".")[0])
        except Exception:
            return np.nan
    month_map = {m.lower(): i for i, m in enumerate(
        ["January","February","March","April","May","June","July","August","September","October","November","December"], 1
    )}
    if s.lower() in month_map:
        return month_map[s.lower()]
    try:
        return int(s)
    except Exception:
        return np.nan


def pretty_scale(code, scale_mult):
    return f"{code} (×{scale_mult.get(code, 1.0):.2f})"


def pick_name_field(gdf: gpd.GeoDataFrame):
    for c in ["AreaID", "areaid", "Name", "name", "Site", "site", "Label", "label"]:
        if c in gdf.columns:
            return c
    return None


# ---------------- CONFIG (edit these) ----------------
OUTPUT_ROOT = r"A:\Project BFA\output_csvs"
SUM_OUTDIR = os.path.join(OUTPUT_ROOT, "summaries")
os.makedirs(SUM_OUTDIR, exist_ok=True)

LAND_PATH  = r"A:\Project BFA\Shapefiles\Land_cover\ne_10m_land.shp"
LAKES_PATH = r"A:\Project BFA\Shapefiles\Land_cover\ne_10m_lakes.shp"

REGION_FILES = {
    "Europe": (r"A:\Project BFA\Shapefiles\buffers_per_region\buffers_Europe_3035.shp", 3035),
    "Siberia": (r"A:\Project BFA\Shapefiles\buffers_per_region\buffers_Siberia_3576.shp", 3576),
    "NorthAmerica": (r"A:\Project BFA\Shapefiles\buffers_per_region\buffers_NorthAmerica_5070.shp", 5070),
}

# Optional filters
YEARS  = None   # e.g. [2020, 2021, 2022]
MONTHS = None   # e.g. [3,4,5,6,7,8,9,10]

SKIP_NAME_PATTERNS = re.compile(r"(summary|summaries|land_area|chronology|master)", re.IGNORECASE)

SCALE_MULT = {
    "original": 1.00,
    "buffer_10pct": 1.10,
    "buffer_20pct": 1.20,
    "buffer_50pct": 1.50,
    "buffer_100pct": 2.00
}

USE_COLS = {
    "Region":"str","AreaID":"str","Buffer_Size":"float64","Buffer_Scale":"str",
    "Year":"object","Month":"object","Tile":"str",
    "Burned_Pixels":"float64","Burned_Area_km2":"float64",
    "Burn_Date_Min":"float64","Burn_Date_Max":"float64",
    "Burn_Date_Median_DOY":"float64",
    "Uncertainty_Min":"float64","Uncertainty_Max":"float64",
    "Uncertainty_Median":"float64",
}


def read_csv_robust(path):
    for enc in ("utf-8","utf-8-sig","latin-1"):
        try:
            df = pd.read_csv(
                path,
                dtype=USE_COLS,
                usecols=lambda c: c in USE_COLS,
                encoding=enc,
                low_memory=False
            )
            return df
        except Exception:
            continue
    print(f"⚠️  Skipping unreadable CSV: {path}")
    return None


# ---- cache land-only unions per CRS ----
_land_union_cache = {}

def land_union_in_crs(target_crs):
    key = str(target_crs)
    if key in _land_union_cache:
        return _land_union_cache[key]

    land = gpd.read_file(LAND_PATH).to_crs(target_crs)
    lakes = gpd.read_file(LAKES_PATH).to_crs(target_crs)

    # union (prefer union_all, else unary_union)
    if union_all is not None:
        land_u  = union_all(list(land.geometry), axis=None)
        lakes_u = union_all(list(lakes.geometry), axis=None)
    else:
        land_u  = unary_union(land.geometry) if unary_union is not None else land.unary_union
        lakes_u = unary_union(lakes.geometry) if unary_union is not None else lakes.unary_union

    land_only = _make_valid(land_u).difference(_make_valid(lakes_u))
    _land_union_cache[key] = land_only
    return land_only


# ---- cache region buffers and AreaID lookup ----
_region_cache = {}

def load_region_buffers(region):
    if region in _region_cache:
        return _region_cache[region]

    shp, epsg = REGION_FILES[region]
    gdf = gpd.read_file(shp)
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg)
    else:
        gdf = gdf.to_crs(epsg)

    name_col = pick_name_field(gdf)

    idx_map = {}
    for idx, row in gdf.iterrows():
        if name_col and pd.notna(row[name_col]):
            idx_map[str(row[name_col])] = row
        idx_map[f"Buffer{int(idx)}"] = row

    _region_cache[region] = (gdf, name_col, idx_map)
    return _region_cache[region]


def scale_buffer_geom(base_row, scale_code):
    factor = SCALE_MULT.get(scale_code, 1.0)
    base_area = float(base_row["Buffer_km2"])  # km²
    r = math.sqrt(base_area * 1e6 / math.pi)   # m
    grow = r * (factor - 1.0)
    geom0 = _make_valid(base_row.geometry)
    try:
        grown = geom0.buffer(grow)
        if scale_code != "original":
            grown = _make_valid(grown.union(geom0))
        return grown
    except Exception:
        return geom0


_land_area_cache = {}

def land_area_km2_for(region, areaid, scale_code):
    key = (region, str(areaid), str(scale_code))
    if key in _land_area_cache:
        return _land_area_cache[key]

    gdf, name_col, idx_map = load_region_buffers(region)
    base_row = idx_map.get(str(areaid), None)

    # try case-insensitive fallback
    if base_row is None:
        low = str(areaid).strip().lower()
        for k, v in idx_map.items():
            if str(k).strip().lower() == low:
                base_row = v
                break

    if base_row is None:
        _land_area_cache[key] = np.nan
        return np.nan

    scaled_geom = scale_buffer_geom(base_row, scale_code)
    land_only = land_union_in_crs(gdf.crs)

    try:
        inter = _make_valid(scaled_geom).intersection(_make_valid(land_only))
        area_km2 = inter.area / 1e6
    except Exception:
        area_km2 = np.nan

    _land_area_cache[key] = area_km2
    return area_km2


def main():
    # 1) Load per-buffer CSVs
    all_csvs = glob.glob(os.path.join(OUTPUT_ROOT, "**", "*.csv"), recursive=True)
    per_buffer_csvs = [p for p in all_csvs if not SKIP_NAME_PATTERNS.search(os.path.basename(p))]
    if not per_buffer_csvs:
        raise SystemExit(f"No per-buffer CSVs found under {OUTPUT_ROOT}")

    frames = []
    for p in per_buffer_csvs:
        df = read_csv_robust(p)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        raise SystemExit("No readable per-buffer CSVs")

    chron = pd.concat(frames, ignore_index=True)

    # 2) Clean types
    chron["Year"] = pd.to_numeric(chron["Year"], errors="coerce").astype("Int64")
    chron["Month"] = chron["Month"].map(coerce_month).astype("Int64")

    chron["Burned_Pixels"] = pd.to_numeric(chron["Burned_Pixels"], errors="coerce").fillna(0).astype(int)
    chron["Burned_Area_km2"] = pd.to_numeric(chron["Burned_Area_km2"], errors="coerce").fillna(0.0)

    for c in ["Burn_Date_Min","Burn_Date_Max","Burn_Date_Median_DOY","Uncertainty_Min","Uncertainty_Max","Uncertainty_Median","Buffer_Size"]:
        if c in chron.columns:
            chron[c] = pd.to_numeric(chron[c], errors="coerce")

    if YEARS is not None:
        chron = chron[chron["Year"].isin(YEARS)]
    if MONTHS is not None:
        chron = chron[chron["Month"].isin(MONTHS)]

    chron = chron.dropna(subset=["Region","AreaID","Buffer_Scale","Year","Month","Buffer_Size"])
    chron = chron[chron["Burned_Pixels"] > 0]  # only events

    # 3) Enhance
    month_names = {i: m for i, m in enumerate(
        ["January","February","March","April","May","June","July","August","September","October","November","December"], 1
    )}
    chron["Month_Name"] = chron["Month"].map(month_names)
    chron["Scale_Pretty"] = chron["Buffer_Scale"].map(lambda s: pretty_scale(s, SCALE_MULT))
    chron["Scale_Mult"] = chron["Buffer_Scale"].map(SCALE_MULT).fillna(1.0)
    chron["Effective_Buffer_Area_km2"] = chron["Buffer_Size"] * (chron["Scale_Mult"] ** 2)

    # 4) Monthly aggregation (sum across tiles; median across tiles)
    group_keys = [
        "Region","AreaID","Buffer_Scale","Scale_Pretty","Buffer_Size","Effective_Buffer_Area_km2",
        "Year","Month","Month_Name"
    ]

    def nanmedian_or_nan(x):
        x = pd.to_numeric(x, errors="coerce")
        return np.nan if x.isna().all() else float(np.nanmedian(x))

    monthly = (chron.groupby(group_keys, as_index=False)
               .agg(
                    Burned_Area_km2=("Burned_Area_km2","sum"),
                    Burned_Pixels=("Burned_Pixels","sum"),
                    Burn_Date_Min=("Burn_Date_Min","min"),
                    Burn_Date_Max=("Burn_Date_Max","max"),
                    Burn_Date_Median_DOY=("Burn_Date_Median_DOY", nanmedian_or_nan),
                    Uncertainty_Min=("Uncertainty_Min","min"),
                    Uncertainty_Max=("Uncertainty_Max","max"),
                    Uncertainty_Median=("Uncertainty_Median", nanmedian_or_nan),
                ))

    monthly["Burn_Date_Min_Date"] = [doy_to_date(y, d) for y, d in zip(monthly["Year"], monthly["Burn_Date_Min"])]
    monthly["Burn_Date_Max_Date"] = [doy_to_date(y, d) for y, d in zip(monthly["Year"], monthly["Burn_Date_Max"])]
    monthly["Burn_Date_Median_Date"] = [doy_to_date(y, d) for y, d in zip(monthly["Year"], monthly["Burn_Date_Median_DOY"])]

    monthly["Median_Month_Name"] = monthly["Burn_Date_Median_Date"].map(month_name_from_date).fillna(monthly["Month_Name"])

    # 5) Land area and % burned
    uniq = monthly[["Region","AreaID","Buffer_Scale"]].drop_duplicates()
    land_rows = []
    for _, r in uniq.iterrows():
        la = land_area_km2_for(r["Region"], r["AreaID"], r["Buffer_Scale"])
        land_rows.append((r["Region"], r["AreaID"], r["Buffer_Scale"], la))
    land_df = pd.DataFrame(land_rows, columns=["Region","AreaID","Buffer_Scale","Land_Area_km2"])

    monthly = monthly.merge(land_df, on=["Region","AreaID","Buffer_Scale"], how="left")
    monthly["Pct_Land_Burned"] = np.where(
        monthly["Land_Area_km2"] > 0,
        100.0 * (monthly["Burned_Area_km2"] / monthly["Land_Area_km2"]),
        np.nan
    )

    # 6) Write ONE CSV per year
    cols = [
        "Region","AreaID",
        "Buffer_Scale","Scale_Pretty",
        "Buffer_Size","Effective_Buffer_Area_km2",
        "Land_Area_km2","Pct_Land_Burned",
        "Year","Month","Month_Name",
        "Burned_Area_km2","Burned_Pixels",
        "Burn_Date_Min","Burn_Date_Min_Date",
        "Burn_Date_Median_DOY","Burn_Date_Median_Date","Median_Month_Name",
        "Burn_Date_Max","Burn_Date_Max_Date",
        "Uncertainty_Min","Uncertainty_Max","Uncertainty_Median",
    ]

    if monthly.empty:
        print("No monthly rows to write (did you filter too hard or have no burns?).")
        return

    for yr, sub in monthly.groupby("Year"):
        fout = os.path.join(SUM_OUTDIR, f"buffer_monthly_{int(yr)}_master.csv")
        sub[cols].sort_values(["Region","AreaID","Month","Buffer_Size","Buffer_Scale"]).to_csv(fout, index=False)
        print("Saved yearly file →", fout)

    print("\nPreview:")
    print(monthly[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
