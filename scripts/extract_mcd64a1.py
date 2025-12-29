#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
extract_mcd64a1.py

Extract MODIS MCD64A1 burned-area chronology per buffer (multiple regions + buffer scales).
Writes one CSV per (tile × buffer × scale × month) only when burned_pixels > 0,
unless --write-zero-csvs is set.

Fixes vs earlier versions:
- If --months is NOT provided: auto-detect available month folders under <hdf_root>/<year>/ as YYYY.MM.DD.
- If multiple HDFs exist for a tile-month: select the HDF with max BurnedCells / NUMBERBURNEDPIXELS,
  instead of "last file wins" (prevents overwriting burn months with zero-burn copies).
- Stronger georeference checks after Translate.
"""

import os
import glob
import re
import tempfile
import warnings
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd

from osgeo import gdal
import rasterio
from rasterio.mask import mask

from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
from shapely import union_all  # shapely 2.x

# make_valid compat (Shapely 2 vs older)
try:
    from shapely import make_valid  # shapely>=2
except Exception:
    try:
        from shapely.validation import make_valid  # shapely<2
    except Exception:
        def make_valid(g):
            try:
                return g.buffer(0)
            except Exception:
                return g

warnings.filterwarnings("ignore", category=UserWarning)
gdal.UseExceptions()

# ---------------- Defaults (edit if you want) ----------------
DEFAULT_YEARS = ['2001','2002','2003','2004','2005','2006','2007','2008','2009','2010']
# Used only if you explicitly pass --months; otherwise we auto-detect folders on disk.
DEFAULT_MONTH_SUFFIXES = ['03.01','04.01','05.01','06.01','07.01','08.01','09.01','10.01']

REGION_META = {
    'Europe': (r"A:\Project BFA\Shapefiles\buffers_per_region\buffers_Europe_3035.shp", 3035),
    'Siberia': (r"A:\Project BFA\Shapefiles\buffers_per_region\buffers_Siberia_3576.shp", 3576),
    'NorthAmerica': (r"A:\Project BFA\Shapefiles\buffers_per_region\buffers_NorthAmerica_5070.shp", 5070),
}

BUFFER_SCALE_FACTORS = {
    'original': 1.00,
    'buffer_10pct': 1.10,
    'buffer_20pct': 1.20,
    'buffer_50pct': 1.50,
    'buffer_100pct': 2.00,
}

HDF_ROOT_DEFAULT = r"A:\Project BFA\hdf files"
OUTPUT_ROOT_DEFAULT = r"A:\Project BFA\output_csvs"

LAND_PATH_DEFAULT = r"A:\Project BFA\Shapefiles\Land_cover\ne_10m_land.shp"
LAKES_PATH_DEFAULT = r"A:\Project BFA\Shapefiles\Land_cover\ne_10m_lakes.shp"

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Extract MODIS MCD64A1 burned area metrics per buffer and write per-buffer CSVs."
    )

    p.add_argument("--years", nargs="+", default=None,
                   help="Years to process, e.g. 2020 2021 2022. If omitted, uses defaults in script.")

    # IMPORTANT: If omitted, we auto-detect folders under each year directory.
    p.add_argument("--months", nargs="+", default=None,
                   help="Month suffixes, e.g. 03.01 04.01 ... If omitted, auto-detect available YYYY.MM.DD folders on disk.")

    p.add_argument("--regions", nargs="+", default=None,
                   help="Regions, e.g. Europe Siberia NorthAmerica. If omitted, uses all REGION_META keys.")

    p.add_argument("--hdf-root", dest="hdf_root", default=None,
                   help="Root folder containing year/month HDF structure.")
    p.add_argument("--out-root", dest="out_root", default=None,
                   help="Output root where per-buffer CSVs will be written.")
    p.add_argument("--land-path", dest="land_path", default=None,
                   help="Natural Earth land shapefile path.")
    p.add_argument("--lakes-path", dest="lakes_path", default=None,
                   help="Natural Earth lakes shapefile path.")

    p.add_argument("--write-zero-csvs", action="store_true",
                   help="If set, write zero-burn CSVs too (default: off).")
    p.add_argument("--overwrite", action="store_true",
                   help="If set, overwrite existing CSVs (default: skip existing).")

    p.add_argument("--tile-buffer-m", type=int, default=15000,
                   help="Buffer (meters) around tile bbox used to clip land mask. Default 15000.")
    p.add_argument("--grid-size", type=float, default=100.0,
                   help="grid_size used in shapely.union_all for stabilizing unions. Default 100.0.")
    p.add_argument("--simplify-m", type=float, default=50.0,
                   help="Simplify tolerance (meters) for land union polygon; 0 disables. Default 50.")

    return p.parse_args()

# ---------------- Geometry helpers ----------------
def _polygon_only(geom):
    if geom is None or geom.is_empty:
        return None
    t = geom.geom_type
    if t in ("Polygon", "MultiPolygon"):
        return geom
    if isinstance(geom, GeometryCollection) and hasattr(geom, "geoms"):
        parts = []
        for g in geom.geoms:
            if g is None or g.is_empty:
                continue
            if g.geom_type == "Polygon":
                parts.append(g)
            elif g.geom_type == "MultiPolygon":
                parts.extend(list(g.geoms))
        if not parts:
            return None
        return MultiPolygon(parts) if len(parts) > 1 else parts[0]
    return None

def _clean_series_to_polys(gser):
    gser = gser.dropna().apply(make_valid).apply(_polygon_only)
    gser = gpd.GeoSeries([g for g in gser if g is not None], crs=gser.crs)
    if len(gser) == 0:
        return gser
    gser = gser.buffer(0)
    gser = gser[~gser.is_empty]
    return gser

def _bbox_filter(gdf, clip_poly):
    if gdf.empty:
        return gdf
    if not hasattr(gdf, "sindex") or gdf.sindex is None:
        return gdf[gdf.intersects(clip_poly)]
    hits = list(gdf.sindex.intersection(clip_poly.bounds))
    if not hits:
        return gdf.iloc[[]]
    cand = gdf.iloc[hits]
    return cand[cand.intersects(clip_poly)]

_land_union_cache = {}

def land_union_in_crs_clipped(land_path, lakes_path, target_crs, clip_poly, grid_size=100.0):
    extent_key = tuple(np.round(gpd.GeoSeries([clip_poly], crs=target_crs).total_bounds, 1))
    key = (str(target_crs), extent_key, float(grid_size))
    if key in _land_union_cache:
        return _land_union_cache[key]

    land  = gpd.read_file(land_path).to_crs(target_crs)
    lakes = gpd.read_file(lakes_path).to_crs(target_crs)

    land  = _bbox_filter(land, clip_poly)
    lakes = _bbox_filter(lakes, clip_poly)

    if not land.empty:
        land = land.copy()
        land["geometry"] = land.geometry.intersection(clip_poly)
        land = land[~land.is_empty]

    if not lakes.empty:
        lakes = lakes.copy()
        lakes["geometry"] = lakes.geometry.intersection(clip_poly)
        lakes = lakes[~lakes.is_empty]

    land_polys  = _clean_series_to_polys(land.geometry)  if not land.empty  else gpd.GeoSeries([], crs=target_crs)
    lakes_polys = _clean_series_to_polys(lakes.geometry) if not lakes.empty else gpd.GeoSeries([], crs=target_crs)

    land_u  = union_all(list(land_polys),  grid_size=grid_size) if len(land_polys)  else MultiPolygon()
    lakes_u = union_all(list(lakes_polys), grid_size=grid_size) if len(lakes_polys) else MultiPolygon()

    land_only = make_valid(land_u).difference(make_valid(lakes_u))
    land_only = _polygon_only(make_valid(land_only)) or MultiPolygon()
    if isinstance(land_only, Polygon):
        land_only = MultiPolygon([land_only])

    _land_union_cache[key] = land_only
    return land_only

# ---------------- MODIS SDS selection + translation ----------------
def _pick_subdatasets(ds):
    subs = ds.GetSubDatasets() or []
    burn_sds = None
    unc_sds = None

    burn_keys = ("burn date", "burn_date")
    unc_keys = ("burn date uncertainty", "uncertainty", "burn_date_uncertainty")

    for name, desc in subs:
        s = (name + " " + desc).lower()
        if burn_sds is None and any(k in s for k in burn_keys):
            burn_sds = name
        if unc_sds is None and any(k in s for k in unc_keys):
            unc_sds = name

    return burn_sds, unc_sds

def translate_sds_to_tif_with_geo(sds_name: str, out_tif: str) -> bool:
    src = gdal.Open(sds_name)
    if src is None:
        return False

    proj = src.GetProjection()
    gt   = src.GetGeoTransform(can_return_null=True)

    out = gdal.Translate(out_tif, src, format="GTiff")
    if out is None:
        return False
    out = None

    ds_out = gdal.Open(out_tif, gdal.GA_Update)
    if ds_out is None:
        return False

    if (not ds_out.GetProjection()) and proj:
        ds_out.SetProjection(proj)
    if (ds_out.GetGeoTransform(can_return_null=True) is None) and (gt is not None):
        ds_out.SetGeoTransform(gt)

    ds_out = None
    return True

# ---------------- Month auto-detection ----------------
_month_folder_re = re.compile(r"^(?P<year>\d{4})\.(?P<mm>\d{2})\.(?P<dd>\d{2})$")

def detect_month_suffixes_for_year(hdf_root: str, year: str):
    """
    Look inside <hdf_root>/<year>/ for folders named YYYY.MM.DD and return suffixes MM.DD sorted.
    """
    year_dir = os.path.join(hdf_root, str(year))
    if not os.path.isdir(year_dir):
        return []

    suffixes = []
    for name in os.listdir(year_dir):
        m = _month_folder_re.match(name)
        if not m:
            continue
        if m.group("year") != str(year):
            continue
        suffixes.append(f"{m.group('mm')}.{m.group('dd')}")

    # sort by month then day
    suffixes = sorted(set(suffixes), key=lambda s: (int(s.split(".")[0]), int(s.split(".")[1])))
    return suffixes

# ---------------- Prefer HDF with burns (handles duplicates) ----------------
def burn_count_from_hdf(hdf_path: str) -> int:
    """
    Read BurnedCells / NUMBERBURNEDPIXELS from HDF metadata; fallback 0 on any error.
    """
    try:
        ds = gdal.Open(hdf_path)
        if ds is None:
            return 0
        md = ds.GetMetadata() or {}
        # MCD64A1 commonly has BurnedCells and NUMBERBURNEDPIXELS
        for k in ("BurnedCells", "NUMBERBURNEDPIXELS"):
            if k in md:
                try:
                    return int(str(md[k]).strip())
                except Exception:
                    pass
        return 0
    except Exception:
        return 0

# ---------------- Core extraction ----------------
def run_extraction(
    run_years,
    run_months,          # can be None => auto detect per year
    run_regions,
    hdf_root,
    out_root,
    land_path,
    lakes_path,
    buffer_scale_factors,
    write_zero_csvs=False,
    overwrite=False,
    tile_buffer_m=15000,
    grid_size=100.0,
    simplify_m=50.0,
):
    all_written_csvs = []

    region_meta_run = {k: REGION_META[k] for k in run_regions if k in REGION_META}
    if not region_meta_run:
        raise ValueError(f"No valid regions selected. Requested={run_regions}, available={list(REGION_META.keys())}")

    tile_re = re.compile(r"(h\d{2}v\d{2})", re.IGNORECASE)

    for year in run_years:
        year = str(year)
        print(f"\n====== Processing Year: {year} ======")

        # AUTO-DETECT months if not provided
        if run_months is None:
            months_this_year = detect_month_suffixes_for_year(hdf_root, year)
            if not months_this_year:
                print(f"WARNING: no month folders detected under {os.path.join(hdf_root, year)}")
            else:
                print(f"Detected months for {year}: {', '.join(months_this_year)}")
        else:
            months_this_year = [str(m) for m in run_months]

        for month_suffix in months_this_year:
            month_int = int(month_suffix.split('.')[0])  # '03.01' -> 3

            month_folder_name = f"{year}.{month_suffix}"
            month_folder = os.path.join(hdf_root, year, month_folder_name)
            print(f"\n----- Processing Month Folder: {month_folder_name} -----")

            if not os.path.isdir(month_folder):
                print(f"WARNING: folder not found → {month_folder} (skipping month)")
                continue

            hdfs = sorted(glob.glob(os.path.join(month_folder, "*.hdf")))
            if not hdfs:
                print(f"No HDF files in {month_folder}")
                continue

            # Group by tile id (hXXvYY)
            tile_dict = {}
            for hdf in hdfs:
                m = tile_re.search(os.path.basename(hdf))
                if m:
                    tile = m.group(1).lower()
                    tile_dict.setdefault(tile, []).append(hdf)

            for tile, tile_files in tile_dict.items():
                print(f"\n--- Tile {tile} ({year}-{month_suffix}) ---")

                with tempfile.TemporaryDirectory() as tmpdir:
                    tif_burn_date   = os.path.join(tmpdir, f"{tile}_burndate.tif")
                    tif_uncertainty = os.path.join(tmpdir, f"{tile}_uncertainty.tif")

                    # If duplicates exist, pick the one with max burn count (prevents "zero overwrites burn")
                    tile_files_sorted = sorted(tile_files)
                    best_hdf = max(tile_files_sorted, key=burn_count_from_hdf)

                    try:
                        ds = gdal.Open(best_hdf)
                        burn_sds, unc_sds = _pick_subdatasets(ds)
                        if burn_sds is None or unc_sds is None:
                            print(f"  Skipping (missing subdatasets): {os.path.basename(best_hdf)}")
                            continue

                        ok1 = translate_sds_to_tif_with_geo(burn_sds, tif_burn_date)
                        ok2 = translate_sds_to_tif_with_geo(unc_sds, tif_uncertainty)
                        if not (ok1 and ok2):
                            print("  → Translate failed → skipping tile.")
                            continue

                    except Exception as e:
                        print(f"  Skipping (GDAL read/translate error): {os.path.basename(best_hdf)} → {e}")
                        continue

                    if (not os.path.isfile(tif_burn_date)) or (not os.path.isfile(tif_uncertainty)):
                        print("  → No valid GeoTIFFs produced for this tile-month → skipping tile.")
                        continue

                    # raster info & pixel area (+ fast skip)
                    with rasterio.open(tif_burn_date) as src_burn_date:
                        if src_burn_date.crs is None:
                            print("  ERROR: GeoTIFF has no CRS after translation → skipping tile.")
                            continue

                        b = src_burn_date.bounds
                        if abs(b.right - b.left) < 1000 and abs(b.top - b.bottom) < 1000:
                            print(f"  ERROR: Suspicious bounds {b} (pixel coords?) → skipping tile.")
                            continue

                        raster_crs  = src_burn_date.crs
                        tile_bounds = src_burn_date.bounds
                        px_x, px_y  = src_burn_date.res
                        pixel_area_km2 = abs(px_x * px_y) / 1e6

                        arr = src_burn_date.read(1, masked=True)
                        if not np.any((arr >= 1) & (arr <= 366)):
                            print(f"  → Tile {tile} has no burns this month → skipping buffers.")
                            continue

                    # Build land union clipped to tile bbox (+tile_buffer_m)
                    tile_poly = box(tile_bounds.left, tile_bounds.bottom, tile_bounds.right, tile_bounds.top)
                    clip_poly = gpd.GeoSeries([tile_poly], crs=raster_crs).buffer(tile_buffer_m).iloc[0]

                    land_union_raster = land_union_in_crs_clipped(
                        land_path, lakes_path, raster_crs, clip_poly, grid_size=grid_size
                    )
                    if simplify_m and simplify_m > 0:
                        try:
                            land_union_raster = land_union_raster.simplify(float(simplify_m))
                        except Exception:
                            pass

                    saved_count = 0
                    zero_count = 0
                    skipped_existing = 0

                    for region, (buffer_path, equal_area_epsg) in region_meta_run.items():
                        print(f"Region: {region}")

                        buffer_gdf = gpd.read_file(buffer_path)
                        buffer_gdf = buffer_gdf.to_crs(equal_area_epsg) if buffer_gdf.crs else buffer_gdf.set_crs(equal_area_epsg)

                        if "Buffer_km2" not in buffer_gdf.columns:
                            raise ValueError(f"'Buffer_km2' missing in {buffer_path}")

                        buffer_gdf["buffer_radius_m"] = np.sqrt(buffer_gdf["Buffer_km2"] * 1e6 / np.pi)

                        for scale_name, mfac in buffer_scale_factors.items():
                            grow_dist = buffer_gdf["buffer_radius_m"] * (mfac - 1.0)
                            buffer_gdf_scaled = buffer_gdf.copy()
                            buffer_gdf_scaled["geometry"] = buffer_gdf.geometry.buffer(grow_dist)
                            if scale_name != "original":
                                buffer_gdf_scaled["geometry"] = buffer_gdf.geometry.union(buffer_gdf_scaled.geometry)

                            buffers_raster = buffer_gdf_scaled.to_crs(raster_crs)
                            intersecting = buffers_raster[buffers_raster.intersects(tile_poly)]
                            print(f"  → {len(intersecting)} buffers intersect this tile in raster CRS.")

                            for idx, row in intersecting.iterrows():
                                geom_landonly = make_valid(row.geometry).intersection(land_union_raster)
                                geom_landonly = _polygon_only(make_valid(geom_landonly))
                                if geom_landonly is None or geom_landonly.is_empty:
                                    continue

                                mask_shapes = [geom_landonly]

                                buffer_area_class = f"A{int(row['Buffer_km2'])}"
                                output_dir = os.path.join(
                                    out_root, year, month_folder_name, region, buffer_area_class, scale_name
                                )
                                os.makedirs(output_dir, exist_ok=True)

                                out_csv = os.path.join(output_dir, f"{tile}_Buffer{idx}_{year}_{month_suffix}.csv")
                                if (not overwrite) and os.path.isfile(out_csv):
                                    skipped_existing += 1
                                    continue

                                try:
                                    with rasterio.open(tif_burn_date) as src_burn:
                                        burn_date_clip, _ = mask(src_burn, mask_shapes, crop=True)
                                except ValueError:
                                    continue

                                bd = burn_date_clip[0]
                                valid = (bd >= 1) & (bd <= 366)

                                burned_pixels = int(valid.sum())
                                burned_area_km2 = burned_pixels * pixel_area_km2
                                areaid_val = str(row["AreaID"]) if "AreaID" in row else f"Buffer{idx}"

                                if burned_pixels == 0:
                                    zero_count += 1
                                    if write_zero_csvs:
                                        pd.DataFrame([{
                                            "Region": region,
                                            "AreaID": areaid_val,
                                            "Buffer_Size": float(row["Buffer_km2"]),
                                            "Buffer_Scale": scale_name,
                                            "Year": int(year),
                                            "Month": int(month_int),
                                            "Tile": tile,
                                            "Burned_Pixels": 0,
                                            "Burned_Area_km2": 0.0,
                                            "Burn_Date_Min": None,
                                            "Burn_Date_Max": None,
                                            "Burn_Date_Median_DOY": None,
                                            "Uncertainty_Min": None,
                                            "Uncertainty_Max": None,
                                            "Uncertainty_Median": None,
                                        }]).to_csv(out_csv, index=False)
                                        all_written_csvs.append(out_csv)
                                    continue

                                try:
                                    with rasterio.open(tif_uncertainty) as src_unc:
                                        uncertainty_clip, _ = mask(src_unc, mask_shapes, crop=True)
                                except ValueError:
                                    bd_vals = bd[valid].astype("float64", copy=False)
                                    burn_date_median_doy = float(np.nanmedian(bd_vals)) if bd_vals.size else None

                                    pd.DataFrame([{
                                        "Region": region,
                                        "AreaID": areaid_val,
                                        "Buffer_Size": float(row["Buffer_km2"]),
                                        "Buffer_Scale": scale_name,
                                        "Year": int(year),
                                        "Month": int(month_int),
                                        "Tile": tile,
                                        "Burned_Pixels": int(burned_pixels),
                                        "Burned_Area_km2": float(burned_area_km2),
                                        "Burn_Date_Min": int(np.nanmin(bd_vals)) if bd_vals.size else None,
                                        "Burn_Date_Max": int(np.nanmax(bd_vals)) if bd_vals.size else None,
                                        "Burn_Date_Median_DOY": burn_date_median_doy,
                                        "Uncertainty_Min": None,
                                        "Uncertainty_Max": None,
                                        "Uncertainty_Median": None,
                                    }]).to_csv(out_csv, index=False)

                                    all_written_csvs.append(out_csv)
                                    saved_count += 1
                                    continue

                                bd_vals = bd[valid].astype("float64", copy=False)
                                un_vals = uncertainty_clip[0][valid].astype("float64", copy=False)

                                burn_date_median_doy = float(np.nanmedian(bd_vals)) if bd_vals.size else None
                                uncertainty_median = float(np.nanmedian(un_vals)) if un_vals.size else None

                                pd.DataFrame([{
                                    "Region": region,
                                    "AreaID": areaid_val,
                                    "Buffer_Size": float(row["Buffer_km2"]),
                                    "Buffer_Scale": scale_name,
                                    "Year": int(year),
                                    "Month": int(month_int),
                                    "Tile": tile,
                                    "Burned_Pixels": int(burned_pixels),
                                    "Burned_Area_km2": float(burned_area_km2),
                                    "Burn_Date_Min": int(np.nanmin(bd_vals)) if bd_vals.size else None,
                                    "Burn_Date_Max": int(np.nanmax(bd_vals)) if bd_vals.size else None,
                                    "Burn_Date_Median_DOY": burn_date_median_doy,
                                    "Uncertainty_Min": float(np.nanmin(un_vals)) if un_vals.size else None,
                                    "Uncertainty_Max": float(np.nanmax(un_vals)) if un_vals.size else None,
                                    "Uncertainty_Median": uncertainty_median,
                                }]).to_csv(out_csv, index=False)

                                all_written_csvs.append(out_csv)
                                saved_count += 1

                    print(f"--- Tile {tile} done: {saved_count} CSVs with burns, "
                          f"{zero_count} zero-burn cases, {skipped_existing} skipped-existing ---")

    print("\n================ SUMMARY ================")
    print(f"Total CSVs written: {len(all_written_csvs)}")
    if all_written_csvs:
        preview = min(15, len(all_written_csvs))
        print(f"First {preview} CSV paths:")
        for p in all_written_csvs[:preview]:
            print("  →", p)
    else:
        print("No CSVs were written.")
    print("========================================\nAll done.\n")

    return all_written_csvs

def main():
    args = parse_args()

    run_years = args.years if args.years is not None else DEFAULT_YEARS

    # If user doesn't pass --months, we AUTO-DETECT months per year
    run_months = None if args.months is None else args.months

    run_regions = args.regions if args.regions is not None else list(REGION_META.keys())

    hdf_root = args.hdf_root if args.hdf_root is not None else HDF_ROOT_DEFAULT
    out_root = args.out_root if args.out_root is not None else OUTPUT_ROOT_DEFAULT
    land_p = args.land_path if args.land_path is not None else LAND_PATH_DEFAULT
    lakes_p = args.lakes_path if args.lakes_path is not None else LAKES_PATH_DEFAULT

    os.makedirs(out_root, exist_ok=True)

    run_extraction(
        run_years=run_years,
        run_months=run_months,
        run_regions=run_regions,
        hdf_root=hdf_root,
        out_root=out_root,
        land_path=land_p,
        lakes_path=lakes_p,
        buffer_scale_factors=BUFFER_SCALE_FACTORS,
        write_zero_csvs=bool(args.write_zero_csvs),
        overwrite=bool(args.overwrite),
        tile_buffer_m=int(args.tile_buffer_m),
        grid_size=float(args.grid_size),
        simplify_m=float(args.simplify_m),
    )

if __name__ == "__main__":
    main()
