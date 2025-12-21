#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract MODIS MCD64A1 Burn Date + Uncertainty into per-buffer CSVs.

Key behaviors:
- Translates HDF subdatasets -> temporary GeoTIFF (avoids rasterio/HDF4 issues)
- Skips tiles with no burns (no DOY in 1..366)
- Intersects buffers with LAND-only polygon (land minus lakes)
- Writes CSVs only when burned_pixels > 0 by default (WRITE_ZERO_CSVS=False)
- Computes Burn_Date_Median_DOY and Uncertainty_Median per output CSV (pixel-level median within that clip)
"""

import os
import glob
import tempfile
import re
import warnings
import math
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
import rasterio
from rasterio.mask import mask

from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection

# ---- Shapely 2.x preferred ----
try:
    from shapely import make_valid, union_all
except Exception:  # pragma: no cover
    make_valid = None
    union_all = None

# ---- Shapely validation fallback (for older stacks) ----
try:
    from shapely.validation import make_valid as make_valid_v1
except Exception:  # pragma: no cover
    make_valid_v1 = None


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
    # last resort
    try:
        return g.buffer(0)
    except Exception:
        return g


def _polygon_only(geom):
    """Keep only Polygon/MultiPolygon parts; drop lines/points/collections."""
    if geom is None or geom.is_empty:
        return None
    t = geom.geom_type
    if t in ("Polygon", "MultiPolygon"):
        return geom
    if isinstance(geom, GeometryCollection) and hasattr(geom, "geoms"):
        polys = [g for g in geom.geoms if g.geom_type in ("Polygon",)]
        if polys:
            return MultiPolygon(polys) if len(polys) > 1 else polys[0]
    return None


def _clean_series_to_polys(gser: gpd.GeoSeries) -> gpd.GeoSeries:
    """Valid -> polygon-only -> buffer(0) -> drop empties."""
    if gser is None or len(gser) == 0:
        return gpd.GeoSeries([], crs=getattr(gser, "crs", None))

    gser = gser.dropna().apply(_make_valid).apply(_polygon_only)
    geoms = [g for g in gser if g is not None and (not g.is_empty)]
    out = gpd.GeoSeries(geoms, crs=gser.crs)
    if len(out) == 0:
        return out
    try:
        out = out.buffer(0)
        out = out[~out.is_empty]
    except Exception:
        pass
    return out


def _bbox_filter(gdf: gpd.GeoDataFrame, clip_poly) -> gpd.GeoDataFrame:
    """Fast bbox filter using spatial index; then intersects."""
    if gdf.empty:
        return gdf
    try:
        sidx = gdf.sindex
    except Exception:
        sidx = None

    if sidx is None:
        return gdf[gdf.intersects(clip_poly)]

    hits = list(sidx.intersection(clip_poly.bounds))
    if not hits:
        return gdf.iloc[[]]
    cand = gdf.iloc[hits]
    return cand[cand.intersects(clip_poly)]


_land_union_cache = {}


def land_union_in_crs_clipped(land_path: str, lakes_path: str, target_crs, clip_poly, grid_size: float = 100.0):
    """
    Build land-only union (land minus lakes) in target CRS, clipped to clip_poly.
    Cached by (CRS, clip bounds, grid_size).
    """
    extent_key = tuple(np.round(gpd.GeoSeries([clip_poly], crs=target_crs).total_bounds, 1))
    key = (str(target_crs), extent_key, float(grid_size))
    if key in _land_union_cache:
        return _land_union_cache[key]

    land = gpd.read_file(land_path).to_crs(target_crs)
    lakes = gpd.read_file(lakes_path).to_crs(target_crs)

    land = _bbox_filter(land, clip_poly)
    lakes = _bbox_filter(lakes, clip_poly)

    if not land.empty:
        land["geometry"] = land.geometry.intersection(clip_poly)
        land = land[~land.is_empty]
    if not lakes.empty:
        lakes["geometry"] = lakes.geometry.intersection(clip_poly)
        lakes = lakes[~lakes.is_empty]

    land_polys = _clean_series_to_polys(land.geometry) if not land.empty else gpd.GeoSeries([], crs=target_crs)
    lakes_polys = _clean_series_to_polys(lakes.geometry) if not lakes.empty else gpd.GeoSeries([], crs=target_crs)

    if union_all is None:
        # fallback for very old shapely (less ideal)
        land_u = land_polys.unary_union if len(land_polys) else MultiPolygon()
        lakes_u = lakes_polys.unary_union if len(lakes_polys) else MultiPolygon()
    else:
        land_u = union_all(list(land_polys), grid_size=grid_size) if len(land_polys) else MultiPolygon()
        lakes_u = union_all(list(lakes_polys), grid_size=grid_size) if len(lakes_polys) else MultiPolygon()

    land_only = _make_valid(land_u).difference(_make_valid(lakes_u))
    land_only = _polygon_only(_make_valid(land_only)) or MultiPolygon()
    if isinstance(land_only, Polygon):
        land_only = MultiPolygon([land_only])

    _land_union_cache[key] = land_only
    return land_only


def _select_subdataset(subdatasets, want_lower: str) -> Optional[str]:
    """Return subdataset name matching key in description or name."""
    for name, desc in subdatasets:
        if want_lower in (desc or "").lower() or want_lower in (name or "").lower():
            return name
    return None


@dataclass
class ExtractConfig:
    years: List[str]
    month_suffixes: List[str]
    region_meta: Dict[str, Tuple[str, int]]
    buffer_scale_factors: Dict[str, float]

    hdf_root: str
    output_root: str

    land_path: str
    lakes_path: str

    write_zero_csvs: bool = False
    tile_buffer_m: float = 15000.0
    land_union_grid_size: float = 100.0
    land_union_simplify_m: float = 50.0


def run_extraction(cfg: ExtractConfig) -> List[str]:
    all_written = []

    for year in cfg.years:
        print(f"\n====== Processing Year: {year} ======")

        for month_suffix in cfg.month_suffixes:
            month_int = int(month_suffix.split(".")[0])  # '03.01' -> 3

            month_folder_name = f"{year}.{month_suffix}"
            month_folder = os.path.join(cfg.hdf_root, year, month_folder_name)
            print(f"\n----- Processing Month Folder: {month_folder_name} -----")

            if not os.path.isdir(month_folder):
                print(f"WARNING: folder not found → {month_folder} (skipping month)")
                continue

            hdfs = sorted(glob.glob(os.path.join(month_folder, "*.hdf")))
            if not hdfs:
                print(f"No HDF files in {month_folder}")
                continue

            # group by tile id hXXvYY
            tile_dict = {}
            tile_re = re.compile(r"(h\d{2}v\d{2})", re.IGNORECASE)
            for hdf in hdfs:
                m = tile_re.search(os.path.basename(hdf))
                if m:
                    tile = m.group(1).lower()
                    tile_dict.setdefault(tile, []).append(hdf)

            for tile, tile_files in tile_dict.items():
                print(f"\n--- Tile {tile} ({year}-{month_suffix}) ---")

                with tempfile.TemporaryDirectory() as tmpdir:
                    tif_burn_date = os.path.join(tmpdir, f"{tile}_burndate.tif")
                    tif_uncertainty = os.path.join(tmpdir, f"{tile}_uncertainty.tif")

                    # Translate subdatasets -> GTiff (last wins)
                    wrote_any = False
                    for hdf in tile_files:
                        ds = gdal.Open(hdf)
                        if ds is None:
                            continue
                        subdatasets = ds.GetSubDatasets()

                        burn_sds = _select_subdataset(subdatasets, "burn date")
                        unc_sds = _select_subdataset(subdatasets, "uncertainty")

                        if burn_sds is None or unc_sds is None:
                            print(f"  Skipping (missing subdatasets): {os.path.basename(hdf)}")
                            continue

                        gdal.Translate(tif_burn_date, burn_sds, format="GTiff")
                        gdal.Translate(tif_uncertainty, unc_sds, format="GTiff")
                        wrote_any = True

                    if not wrote_any or (not os.path.isfile(tif_burn_date)):
                        print("  → No readable burn-date output for this tile → skipping.")
                        continue

                    # raster info + fast skip
                    with rasterio.open(tif_burn_date) as src_bd:
                        raster_crs = src_bd.crs
                        tile_bounds = src_bd.bounds
                        px_x, px_y = src_bd.res
                        pixel_area_km2 = abs(px_x * px_y) / 1e6

                        arr = src_bd.read(1, masked=True)
                        if not np.any((arr >= 1) & (arr <= 366)):
                            print(f"  → Tile {tile} has no burns this month → skipping buffers.")
                            continue

                    tile_poly = box(tile_bounds.left, tile_bounds.bottom, tile_bounds.right, tile_bounds.top)
                    clip_poly = gpd.GeoSeries([tile_poly], crs=raster_crs).buffer(cfg.tile_buffer_m).iloc[0]

                    land_union_raster = land_union_in_crs_clipped(
                        cfg.land_path, cfg.lakes_path, raster_crs, clip_poly,
                        grid_size=cfg.land_union_grid_size
                    )
                    try:
                        land_union_raster = land_union_raster.simplify(cfg.land_union_simplify_m)
                    except Exception:
                        pass

                    saved_count = 0
                    zero_count = 0

                    for region, (buffer_path, equal_area_epsg) in cfg.region_meta.items():
                        print(f"Region: {region}")

                        buffer_gdf = gpd.read_file(buffer_path)
                        if buffer_gdf.crs is None:
                            buffer_gdf = buffer_gdf.set_crs(equal_area_epsg)
                        else:
                            buffer_gdf = buffer_gdf.to_crs(equal_area_epsg)

                        if "Buffer_km2" not in buffer_gdf.columns:
                            raise ValueError(f"'Buffer_km2' missing in {buffer_path}")

                        buffer_gdf["buffer_radius_m"] = np.sqrt(buffer_gdf["Buffer_km2"] * 1e6 / np.pi)

                        for scale_name, mfac in cfg.buffer_scale_factors.items():
                            grow_dist = buffer_gdf["buffer_radius_m"] * (mfac - 1.0)
                            buffer_gdf_scaled = buffer_gdf.copy()
                            buffer_gdf_scaled["geometry"] = buffer_gdf.geometry.buffer(grow_dist)
                            if scale_name != "original":
                                buffer_gdf_scaled["geometry"] = buffer_gdf.geometry.union(buffer_gdf_scaled.geometry)

                            buffers_raster = buffer_gdf_scaled.to_crs(raster_crs)
                            intersecting = buffers_raster[buffers_raster.intersects(tile_poly)]
                            print(f"  → {len(intersecting)} buffers intersect this tile in raster CRS.")

                            for idx, row in intersecting.iterrows():
                                geom_landonly = _make_valid(row.geometry).intersection(_make_valid(land_union_raster))
                                geom_landonly = _polygon_only(_make_valid(geom_landonly))
                                if geom_landonly is None or geom_landonly.is_empty:
                                    continue

                                mask_shapes = [geom_landonly]

                                buffer_area_class = f"A{int(row['Buffer_km2'])}"
                                output_dir = os.path.join(
                                    cfg.output_root, year, month_folder_name, region, buffer_area_class, scale_name
                                )
                                os.makedirs(output_dir, exist_ok=True)

                                out_csv = os.path.join(output_dir, f"{tile}_Buffer{idx}_{year}_{month_suffix}.csv")

                                # Clip burn date first
                                try:
                                    with rasterio.open(tif_burn_date) as src_burn:
                                        burn_date_clip, _ = mask(src_burn, mask_shapes, crop=True)
                                except ValueError:
                                    continue

                                bd = burn_date_clip[0]
                                valid = (bd >= 1) & (bd <= 366)
                                burned_pixels = int(valid.sum())

                                if burned_pixels == 0:
                                    zero_count += 1
                                    if cfg.write_zero_csvs:
                                        pd.DataFrame([{
                                            "Region": region,
                                            "AreaID": str(row["AreaID"]) if "AreaID" in row else f"Buffer{idx}",
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

                                        print(f"    wrote CSV (zero): {out_csv}")
                                        all_written.append(out_csv)
                                    continue

                                burned_area_km2 = burned_pixels * pixel_area_km2

                                # Clip uncertainty only if burns exist
                                try:
                                    with rasterio.open(tif_uncertainty) as src_unc:
                                        uncertainty_clip, _ = mask(src_unc, mask_shapes, crop=True)
                                except ValueError:
                                    bd_vals = bd[valid]
                                    burn_date_median_doy = float(np.nanmedian(bd_vals)) if bd_vals.size else None
                                    pd.DataFrame([{
                                        "Region": region,
                                        "AreaID": str(row["AreaID"]) if "AreaID" in row else f"Buffer{idx}",
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

                                    print(f"    wrote CSV: {out_csv}  (burned_pixels={burned_pixels}, no-uncertainty)")
                                    all_written.append(out_csv)
                                    saved_count += 1
                                    continue

                                bd_vals = bd[valid]
                                un_vals = uncertainty_clip[0][valid]

                                burn_date_median_doy = float(np.nanmedian(bd_vals)) if bd_vals.size else None
                                uncertainty_median = float(np.nanmedian(un_vals)) if un_vals.size else None

                                pd.DataFrame([{
                                    "Region": region,
                                    "AreaID": str(row["AreaID"]) if "AreaID" in row else f"Buffer{idx}",
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

                                print(f"    wrote CSV: {out_csv}  (burned_pixels={burned_pixels})")
                                all_written.append(out_csv)
                                saved_count += 1

                    print(f"--- Tile {tile} done: {saved_count} CSVs with burns, {zero_count} zero-burn buffers ---")

    print("\n================ SUMMARY ================")
    print(f"Total CSVs written: {len(all_written)}")
    if all_written:
        preview = min(15, len(all_written))
        print(f"First {preview} CSV paths:")
        for p in all_written[:preview]:
            print("  →", p)
    else:
        print("No CSVs were written.")
    print("========================================\nAll done.")

    return all_written


def main():
    # ---- EDIT DEFAULTS HERE (or later replace with argparse/config file) ----
    cfg = ExtractConfig(
        years=["2020"],
        month_suffixes=["03.01", "04.01", "05.01", "06.01", "07.01", "08.01", "09.01", "10.01"],
        region_meta={
            "Europe": (r"A:\Project BFA\Shapefiles\buffers_per_region\buffers_Europe_3035.shp", 3035),
            "Siberia": (r"A:\Project BFA\Shapefiles\buffers_per_region\buffers_Siberia_3576.shp", 3576),
            "NorthAmerica": (r"A:\Project BFA\Shapefiles\buffers_per_region\buffers_NorthAmerica_5070.shp", 5070),
        },
        buffer_scale_factors={
            "original": 1.00,
            "buffer_10pct": 1.10,
            "buffer_20pct": 1.20,
            "buffer_50pct": 1.50,
            "buffer_100pct": 2.00,
        },
        hdf_root=r"A:\Project BFA\hdf files",
        output_root=r"A:\Project BFA\output_csvs",
        land_path=r"A:\Project BFA\Shapefiles\Land_cover\ne_10m_land.shp",
        lakes_path=r"A:\Project BFA\Shapefiles\Land_cover\ne_10m_lakes.shp",
        write_zero_csvs=False,  # IMPORTANT: default = don't write zero-burn CSVs
    )

    os.makedirs(cfg.output_root, exist_ok=True)
    run_extraction(cfg)


if __name__ == "__main__":
    main()
