# MODIS Fire Chronology (MCD64A1)

Pipeline to extract MODIS MCD64A1 burned area chronology per buffer
(multiple regions and buffer scales), and to aggregate results into
yearly master tables suitable for further analysis. This code was 
used for extraction of fire history for 25 years for 22 research
sites of DELA (Dendrochronological Laboratory of SLU Alnarp), Sweden
across North America, Europe and Russia from 2001 to 2025.

## Structure

- `scripts/extract_mcd64a1.py` — Extract per-buffer fire chronology CSVs from MODIS MCD64A1 HDF tiles  
- `scripts/aggregate_fire_chronology.py` — Aggregate per-buffer CSVs into yearly master tables  

## Required external data

This repository contains **code only**. The following datasets must be provided by the user and are **not included**:

- **MODIS MCD64A1 burned area products** (HDF format)  
  Organized by year and acquisition date.

- **Region-specific buffer shapefiles**  
  One shapefile per region, containing a `Buffer_km2` attribute and (optionally) a site identifier
  (e.g. `AreaID`, `Name`, `Site`).

- **Land–water masks (Natural Earth)**  
  - `ne_10m_land.shp`  
  - `ne_10m_lakes.shp`

- **MODIS tile grid shapefile**  
  Used for tile identification and quality control.

## Configuration

All input and output paths are defined directly inside the scripts.
Edit the configuration sections at the top of each script before running:

- MODIS HDF root directory
- Buffer shapefile paths
- Natural Earth land/lakes shapefiles
- Output directory

No input data or generated outputs are stored in this repository.

## Notes

This pipeline is actively maintained and may be updated if bugs are identified or new requirements emerge.





