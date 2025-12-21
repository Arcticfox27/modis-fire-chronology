# MODIS Fire Chronology (MCD64A1)

Pipeline to extract MODIS MCD64A1 burned area chronology per buffer
(multiple regions and buffer scales), and aggregate results into
yearly master tables.

## Structure

- `scripts/extract_mcd64a1.py` — Extract per-buffer CSVs from MCD64A1 HDF tiles  
- `scripts/aggregate_fire_chronology.py` — Aggregate per-buffer CSVs into yearly master outputs  

## Configuration

All input and output paths are defined directly inside the scripts.
Edit the following sections before running:

- MODIS HDF root directory
- Buffer shapefile paths
- Natural Earth land/lakes shapefiles
- Output directory

No data are stored in this repository.

## Run (Windows)

```bat
python scripts\extract_mcd64a1.py
python scripts\aggregate_fire_chronology.py




