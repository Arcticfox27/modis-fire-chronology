#!/usr/bin/env bash
set -euo pipefail

# rename_laads_doy_to_month.sh
#
# Convert LAADS DOY folders (e.g. 2025/060) into
# calendar month folders expected by the pipeline (e.g. 2025/2025.03.01).
#
# Usage:
#   bash scripts/rename_laads_doy_to_month.sh 2025 /mnt/a/Project_BFA/hdf_files
#
# This is a RENAMING step (no data copied).
# Safe to re-run: existing target folders are skipped.

YEAR="${1:-}"
ROOT="${2:-}"

if [[ -z "${YEAR}" || -z "${ROOT}" ]]; then
  echo "Usage: $0 <YEAR> <ROOT>"
  echo "Example: $0 2025 /mnt/a/Project_BFA/hdf_files"
  exit 1
fi

YEAR_DIR="${ROOT}/${YEAR}"

if [[ ! -d "${YEAR_DIR}" ]]; then
  echo "Year directory not found: ${YEAR_DIR}"
  exit 1
fi

# Mapping: DOY -> MM
declare -A DOY2MONTH=(
  ["001"]="01"
  ["032"]="02"
  ["060"]="03"
  ["091"]="04"
  ["121"]="05"
  ["152"]="06"
  ["182"]="07"
  ["213"]="08"
  ["244"]="09"
  ["274"]="10"
  ["305"]="11"
  ["335"]="12"
)

echo "Renaming LAADS DOY folders for year ${YEAR}"
echo "Root: ${YEAR_DIR}"
echo

for doy in "${!DOY2MONTH[@]}"; do
  src="${YEAR_DIR}/${YEAR}.${doy}"
  month="${DOY2MONTH[$doy]}"
  dst="${YEAR_DIR}/${YEAR}.${month}.01"

  if [[ ! -d "${src}" ]]; then
    echo "Skip (missing): ${src}"
    continue
  fi

  if [[ -d "${dst}" ]]; then
    echo "Skip (already exists): ${dst}"
    continue
  fi

  echo "Rename: ${src} -> ${dst}"
  mv "${src}" "${dst}"
done

echo
echo "Done."
echo "You can now run the extractor without any changes."
