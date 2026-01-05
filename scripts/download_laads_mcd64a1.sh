#!/usr/bin/env bash
set -euo pipefail

# download_laads_mcd64a1.sh
#
# Download MODIS MCD64A1.061 monthly HDFs for a given year from LAADS DAAC.
# LAADS layout uses DOY (day-of-year) directories (e.g., 001, 032, 060...).
# This script downloads only those month-start DOY folders and only *.hdf files.
#
# Requirements:
#   - curl
#   - python3
#
# Auth:
#   Export a LAADS/Earthdata token in your shell (recommended):
#     export LAADS_TOKEN="eyJ...."
#
# Usage:
#   ./scripts/download_laads_mcd64a1.sh 2025 "/mnt/a/Project_BFA/hdf_files"
#
# Optional:
#   Restrict to a subset of months (by DOY folders) by editing DOYS below.

YEAR="${1:-}"
DEST_ROOT="${2:-}"

if [[ -z "${YEAR}" || -z "${DEST_ROOT}" ]]; then
  echo "Usage: $0 <YEAR> <DEST_ROOT>"
  echo "Example: $0 2025 /mnt/a/Project_BFA/hdf_files"
  exit 1
fi

: "${LAADS_TOKEN:?Environment variable LAADS_TOKEN is not set. Export it first.}"

PRODUCT="MCD64A1"
COLLECTION="61"
BASE_URL="https://ladsweb.modaps.eosdis.nasa.gov/archive/allData/${COLLECTION}/${PRODUCT}/${YEAR}"

# Month-start DOY folders for a non-leap year (common LAADS monthly layout):
# Jan 01=001, Feb 01=032, Mar 01=060, Apr 01=091, May 01=121, Jun 01=152,
# Jul 01=182, Aug 01=213, Sep 01=244, Oct 01=274, Nov 01=305, Dec 01=335
#
# For leap years, some products may still use these starts; if a folder is missing,
# the script will warn and continue.
#
# For March–October only, use:
# DOYS=(060 091 121 152 182 213 244 274)
DOYS=(001 032 060 091 121 152 182 213 244 274 305 335)

mkdir -p "${DEST_ROOT}/${YEAR}"

echo "Downloading ${PRODUCT}.${COLLECTION} for year ${YEAR}"
echo "Destination: ${DEST_ROOT}/${YEAR}"
echo "Base URL: ${BASE_URL}"
echo

download_doy () {
  local doy="$1"
  local outdir="${DEST_ROOT}/${YEAR}/${YEAR}.${doy}"
  local url="${BASE_URL}/${doy}/"

  mkdir -p "${outdir}"

  echo "== DOY ${doy} =="
  echo "URL: ${url}"
  echo "OUT: ${outdir}"

  # Fetch JSON listing (fail gracefully if folder doesn't exist)
  local json
  if ! json="$(curl -fsSL -H "Authorization: Bearer ${LAADS_TOKEN}" "${url}.json")"; then
    echo "  WARNING: ${url}.json not found or not accessible (skipping DOY ${doy})"
    echo
    return 0
  fi

  # Extract .hdf filenames from JSON
  local files
  files="$(python3 -c '
import json, sys
j = json.load(sys.stdin)
out=[]
for it in j.get("content", []):
    name = str(it.get("name",""))
    size = int(it.get("size", 0) or 0)
    if size > 0 and name.lower().endswith(".hdf"):
        out.append(name)
sys.stdout.write("\n".join(out))
' <<< "${json}")"

  local n
  n="$(printf "%s\n" "${files}" | sed "/^$/d" | wc -l | tr -d " ")"
  if [[ "${n}" == "0" ]]; then
    echo "  No .hdf files listed for DOY ${doy}"
    echo
    return 0
  fi

  echo "  Found ${n} HDF files"

  # Download each file (resume supported). Continue even if one file fails.
  while IFS= read -r f; do
    [[ -z "${f}" ]] && continue

    if [[ -f "${outdir}/${f}" ]]; then
      echo "  Skip: ${f}"
      continue
    fi

    echo "  Get : ${f}"

    # Download to temp first, then atomically move into place.
    # Retry handles transient 502/503/504 and flaky network.
    local tmp="${outdir}/${f}.part"
    if ! curl -fL --progress-bar \
          --retry 8 --retry-all-errors --retry-delay 2 \
          --connect-timeout 30 --max-time 0 \
          -H "Authorization: Bearer ${LAADS_TOKEN}" -C - \
          "${url}${f}" -o "${tmp}"; then
      echo "  WARNING: failed download after retries: ${f}"
      rm -f "${tmp}" 2>/dev/null || true
      continue
    fi

    mv -f "${tmp}" "${outdir}/${f}"
  done <<< "${files}"

  echo
}

for doy in "${DOYS[@]}"; do
  download_doy "${doy}"
done

echo "Done."
echo "Downloaded folders are stored as: ${DEST_ROOT}/${YEAR}/${YEAR}.<DOY>/"
echo "You can symlink or map these to your extractor's expected month naming if needed."
