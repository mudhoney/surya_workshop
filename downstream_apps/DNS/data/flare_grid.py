#!/usr/bin/env python3
"""
flare_grid.py
-------------
Read a CSV whose 4th column contains a solar-flare heliographic location in
the "S23E54" format, work out which cell of an 8x8 heliographic grid the flare
falls in, and write a new CSV with an extra "cell_number" column.

Grid definition (matches the diagrams):
  - Longitude runs -90 (E limb) .. +90 (W limb), split into 8 columns of 22.5 deg.
  - Latitude  runs +90 (N pole) .. -90 (S pole), split into 8 rows of 22.5 deg.
  - Cells are numbered 1..64, left-to-right then top-to-bottom, from top-left.
  - cell_number = (row-1)*8 + column, with row 1 = northernmost, column 1 = easternmost.

Location format: <NS><lat><EW><lon>, e.g. S23E54, N05W12, S00E00.
  N -> +latitude, S -> -latitude, E -> -longitude, W -> +longitude.

Usage:
  python3 flare_grid.py input.csv output.csv
  python3 flare_grid.py input.csv output.csv --loc-col 4 --has-header
"""

import argparse
import csv
import re
import sys

# ---- coordinate parsing -----------------------------------------------------

# Matches e.g. S23E54, N5W120, s07e004 ; captures hemisphere letters + numbers.
_LOC_RE = re.compile(r'^\s*([NS])\s*(\d{1,2})\s*([EW])\s*(\d{1,3})\s*$', re.IGNORECASE)


def parse_location(text):
    """Return (lat, lon) in degrees, or None if the string is not parseable.

    lat: + north, - south      lon: + west, - east
    """
    if text is None:
        return None
    m = _LOC_RE.match(str(text))
    if not m:
        return None
    ns, lat_s, ew, lon_s = m.groups()
    lat = int(lat_s)
    lon = int(lon_s)
    if ns.upper() == 'S':
        lat = -lat
    if ew.upper() == 'E':
        lon = -lon
    # Reject values clearly off the disk, but tolerate limb events reported
    # up to 1 deg past +/-90 (common when an AR is just rounding the limb):
    # those get clamped onto the limb column/row rather than dropped.
    if not (-91 <= lat <= 91) or not (-91 <= lon <= 91):
        return None
    lat = max(-90, min(90, lat))
    lon = max(-90, min(90, lon))
    return lat, lon


# ---- grid mapping -----------------------------------------------------------

STEP = 180.0 / 8.0  # 22.5 deg per cell


def latlon_to_cell(lat, lon):
    """Map (lat, lon) to a cell number 1..64.

    Column: 0..7 from lon -90 (E) to +90 (W).
    Row:    0..7 from lat +90 (N) to -90 (S).
    Points exactly on the max edge (+90) are clamped into the last cell.
    """
    col = int((lon + 90.0) // STEP)          # 0..8  (8 only when lon == +90)
    row = int((90.0 - lat) // STEP)          # 0..8  (8 only when lat == -90)
    col = min(max(col, 0), 7)
    row = min(max(row, 0), 7)
    return row * 8 + col + 1


def location_to_cell(text):
    """Convenience: raw 'S23E54' string -> cell number, or None."""
    parsed = parse_location(text)
    if parsed is None:
        return None
    return latlon_to_cell(*parsed)


# ---- CSV driver -------------------------------------------------------------

def process(in_path, out_path, loc_col=4, has_header=True,
            new_col_name="cell_number"):
    """Read in_path, append the cell number, write out_path.

    loc_col is 1-based (column 4 by default).
    """
    idx = loc_col - 1
    n_ok = n_bad = 0

    with open(in_path, newline='', encoding='utf-8-sig') as fin, \
         open(out_path, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)

        rows = iter(reader)

        if has_header:
            try:
                header = next(rows)
            except StopIteration:
                print("Input CSV is empty.", file=sys.stderr)
                return
            writer.writerow(header + [new_col_name])

        for row in rows:
            if idx >= len(row):
                # location column missing on this row
                writer.writerow(row + [""])
                n_bad += 1
                continue
            cell = location_to_cell(row[idx])
            if cell is None:
                writer.writerow(row + [""])   # leave blank for unparseable
                n_bad += 1
            else:
                writer.writerow(row + [cell])
                n_ok += 1

    print(f"Done. {n_ok} rows mapped, {n_bad} rows left blank -> {out_path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="input CSV path")
    p.add_argument("output", help="output CSV path")
    p.add_argument("--loc-col", type=int, default=4,
                   help="1-based index of the location column (default 4)")
    p.add_argument("--no-header", dest="has_header", action="store_false",
                   help="set if the CSV has no header row")
    p.add_argument("--col-name", default="cell_number",
                   help="name of the appended column (default 'cell_number')")
    args = p.parse_args(argv)

    process(args.input, args.output, loc_col=args.loc_col,
            has_header=args.has_header, new_col_name=args.col_name)


if __name__ == "__main__":
    main()