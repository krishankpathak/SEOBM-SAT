# src/parsers/parse_starlink.py
# =====================================================
# Layer-2 Starlink Geodetic → ECI Parser
# (Latitude, Longitude, Altitude, Epoch)
# =====================================================

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timezone

# -----------------------------------------------------
# Path setup
# -----------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
sys.path.append(PROJECT_ROOT)

BASE_DIR = os.path.join(
    PROJECT_ROOT, "data", "layer2_public", "starlink_kaggle"
)
RAW_DIR = os.path.join(BASE_DIR, "raw_csv")
OUT_DIR = os.path.join(BASE_DIR, "parsed_states")
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------------------------------
# Earth constants
# -----------------------------------------------------
R_EARTH = 6378.137          # km
OMEGA_EARTH = 7.2921159e-5  # rad/s

# -----------------------------------------------------
# Coordinate transforms
# -----------------------------------------------------
def geodetic_to_ecef(lat, lon, alt):
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)

    r = R_EARTH + alt

    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)

    return np.array([x, y, z])


def ecef_to_eci(r_ecef, epoch):
    # Convert time to seconds since midnight UTC
    t = epoch.timestamp()
    theta = OMEGA_EARTH * t

    R3 = np.array([
        [ np.cos(theta), -np.sin(theta), 0],
        [ np.sin(theta),  np.cos(theta), 0],
        [ 0,              0,             1]
    ])

    return R3 @ r_ecef

# -----------------------------------------------------
# Main
# -----------------------------------------------------
def main():
    csv_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise RuntimeError("No CSV files found in raw_csv.")

    for file in csv_files:
        print(f"\nParsing {file} ...")
        df = pd.read_csv(os.path.join(RAW_DIR, file))

        r_list = []

        for _, row in df.iterrows():
            try:
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
                alt = float(row["Altitude_km"])

                epoch = datetime.fromisoformat(row["Epoch"]).replace(tzinfo=timezone.utc)

                r_ecef = geodetic_to_ecef(lat, lon, alt)
                r_eci = ecef_to_eci(r_ecef, epoch)

                r_list.append(r_eci)

            except Exception:
                continue

        if not r_list:
            print("  ⚠ Skipped (no valid geodetic records)")
            continue

        r_stack = np.stack(r_list)

        np.savez(
            os.path.join(OUT_DIR, file.replace(".csv", ".npz")),
            r=r_stack
        )

        print(f"  ✅ Saved {len(r_stack)} ECI position vectors")

    print("\n✅ Layer-2 Starlink geodetic parsing COMPLETE.")

if __name__ == "__main__":
    main()
