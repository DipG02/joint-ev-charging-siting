#!/usr/bin/env python3
"""Verify the curated p=200 baseline case-study outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS = ROOT / "plots"


def close(actual: float, expected: float, tolerance: float = 1e-6) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"expected {expected}, found {actual}")


def main() -> None:
    dwc = pd.read_csv(RESULTS / "DWC_Selected_Segments.csv")
    scs = pd.read_csv(RESULTS / "SCS_Selected_Stations.csv")
    dwc_links = pd.read_csv(RESULTS / "DWC_Network_FromTo_Table.csv")
    scs_links = pd.read_csv(RESULTS / "SCS_Network_FromTo_Table.csv")
    dwc_subs = pd.read_csv(RESULTS / "DWC_Selected_Substations.csv")
    scs_subs = pd.read_csv(RESULTS / "SCS_Selected_Substations.csv")
    metadata = pd.read_csv(RESULTS / "run_metadata.csv").iloc[0]

    assert len(dwc) == 96
    assert len(scs) == 104
    assert len(dwc_links) == 96
    assert len(scs_links) == 104
    assert len(dwc_subs) == 64
    assert len(scs_subs) == 103
    assert int(metadata["p_param"]) == 200
    assert int(metadata["n_dwc_selected"]) == 96
    assert int(metadata["n_scs_selected"]) == 104

    connection_counts = dwc["Conn_Type"].value_counts().to_dict()
    assert connection_counts == {"Substation Tap": 64, "Adjacent Link": 32}

    close(
        float(dwc_links.loc[
            dwc_links["Conn_Type"] == "Substation Tap", "Dist_Miles"
        ].sum()),
        32.74850968446185,
    )
    close(float(scs_links["Dist_Miles"].sum()), 61.11237707986824)
    close(float(metadata["obj_val"]), 2_858_008_983.020651, tolerance=0.01)

    expected_plots = {
        "map_proposed.png",
        "total_units.png",
        "traffic_split.png",
    }
    actual_plots = {path.name for path in PLOTS.glob("*.png") if path.stat().st_size}
    missing = expected_plots - actual_plots
    if missing:
        raise AssertionError(f"missing or empty plot(s): {sorted(missing)}")

    print("Baseline p=200 case study verified.")
    print("  DWC/SCS selections: 96/104")
    print("  DWC direct/adjacent: 64/32")
    print("  Direct feeder miles (DWC/SCS): 32.75/61.11")
    print("  Lifetime portfolio NPV: $2.858 billion")


if __name__ == "__main__":
    main()
