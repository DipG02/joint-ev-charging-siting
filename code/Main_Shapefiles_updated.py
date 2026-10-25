#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# Main_shapefiles.py
#
# ArcGIS-ready shapefile package generator for the DWC/SCS optimizer.
#
# Reads one saved Results_* directory from Main_optimizer.py / Main.py and writes
# all GIS layers into one folder, then creates a ZIP package that can be
# extracted and imported directly into ArcGIS Pro.
#
# Main outputs:
#   01_DWC_Selected_Segments_Lines.shp
#   02_DWC_Selected_Segments_Join_Table.csv
#   03_SCS_Selected_Stations.shp
#   04_Selected_Substations_All.shp
#   05_DWC_SubstationTaps.shp
#   06_SCS_SubstationTaps.shp
#   07_Charging_Substation_Taps_All.shp
#   08_DWC_AdjacencyLinks.shp
#   09_SCS_Existing_Stations.shp                  optional
#   10_DWC_Selected_Segments_Midpoints.shp         optional diagnostic
#   README_ArcGIS_Import.txt
#   shapefile_manifest.csv
#   ArcGIS_Shapefile_Package_<run_tag>.zip
#
# Key logic:
#   - Selected DWC segments are joined back to the original highway polyline
#     layer using optimizer Segment_ID == original SEGMENT_ID_OG.
#   - DWC segments are classified into:
#       Direct_Substation
#       Adjacent_Powered
#   - Selected substations are collapsed into one layer with:
#       tech_role = DWC, SCS, or DWC+SCS
# ═══════════════════════════════════════════════════════════════════════════════

import os
import re
import zipfile
import textwrap
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString

warnings.filterwarnings("ignore", category=FutureWarning)


# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Override these paths with environment variables or edit them locally.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.environ.get(
    "DWC_RESULTS_DIR",
    os.path.join(
        REPO_ROOT,
        "results",
        "Results_baseline_p200_etaDWC0.50_etaSCS0.50_gev15pct_gaadt011bps",
    ),
)

# Original ArcGIS file geodatabase containing the true highway segment polylines.
ORIGINAL_HIGHWAY_GDB = os.environ.get("DWC_HIGHWAY_GDB", "")

# Layer name inside the geodatabase.
ORIGINAL_HIGHWAY_LAYER = "NHS_Interstate_5_miles_FINAL_filteredSegments"

# CRS used by your projected source data.
CRS = "ESRI:102003"

# If True, also writes DWC selected segment midpoint points for diagnostics.
WRITE_DWC_MIDPOINT_DIAGNOSTIC = True

# If True, existing SCS points are written when available.
WRITE_EXISTING_SCS = True


# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_str_for_path(x):
    x = str(x)
    x = re.sub(r"[^A-Za-z0-9_.-]+", "_", x)
    return x.strip("_")


def inpath(fname):
    return os.path.join(RESULTS_DIR, fname)


def _read_csv_optional(fname, required=False):
    path = inpath(fname)
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"Required file missing: {path}")
        print(f"  [missing optional] {fname}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    print(f"  Loaded {fname:<40s} {len(df):>8,} rows")
    return df


def _clean_for_shapefile(gdf):
    """
    Clean field names and values for ESRI Shapefile compatibility.
    Shapefiles truncate field names to 10 characters, so we intentionally
    use short stable names for important fields.
    """
    if gdf is None or gdf.empty:
        return gdf

    gdf = gdf.copy()

    # Replace unsupported object values.
    for col in gdf.columns:
        if col == "geometry":
            continue

        if pd.api.types.is_bool_dtype(gdf[col]):
            gdf[col] = gdf[col].astype(int)

        elif pd.api.types.is_object_dtype(gdf[col]):
            gdf[col] = gdf[col].astype(str)
            gdf[col] = gdf[col].replace({"nan": "", "None": "", "<NA>": ""})

    return gdf


def save_shp(gdf, shp_dir, name):
    if gdf is None or gdf.empty:
        print(f"  [skip] {name} — no features")
        return None

    gdf = _clean_for_shapefile(gdf)

    path = os.path.join(shp_dir, name)
    gdf.to_file(path, driver="ESRI Shapefile", encoding="utf-8")

    # Write .cpg explicitly for ArcGIS.
    cpg_path = os.path.splitext(path)[0] + ".cpg"
    with open(cpg_path, "w", encoding="utf-8") as f:
        f.write("UTF-8")

    print(f"  Saved {name:<45s} {len(gdf):>8,} features")
    return path


def _valid_xy(x, y):
    return pd.notna(x) and pd.notna(y) and np.isfinite(float(x)) and np.isfinite(float(y))


def _make_point_gdf(df, x_col, y_col, crs=CRS):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return gpd.GeoDataFrame(columns=list(df.columns) + ["geometry"], geometry="geometry", crs=crs)

    out = df.copy()
    out["geometry"] = [
        Point(float(x), float(y)) if _valid_xy(x, y) else None
        for x, y in zip(out[x_col], out[y_col])
    ]
    out = out.dropna(subset=["geometry"])
    return gpd.GeoDataFrame(out, geometry="geometry", crs=crs)


def _make_line_gdf(df, crs=CRS):
    needed = ["FromX", "FromY", "ToX", "ToY"]
    if df.empty or any(c not in df.columns for c in needed):
        return gpd.GeoDataFrame(columns=list(df.columns) + ["geometry"], geometry="geometry", crs=crs)

    out = df.copy()
    geoms = []
    for _, r in out.iterrows():
        if _valid_xy(r["FromX"], r["FromY"]) and _valid_xy(r["ToX"], r["ToY"]):
            geoms.append(LineString([
                (float(r["FromX"]), float(r["FromY"])),
                (float(r["ToX"], float(r["ToY"])) if False else float(r["ToX"]), float(r["ToY"]))
            ]))
        else:
            geoms.append(None)

    out["geometry"] = geoms
    out = out.dropna(subset=["geometry"])
    return gpd.GeoDataFrame(out, geometry="geometry", crs=crs)


def _standardize_id_series(s):
    """
    Convert IDs to nullable integer when possible, then string-safe fallback.
    """
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def _first_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _shorten_columns_for_shp(gdf, rename_map):
    """
    Rename selected columns to stable <=10 character names.
    Keeps geometry unchanged.
    """
    existing = {k: v for k, v in rename_map.items() if k in gdf.columns}
    return gdf.rename(columns=existing)


# ═══════════════════════════════════════════════════════════════════════════════
# README, MANIFEST, ZIP
# ═══════════════════════════════════════════════════════════════════════════════

def write_readme_and_manifest(shp_dir, run_tag, results_dir, highway_layer):
    readme_path = os.path.join(shp_dir, "README_ArcGIS_Import.txt")

    readme = f"""
    ArcGIS import package for DWC/SCS optimizer run

    Run tag:
      {run_tag}

    Source results directory:
      {results_dir}

    Original highway polyline layer:
      {highway_layer}

    Recommended ArcGIS Pro workflow:
      1. Extract this ZIP folder.
      2. In ArcGIS Pro, use Add Data.
      3. Add the .shp files directly from the extracted folder.
      4. For selected DWC highway segments, use:
           01_DWC_Selected_Segments_Lines.shp
      5. Symbolize DWC selected segments by:
           dwc_conn
         where:
           Direct_Substation = DWC segment directly connected to a substation
           Adjacent_Powered  = DWC segment powered through an adjacent selected DWC segment
      6. Symbolize selected substations by:
           tech_role
         where:
           DWC, SCS, or DWC+SCS
      7. Use:
           07_Charging_Substation_Taps_All.shp
         to draw all substation-to-charger tap lines.
      8. Use:
           08_DWC_AdjacencyLinks.shp
         to draw segment-to-segment DWC power links.

    Main layers:
      01_DWC_Selected_Segments_Lines.shp
      03_SCS_Selected_Stations.shp
      04_Selected_Substations_All.shp
      07_Charging_Substation_Taps_All.shp
      08_DWC_AdjacencyLinks.shp

    Helper CSV:
      02_DWC_Selected_Segments_Join_Table.csv

    Notes:
      A shapefile is a group of files. Do not move only the .shp file without
      its .shx, .dbf, .prj, and .cpg sidecar files.
    """

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(textwrap.dedent(readme).strip() + "\n")

    manifest_rows = []
    for fname in sorted(os.listdir(shp_dir)):
        fpath = os.path.join(shp_dir, fname)
        if os.path.isfile(fpath):
            manifest_rows.append({
                "file_name": fname,
                "file_size_bytes": os.path.getsize(fpath)
            })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(os.path.join(shp_dir, "shapefile_manifest.csv"), index=False)


def zip_output_folder(shp_dir):
    zip_path = f"{shp_dir}.zip"

    if os.path.exists(zip_path):
        os.remove(zip_path)

    parent = os.path.dirname(shp_dir)
    if parent == "":
        parent = "."

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(shp_dir):
            for fname in files:
                full_path = os.path.join(root, fname)
                rel_path = os.path.relpath(full_path, start=parent)
                zf.write(full_path, rel_path)

    print(f"\nZIP package written:")
    print(f"  {zip_path}")

    return zip_path


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_dwc_join_table(df_dwc, run_tag):
    """
    Helper table that can be manually joined in ArcGIS:
      original highway SEGMENT_ID_OG == Segment_ID
    """
    if df_dwc.empty:
        return pd.DataFrame()

    out = df_dwc.copy()
    out["Segment_ID"] = _standardize_id_series(out["Segment_ID"])
    out["SEGMENT_ID_OG"] = out["Segment_ID"]
    out["dwc_selected"] = 1

    out["dwc_conn_type"] = np.where(
        out["Conn_Type"].astype(str).str.lower().eq("substation tap"),
        "Direct_Substation",
        np.where(
            out["Conn_Type"].astype(str).str.lower().eq("adjacent link"),
            "Adjacent_Powered",
            "Unknown"
        )
    )

    out["run_tag"] = run_tag

    keep_cols = [
        "Segment_ID",
        "SEGMENT_ID_OG",
        "dwc_selected",
        "dwc_conn_type",
        "Conn_Type",
        "From_ID",
        "AADT_COM",
        "MW_Load",
        "E_traff_DWC",
        "E_del_DWC",
        "Lambda",
        "LCOE",
        "Capex_Total",
        "Ann_Revenue",
        "Ann_Opex",
        "run_tag",
    ]

    keep_cols = [c for c in keep_cols if c in out.columns]
    return out[keep_cols].copy()


def load_original_highway_layer():
    if not ORIGINAL_HIGHWAY_GDB:
        raise ValueError(
            "Set DWC_HIGHWAY_GDB to the original ArcGIS file geodatabase path."
        )
    print("\nLoading original highway polyline layer:")
    print(f"  GDB   : {ORIGINAL_HIGHWAY_GDB}")
    print(f"  Layer : {ORIGINAL_HIGHWAY_LAYER}")

    gdf = gpd.read_file(ORIGINAL_HIGHWAY_GDB, layer=ORIGINAL_HIGHWAY_LAYER)

    if gdf.crs is None:
        print(f"  WARNING: original layer has no CRS. Assigning {CRS}.")
        gdf = gdf.set_crs(CRS)
    else:
        print(f"  Original CRS: {gdf.crs}")

    print(f"  Loaded original highway layer: {len(gdf):,} features")
    print(f"  Columns include: {list(gdf.columns)[:20]}{' ...' if len(gdf.columns) > 20 else ''}")

    return gdf


def build_dwc_selected_lines(gdf_highway, df_join, run_tag):
    """
    Join selected DWC table to true highway polyline geometry.
    """
    if df_join.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=gdf_highway.crs)

    id_candidates = [
        "SEGMENT_ID_OG",
        "Segment_ID_OG",
        "SEGMENT_ID",
        "Segment_ID",
        "OBJECTID",
        "FID",
    ]

    highway_id_col = _first_existing_col(gdf_highway, id_candidates)

    if highway_id_col is None:
        raise KeyError(
            "Could not find a segment ID field in the original highway layer. "
            f"Tried: {id_candidates}. Available columns: {list(gdf_highway.columns)}"
        )

    print(f"\nJoining selected DWC segments to original polylines:")
    print(f"  Highway ID field: {highway_id_col}")
    print(f"  Optimizer ID    : Segment_ID")

    hw = gdf_highway.copy()
    hw["_join_seg_id"] = _standardize_id_series(hw[highway_id_col])

    jt = df_join.copy()
    jt["_join_seg_id"] = _standardize_id_series(jt["Segment_ID"])

    # Remove duplicate non-geometry fields from highway if they conflict.
    join_cols = [c for c in jt.columns if c != "_join_seg_id"]

    merged = hw.merge(
        jt[join_cols + ["_join_seg_id"]],
        on="_join_seg_id",
        how="inner",
        suffixes=("", "_opt")
    )

    if merged.empty:
        raise ValueError(
            "No selected DWC segments matched the original highway polyline layer. "
            "Check whether original SEGMENT_ID_OG matches optimizer Segment_ID."
        )

    print(f"  Selected DWC rows          : {len(df_join):,}")
    print(f"  Matched highway polylines  : {len(merged):,}")

    unmatched = set(jt["_join_seg_id"].dropna().astype(int)) - set(merged["_join_seg_id"].dropna().astype(int))
    if unmatched:
        print(f"  WARNING: {len(unmatched):,} selected DWC Segment_ID values did not match original geometry.")
        print(f"  First few unmatched IDs: {sorted(list(unmatched))[:20]}")

    out = gpd.GeoDataFrame(merged, geometry="geometry", crs=gdf_highway.crs)

    # Stable short names for shapefile.
    out = _shorten_columns_for_shp(out, {
        "Segment_ID": "seg_id",
        "SEGMENT_ID_OG": "seg_id_og",
        "dwc_selected": "dwc_sel",
        "dwc_conn_type": "dwc_conn",
        "Conn_Type": "conn_type",
        "From_ID": "from_id",
        "AADT_COM": "aadt_com",
        "MW_Load": "mw_load",
        "E_traff_DWC": "e_traf",
        "E_del_DWC": "e_del",
        "Lambda": "lambda",
        "LCOE": "lcoe",
        "Capex_Total": "capex_tot",
        "Ann_Revenue": "ann_rev",
        "Ann_Opex": "ann_opex",
        "run_tag": "run_tag",
    })

    # Keep core fields plus any original useful geometry fields that survived.
    preferred = [
        "seg_id",
        "seg_id_og",
        "dwc_sel",
        "dwc_conn",
        "conn_type",
        "from_id",
        "aadt_com",
        "mw_load",
        "e_traf",
        "e_del",
        "lambda",
        "lcoe",
        "capex_tot",
        "ann_rev",
        "ann_opex",
        "run_tag",
        "geometry",
    ]

    existing = [c for c in preferred if c in out.columns]
    out = out[existing].copy()

    return out


def build_dwc_midpoint_points(df_dwc, run_tag):
    if df_dwc.empty or "MID_X" not in df_dwc.columns or "MID_Y" not in df_dwc.columns:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=CRS)

    out = df_dwc.copy()
    out["Segment_ID"] = _standardize_id_series(out["Segment_ID"])
    out["dwc_conn_type"] = np.where(
        out["Conn_Type"].astype(str).str.lower().eq("substation tap"),
        "Direct_Substation",
        np.where(
            out["Conn_Type"].astype(str).str.lower().eq("adjacent link"),
            "Adjacent_Powered",
            "Unknown"
        )
    )
    out["run_tag"] = run_tag

    gdf = _make_point_gdf(out, "MID_X", "MID_Y", crs=CRS)
    gdf = _shorten_columns_for_shp(gdf, {
        "Segment_ID": "seg_id",
        "dwc_conn_type": "dwc_conn",
        "Conn_Type": "conn_type",
        "From_ID": "from_id",
        "AADT_COM": "aadt_com",
        "MW_Load": "mw_load",
        "E_traff_DWC": "e_traf",
        "E_del_DWC": "e_del",
        "Lambda": "lambda",
        "LCOE": "lcoe",
        "Capex_Total": "capex_tot",
        "Ann_Revenue": "ann_rev",
        "Ann_Opex": "ann_opex",
        "MID_X": "mid_x",
        "MID_Y": "mid_y",
        "run_tag": "run_tag",
    })

    preferred = [
        "seg_id", "dwc_conn", "conn_type", "from_id", "aadt_com",
        "mw_load", "e_traf", "e_del", "lambda", "lcoe", "capex_tot",
        "ann_rev", "ann_opex", "mid_x", "mid_y", "run_tag", "geometry"
    ]
    return gdf[[c for c in preferred if c in gdf.columns]].copy()


def build_scs_selected_points(df_scs, run_tag):
    if df_scs.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=CRS)

    out = df_scs.copy()
    out["run_tag"] = run_tag
    gdf = _make_point_gdf(out, "MID_X", "MID_Y", crs=CRS)

    gdf = _shorten_columns_for_shp(gdf, {
        "SCS_ID": "scs_id",
        "Type": "type",
        "Sub_ID": "sub_id",
        "Dist_to_Sub_mi": "dist_mi",
        "Host_Seg": "host_seg",
        "E_traff_SCS": "e_traf",
        "E_del_SCS": "e_del",
        "Lambda": "lambda",
        "LCOE": "lcoe",
        "MID_X": "mid_x",
        "MID_Y": "mid_y",
        "Capex_Total": "capex_tot",
        "Ann_Revenue": "ann_rev",
        "Ann_Opex": "ann_opex",
        "run_tag": "run_tag",
    })

    preferred = [
        "scs_id", "type", "sub_id", "dist_mi", "host_seg", "e_traf",
        "e_del", "lambda", "lcoe", "mid_x", "mid_y", "capex_tot",
        "ann_rev", "ann_opex", "run_tag", "geometry"
    ]
    return gdf[[c for c in preferred if c in gdf.columns]].copy()


def build_selected_substations_all(df_dwc_sub, df_scs_sub, run_tag):
    """
    Collapse DWC and SCS selected substations into one point layer.
    If a substation appears for both technologies, label it DWC+SCS.
    """
    rows = []

    def _append_rows(df, tech):
        if df.empty:
            return

        sub_col = _first_existing_col(df, ["Sub_ID", "Sub_ID_OG", "From_ID", "NEAR_FID"])
        x_col = _first_existing_col(df, ["X", "MID_X", "FromX"])
        y_col = _first_existing_col(df, ["Y", "MID_Y", "FromY"])

        if sub_col is None or x_col is None or y_col is None:
            print(f"  WARNING: Cannot parse selected {tech} substations. Columns: {list(df.columns)}")
            return

        for _, r in df.iterrows():
            if not _valid_xy(r[x_col], r[y_col]):
                continue
            rows.append({
                "sub_id": int(r[sub_col]) if pd.notna(r[sub_col]) else None,
                "tech": tech,
                "x": float(r[x_col]),
                "y": float(r[y_col]),
            })

    _append_rows(df_dwc_sub, "DWC")
    _append_rows(df_scs_sub, "SCS")

    if not rows:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=CRS)

    raw = pd.DataFrame(rows).dropna(subset=["sub_id"])
    raw["sub_id"] = raw["sub_id"].astype(int)

    collapsed_rows = []
    for sub_id, grp in raw.groupby("sub_id"):
        techs = sorted(set(grp["tech"].astype(str)))
        is_dwc = int("DWC" in techs)
        is_scs = int("SCS" in techs)

        if is_dwc and is_scs:
            tech_role = "DWC+SCS"
        elif is_dwc:
            tech_role = "DWC"
        elif is_scs:
            tech_role = "SCS"
        else:
            tech_role = "Unknown"

        collapsed_rows.append({
            "sub_id": sub_id,
            "tech_role": tech_role,
            "is_dwc": is_dwc,
            "is_scs": is_scs,
            "x": float(grp["x"].iloc[0]),
            "y": float(grp["y"].iloc[0]),
            "run_tag": run_tag,
        })

    out = pd.DataFrame(collapsed_rows)
    gdf = _make_point_gdf(out, "x", "y", crs=CRS)
    return gdf[["sub_id", "tech_role", "is_dwc", "is_scs", "x", "y", "run_tag", "geometry"]].copy()


def build_tap_layers(df_conn, df_scs_conn, run_tag):
    """
    Returns:
      dwc_taps
      scs_taps
      all_taps
      dwc_adjacency
    """
    # DWC substation taps.
    if not df_conn.empty and "Conn_Type" in df_conn.columns:
        dwc_tap_rows = df_conn[df_conn["Conn_Type"].astype(str).eq("Substation Tap")].copy()
        dwc_adj_rows = df_conn[df_conn["Conn_Type"].astype(str).eq("Adjacent Link")].copy()
    else:
        dwc_tap_rows = pd.DataFrame()
        dwc_adj_rows = pd.DataFrame()

    if not dwc_tap_rows.empty:
        dwc_tap_rows["tech"] = "DWC"
        dwc_tap_rows["run_tag"] = run_tag

    if not dwc_adj_rows.empty:
        dwc_adj_rows["tech"] = "DWC"
        dwc_adj_rows["run_tag"] = run_tag

    dwc_taps = _make_line_gdf(dwc_tap_rows, crs=CRS)
    dwc_adj = _make_line_gdf(dwc_adj_rows, crs=CRS)

    # SCS taps.
    scs_tap_rows = df_scs_conn.copy() if not df_scs_conn.empty else pd.DataFrame()
    if not scs_tap_rows.empty:
        scs_tap_rows["tech"] = "SCS"
        scs_tap_rows["run_tag"] = run_tag
        if "Conn_Type" not in scs_tap_rows.columns:
            scs_tap_rows["Conn_Type"] = "SCS Tap"

    scs_taps = _make_line_gdf(scs_tap_rows, crs=CRS)

    # Combined taps.
    all_frames = []
    if not dwc_taps.empty:
        all_frames.append(pd.DataFrame(dwc_taps.drop(columns="geometry")).assign(
            geometry=dwc_taps.geometry
        ))
    if not scs_taps.empty:
        all_frames.append(pd.DataFrame(scs_taps.drop(columns="geometry")).assign(
            geometry=scs_taps.geometry
        ))

    if all_frames:
        all_taps_df = pd.concat(all_frames, ignore_index=True)
        all_taps = gpd.GeoDataFrame(all_taps_df, geometry="geometry", crs=CRS)
    else:
        all_taps = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=CRS)

    def _format_line_gdf(gdf):
        if gdf.empty:
            return gdf

        gdf = _shorten_columns_for_shp(gdf, {
            "Conn_Type": "conn_type",
            "From_Type": "from_type",
            "From_ID": "from_id",
            "To_ID": "to_id",
            "FromX": "from_x",
            "FromY": "from_y",
            "ToX": "to_x",
            "ToY": "to_y",
            "Dist_Miles": "dist_mi",
            "tech": "tech",
            "run_tag": "run_tag",
        })

        preferred = [
            "tech", "conn_type", "from_type", "from_id", "to_id",
            "dist_mi", "from_x", "from_y", "to_x", "to_y",
            "run_tag", "geometry"
        ]
        return gdf[[c for c in preferred if c in gdf.columns]].copy()

    return (
        _format_line_gdf(dwc_taps),
        _format_line_gdf(scs_taps),
        _format_line_gdf(all_taps),
        _format_line_gdf(dwc_adj),
    )


def build_existing_scs_points(df_exist, run_tag):
    if df_exist.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=CRS)

    x_col = _first_existing_col(df_exist, ["MID_X", "X"])
    y_col = _first_existing_col(df_exist, ["MID_Y", "Y"])

    if x_col is None or y_col is None:
        print(f"  WARNING: Existing SCS file has no coordinate columns. Columns: {list(df_exist.columns)}")
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=CRS)

    out = df_exist.copy()
    out["run_tag"] = run_tag
    gdf = _make_point_gdf(out, x_col, y_col, crs=CRS)

    gdf = _shorten_columns_for_shp(gdf, {
        "SCS_ID": "scs_id",
        "EXISTING_SCS_ID": "exist_id",
        "EXISTING_SCS_DIST": "exist_dst",
        "MID_X": "mid_x",
        "MID_Y": "mid_y",
        "X": "x",
        "Y": "y",
        "run_tag": "run_tag",
    })

    # Keep all columns if uncertain, but geometry last.
    cols = [c for c in gdf.columns if c != "geometry"] + ["geometry"]
    return gdf[cols].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  DWC/SCS ArcGIS Shapefile Package Generator")
    print("=" * 80)

    if not os.path.isdir(RESULTS_DIR):
        raise NotADirectoryError(f"RESULTS_DIR does not exist: {RESULTS_DIR}")

    print(f"\nReading optimizer outputs from:")
    print(f"  {RESULTS_DIR}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = _read_csv_optional("run_metadata.csv", required=True).iloc[0]

    run_tag = str(meta.get("run_tag", os.path.basename(RESULTS_DIR).replace("Results_", "")))
    run_tag_safe = _safe_str_for_path(run_tag)

    use_existing_scs = bool(meta.get("use_existing_scs", False))
    eta_dwc = float(meta.get("eta_dwc", np.nan))
    eta_scs = float(meta.get("eta_scs", np.nan))
    p_param = int(float(meta.get("p_param", -1)))

    base_parent = os.path.dirname(os.path.abspath(RESULTS_DIR))
    shp_dir = os.path.join(base_parent, f"ArcGIS_Shapefile_Package_{run_tag_safe}")
    os.makedirs(shp_dir, exist_ok=True)

    print(f"\nRun metadata:")
    print(f"  run_tag          : {run_tag}")
    print(f"  p_param          : {p_param}")
    print(f"  eta_dwc          : {eta_dwc}")
    print(f"  eta_scs          : {eta_scs}")
    print(f"  use_existing_scs : {use_existing_scs}")
    print(f"\nWriting all GIS files to:")
    print(f"  {shp_dir}")

    # ── Load optimizer CSVs ──────────────────────────────────────────────────
    print("\nLoading optimizer CSV outputs:")
    df_dwc = _read_csv_optional("DWC_Selected_Segments.csv")
    df_scs = _read_csv_optional("SCS_Selected_Stations.csv")
    df_dwc_sub = _read_csv_optional("DWC_Selected_Substations.csv")
    df_scs_sub = _read_csv_optional("SCS_Selected_Substations.csv")
    df_conn = _read_csv_optional("DWC_Network_FromTo_Table.csv")
    df_scs_conn = _read_csv_optional("SCS_Network_FromTo_Table.csv")

    # Existing SCS can be in either file depending on what you want.
    # Prefer SCS_Existing_Stations.csv if available; fallback to SCS_Exist_Coords.csv.
    df_exist = pd.DataFrame()
    if WRITE_EXISTING_SCS and use_existing_scs:
        if os.path.exists(inpath("SCS_Existing_Stations.csv")):
            df_exist = _read_csv_optional("SCS_Existing_Stations.csv")
        elif os.path.exists(inpath("SCS_Exist_Coords.csv")):
            df_exist = _read_csv_optional("SCS_Exist_Coords.csv")
        else:
            print("  [missing optional] SCS_Existing_Stations.csv / SCS_Exist_Coords.csv")

    # ── 1 and 2. DWC selected lines and helper join table ─────────────────────
    print("\nGenerating GIS layers:")

    df_dwc_join = build_dwc_join_table(df_dwc, run_tag)
    join_csv_path = os.path.join(shp_dir, "02_DWC_Selected_Segments_Join_Table.csv")
    df_dwc_join.to_csv(join_csv_path, index=False)
    print(f"  Saved 02_DWC_Selected_Segments_Join_Table.csv {len(df_dwc_join):>8,} rows")

    if not df_dwc_join.empty:
        gdf_highway = load_original_highway_layer()
        gdf_dwc_lines = build_dwc_selected_lines(gdf_highway, df_dwc_join, run_tag)

        # Force output to target CRS if needed.
        if gdf_dwc_lines.crs is not None and str(gdf_dwc_lines.crs) != CRS:
            try:
                gdf_dwc_lines = gdf_dwc_lines.to_crs(CRS)
            except Exception:
                print("  WARNING: could not reproject DWC lines; keeping original CRS.")

        save_shp(gdf_dwc_lines, shp_dir, "01_DWC_Selected_Segments_Lines.shp")
    else:
        print("  [skip] 01_DWC_Selected_Segments_Lines.shp — no selected DWC rows")

    # Optional midpoint diagnostic.
    if WRITE_DWC_MIDPOINT_DIAGNOSTIC:
        gdf_dwc_mid = build_dwc_midpoint_points(df_dwc, run_tag)
        save_shp(gdf_dwc_mid, shp_dir, "10_DWC_Selected_Segments_Midpoints.shp")

    # ── 3. SCS selected sites ─────────────────────────────────────────────────
    gdf_scs = build_scs_selected_points(df_scs, run_tag)
    save_shp(gdf_scs, shp_dir, "03_SCS_Selected_Stations.shp")

    # ── 4. Selected substations all, with DWC/SCS labels ──────────────────────
    gdf_subs_all = build_selected_substations_all(df_dwc_sub, df_scs_sub, run_tag)
    save_shp(gdf_subs_all, shp_dir, "04_Selected_Substations_All.shp")

    # ── 5, 6, 7, 8. Tap and adjacency line layers ─────────────────────────────
    gdf_dwc_taps, gdf_scs_taps, gdf_all_taps, gdf_dwc_adj = build_tap_layers(
        df_conn=df_conn,
        df_scs_conn=df_scs_conn,
        run_tag=run_tag
    )

    save_shp(gdf_dwc_taps, shp_dir, "05_DWC_SubstationTaps.shp")
    save_shp(gdf_scs_taps, shp_dir, "06_SCS_SubstationTaps.shp")
    save_shp(gdf_all_taps, shp_dir, "07_Charging_Substation_Taps_All.shp")
    save_shp(gdf_dwc_adj, shp_dir, "08_DWC_AdjacencyLinks.shp")

    # ── 9. Existing SCS points, optional ──────────────────────────────────────
    if WRITE_EXISTING_SCS and use_existing_scs:
        gdf_exist = build_existing_scs_points(df_exist, run_tag)
        save_shp(gdf_exist, shp_dir, "09_SCS_Existing_Stations.shp")

    # ── README, manifest, ZIP ─────────────────────────────────────────────────
    write_readme_and_manifest(
        shp_dir=shp_dir,
        run_tag=run_tag,
        results_dir=RESULTS_DIR,
        highway_layer=f"{ORIGINAL_HIGHWAY_GDB} | {ORIGINAL_HIGHWAY_LAYER}"
    )

    zip_path = zip_output_folder(shp_dir)

    print("\n" + "=" * 80)
    print("  DONE")
    print("=" * 80)
    print(f"\nAll GIS files written to:")
    print(f"  {shp_dir}")
    print(f"\nArcGIS-ready ZIP package:")
    print(f"  {zip_path}")
    print("\nRecommended ArcGIS layers:")
    print("  01_DWC_Selected_Segments_Lines.shp")
    print("  03_SCS_Selected_Stations.shp")
    print("  04_Selected_Substations_All.shp")
    print("  07_Charging_Substation_Taps_All.shp")
    print("  08_DWC_AdjacencyLinks.shp")


if __name__ == "__main__":
    main()
