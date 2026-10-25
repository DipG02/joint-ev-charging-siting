#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# Figure5_statewise_policy_plots.py
#
# Figure 5: Statewise policy decision framework from state-level optimizer runs.
#
# Reads outputs created by Main_statewiseruns.py:
#   Statewise_UnlimitedBudget_Results/
#       Results_state_TX_.../
#           run_metadata.csv
#           DWC_Selected_Segments.csv
#           SCS_Selected_Stations.csv
#           DWC_Selection_Master_Diagnostic.csv
#           ...
#       statewise_run_summary.csv
#
# Main outputs:
#   Figure5_StatewisePolicy/
#       Figure5_statewise_policy_framework.png
#       Figure5_statewise_policy_framework.pdf
#       statewise_policy_summary.csv
#
# Panels:
#   a. Statewise policy class map
#   b. DWC opportunity intensity map
#   c. SCS opportunity intensity map
#   d. Ranked state deployment composition
#   e. State feature space: traffic vs substation distance
#   f. Policy decision matrix: traffic × grid proximity
# ═══════════════════════════════════════════════════════════════════════════════

import os
import re
import glob
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

RESULTS_ROOT = "./Statewise_UnlimitedBudget_Results"

OUT_DIR = "./Figure5_StatewisePolicy"

# Optional: set to a Census / ArcGIS state shapefile path.
# Example:
# STATE_SHAPEFILE = "/path/to/cb_2023_us_state_500k.shp"
STATE_SHAPEFILE = None

# If using a shapefile, this should be the postal abbreviation column.
# Census shapefiles usually use "STUSPS".
SHAPE_STATE_ABBR_COL = "STUSPS"

# Keep Alaska/Hawaii out of map panels by default for contiguous-US paper figures.
EXCLUDE_MAP_STATES = {"AK", "HI", "PR", "VI", "GU", "MP", "AS"}

# Ranking panel shows top N states by total selected units.
TOP_N_STATES_BAR = 15

# Policy class thresholds.
# DWC share = N_DWC / (N_DWC + N_SCS)
DWC_PRIORITY_THRESHOLD = 0.67
SCS_PRIORITY_THRESHOLD = 0.33

# A state with extremely low selected density can be labelled limited opportunity.
# selected_density = (N_DWC + N_SCS) / number of candidate segments × 1000
LIMITED_DENSITY_THRESHOLD_PER_1000_SEGMENTS = 1.0

# Export settings.
SAVE_INDIVIDUAL_PANELS = True
DPI = 400


# ═══════════════════════════════════════════════════════════════════════════════
# STYLE
# ═══════════════════════════════════════════════════════════════════════════════

C_DWC = "#0B6B53"
C_DWC_LIGHT = "#7DBBA4"
C_SCS = "#C9A33A"
C_SCS_LIGHT = "#E7D38A"
C_HYBRID = "#5B8DB8"
C_LIMITED = "#D5D8DC"
C_ROAD = "#AEB6BF"
C_TEXT = "#111111"
C_MUTED = "#5F666D"
C_GRID = "#D9DEE4"
C_EDGE = "#222222"
C_BG = "#FFFFFF"

CLASS_COLORS = {
    "DWC-priority": C_DWC,
    "SCS-priority": C_SCS,
    "Hybrid": C_HYBRID,
    "Limited": C_LIMITED,
}

CLASS_ORDER = ["DWC-priority", "Hybrid", "SCS-priority", "Limited"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"],
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "axes.titleweight": "bold",
    "axes.edgecolor": C_EDGE,
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": C_BG,
    "axes.facecolor": C_BG,
    "savefig.facecolor": C_BG,
})


# ═══════════════════════════════════════════════════════════════════════════════
# STATE METADATA
# ═══════════════════════════════════════════════════════════════════════════════

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania",
    "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia",
}

# Approximate tile-map positions. Fallback when no shapefile is supplied.
STATE_TILE_POS = {
    "WA": (0, 0), "MT": (2, 0), "ND": (3, 0), "MN": (4, 0), "WI": (5, 0), "MI": (6, 0),
    "ME": (10, 0),

    "OR": (0, 1), "ID": (1, 1), "WY": (2, 1), "SD": (3, 1), "IA": (4, 1),
    "IL": (5, 1), "IN": (6, 1), "OH": (7, 1), "PA": (8, 1), "NY": (9, 1),
    "VT": (10, 1), "NH": (11, 1),

    "CA": (0, 2), "NV": (1, 2), "UT": (2, 2), "CO": (3, 2), "NE": (4, 2),
    "MO": (5, 2), "KY": (6, 2), "WV": (7, 2), "VA": (8, 2), "MD": (9, 2),
    "NJ": (10, 2), "MA": (11, 2),

    "AZ": (1, 3), "NM": (2, 3), "KS": (4, 3), "AR": (5, 3), "TN": (6, 3),
    "NC": (8, 3), "DE": (9, 3), "CT": (10, 3), "RI": (11, 3),

    "OK": (4, 4), "LA": (5, 4), "MS": (6, 4), "AL": (7, 4), "GA": (8, 4),
    "SC": (9, 4), "DC": (10, 4),

    "TX": (4, 5), "FL": (9, 5),
}


# ═══════════════════════════════════════════════════════════════════════════════
# FILE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _read_csv_if_exists(path):
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()


def _first_existing(results_dir, names):
    for name in names:
        p = os.path.join(results_dir, name)
        if os.path.exists(p):
            return p
    return None


def _extract_state_from_run_tag(run_tag):
    if not isinstance(run_tag, str):
        return None
    m = re.search(r"state_([A-Z]{2})_", run_tag)
    if m:
        return m.group(1)
    return None


def _safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except Exception:
        return default


def discover_result_dirs(results_root):
    dirs = sorted(glob.glob(os.path.join(results_root, "Results_state_*")))
    if not dirs:
        dirs = sorted(glob.glob(os.path.join(results_root, "Results_*")))
    dirs = [d for d in dirs if os.path.isdir(d)]
    if not dirs:
        raise FileNotFoundError(f"No Results_* folders found in {results_root}")
    return dirs


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING AND SUMMARY BUILDING
# ═══════════════════════════════════════════════════════════════════════════════

def summarize_one_state(results_dir):
    meta_path = os.path.join(results_dir, "run_metadata.csv")
    meta = _read_csv_if_exists(meta_path)

    if meta.empty:
        raise FileNotFoundError(f"Missing run_metadata.csv in {results_dir}")

    meta_row = meta.iloc[0].to_dict()
    run_tag = str(meta_row.get("run_tag", os.path.basename(results_dir)))
    state_abbr = _extract_state_from_run_tag(run_tag)

    if state_abbr is None:
        # Fallback: parse folder name.
        state_abbr = _extract_state_from_run_tag(os.path.basename(results_dir))

    if state_abbr is None:
        raise ValueError(f"Could not infer state abbreviation from run_tag/folder: {results_dir}")

    state_abbr = state_abbr.upper()
    state_name = STATE_NAMES.get(state_abbr, state_abbr)

    dwc_path = _first_existing(
        results_dir,
        ["DWC_Selected_Segments.csv", "DWC_Selectd_Segments.csv"],
    )
    scs_path = _first_existing(
        results_dir,
        ["SCS_Selected_Stations.csv", "SCS_Selected_Substations.csv"],
    )
    diag_path = _first_existing(
        results_dir,
        ["DWC_Selection_Master_Diagnostic.csv", "DWC_Selection_Master_Disgnostic.csv"],
    )

    df_dwc = pd.read_csv(dwc_path, low_memory=False) if dwc_path else pd.DataFrame()
    df_scs = pd.read_csv(scs_path, low_memory=False) if scs_path else pd.DataFrame()
    df_diag = pd.read_csv(diag_path, low_memory=False) if diag_path else pd.DataFrame()

    n_dwc = _safe_int(meta_row.get("n_dwc_selected", len(df_dwc)), len(df_dwc))
    n_scs = _safe_int(meta_row.get("n_scs_selected", len(df_scs)), len(df_scs))

    n_segments = len(df_diag) if not df_diag.empty else np.nan
    selected_total = n_dwc + n_scs
    dwc_share = n_dwc / selected_total if selected_total > 0 else 0.0
    scs_share = n_scs / selected_total if selected_total > 0 else 0.0

    selected_density = (
        selected_total / n_segments * 1000.0
        if n_segments and n_segments > 0
        else np.nan
    )
    dwc_density = (
        n_dwc / n_segments * 1000.0
        if n_segments and n_segments > 0
        else np.nan
    )
    scs_density = (
        n_scs / n_segments * 1000.0
        if n_segments and n_segments > 0
        else np.nan
    )

    # State-level feature medians.
    state_aadt_median = np.nan
    state_subdist_median = np.nan
    state_lambda_median = np.nan
    state_revenue_median = np.nan

    if not df_diag.empty:
        if "AADT_COM" in df_diag.columns:
            state_aadt_median = pd.to_numeric(df_diag["AADT_COM"], errors="coerce").median()
        if "Min_Sub_Dist" in df_diag.columns:
            state_subdist_median = pd.to_numeric(df_diag["Min_Sub_Dist"], errors="coerce").median()
        if "Lambda" in df_diag.columns:
            state_lambda_median = pd.to_numeric(df_diag["Lambda"], errors="coerce").median()
        if "Total_Revenue" in df_diag.columns:
            state_revenue_median = pd.to_numeric(df_diag["Total_Revenue"], errors="coerce").median()

    # Selected DWC feature medians.
    dwc_aadt_median = np.nan
    dwc_subdist_median = np.nan
    dwc_lambda_median = np.nan
    dwc_lcoe_median = np.nan
    dwc_e_del_total = np.nan
    dwc_capex_total = np.nan

    if not df_dwc.empty:
        if "AADT_COM" in df_dwc.columns:
            dwc_aadt_median = pd.to_numeric(df_dwc["AADT_COM"], errors="coerce").median()
        if "Lambda" in df_dwc.columns:
            dwc_lambda_median = pd.to_numeric(df_dwc["Lambda"], errors="coerce").median()
        if "LCOE" in df_dwc.columns:
            dwc_lcoe_median = pd.to_numeric(df_dwc["LCOE"], errors="coerce").median()
        if "E_del_DWC" in df_dwc.columns:
            dwc_e_del_total = pd.to_numeric(df_dwc["E_del_DWC"], errors="coerce").sum()
        if "Capex_Total" in df_dwc.columns:
            dwc_capex_total = pd.to_numeric(df_dwc["Capex_Total"], errors="coerce").sum()

        # DWC selected segments may not have Min_Sub_Dist. Merge from diagnostic.
        if "Segment_ID" in df_dwc.columns and not df_diag.empty and "Segment_ID" in df_diag.columns:
            if "Min_Sub_Dist" in df_diag.columns:
                m = df_dwc[["Segment_ID"]].merge(
                    df_diag[["Segment_ID", "Min_Sub_Dist"]],
                    on="Segment_ID",
                    how="left",
                )
                dwc_subdist_median = pd.to_numeric(m["Min_Sub_Dist"], errors="coerce").median()

    # Selected SCS feature medians.
    scs_aadt_median = np.nan
    scs_subdist_median = np.nan
    scs_lambda_median = np.nan
    scs_lcoe_median = np.nan
    scs_e_del_total = np.nan
    scs_capex_total = np.nan

    if not df_scs.empty:
        if "Lambda" in df_scs.columns:
            scs_lambda_median = pd.to_numeric(df_scs["Lambda"], errors="coerce").median()
        if "LCOE" in df_scs.columns:
            scs_lcoe_median = pd.to_numeric(df_scs["LCOE"], errors="coerce").median()
        if "E_del_SCS" in df_scs.columns:
            scs_e_del_total = pd.to_numeric(df_scs["E_del_SCS"], errors="coerce").sum()
        if "Capex_Total" in df_scs.columns:
            scs_capex_total = pd.to_numeric(df_scs["Capex_Total"], errors="coerce").sum()

        # Merge Host_Seg to diagnostic for AADT and substation distance.
        if "Host_Seg" in df_scs.columns and not df_diag.empty and "Segment_ID" in df_diag.columns:
            merge_cols = ["Segment_ID"]
            for c in ["AADT_COM", "Min_Sub_Dist"]:
                if c in df_diag.columns:
                    merge_cols.append(c)
            if len(merge_cols) > 1:
                m = df_scs[["Host_Seg"]].rename(columns={"Host_Seg": "Segment_ID"}).merge(
                    df_diag[merge_cols],
                    on="Segment_ID",
                    how="left",
                )
                if "AADT_COM" in m.columns:
                    scs_aadt_median = pd.to_numeric(m["AADT_COM"], errors="coerce").median()
                if "Min_Sub_Dist" in m.columns:
                    scs_subdist_median = pd.to_numeric(m["Min_Sub_Dist"], errors="coerce").median()

    if selected_total == 0 or (
        not pd.isna(selected_density)
        and selected_density < LIMITED_DENSITY_THRESHOLD_PER_1000_SEGMENTS
    ):
        policy_class = "Limited"
    elif dwc_share >= DWC_PRIORITY_THRESHOLD:
        policy_class = "DWC-priority"
    elif dwc_share <= SCS_PRIORITY_THRESHOLD:
        policy_class = "SCS-priority"
    else:
        policy_class = "Hybrid"

    row = {
        "state": state_abbr,
        "state_name": state_name,
        "run_tag": run_tag,
        "results_dir": results_dir,

        "n_segments": n_segments,
        "n_dwc": n_dwc,
        "n_scs": n_scs,
        "selected_total": selected_total,
        "dwc_share": dwc_share,
        "scs_share": scs_share,
        "selected_density_per_1000_segments": selected_density,
        "dwc_density_per_1000_segments": dwc_density,
        "scs_density_per_1000_segments": scs_density,
        "policy_class": policy_class,

        "obj_val": _safe_float(meta_row.get("obj_val", np.nan)),
        "project_life": _safe_float(meta_row.get("project_life", np.nan)),
        "g_ev": _safe_float(meta_row.get("g_ev", np.nan)),
        "g_aadt": _safe_float(meta_row.get("g_aadt", np.nan)),
        "eta_dwc": _safe_float(meta_row.get("eta_dwc", np.nan)),
        "eta_scs": _safe_float(meta_row.get("eta_scs", np.nan)),
        "p_param": _safe_float(meta_row.get("p_param", np.nan)),

        "total_rev_dwc": _safe_float(meta_row.get("total_rev_dwc", np.nan)),
        "total_rev_scs": _safe_float(meta_row.get("total_rev_scs", np.nan)),
        "total_opex_dwc": _safe_float(meta_row.get("total_opex_dwc", np.nan)),
        "total_opex_scs": _safe_float(meta_row.get("total_opex_scs", np.nan)),

        "state_aadt_median": state_aadt_median,
        "state_subdist_median": state_subdist_median,
        "state_lambda_median": state_lambda_median,
        "state_revenue_median": state_revenue_median,

        "dwc_aadt_median": dwc_aadt_median,
        "dwc_subdist_median": dwc_subdist_median,
        "dwc_lambda_median": dwc_lambda_median,
        "dwc_lcoe_median": dwc_lcoe_median,
        "dwc_e_del_total": dwc_e_del_total,
        "dwc_capex_total": dwc_capex_total,

        "scs_aadt_median": scs_aadt_median,
        "scs_subdist_median": scs_subdist_median,
        "scs_lambda_median": scs_lambda_median,
        "scs_lcoe_median": scs_lcoe_median,
        "scs_e_del_total": scs_e_del_total,
        "scs_capex_total": scs_capex_total,
    }

    return row


def build_statewise_summary(results_root):
    result_dirs = discover_result_dirs(results_root)

    rows = []
    failures = []

    for d in result_dirs:
        try:
            rows.append(summarize_one_state(d))
        except Exception as e:
            failures.append((d, str(e)))
            print(f"WARNING: failed to summarize {d}: {e}")

    if not rows:
        raise RuntimeError("No state results could be summarized.")

    df = pd.DataFrame(rows)

    # Remove non-map states from default plotting but keep them in saved CSV.
    df["include_in_contiguous_map"] = ~df["state"].isin(EXCLUDE_MAP_STATES)

    # Useful normalized columns.
    df["net_technology_preference"] = df["dwc_share"] - df["scs_share"]

    # Stable class order.
    df["policy_class"] = pd.Categorical(
        df["policy_class"],
        categories=CLASS_ORDER,
        ordered=True,
    )

    if failures:
        fail_path = os.path.join(OUT_DIR, "statewise_summary_failures.txt")
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(fail_path, "w") as f:
            for d, err in failures:
                f.write(f"{d}\n{err}\n\n")
        print(f"Summary failures written to: {fail_path}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAP HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def try_load_state_shapes():
    if STATE_SHAPEFILE is None:
        return None

    if not os.path.exists(STATE_SHAPEFILE):
        print(f"WARNING: STATE_SHAPEFILE not found: {STATE_SHAPEFILE}")
        return None

    try:
        import geopandas as gpd
    except Exception:
        print("WARNING: geopandas is not available. Using tile-map fallback.")
        return None

    gdf = gpd.read_file(STATE_SHAPEFILE)

    if SHAPE_STATE_ABBR_COL not in gdf.columns:
        print(f"WARNING: shapefile does not contain {SHAPE_STATE_ABBR_COL}.")
        print(f"Available columns: {list(gdf.columns)}")
        return None

    gdf[SHAPE_STATE_ABBR_COL] = gdf[SHAPE_STATE_ABBR_COL].astype(str).str.upper()
    gdf = gdf[~gdf[SHAPE_STATE_ABBR_COL].isin(EXCLUDE_MAP_STATES)].copy()

    # Project to a US-friendly projection if possible.
    try:
        gdf = gdf.to_crs("EPSG:5070")
    except Exception:
        pass

    return gdf


def draw_numeric_state_map(ax, df, value_col, title, cmap, gdf=None, vmin=None, vmax=None):
    plot_df = df[df["include_in_contiguous_map"]].copy()

    if vmin is None:
        vmin = np.nanmin(plot_df[value_col].values)
    if vmax is None:
        vmax = np.nanmax(plot_df[value_col].values)
    if not np.isfinite(vmin):
        vmin = 0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if gdf is not None:
        g = gdf.merge(
            plot_df[["state", value_col]],
            left_on=SHAPE_STATE_ABBR_COL,
            right_on="state",
            how="left",
        )
        g.plot(
            ax=ax,
            column=value_col,
            cmap=cmap,
            norm=norm,
            edgecolor="#FFFFFF",
            linewidth=0.45,
            missing_kwds={"color": "#F3F4F5"},
        )
        ax.set_axis_off()
        ax.set_title(title, loc="left", pad=2)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.030, pad=0.01)
        cbar.ax.tick_params(labelsize=6.5, length=2)
        return

    draw_numeric_tile_map(ax, plot_df, value_col, title, cmap, norm)


def draw_categorical_state_map(ax, df, title, gdf=None):
    plot_df = df[df["include_in_contiguous_map"]].copy()

    if gdf is not None:
        g = gdf.merge(
            plot_df[["state", "policy_class"]],
            left_on=SHAPE_STATE_ABBR_COL,
            right_on="state",
            how="left",
        )
        g["plot_color"] = g["policy_class"].map(CLASS_COLORS).fillna("#F3F4F5")
        g.plot(
            ax=ax,
            color=g["plot_color"],
            edgecolor="#FFFFFF",
            linewidth=0.45,
        )
        ax.set_axis_off()
        ax.set_title(title, loc="left", pad=2)

        handles = [
            patches.Patch(facecolor=CLASS_COLORS[k], edgecolor=C_EDGE, linewidth=0.4, label=k)
            for k in CLASS_ORDER
        ]
        ax.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=(0.0, -0.02),
            frameon=False,
            ncol=2,
            handlelength=1.0,
            columnspacing=0.8,
            labelspacing=0.3,
        )
        return

    draw_categorical_tile_map(ax, plot_df, title)


def draw_numeric_tile_map(ax, df, value_col, title, cmap, norm):
    ax.set_aspect("equal")
    ax.set_axis_off()

    for _, r in df.iterrows():
        st = r["state"]
        if st not in STATE_TILE_POS:
            continue
        x, y = STATE_TILE_POS[st]
        val = r[value_col]
        color = cmap(norm(val)) if pd.notna(val) else "#F3F4F5"

        rect = patches.Rectangle(
            (x, -y),
            0.92,
            0.92,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.46,
            -y + 0.46,
            st,
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color="#111111",
        )

    ax.set_xlim(-0.2, 12.1)
    ax.set_ylim(-6.1, 0.9)
    ax.set_title(title, loc="left", pad=2)

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
    cbar.ax.tick_params(labelsize=6.5, length=2)


def draw_categorical_tile_map(ax, df, title):
    ax.set_aspect("equal")
    ax.set_axis_off()

    for _, r in df.iterrows():
        st = r["state"]
        if st not in STATE_TILE_POS:
            continue
        x, y = STATE_TILE_POS[st]
        cls = str(r["policy_class"])
        color = CLASS_COLORS.get(cls, "#F3F4F5")

        rect = patches.Rectangle(
            (x, -y),
            0.92,
            0.92,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
        )
        ax.add_patch(rect)
        ax.text(
            x + 0.46,
            -y + 0.46,
            st,
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color="#111111",
        )

    ax.set_xlim(-0.2, 12.1)
    ax.set_ylim(-6.1, 0.9)
    ax.set_title(title, loc="left", pad=2)

    handles = [
        patches.Patch(facecolor=CLASS_COLORS[k], edgecolor=C_EDGE, linewidth=0.4, label=k)
        for k in CLASS_ORDER
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.03),
        frameon=False,
        ncol=2,
        handlelength=1.0,
        columnspacing=0.8,
        labelspacing=0.3,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# NON-MAP PANEL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def panel_ranked_deployment(ax, df):
    d = df[df["include_in_contiguous_map"]].copy()
    d = d.sort_values("selected_total", ascending=False).head(TOP_N_STATES_BAR)
    d = d.sort_values("selected_total", ascending=True)

    y = np.arange(len(d))
    ax.barh(
        y,
        d["n_scs"],
        color=C_SCS,
        edgecolor=C_EDGE,
        linewidth=0.4,
        label="SCS",
    )
    ax.barh(
        y,
        d["n_dwc"],
        left=d["n_scs"],
        color=C_DWC,
        edgecolor=C_EDGE,
        linewidth=0.4,
        label="DWC",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(d["state"])
    ax.set_xlabel("Selected units")
    ax.set_title("d. Highest latent deployment volumes", loc="left", pad=2)
    ax.grid(axis="x", color=C_GRID, linewidth=0.5)
    ax.legend(frameon=False, loc="lower right")

    for yi, (_, r) in zip(y, d.iterrows()):
        total = r["selected_total"]
        ax.text(
            total + max(d["selected_total"]) * 0.01,
            yi,
            f"{int(total)}",
            va="center",
            ha="left",
            fontsize=6.8,
        )


def panel_feature_space(ax, df):
    d = df[df["include_in_contiguous_map"]].copy()
    d = d.dropna(subset=["state_aadt_median", "state_subdist_median"])

    size_raw = d["selected_total"].clip(lower=1)
    if size_raw.max() > size_raw.min():
        sizes = 25 + 220 * (size_raw - size_raw.min()) / (size_raw.max() - size_raw.min())
    else:
        sizes = np.full(len(d), 80.0)

    for cls in CLASS_ORDER:
        sub = d[d["policy_class"].astype(str) == cls]
        if sub.empty:
            continue
        s_sub = sizes[sub.index]
        ax.scatter(
            sub["state_aadt_median"],
            sub["state_subdist_median"],
            s=s_sub,
            color=CLASS_COLORS[cls],
            edgecolor=C_EDGE,
            linewidth=0.4,
            alpha=0.88,
            label=cls,
        )

        # Label the most important states in each class by selected total.
        lab = sub.sort_values("selected_total", ascending=False).head(3)
        for _, r in lab.iterrows():
            ax.text(
                r["state_aadt_median"],
                r["state_subdist_median"],
                f" {r['state']}",
                fontsize=6.5,
                va="center",
                ha="left",
                color=C_TEXT,
            )

    ax.set_xlabel("Median commercial traffic, AADT")
    ax.set_ylabel("Median nearest-substation distance (mi)")
    ax.set_title("e. State feature space and technology preference", loc="left", pad=2)
    ax.grid(True, color=C_GRID, linewidth=0.5)

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.0f}")
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CLASS_COLORS[k],
            markeredgecolor=C_EDGE,
            markersize=5.5,
            label=k,
        )
        for k in CLASS_ORDER
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", ncol=1)


def _tertile_labels(series, low_label, mid_label, high_label):
    vals = pd.to_numeric(series, errors="coerce")
    if vals.dropna().nunique() < 3:
        return pd.Series(["Medium"] * len(vals), index=series.index)

    try:
        out = pd.qcut(vals, q=3, labels=[low_label, mid_label, high_label], duplicates="drop")
        return out.astype(str)
    except Exception:
        med = vals.median()
        return pd.Series(np.where(vals >= med, high_label, low_label), index=series.index)


def panel_policy_matrix(ax, df):
    d = df[df["include_in_contiguous_map"]].copy()
    d = d.dropna(subset=["state_aadt_median", "state_subdist_median", "dwc_share"])

    # Traffic: low to high.
    d["traffic_bin"] = _tertile_labels(
        d["state_aadt_median"],
        "Low traffic",
        "Medium traffic",
        "High traffic",
    )

    # Grid proximity: nearest-substation distance is inverted.
    # Small distance = near grid; large distance = far grid.
    d["grid_bin"] = _tertile_labels(
        d["state_subdist_median"],
        "Near grid",
        "Moderate grid",
        "Far grid",
    )

    traffic_order = ["Low traffic", "Medium traffic", "High traffic"]
    grid_order = ["Near grid", "Moderate grid", "Far grid"]

    mat = np.full((len(traffic_order), len(grid_order)), np.nan)
    count_mat = np.zeros_like(mat)

    for i, tb in enumerate(traffic_order):
        for j, gb in enumerate(grid_order):
            sub = d[(d["traffic_bin"] == tb) & (d["grid_bin"] == gb)]
            if not sub.empty:
                mat[i, j] = sub["dwc_share"].mean()
                count_mat[i, j] = len(sub)

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "scs_to_dwc",
        [C_SCS, "#F4F1DC", C_DWC],
    )
    im = ax.imshow(mat, vmin=0, vmax=1, cmap=cmap, aspect="auto")

    ax.set_xticks(np.arange(len(grid_order)))
    ax.set_xticklabels(grid_order, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(traffic_order)))
    ax.set_yticklabels(traffic_order)

    ax.set_title("f. Policy decision matrix", loc="left", pad=2)

    for i in range(len(traffic_order)):
        for j in range(len(grid_order)):
            if np.isfinite(mat[i, j]):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]*100:.0f}% DWC\nn={int(count_mat[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=C_TEXT,
                )
            else:
                ax.text(j, i, "No states", ha="center", va="center", fontsize=7, color=C_MUTED)

    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Mean DWC share", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, length=2)

    # Add directional labels.
    ax.text(
        0.5,
        -0.42,
        "Policy interpretation: DWC is strongest where long-run throughput can amortize corridor capital.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
        color=C_MUTED,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def make_figure5(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    gdf = try_load_state_shapes()

    # Numeric map ranges use shared max where appropriate.
    dmap = df[df["include_in_contiguous_map"]].copy()
    max_dwc_density = np.nanmax(dmap["dwc_density_per_1000_segments"].values)
    max_scs_density = np.nanmax(dmap["scs_density_per_1000_segments"].values)

    if not np.isfinite(max_dwc_density):
        max_dwc_density = 1
    if not np.isfinite(max_scs_density):
        max_scs_density = 1

    fig = plt.figure(figsize=(7.2, 7.6))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 1.05, 1.05],
        width_ratios=[1.0, 1.0],
        hspace=0.36,
        wspace=0.24,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])

    draw_categorical_state_map(
        ax_a,
        df,
        "a. Statewise policy class",
        gdf=gdf,
    )

    cmap_dwc = mpl.colors.LinearSegmentedColormap.from_list(
        "dwc_density",
        ["#F2F7F4", C_DWC_LIGHT, C_DWC],
    )
    cmap_scs = mpl.colors.LinearSegmentedColormap.from_list(
        "scs_density",
        ["#FAF7E8", C_SCS_LIGHT, C_SCS],
    )

    draw_numeric_state_map(
        ax_b,
        df,
        "dwc_density_per_1000_segments",
        "b. DWC opportunity intensity",
        cmap=cmap_dwc,
        gdf=gdf,
        vmin=0,
        vmax=max_dwc_density,
    )

    draw_numeric_state_map(
        ax_c,
        df,
        "scs_density_per_1000_segments",
        "c. SCS opportunity intensity",
        cmap=cmap_scs,
        gdf=gdf,
        vmin=0,
        vmax=max_scs_density,
    )

    panel_ranked_deployment(ax_d, df)
    panel_feature_space(ax_e, df)
    panel_policy_matrix(ax_f, df)

    fig.text(
        0.01,
        0.995,
        "Figure 5 | Statewise latent deployment preferences and policy decision rules",
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
        color=C_TEXT,
    )

    fig.text(
        0.01,
        0.972,
        "Unlimited-budget state runs isolate where DWC, SCS, or hybrid deployment is economically selected under common assumptions.",
        ha="left",
        va="top",
        fontsize=7.5,
        color=C_MUTED,
    )

    png_path = os.path.join(out_dir, "Figure5_statewise_policy_framework.png")
    pdf_path = os.path.join(out_dir, "Figure5_statewise_policy_framework.pdf")

    fig.savefig(png_path, dpi=DPI, bbox_inches="tight", facecolor=C_BG)
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def save_individual_panels(df, out_dir):
    if not SAVE_INDIVIDUAL_PANELS:
        return

    panel_dir = os.path.join(out_dir, "individual_panels")
    os.makedirs(panel_dir, exist_ok=True)

    gdf = try_load_state_shapes()
    dmap = df[df["include_in_contiguous_map"]].copy()

    cmap_dwc = mpl.colors.LinearSegmentedColormap.from_list(
        "dwc_density",
        ["#F2F7F4", C_DWC_LIGHT, C_DWC],
    )
    cmap_scs = mpl.colors.LinearSegmentedColormap.from_list(
        "scs_density",
        ["#FAF7E8", C_SCS_LIGHT, C_SCS],
    )

    panels = []

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    draw_categorical_state_map(ax, df, "a. Statewise policy class", gdf=gdf)
    panels.append((fig, "panel_a_policy_class"))

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    draw_numeric_state_map(
        ax, df, "dwc_density_per_1000_segments",
        "b. DWC opportunity intensity",
        cmap_dwc, gdf=gdf, vmin=0,
        vmax=np.nanmax(dmap["dwc_density_per_1000_segments"].values),
    )
    panels.append((fig, "panel_b_dwc_intensity"))

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    draw_numeric_state_map(
        ax, df, "scs_density_per_1000_segments",
        "c. SCS opportunity intensity",
        cmap_scs, gdf=gdf, vmin=0,
        vmax=np.nanmax(dmap["scs_density_per_1000_segments"].values),
    )
    panels.append((fig, "panel_c_scs_intensity"))

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    panel_ranked_deployment(ax, df)
    panels.append((fig, "panel_d_ranked_units"))

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    panel_feature_space(ax, df)
    panels.append((fig, "panel_e_feature_space"))

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    panel_policy_matrix(ax, df)
    panels.append((fig, "panel_f_policy_matrix"))

    for fig, name in panels:
        fig.savefig(os.path.join(panel_dir, f"{name}.png"), dpi=DPI, bbox_inches="tight", facecolor=C_BG)
        fig.savefig(os.path.join(panel_dir, f"{name}.pdf"), dpi=DPI, bbox_inches="tight", facecolor=C_BG)
        plt.close(fig)

    print(f"Individual panels saved to: {panel_dir}")


def print_quick_summary(df):
    print("\n" + "═" * 80)
    print("STATEWISE POLICY SUMMARY")
    print("═" * 80)

    class_counts = df["policy_class"].astype(str).value_counts().reindex(CLASS_ORDER).fillna(0).astype(int)
    print("\nPolicy classes:")
    print(class_counts.to_string())

    top_dwc = df.sort_values("n_dwc", ascending=False).head(10)[
        ["state", "n_dwc", "n_scs", "dwc_share", "selected_density_per_1000_segments"]
    ]
    print("\nTop DWC states:")
    print(top_dwc.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    top_scs = df.sort_values("n_scs", ascending=False).head(10)[
        ["state", "n_dwc", "n_scs", "dwc_share", "selected_density_per_1000_segments"]
    ]
    print("\nTop SCS states:")
    print(top_scs.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))

    print("═" * 80 + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 80)
    print("Figure 5 statewise policy plotting")
    print(f"Results root : {RESULTS_ROOT}")
    print(f"Output dir   : {OUT_DIR}")
    print(f"Shapefile    : {STATE_SHAPEFILE}")
    print("=" * 80)

    df = build_statewise_summary(RESULTS_ROOT)

    summary_path = os.path.join(OUT_DIR, "statewise_policy_summary.csv")
    df.to_csv(summary_path, index=False)
    print(f"Saved summary table: {summary_path}")

    print_quick_summary(df)

    make_figure5(df, OUT_DIR)
    save_individual_panels(df, OUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()