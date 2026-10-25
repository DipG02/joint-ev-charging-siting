#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# Policy_Insight_Analysis.py
#
# Portfolio-centered post-processing for the DWC + SCS placement project.
#
# Core correction relative to earlier versions:
#   The optimizer is a fixed-budget portfolio selection model. Therefore the
#   main denominator is NOT all ~9000 candidate segments. The main denominator is
#   the selected deployment portfolio, constrained by p_param, usually 200.
#
# This script produces:
#   1. A joined baseline segment table using the original input segment file.
#   2. DWC/SCS selected portfolio composition tables.
#   3. Budget-allocation heatmaps:
#        cell = selected DWC in bin / total selected DWC
#   4. Enrichment heatmaps:
#        enrichment = selected portfolio share / candidate-universe share
#   5. Portfolio composition bars:
#        all candidates vs selected DWC vs selected SCS
#   6. GSA run-level portfolio response:
#        each GSA run becomes one row with n_DWC, n_SCS, shares of budget,
#        median selected AADT, median selected distance, etc.
#   7. Run-level tornado plots:
#        which sampled parameters change DWC/SCS share of the 200-budget
#        portfolio.
#
# This script does NOT:
#   - rerun the MILP
#   - call Gurobi
#   - modify optimizer outputs
#   - treat selection as a random rare event over 9000 segments for the main story
#
# Usage:
#   python Policy_Insight_Analysis.py \
#       --baseline_dir "results/Results_baseline_p200" \
#       --gsa_dir "results/GSA_example" \
#       --input_segments "data/processed/NHS_US_5miles_with_SCS.csv" \
#       --p_param 200
#
# ═══════════════════════════════════════════════════════════════════════════════

import os
import re
import sys
import json
import math
import argparse
import datetime
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# 0. STYLE AND COLORS
# ═══════════════════════════════════════════════════════════════════════════════

def try_import_main_plots():
    """
    Import colors from Main_plots.py if available. The script remains standalone
    if Main_plots.py is not importable.
    """
    candidates = [Path.cwd(), Path(__file__).resolve().parent]
    for p in candidates:
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))

    try:
        import Main_plots as mp
        return {
            "C_DWC": getattr(mp, "C_DWC", "#2c3e50"),
            "C_SCS": getattr(mp, "C_SCS", "#17a589"),
            "C_EXIST": getattr(mp, "C_EXIST", "#e67e22"),
            "C_SUB": getattr(mp, "C_SUB", "#000000"),
            "C_REV": getattr(mp, "C_REV", "#27ae60"),
            "C_CAPEX": getattr(mp, "C_CAPEX", "#e74c3c"),
            "C_OPEX": getattr(mp, "C_OPEX", "#e67e22"),
            "C_PROFIT": getattr(mp, "C_PROFIT", "#2980b9"),
            "C_GREY": getattr(mp, "C_GREY", "#bdc3c7"),
            "C_GOLD": getattr(mp, "C_GOLD", "#f39c12"),
            "C_CYAN": getattr(mp, "C_CYAN", "cyan"),
            "_imported": True,
        }
    except Exception:
        return {
            "C_DWC": "#2c3e50",
            "C_SCS": "#17a589",
            "C_EXIST": "#e67e22",
            "C_SUB": "#000000",
            "C_REV": "#27ae60",
            "C_CAPEX": "#e74c3c",
            "C_OPEX": "#e67e22",
            "C_PROFIT": "#2980b9",
            "C_GREY": "#bdc3c7",
            "C_GOLD": "#f39c12",
            "C_CYAN": "cyan",
            "_imported": False,
        }


COL = try_import_main_plots()
C_DWC    = COL["C_DWC"]
C_SCS    = COL["C_SCS"]
C_EXIST  = COL["C_EXIST"]
C_SUB    = COL["C_SUB"]
C_REV    = COL["C_REV"]
C_CAPEX  = COL["C_CAPEX"]
C_OPEX   = COL["C_OPEX"]
C_PROFIT = COL["C_PROFIT"]
C_GREY   = COL["C_GREY"]
C_GOLD   = COL["C_GOLD"]
C_CYAN   = COL["C_CYAN"]

C_TEXT   = "#202124"
C_MUTED  = "#6B7280"
C_GRID   = "#D9DDE3"
C_LIGHT  = "#F7F8FA"
C_LIGHT2 = "#EEF1F4"
C_BAD    = "#8C2D04"
C_GOOD   = "#276419"

plt.rcParams.update({
    "font.family"       : "sans-serif",
    "font.sans-serif"   : ["DejaVu Sans", "Liberation Sans", "Arial", "sans-serif"],
    "font.size"         : 10.5,
    "axes.labelsize"    : 10.5,
    "axes.titlesize"    : 12.5,
    "axes.titleweight"  : "bold",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.linewidth"    : 0.85,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "legend.frameon"    : False,
    "figure.dpi"        : 160,
    "savefig.dpi"       : 300,
    "savefig.bbox"      : "tight",
})


# ═══════════════════════════════════════════════════════════════════════════════
# 1. HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def read_csv(path, required=True):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def savefig(fig, path):
    mkdir(Path(path).parent)
    fig.savefig(path, dpi=300, bbox_inches="tight", transparent=False)
    plt.close(fig)
    print(f"  saved: {Path(path).name}")


def safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def find_col(df, candidates, required=False, label=None):
    if df is None or df.empty:
        if required:
            raise ValueError(f"Cannot find {label or candidates}: empty dataframe.")
        return None

    norm = {c.lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace("_", "").replace(" ", "")
        if key in norm:
            return norm[key]

    for cand in candidates:
        key = cand.lower().replace("_", "").replace(" ", "")
        for nk, orig in norm.items():
            if key in nk or nk in key:
                return orig

    if required:
        raise KeyError(
            f"Column not found for {label or candidates}. "
            f"Tried {candidates}. Available columns: {list(df.columns)}"
        )
    return None


def parse_run_index(text):
    if text is None:
        return None
    m = re.search(r"(\d+)", str(text))
    if not m:
        return None
    return int(m.group(1))


def short_num(x, digits=1):
    if pd.isna(x):
        return ""
    x = float(x)
    if abs(x) >= 1e6:
        return f"{x/1e6:.{digits}f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.{digits}f}k"
    return f"{x:.{digits}f}"


def percent_str(x, digits=1):
    if pd.isna(x):
        return ""
    return f"{100*x:.{digits}f}%"


def infer_p_param_from_metadata(df_meta, fallback=200):
    if df_meta is None or df_meta.empty:
        return fallback

    # Support key-value metadata or single-row metadata.
    cols = [c.lower() for c in df_meta.columns]
    if "key" in cols and "value" in cols:
        kcol = df_meta.columns[cols.index("key")]
        vcol = df_meta.columns[cols.index("value")]
        tmp = df_meta.copy()
        tmp[kcol] = tmp[kcol].astype(str).str.lower()
        hit = tmp[tmp[kcol].isin(["p_param", "budget", "deployment_budget"])]
        if not hit.empty:
            val = pd.to_numeric(hit[vcol].iloc[0], errors="coerce")
            if not pd.isna(val):
                return int(val)

    for c in df_meta.columns:
        if c.lower() in ["p_param", "budget", "deployment_budget"]:
            val = pd.to_numeric(df_meta[c].iloc[0], errors="coerce")
            if not pd.isna(val):
                return int(val)

    return fallback


def lmh_bins(s, labels=("Low", "Medium", "High")):
    """
    Tercile bins, returned as ordered categorical plus cutoffs.
    """
    x = safe_num(s)
    q1, q2 = x.quantile([1/3, 2/3])
    out = pd.Series(index=x.index, dtype=object)
    out[x <= q1] = labels[0]
    out[(x > q1) & (x <= q2)] = labels[1]
    out[x > q2] = labels[2]
    out = pd.Categorical(out, categories=list(labels), ordered=True)
    return out, float(q1), float(q2)


def add_driver_bins(df):
    out = df.copy()
    thresholds = []

    mappings = [
        ("AADT_COM", "AADT_bin", "AADT"),
        ("Min_Sub_Dist", "Distance_bin", "Substation distance"),
        ("Electricity_Cost", "Electricity_bin", "Electricity cost"),
        ("Lambda", "Lambda_bin", "Locational cost"),
    ]

    for col, bin_col, label in mappings:
        if col not in out.columns:
            continue
        out[bin_col], q1, q2 = lmh_bins(out[col])
        thresholds.append({
            "variable": label,
            "source_column": col,
            "low_medium_cutoff": q1,
            "medium_high_cutoff": q2,
        })

    return out, pd.DataFrame(thresholds)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BUILD JOINED BASELINE TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def build_joined_segment_table(baseline_dir, input_segments_path, p_param_fallback=200):
    """
    Start from the original input file, not the diagnostic file.

    DWC selection:
        DWC_Selected_Segments.Segment_ID == input.SEGMENT_ID_OG

    SCS selection:
        SCS_Selected_Stations.Host_Seg == input.SEGMENT_ID_OG

    Electricity:
        Total_Electricity_Rate_Nom_2024 from input file, converted to $/kWh
        if the values look like cents/kWh.
    """
    baseline_dir = Path(baseline_dir)

    df_in = read_csv(input_segments_path, required=True)
    df_diag = read_csv(baseline_dir / "DWC_Selection_Master_Diagnostic.csv", required=True)
    df_dwc = read_csv(baseline_dir / "DWC_Selected_Segments.csv", required=False)
    df_scs = read_csv(baseline_dir / "SCS_Selected_Stations.csv", required=False)
    df_meta = read_csv(baseline_dir / "run_metadata.csv", required=False)

    p_param = infer_p_param_from_metadata(df_meta, fallback=p_param_fallback)

    seg_in_col = find_col(df_in, ["SEGMENT_ID_OG", "Segment_ID"], required=True, label="input segment id")
    df = df_in.copy().rename(columns={seg_in_col: "Segment_ID"})

    aadt_col = find_col(df, ["AADT_COM", "AADT"], required=True, label="AADT")
    df["AADT_COM"] = safe_num(df[aadt_col])

    elec_col = find_col(
        df,
        [
            "Total_Electricity_Rate_Nom_2024",
            "Cmrcl_Electricity_Rate_Nom_2024",
            "Electricity_Rate",
            "LCOE",
        ],
        required=True,
        label="electricity price from input file",
    )
    electricity_raw = safe_num(df[elec_col])
    if electricity_raw.median(skipna=True) > 1.0:
        df["Electricity_Cost"] = electricity_raw / 100.0
        electricity_unit = "$/kWh, converted from cents/kWh"
    else:
        df["Electricity_Cost"] = electricity_raw
        electricity_unit = "$/kWh"

    # Locational cost fallback from input.
    norm_col = find_col(df, ["Norm_index", "Lambda", "lambda"])
    if norm_col:
        df["Norm_index_CANON"] = safe_num(df[norm_col])
    else:
        df["Norm_index_CANON"] = np.nan

    # Diagnostic join for optimizer-computed substation distance and Lambda.
    diag_seg_col = find_col(df_diag, ["Segment_ID"], required=True, label="diagnostic segment id")
    diag_keep = [diag_seg_col]
    for c in ["X", "Y", "Min_Sub_Dist", "Total_Revenue", "Lambda", "Rev_Dist_Ratio", "Traff_Dist_Ratio", "Is_DWC"]:
        if c in df_diag.columns:
            diag_keep.append(c)

    diag = df_diag[diag_keep].copy().rename(columns={diag_seg_col: "Segment_ID"})
    diag = diag.rename(columns={c: f"diag_{c}" for c in diag.columns if c != "Segment_ID"})
    df = df.merge(diag, on="Segment_ID", how="left")

    if "diag_Min_Sub_Dist" in df.columns:
        df["Min_Sub_Dist"] = safe_num(df["diag_Min_Sub_Dist"])
    else:
        # Last resort only.
        scs_dist_col = find_col(df, ["NEAR_SCS_Dist"])
        df["Min_Sub_Dist"] = safe_num(df[scs_dist_col]) if scs_dist_col else np.nan

    if "diag_Lambda" in df.columns:
        df["Lambda"] = safe_num(df["diag_Lambda"])
    else:
        df["Lambda"] = df["Norm_index_CANON"]

    if "diag_X" in df.columns:
        df["X_plot"] = safe_num(df["diag_X"])
    else:
        xcol = find_col(df, ["MID_X", "X"])
        df["X_plot"] = safe_num(df[xcol]) if xcol else np.nan

    if "diag_Y" in df.columns:
        df["Y_plot"] = safe_num(df["diag_Y"])
    else:
        ycol = find_col(df, ["MID_Y", "Y"])
        df["Y_plot"] = safe_num(df[ycol]) if ycol else np.nan

    # DWC selected.
    df["is_DWC_selected"] = 0
    if not df_dwc.empty and "Segment_ID" in df_dwc.columns:
        dwc_ids = set(safe_num(df_dwc["Segment_ID"]).dropna().astype(int).tolist())
        df["is_DWC_selected"] = safe_num(df["Segment_ID"]).astype("Int64").isin(dwc_ids).astype(int)

        dwc_fin = df_dwc.copy()
        dwc_fin = dwc_fin.rename(columns={c: f"DWC_{c}" for c in dwc_fin.columns if c != "Segment_ID"})
        df = df.merge(dwc_fin, on="Segment_ID", how="left")

    # SCS selected by host segment.
    df["is_SCS_selected"] = 0
    if not df_scs.empty:
        host_col = find_col(df_scs, ["Host_Seg"], required=True, label="SCS host segment")
        scs = df_scs.copy().rename(columns={host_col: "Segment_ID"})
        scs_ids = set(safe_num(scs["Segment_ID"]).dropna().astype(int).tolist())
        df["is_SCS_selected"] = safe_num(df["Segment_ID"]).astype("Int64").isin(scs_ids).astype(int)

        scs = scs.rename(columns={c: f"SCS_{c}" for c in scs.columns if c != "Segment_ID"})
        df = df.merge(scs, on="Segment_ID", how="left")

    df["placement_class"] = "None"
    df.loc[df["is_SCS_selected"] == 1, "placement_class"] = "SCS"
    df.loc[df["is_DWC_selected"] == 1, "placement_class"] = "DWC"

    df, thresholds = add_driver_bins(df)

    info = {
        "electricity_column_used": elec_col,
        "electricity_unit": electricity_unit,
        "main_plots_imported": bool(COL["_imported"]),
        "n_candidate_segments": int(len(df)),
        "n_dwc_selected": int(df["is_DWC_selected"].sum()),
        "n_scs_selected": int(df["is_SCS_selected"].sum()),
        "n_selected_total": int(df["is_DWC_selected"].sum() + df["is_SCS_selected"].sum()),
        "p_param": int(p_param),
        "unused_or_rounding_budget": int(p_param - df["is_DWC_selected"].sum() - df["is_SCS_selected"].sum()),
    }

    return df, df_dwc, df_scs, df_meta, thresholds, info


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BASELINE PORTFOLIO TABLES
# ═══════════════════════════════════════════════════════════════════════════════

def portfolio_share_table(df, row_bin, col_bin, selected_col="is_DWC_selected"):
    """
    Returns a table where each cell contains:
      - candidate count in bin
      - selected count in bin
      - selected portfolio share
      - candidate universe share
      - enrichment ratio
    """
    d = df.dropna(subset=[row_bin, col_bin]).copy()

    total_candidates = len(d)
    total_selected = int(d[selected_col].sum())

    g = (
        d.groupby([row_bin, col_bin], observed=False)
        .agg(
            candidate_n=("Segment_ID", "count"),
            selected_n=(selected_col, "sum"),
            median_AADT=("AADT_COM", "median"),
            median_distance=("Min_Sub_Dist", "median"),
            median_electricity=("Electricity_Cost", "median"),
            median_lambda=("Lambda", "median"),
        )
        .reset_index()
    )

    g["candidate_share"] = g["candidate_n"] / max(total_candidates, 1)
    g["selected_portfolio_share"] = g["selected_n"] / max(total_selected, 1)
    g["within_bin_selection_rate"] = g["selected_n"] / g["candidate_n"].clip(lower=1)
    g["enrichment"] = g["selected_portfolio_share"] / g["candidate_share"].replace(0, np.nan)
    g["selected_total"] = total_selected
    return g


def make_all_baseline_portfolio_tables(df):
    pairs = [
        ("AADT_bin", "Distance_bin", "AADT_x_Distance"),
        ("AADT_bin", "Lambda_bin", "AADT_x_Lambda"),
        ("AADT_bin", "Electricity_bin", "AADT_x_Electricity"),
    ]

    tables = {}
    for r, c, name in pairs:
        if r in df.columns and c in df.columns:
            tables[f"DWC_{name}"] = portfolio_share_table(df, r, c, "is_DWC_selected")
            tables[f"SCS_{name}"] = portfolio_share_table(df, r, c, "is_SCS_selected")

    return tables


def make_portfolio_composition_table(df):
    """
    Compares all candidates, selected DWC, and selected SCS by Low/Medium/High bins.
    """
    rows = []
    feature_specs = [
        ("AADT", "AADT_bin"),
        ("Substation distance", "Distance_bin"),
        ("Electricity cost", "Electricity_bin"),
        ("Locational cost", "Lambda_bin"),
    ]

    groups = {
        "All candidates": df,
        "Selected DWC": df[df["is_DWC_selected"] == 1],
        "Selected SCS": df[df["is_SCS_selected"] == 1],
    }

    for feature, bin_col in feature_specs:
        if bin_col not in df.columns:
            continue
        for group_name, gdf in groups.items():
            total = len(gdf)
            counts = gdf[bin_col].value_counts(dropna=False)
            for level in ["Low", "Medium", "High"]:
                rows.append({
                    "feature": feature,
                    "group": group_name,
                    "bin": level,
                    "n": int(counts.get(level, 0)),
                    "share": float(counts.get(level, 0) / max(total, 1)),
                    "total": int(total),
                })

    return pd.DataFrame(rows)


def make_policy_readout_matrix(df):
    """
    User-facing table with:
      Substation distance | Electricity cost | AADT | selected DWC share | enrichment | interpretation

    This is portfolio-centered:
        DWC portfolio share = selected DWC in bin / all selected DWC.
    """
    required = ["Distance_bin", "Electricity_bin", "AADT_bin"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing bin column: {c}")

    g = (
        df.groupby(["Distance_bin", "Electricity_bin", "AADT_bin"], observed=False)
        .agg(
            candidate_n=("Segment_ID", "count"),
            selected_DWC_n=("is_DWC_selected", "sum"),
            selected_SCS_n=("is_SCS_selected", "sum"),
            median_AADT=("AADT_COM", "median"),
            median_distance=("Min_Sub_Dist", "median"),
            median_electricity=("Electricity_Cost", "median"),
            median_lambda=("Lambda", "median"),
        )
        .reset_index()
        .rename(columns={
            "Distance_bin": "Substation distance",
            "Electricity_bin": "Electricity cost",
            "AADT_bin": "AADT",
        })
    )

    n_candidate = len(df)
    n_dwc = max(int(df["is_DWC_selected"].sum()), 1)
    n_scs = max(int(df["is_SCS_selected"].sum()), 1)

    g["candidate_share"] = g["candidate_n"] / max(n_candidate, 1)
    g["DWC_portfolio_share"] = g["selected_DWC_n"] / n_dwc
    g["SCS_portfolio_share"] = g["selected_SCS_n"] / n_scs
    g["DWC_enrichment"] = g["DWC_portfolio_share"] / g["candidate_share"].replace(0, np.nan)
    g["SCS_enrichment"] = g["SCS_portfolio_share"] / g["candidate_share"].replace(0, np.nan)

    def readout(row):
        aadt = str(row["AADT"])
        dist = str(row["Substation distance"])
        elec = str(row["Electricity cost"])

        if aadt == "High" and dist == "Low" and elec in ["Low", "Medium"]:
            return "Core first-wave DWC condition"
        if aadt == "High" and dist == "High" and elec == "High":
            return "High demand, but grid and energy-cost support likely needed"
        if aadt == "High" and dist == "High":
            return "High demand, grid-access support likely needed"
        if aadt == "High" and elec == "High":
            return "High demand, electricity-cost support may improve economics"
        if aadt == "Low":
            return "Not first-wave unless non-economic criteria dominate"
        return "Intermediate case; inspect local economics"

    g["Policy readout"] = g.apply(readout, axis=1)

    # Prioritize high AADT combinations, then DWC portfolio share.
    aadt_rank = {"High": 3, "Medium": 2, "Low": 1}
    g["_rank"] = g["AADT"].map(aadt_rank).astype(float) * 100 + g["DWC_portfolio_share"] * 100
    return g.sort_values("_rank", ascending=False).drop(columns=["_rank"])


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FIGURES: BASELINE PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════

def _pivot_for_heatmap(table, row_bin, col_bin, value_col):
    p = table.pivot(index=row_bin, columns=col_bin, values=value_col)
    p = p.reindex(index=["Low", "Medium", "High"], columns=["Low", "Medium", "High"])
    return p


def draw_heatmap(ax, pivot, title, cbar_label, cmap, fmt="{:.1%}", vmin=None, vmax=None):
    data = pivot.values.astype(float)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    ax.set_title(title, fontsize=11.5)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                txt = ""
            else:
                txt = fmt.format(val)
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=C_TEXT)

    return im


def plot_budget_allocation_heatmaps(tables, fig_dir):
    """
    Main baseline figure:
    Cell value = selected DWC in bin / total selected DWC.
    """
    specs = [
        ("DWC_AADT_x_Distance", "AADT_bin", "Distance_bin", "AADT × distance"),
        ("DWC_AADT_x_Lambda", "AADT_bin", "Lambda_bin", "AADT × locational cost"),
        ("DWC_AADT_x_Electricity", "AADT_bin", "Electricity_bin", "AADT × electricity cost"),
    ]

    available = [s for s in specs if s[0] in tables]
    if not available:
        print("  skipped budget allocation heatmaps: no tables")
        return

    cmap = LinearSegmentedColormap.from_list("dwc_alloc", ["#F5F7FA", "#A8B6C3", C_DWC])

    fig, axes = plt.subplots(1, len(available), figsize=(4.2 * len(available), 3.8), constrained_layout=True)
    if len(available) == 1:
        axes = [axes]

    im = None
    for ax, (key, rbin, cbin, title) in zip(axes, available):
        p = _pivot_for_heatmap(tables[key], rbin, cbin, "selected_portfolio_share")
        im = draw_heatmap(
            ax,
            p,
            title,
            "DWC portfolio share",
            cmap,
            fmt="{:.0%}",
            vmin=0,
            vmax=max(0.01, np.nanmax(p.values)),
        )
        ax.set_xlabel(cbin.replace("_bin", "").replace("Distance", "Substation distance").replace("Lambda", "Locational cost").replace("Electricity", "Electricity cost"))
        ax.set_ylabel("AADT")

    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label("Share of selected DWC portfolio", rotation=270, labelpad=16)
    fig.suptitle("DWC budget allocation across observed driver bins", fontsize=14, fontweight="bold")
    savefig(fig, Path(fig_dir) / "fig_A_dwc_budget_allocation_heatmaps.png")


def plot_enrichment_heatmaps(tables, fig_dir):
    """
    Main validation figure:
    Enrichment = selected portfolio share / candidate universe share.
    """
    specs = [
        ("DWC_AADT_x_Distance", "AADT_bin", "Distance_bin", "AADT × distance"),
        ("DWC_AADT_x_Lambda", "AADT_bin", "Lambda_bin", "AADT × locational cost"),
        ("DWC_AADT_x_Electricity", "AADT_bin", "Electricity_bin", "AADT × electricity cost"),
    ]

    available = [s for s in specs if s[0] in tables]
    if not available:
        print("  skipped enrichment heatmaps: no tables")
        return

    cmap = LinearSegmentedColormap.from_list("enrich", ["#F7F7F7", "#B7C7D8", C_PROFIT, C_DWC])

    # Robust max keeps one extreme cell from washing out everything.
    all_vals = []
    for key, rbin, cbin, _ in available:
        vals = tables[key]["enrichment"].replace([np.inf, -np.inf], np.nan).dropna()
        all_vals.extend(vals.tolist())
    vmax = max(1.0, np.nanpercentile(all_vals, 95)) if all_vals else 1.0

    fig, axes = plt.subplots(1, len(available), figsize=(4.2 * len(available), 3.8), constrained_layout=True)
    if len(available) == 1:
        axes = [axes]

    im = None
    for ax, (key, rbin, cbin, title) in zip(axes, available):
        p = _pivot_for_heatmap(tables[key], rbin, cbin, "enrichment")
        im = draw_heatmap(
            ax,
            p,
            title,
            "Enrichment",
            cmap,
            fmt="{:.1f}×",
            vmin=0,
            vmax=vmax,
        )
        ax.set_xlabel(cbin.replace("_bin", "").replace("Distance", "Substation distance").replace("Lambda", "Locational cost").replace("Electricity", "Electricity cost"))
        ax.set_ylabel("AADT")

    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02)
    cbar.set_label("Portfolio enrichment ratio", rotation=270, labelpad=16)
    fig.suptitle("Where DWC is over-selected relative to candidate availability", fontsize=14, fontweight="bold")
    savefig(fig, Path(fig_dir) / "fig_B_dwc_portfolio_enrichment_heatmaps.png")


def plot_portfolio_composition(comp, fig_dir):
    """
    Stacked bars comparing all candidates vs selected portfolio.
    """
    if comp.empty:
        print("  skipped portfolio composition: empty table")
        return

    feature_order = ["AADT", "Substation distance", "Electricity cost", "Locational cost"]
    group_order = ["All candidates", "Selected DWC", "Selected SCS"]
    bin_order = ["Low", "Medium", "High"]
    colors = {"Low": "#D7E3EF", "Medium": "#7F9DBA", "High": C_DWC}

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.8), constrained_layout=True)
    axes = axes.ravel()

    for ax, feat in zip(axes, feature_order):
        d = comp[comp["feature"] == feat].copy()
        y = np.arange(len(group_order))
        left = np.zeros(len(group_order))

        for b in bin_order:
            vals = []
            for g in group_order:
                hit = d[(d["group"] == g) & (d["bin"] == b)]
                vals.append(float(hit["share"].iloc[0]) if not hit.empty else 0.0)
            ax.barh(y, vals, left=left, color=colors[b], edgecolor="white", height=0.58, label=b)
            for i, v in enumerate(vals):
                if v > 0.08:
                    ax.text(left[i] + v/2, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color=C_TEXT)
            left += np.array(vals)

        ax.set_yticks(y)
        ax.set_yticklabels(group_order)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title(feat)
        ax.grid(axis="x", linestyle=":", color=C_GRID, lw=0.6)

    handles = [patches.Patch(color=colors[b], label=b) for b in bin_order]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Portfolio composition compared with candidate universe", fontsize=14, fontweight="bold")
    savefig(fig, Path(fig_dir) / "fig_C_portfolio_composition_bars.png")


def plot_policy_readout_matrix(policy, fig_dir, max_rows=12):
    """
    Table figure, portfolio-centered.
    """
    d = policy.copy()

    # Show high-AADT rows first and any rows with high DWC portfolio share.
    high = d[d["AADT"] == "High"].copy()
    rest = d[d["AADT"] != "High"].sort_values("DWC_portfolio_share", ascending=False)
    show = pd.concat([high, rest], ignore_index=True).head(max_rows)

    cols = [
        "Substation distance",
        "Electricity cost",
        "AADT",
        "candidate_n",
        "selected_DWC_n",
        "DWC_portfolio_share",
        "DWC_enrichment",
        "Policy readout",
    ]
    show = show[cols].copy()

    show["candidate_n"] = show["candidate_n"].map(lambda x: f"{int(x):,}")
    show["selected_DWC_n"] = show["selected_DWC_n"].map(lambda x: f"{int(x):,}")
    show["DWC_portfolio_share"] = show["DWC_portfolio_share"].map(lambda x: f"{100*x:.1f}%")
    show["DWC_enrichment"] = show["DWC_enrichment"].map(lambda x: "" if pd.isna(x) else f"{x:.1f}×")

    headers = [
        "Substation\ndistance",
        "Electricity\ncost",
        "AADT",
        "Candidate\nn",
        "Selected\nDWC n",
        "DWC budget\nshare",
        "Enrichment",
        "Policy readout",
    ]

    fig_h = max(4.4, 0.44 * len(show) + 1.7)
    fig, ax = plt.subplots(figsize=(12.2, fig_h))
    ax.axis("off")

    ax.text(0.01, 0.97, "Policy readout from selected DWC portfolio allocation",
            ha="left", va="top", fontsize=13.5, fontweight="bold")

    tbl = ax.table(
        cellText=show.values.tolist(),
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.105, 0.105, 0.07, 0.085, 0.09, 0.095, 0.085, 0.36],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.8)
    tbl.scale(1, 1.55)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        cell.set_linewidth(0.55)
        if r == 0:
            cell.set_facecolor(C_DWC)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("white" if r % 2 else C_LIGHT)
            if c == 7:
                cell.set_text_props(ha="left")
            if c in [4, 5, 6]:
                cell.set_text_props(color=C_DWC)

    ax.text(0.01, 0.04,
            "Budget share = selected DWC in bin / total selected DWC. "
            "Enrichment = DWC budget share / candidate-universe share.",
            ha="left", va="bottom", fontsize=8.7, color=C_MUTED)

    savefig(fig, Path(fig_dir) / "fig_D_policy_readout_matrix_portfolio.png")


def plot_baseline_context(df, fig_dir):
    d = df.dropna(subset=["Min_Sub_Dist", "AADT_COM"]).copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    none = d[d["placement_class"] == "None"]
    ax.scatter(
        none["Min_Sub_Dist"], none["AADT_COM"],
        s=6, color="#AEB6BF", alpha=0.13, linewidths=0,
        rasterized=True, label="Candidate, not selected",
    )

    dwc = d[d["placement_class"] == "DWC"]
    scs = d[d["placement_class"] == "SCS"]

    if not dwc.empty:
        ax.scatter(
            dwc["Min_Sub_Dist"], dwc["AADT_COM"],
            s=40, facecolors="none", edgecolors=C_DWC,
            linewidths=1.2, label=f"DWC selected ({len(dwc)})", zorder=4,
        )
    if not scs.empty:
        ax.scatter(
            scs["Min_Sub_Dist"], scs["AADT_COM"],
            s=50, marker="^", color=C_SCS,
            edgecolors="white", linewidths=0.45, label=f"SCS selected ({len(scs)})", zorder=5,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Substation distance")
    ax.set_ylabel("AADT")
    ax.set_title("Baseline selected portfolio in candidate space")
    ax.legend(loc="lower right")
    ax.grid(axis="both", linestyle=":", color=C_GRID, lw=0.55)

    savefig(fig, Path(fig_dir) / "fig_E_baseline_selected_portfolio_context.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. GSA RUN-LEVEL PORTFOLIO RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

PARAM_PRETTY = {
    "ETA_DWC": r"$\eta^{DWC}$",
    "ETA_SCS_SHARE": r"$\eta^{SCS}_{share}$",
    "g_aadt": r"$g^{AADT}$",
    "g_ev": r"$g^{EV}$",
    "project_life": "Project life",
    "C_DWC_base": "DWC capex",
    "C_SCS_base_per_port": "SCS capex/port",
    "N_SCS_stalls": "SCS ports",
    "g_lcoe": r"$g^{LCOE}$",
}


def find_run_result_dirs(gsa_dir):
    """
    Finds directories containing selected-output files under GSA/Runs.
    Robust to:
      Runs/run_0001/DWC_Selected_Segments.csv
      Runs/run_0001/Results_xxx/DWC_Selected_Segments.csv
    """
    gsa_dir = Path(gsa_dir)
    root = gsa_dir / "Runs"
    if not root.exists():
        return []

    dirs = set()
    for p in root.glob("**/DWC_Selected_Segments.csv"):
        dirs.add(p.parent)
    for p in root.glob("**/SCS_Selected_Stations.csv"):
        dirs.add(p.parent)

    return sorted(dirs, key=lambda p: str(p))


def match_params_to_run(run_dir, samples, ordinal):
    """
    Robust matching:
      1. run_tag column equals directory name or ancestor name.
      2. run_id column equals parsed numeric id.
      3. fallback: ordinal row order.
    """
    if samples is None or samples.empty:
        return {}

    names = [run_dir.name] + [p.name for p in run_dir.parents]
    cols_lower = {c.lower(): c for c in samples.columns}

    if "run_tag" in cols_lower:
        c = cols_lower["run_tag"]
        for nm in names:
            hit = samples[samples[c].astype(str) == str(nm)]
            if not hit.empty:
                return hit.iloc[0].to_dict()

    parsed = parse_run_index(run_dir.name)
    if parsed is not None and "run_id" in cols_lower:
        c = cols_lower["run_id"]
        hit = samples[pd.to_numeric(samples[c], errors="coerce") == parsed]
        if not hit.empty:
            return hit.iloc[0].to_dict()

    if ordinal < len(samples):
        return samples.iloc[ordinal].to_dict()

    return {}


def summarize_selected_attrs(selected_ids, df_master):
    if not selected_ids:
        return {
            "median_AADT": np.nan,
            "median_distance": np.nan,
            "median_lambda": np.nan,
            "median_electricity": np.nan,
        }
    g = df_master[df_master["Segment_ID"].astype("Int64").isin(selected_ids)]
    if g.empty:
        return {
            "median_AADT": np.nan,
            "median_distance": np.nan,
            "median_lambda": np.nan,
            "median_electricity": np.nan,
        }
    return {
        "median_AADT": float(g["AADT_COM"].median()),
        "median_distance": float(g["Min_Sub_Dist"].median()),
        "median_lambda": float(g["Lambda"].median()),
        "median_electricity": float(g["Electricity_Cost"].median()),
    }


def build_gsa_run_level_table(gsa_dir, df_master, p_param):
    gsa_dir = Path(gsa_dir)

    samples_path = gsa_dir / "lhs_parameter_samples.csv"
    samples = read_csv(samples_path, required=False)

    run_dirs = find_run_result_dirs(gsa_dir)
    rows = []

    for ordinal, rd in enumerate(run_dirs):
        dwc_file = rd / "DWC_Selected_Segments.csv"
        scs_file = rd / "SCS_Selected_Stations.csv"

        dwc = read_csv(dwc_file, required=False)
        scs = read_csv(scs_file, required=False)

        dwc_ids = []
        if not dwc.empty and "Segment_ID" in dwc.columns:
            dwc_ids = safe_num(dwc["Segment_ID"]).dropna().astype(int).tolist()

        scs_host_ids = []
        if not scs.empty:
            host_col = find_col(scs, ["Host_Seg"], required=False)
            if host_col:
                scs_host_ids = safe_num(scs[host_col]).dropna().astype(int).tolist()

        row = {
            "run_dir": str(rd),
            "run_name": rd.name,
            "run_ordinal": ordinal,
            "n_DWC": int(len(dwc_ids)),
            "n_SCS": int(len(scs_host_ids)),
            "n_selected_total": int(len(dwc_ids) + len(scs_host_ids)),
            "DWC_share_budget": float(len(dwc_ids) / max(p_param, 1)),
            "SCS_share_budget": float(len(scs_host_ids) / max(p_param, 1)),
            "selected_share_budget": float((len(dwc_ids) + len(scs_host_ids)) / max(p_param, 1)),
        }

        dwc_attrs = summarize_selected_attrs(dwc_ids, df_master)
        scs_attrs = summarize_selected_attrs(scs_host_ids, df_master)
        for k, v in dwc_attrs.items():
            row[f"DWC_{k}"] = v
        for k, v in scs_attrs.items():
            row[f"SCS_{k}"] = v

        params = match_params_to_run(rd, samples, ordinal)
        for k, v in params.items():
            if str(k).lower() not in ["run_tag", "run_id"]:
                row[k] = v

        rows.append(row)

    out = pd.DataFrame(rows)
    return out, samples


def fit_run_level_ols(run_df, outcome, param_cols):
    """
    OLS of a run-level portfolio outcome on standardized sampled parameters.
    """
    import statsmodels.api as sm

    cols = [outcome] + param_cols
    d = run_df[cols].copy()
    for c in cols:
        d[c] = safe_num(d[c])
    d = d.dropna()

    if len(d) < max(10, len(param_cols) + 3):
        return pd.DataFrame()

    y = d[outcome].astype(float)
    X_raw = d[param_cols].astype(float)

    # Drop constant columns.
    usable = [c for c in X_raw.columns if X_raw[c].std(ddof=0) > 1e-12]
    if not usable:
        return pd.DataFrame()

    X_raw = X_raw[usable]
    mu = X_raw.mean()
    sd = X_raw.std(ddof=0).replace(0, 1.0)
    X = (X_raw - mu) / sd
    X = sm.add_constant(X, has_constant="add")

    res = sm.OLS(y, X).fit(cov_type="HC3")

    rows = []
    for p in usable:
        beta = float(res.params[p])
        rows.append({
            "outcome": outcome,
            "parameter": p,
            "coef_standardized": beta,
            "abs_coef_standardized": abs(beta),
            "se": float(res.bse[p]),
            "p_value": float(res.pvalues[p]),
            "n_runs": int(len(d)),
            "r2": float(res.rsquared),
            "mean_raw": float(mu[p]),
            "std_raw": float(sd[p]),
        })

    return pd.DataFrame(rows).sort_values("abs_coef_standardized", ascending=False)


def run_level_regressions(run_df, samples):
    if run_df.empty or samples.empty:
        return pd.DataFrame()

    param_cols = []
    for c in samples.columns:
        if c.lower() in ["run_tag", "run_id"]:
            continue
        if c in run_df.columns and safe_num(run_df[c]).notna().sum() > 5:
            param_cols.append(c)

    outcomes = [
        "DWC_share_budget",
        "SCS_share_budget",
        "selected_share_budget",
        "DWC_median_AADT",
        "DWC_median_distance",
        "DWC_median_lambda",
        "DWC_median_electricity",
    ]

    outs = []
    for y in outcomes:
        if y in run_df.columns and safe_num(run_df[y]).notna().sum() > 5:
            out = fit_run_level_ols(run_df, y, param_cols)
            if not out.empty:
                outs.append(out)

    return pd.concat(outs, ignore_index=True) if outs else pd.DataFrame()


def assign_run_level_parameter_codes(coef_df, outcome="DWC_share_budget", top_k=6):
    if coef_df.empty:
        return pd.DataFrame(), {}, coef_df

    base = coef_df[coef_df["outcome"] == outcome].copy()
    if base.empty:
        base = coef_df.copy()

    top = base.sort_values("abs_coef_standardized", ascending=False).head(top_k).copy()
    top["P_code"] = [f"P{i+1}" for i in range(len(top))]
    code_map = dict(zip(top["parameter"], top["P_code"]))

    out = coef_df.copy()
    out["P_code"] = out["parameter"].map(code_map)
    out["pretty_name"] = out["parameter"].map(PARAM_PRETTY).fillna(out["parameter"])
    top["pretty_name"] = top["parameter"].map(PARAM_PRETTY).fillna(top["parameter"])

    return top, code_map, out


def plot_run_level_tornado(coef_df, code_map, fig_dir):
    """
    Run-level tornado: response of DWC/SCS budget share to sampled parameters.
    """
    if coef_df.empty or not code_map:
        print("  skipped run-level tornado: no run-level coefficients")
        return

    rows = []
    for p, code in code_map.items():
        d = {"parameter": p, "P_code": code}
        for outcome in ["DWC_share_budget", "SCS_share_budget"]:
            hit = coef_df[(coef_df["outcome"] == outcome) & (coef_df["parameter"] == p)]
            d[outcome] = float(hit["coef_standardized"].iloc[0]) if not hit.empty else np.nan
        d["max_abs"] = np.nanmax([abs(d.get("DWC_share_budget", np.nan)), abs(d.get("SCS_share_budget", np.nan))])
        rows.append(d)

    d = pd.DataFrame(rows).sort_values("max_abs", ascending=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = np.arange(len(d))
    h = 0.34

    ax.axvline(0, color=C_TEXT, lw=0.9)
    ax.barh(y - h/2, d["DWC_share_budget"], height=h, color=C_DWC, label="DWC budget share")
    ax.barh(y + h/2, d["SCS_share_budget"], height=h, color=C_SCS, label="SCS budget share")

    ax.set_yticks(y)
    ax.set_yticklabels(d["P_code"])
    ax.set_xlabel("Change in budget share per 1 s.d. parameter increase")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("Run-level GSA: portfolio response to sampled parameters")
    ax.grid(axis="x", linestyle=":", color=C_GRID, lw=0.7)
    ax.legend(loc="lower right")

    savefig(fig, Path(fig_dir) / "fig_F_run_level_budget_share_tornado.png")


def plot_run_level_parameter_bin_response(run_df, code_map, fig_dir):
    """
    For each top parameter, bin runs into low/medium/high and show mean DWC/SCS
    share of p_param.
    """
    if run_df.empty or not code_map:
        print("  skipped parameter-bin response: no run table or parameter codes")
        return

    rows = []
    for p, code in code_map.items():
        if p not in run_df.columns:
            continue
        x = safe_num(run_df[p])
        if x.notna().sum() < 5:
            continue
        bins, q1, q2 = lmh_bins(x)
        tmp = run_df.copy()
        tmp["_bin"] = bins
        g = (
            tmp.groupby("_bin", observed=False)
            .agg(
                DWC_share_budget=("DWC_share_budget", "mean"),
                SCS_share_budget=("SCS_share_budget", "mean"),
                n_runs=("run_dir", "count"),
            )
            .reset_index()
        )
        for _, r in g.iterrows():
            rows.append({
                "parameter": p,
                "P_code": code,
                "case": str(r["_bin"]),
                "DWC_share_budget": float(r["DWC_share_budget"]),
                "SCS_share_budget": float(r["SCS_share_budget"]),
                "n_runs": int(r["n_runs"]),
            })

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        return

    case_order = ["Low", "Medium", "High"]
    case_x = {"Low": -0.18, "Medium": 0.0, "High": 0.18}
    marker = {"Low": "o", "Medium": "s", "High": "^"}
    alpha = {"Low": 0.55, "Medium": 0.85, "High": 1.0}

    codes = list(code_map.values())
    xbase = np.arange(len(codes))

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)

    for ax, col, color, title in [
        (axes[0], "DWC_share_budget", C_DWC, "DWC budget share"),
        (axes[1], "SCS_share_budget", C_SCS, "SCS budget share"),
    ]:
        for i, code in enumerate(codes):
            g = plot_df[plot_df["P_code"] == code]
            vals = []
            for case in case_order:
                hit = g[g["case"] == case]
                if hit.empty:
                    continue
                val = float(hit[col].iloc[0])
                vals.append(val)
                ax.scatter(
                    i + case_x[case], val,
                    s=48, marker=marker[case], color=color,
                    edgecolors="white", linewidths=0.5, alpha=alpha[case],
                    label=case if i == 0 else None,
                    zorder=3,
                )
            if vals:
                ax.plot([i, i], [min(vals), max(vals)], color=color, lw=2.0, alpha=0.55)

        ax.set_xticks(xbase)
        ax.set_xticklabels(codes)
        ax.set_ylim(0, max(1.0, plot_df[[ "DWC_share_budget", "SCS_share_budget"]].max().max() * 1.15))
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title(title)
        ax.grid(axis="y", linestyle=":", color=C_GRID, lw=0.7)

    axes[0].set_ylabel("Mean share of p_param budget")
    axes[1].legend(title="Parameter bin", loc="upper right", frameon=False)
    fig.suptitle("GSA run-level portfolio response by parameter bin", fontsize=14, fontweight="bold")

    savefig(fig, Path(fig_dir) / "fig_G_parameter_bin_budget_response.png")


def plot_run_parameter_key(top_params, fig_dir):
    if top_params.empty:
        return

    fig_h = max(2.4, 0.42 * len(top_params) + 1.0)
    fig, ax = plt.subplots(figsize=(6.8, fig_h))
    ax.axis("off")

    ax.text(0.02, 0.95, "Parameter key for run-level GSA figures",
            ha="left", va="top", fontsize=13.5, fontweight="bold")

    ax.text(0.06, 0.80, "Code", fontsize=10, fontweight="bold")
    ax.text(0.22, 0.80, "Parameter", fontsize=10, fontweight="bold")
    ax.text(0.68, 0.80, "Outcome used for ranking", fontsize=10, fontweight="bold")
    ax.plot([0.04, 0.96], [0.755, 0.755], color=C_GRID, lw=0.8)

    y0 = 0.68
    dy = 0.105 if len(top_params) <= 6 else 0.085
    for i, (_, r) in enumerate(top_params.iterrows()):
        y = y0 - i * dy
        ax.text(0.08, y, r["P_code"], ha="center", va="center",
                fontsize=10.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.23", facecolor=C_LIGHT, edgecolor=C_GRID))
        ax.text(0.22, y, str(r["pretty_name"]), ha="left", va="center", fontsize=10.5)
        ax.text(0.68, y, "DWC budget share", ha="left", va="center", fontsize=9.5, color=C_MUTED)

    savefig(fig, Path(fig_dir) / "fig_H_run_level_parameter_key.png")

# ═══════════════════════════════════════════════════════════════════════════════
# EXTRA PORTFOLIO-RESPONSE FIGURES
# Inspired by Sensitivity_Plots.py, but adapted to fixed-budget portfolio outputs.
# These do NOT use segment-level rare-event probabilities.
# ═══════════════════════════════════════════════════════════════════════════════

def _get_sampled_param_cols_from_samples(samples, run_df=None):
    """Return sampled GSA parameter columns, excluding run identifiers."""
    if samples is None or samples.empty:
        return []
    cols = []
    for c in samples.columns:
        if str(c).lower() in ["run_tag", "run_id", "eta_scs"]:
            continue
        if run_df is not None and c not in run_df.columns:
            continue
        vals = safe_num(samples[c])
        if vals.notna().sum() > 5 and vals.std(ddof=0) > 1e-12:
            cols.append(c)
    return cols


def _fit_standardized_ols_for_outcome(run_df, samples, outcome):
    """
    Fit run-level OLS:
        outcome_r ~ sampled parameters_r

    Predictors are standardized. The outcome is a portfolio quantity, e.g.
        DWC_share_budget or SCS_share_budget.

    Returns:
        result, param_cols, means, stds
    """
    import statsmodels.api as sm

    param_cols = _get_sampled_param_cols_from_samples(samples, run_df)
    cols = [outcome] + param_cols

    d = run_df[cols].copy()
    for c in cols:
        d[c] = safe_num(d[c])
    d = d.dropna()

    if len(d) < max(10, len(param_cols) + 3):
        return None, [], None, None

    X_raw = d[param_cols].astype(float)
    usable = [c for c in X_raw.columns if X_raw[c].std(ddof=0) > 1e-12]
    if not usable:
        return None, [], None, None

    X_raw = X_raw[usable]
    means = X_raw.mean()
    stds = X_raw.std(ddof=0).replace(0, 1.0)

    X = (X_raw - means) / stds
    X = sm.add_constant(X, has_constant="add")
    y = d[outcome].astype(float)

    res = sm.OLS(y, X).fit(cov_type="HC3")
    return res, usable, means, stds


def _predict_ols_share(res, param_cols, means, stds, values_raw):
    """
    Predict portfolio share from a standardized OLS model.

    values_raw is a dict of raw parameter values for only the parameters being
    varied. All other parameters are held at their mean.
    """
    x = {"const": 1.0}
    for p in param_cols:
        raw = values_raw.get(p, means[p])
        x[p] = (raw - means[p]) / stds[p]

    X = pd.DataFrame([x])
    # preserve model column order
    X = X[res.params.index]
    yhat = float(res.predict(X).iloc[0])
    return float(np.clip(yhat, 0.0, 1.0))


def _fmt_param_value(p, x):
    """Compact axis tick label for common GSA parameters."""
    if p in ["g_aadt", "g_ev", "g_lcoe"]:
        return f"{100*x:.1f}%"
    if p in ["ETA_DWC", "ETA_SCS_SHARE"]:
        return f"{x:.2f}"
    if p == "project_life":
        return f"{int(round(x))}"
    if p == "C_DWC_base":
        return f"${x/1e6:.1f}M"
    if p == "C_SCS_base_per_port":
        return f"${x/1e3:.0f}k"
    if p == "N_SCS_stalls":
        return f"{int(round(x))}"
    return short_num(x, 2)


def _pretty_param(p):
    """Use existing PARAM_PRETTY if available."""
    try:
        return PARAM_PRETTY.get(p, p)
    except NameError:
        return p


def plot_portfolio_influence_ranking_nature(coef_df, code_map, fig_dir):
    """
    Nature-style Cleveland dot plot for run-level portfolio response.

    Uses run-level coefficients, not segment-level MNLogit coefficients.

    Inputs
    ------
    coef_df : output of run_level_regressions(), with columns:
              outcome, parameter, coef_standardized, se, p_value
    code_map : dict {parameter: P_code}
    """
    if coef_df is None or coef_df.empty or not code_map:
        print("  skipped portfolio influence ranking: missing coefficients or code_map")
        return

    rows = []
    for p, code in code_map.items():
        row = {"parameter": p, "P_code": code, "pretty": _pretty_param(p)}
        for out in ["DWC_share_budget", "SCS_share_budget"]:
            hit = coef_df[(coef_df["outcome"] == out) & (coef_df["parameter"] == p)]
            if hit.empty:
                row[f"{out}_coef"] = np.nan
                row[f"{out}_se"] = np.nan
                row[f"{out}_p"] = np.nan
            else:
                row[f"{out}_coef"] = float(hit["coef_standardized"].iloc[0])
                row[f"{out}_se"] = float(hit["se"].iloc[0])
                row[f"{out}_p"] = float(hit["p_value"].iloc[0])
        row["sort_abs"] = np.nanmax([
            abs(row.get("DWC_share_budget_coef", np.nan)),
            abs(row.get("SCS_share_budget_coef", np.nan)),
        ])
        rows.append(row)

    d = pd.DataFrame(rows).sort_values("sort_abs", ascending=True)
    y = np.arange(len(d))

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), sharey=True)
    specs = [
        ("DWC_share_budget", "DWC budget share", C_DWC),
        ("SCS_share_budget", "SCS budget share", C_SCS),
    ]

    for ax, (out, title, color) in zip(axes, specs):
        ax.axvline(0, color=C_TEXT, lw=0.8)

        for i, (_, r) in enumerate(d.iterrows()):
            beta = r[f"{out}_coef"]
            se = r[f"{out}_se"]
            if pd.isna(beta) or pd.isna(se):
                continue
            lo = beta - 1.96 * se
            hi = beta + 1.96 * se

            ax.plot([lo, hi], [i, i], color=color, lw=1.6, alpha=0.85, solid_capstyle="round")
            ax.scatter(beta, i, color=color, s=38, edgecolors="white", linewidths=0.5, zorder=3)

        ax.set_title(title)
        ax.set_xlabel("Change in budget share per 1 s.d. increase")
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.grid(axis="x", linestyle=":", color=C_GRID, lw=0.7)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{r.P_code}" for _, r in d.iterrows()])
    fig.suptitle("Run-level GSA influence on fixed-budget portfolio allocation",
                 fontsize=14, fontweight="bold")

    # compact note, outside axes
    fig.text(
        0.01, 0.01,
        "Bars show 95% robust confidence intervals from run-level OLS. "
        "Outcome is share of p_param budget, not probability over all candidate segments.",
        ha="left", va="bottom", fontsize=8.5, color=C_MUTED,
    )

    savefig(fig, Path(fig_dir) / "fig_I_portfolio_influence_ranking_CI.png")


def plot_portfolio_phase_diagram_nature(run_df, samples, code_map, fig_dir):
    """
    Two-parameter portfolio phase diagram.

    Uses top two parameters in code_map. Each panel shows predicted share of
    p_param budget allocated to DWC or SCS, holding all other sampled parameters
    at their mean.

    This replaces segment-level P(DWC) / P(SCS) phase surfaces.
    """
    if run_df is None or run_df.empty or samples is None or samples.empty or len(code_map) < 2:
        print("  skipped portfolio phase diagram: missing run table, samples, or code_map")
        return

    params = list(code_map.keys())[:2]
    px, py = params[0], params[1]
    cx, cy = code_map[px], code_map[py]

    res_d, cols_d, mu_d, sd_d = _fit_standardized_ols_for_outcome(run_df, samples, "DWC_share_budget")
    res_s, cols_s, mu_s, sd_s = _fit_standardized_ols_for_outcome(run_df, samples, "SCS_share_budget")
    if res_d is None or res_s is None:
        print("  skipped portfolio phase diagram: OLS model could not be fit")
        return

    x = safe_num(run_df[px]).dropna()
    y = safe_num(run_df[py]).dropna()
    if x.empty or y.empty:
        print("  skipped portfolio phase diagram: top parameters missing from run_df")
        return

    x_grid = np.linspace(x.quantile(0.05), x.quantile(0.95), 160)
    y_grid = np.linspace(y.quantile(0.05), y.quantile(0.95), 160)
    XX, YY = np.meshgrid(x_grid, y_grid)

    Z_d = np.zeros_like(XX, dtype=float)
    Z_s = np.zeros_like(XX, dtype=float)

    for i in range(XX.shape[0]):
        for j in range(XX.shape[1]):
            vals = {px: XX[i, j], py: YY[i, j]}
            Z_d[i, j] = _predict_ols_share(res_d, cols_d, mu_d, sd_d, vals)
            Z_s[i, j] = _predict_ols_share(res_s, cols_s, mu_s, sd_s, vals)

    cmap_d = LinearSegmentedColormap.from_list("dwc_phase", ["#F4F6F7", "#A8B6C3", C_DWC])
    cmap_s = LinearSegmentedColormap.from_list("scs_phase", ["#F4F6F7", "#9BD8C5", C_SCS])

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), constrained_layout=True)

    for ax, Z, cmap, title in [
        (axes[0], Z_d, cmap_d, "DWC budget share"),
        (axes[1], Z_s, cmap_s, "SCS budget share"),
    ]:
        im = ax.contourf(XX, YY, Z, levels=18, cmap=cmap, vmin=0, vmax=max(0.01, np.nanmax(Z)))
        ct = ax.contour(XX, YY, Z, levels=6, colors="white", linewidths=0.55, alpha=0.85)
        ax.clabel(ct, inline=True, fontsize=7, fmt=lambda v: f"{100*v:.0f}%")

        ax.scatter([x.median()], [y.median()], marker="X", s=50,
                   color="white", edgecolors=C_TEXT, linewidths=0.7, zorder=5)
        ax.set_xlabel(f"{cx}")
        ax.set_ylabel(f"{cy}")
        ax.set_title(title)

        ax.set_xticks(np.linspace(x_grid.min(), x_grid.max(), 3))
        ax.set_xticklabels([_fmt_param_value(px, t) for t in np.linspace(x_grid.min(), x_grid.max(), 3)])
        ax.set_yticks(np.linspace(y_grid.min(), y_grid.max(), 3))
        ax.set_yticklabels([_fmt_param_value(py, t) for t in np.linspace(y_grid.min(), y_grid.max(), 3)])

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Predicted share of p_param budget", rotation=270, labelpad=14)
        cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    fig.suptitle("Portfolio phase diagram for top GSA parameters",
                 fontsize=14, fontweight="bold")
    savefig(fig, Path(fig_dir) / "fig_J_portfolio_phase_diagram.png")


def plot_portfolio_marginal_sweeps_nature(run_df, samples, code_map, fig_dir):
    """
    Small-multiple marginal sweeps.

    For each top parameter:
      x-axis: actual sampled parameter value
      y-axis: predicted share of p_param budget

    This replaces marginal probability sweeps.
    """
    if run_df is None or run_df.empty or samples is None or samples.empty or not code_map:
        print("  skipped portfolio marginal sweeps: missing run table, samples, or code_map")
        return

    res_d, cols_d, mu_d, sd_d = _fit_standardized_ols_for_outcome(run_df, samples, "DWC_share_budget")
    res_s, cols_s, mu_s, sd_s = _fit_standardized_ols_for_outcome(run_df, samples, "SCS_share_budget")
    if res_d is None or res_s is None:
        print("  skipped portfolio marginal sweeps: OLS models could not be fit")
        return

    params = list(code_map.keys())[:6]
    n = len(params)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(10.2, 3.0 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    all_y = []

    plot_data = {}
    for p in params:
        s = safe_num(run_df[p]).dropna()
        if s.empty:
            continue
        xs = np.linspace(s.quantile(0.05), s.quantile(0.95), 120)
        yd = np.array([_predict_ols_share(res_d, cols_d, mu_d, sd_d, {p: x}) for x in xs])
        ys = np.array([_predict_ols_share(res_s, cols_s, mu_s, sd_s, {p: x}) for x in xs])
        plot_data[p] = (xs, yd, ys)
        all_y.extend(yd.tolist())
        all_y.extend(ys.tolist())

    ymax = max(0.01, np.nanpercentile(all_y, 98) * 1.2) if all_y else 1.0

    for ax, p in zip(axes, params):
        if p not in plot_data:
            ax.axis("off")
            continue

        xs, yd, ys = plot_data[p]
        code = code_map[p]

        ax.plot(xs, yd, color=C_DWC, lw=1.8, label="DWC")
        ax.plot(xs, ys, color=C_SCS, lw=1.8, ls="--", label="SCS")
        ax.axvline(np.median(xs), color=C_MUTED, lw=0.75, ls=":", alpha=0.75)

        ax.set_title(f"{code}: {_pretty_param(p)}", fontsize=10.5)
        ax.set_ylim(0, ymax)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        ax.grid(axis="y", linestyle=":", color=C_GRID, lw=0.65)

        ticks = np.linspace(xs.min(), xs.max(), 3)
        ax.set_xticks(ticks)
        ax.set_xticklabels([_fmt_param_value(p, t) for t in ticks], fontsize=8.5)

    for ax in axes[len(params):]:
        ax.axis("off")

    axes[0].set_ylabel("Predicted budget share")
    axes[0].legend(loc="best", frameon=False)

    fig.suptitle("Marginal portfolio response to sampled parameters",
                 fontsize=14, fontweight="bold")
    savefig(fig, Path(fig_dir) / "fig_K_portfolio_marginal_sweeps.png")


def bootstrap_portfolio_rank_stability(run_df, samples, n_boot=200, seed=42, top_only=False):
    """
    Bootstrap run-level parameter ranking.

    Resamples GSA runs with replacement, fits run-level OLS for DWC and SCS
    budget share, and records the rank of each sampled parameter by |beta|.

    Returns a dataframe suitable for plot_portfolio_bootstrap_rank_stability().
    """
    if run_df is None or run_df.empty or samples is None or samples.empty:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    param_cols = _get_sampled_param_cols_from_samples(samples, run_df)
    if not param_cols:
        return pd.DataFrame()

    outcomes = ["DWC_share_budget", "SCS_share_budget"]
    boot_rows = []

    n = len(run_df)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_df = run_df.iloc[idx].copy()

        for outcome in outcomes:
            res, usable, mu, sd = _fit_standardized_ols_for_outcome(boot_df, samples, outcome)
            if res is None:
                continue

            coefs = []
            for p in usable:
                if p in res.params.index:
                    coefs.append((p, float(res.params[p])))

            if not coefs:
                continue

            ranked = (
                pd.DataFrame(coefs, columns=["parameter", "coef"])
                .assign(abs_coef=lambda x: x["coef"].abs())
                .sort_values("abs_coef", ascending=False)
                .reset_index(drop=True)
            )
            ranked["rank"] = np.arange(1, len(ranked) + 1)

            for _, r in ranked.iterrows():
                boot_rows.append({
                    "bootstrap": b,
                    "outcome": outcome,
                    "parameter": r["parameter"],
                    "coef": r["coef"],
                    "rank": int(r["rank"]),
                })

    dfb = pd.DataFrame(boot_rows)
    if dfb.empty:
        return dfb

    summary = (
        dfb.groupby(["outcome", "parameter"])
        .agg(
            mean_rank=("rank", "mean"),
            std_rank=("rank", "std"),
            pct_top1=("rank", lambda x: float((x <= 1).mean())),
            pct_top3=("rank", lambda x: float((x <= 3).mean())),
            mean_coef=("coef", "mean"),
            n_valid_bootstraps=("bootstrap", "nunique"),
        )
        .reset_index()
    )
    return summary


def plot_portfolio_bootstrap_rank_stability(df_boot, fig_dir):
    """
    Bootstrap rank-stability plot for portfolio-response ranking.
    """
    if df_boot is None or df_boot.empty:
        print("  skipped bootstrap rank stability: empty bootstrap table")
        return

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), sharey=False)
    specs = [
        ("DWC_share_budget", "DWC budget-share ranking", C_DWC),
        ("SCS_share_budget", "SCS budget-share ranking", C_SCS),
    ]

    for ax, (outcome, title, color) in zip(axes, specs):
        d = df_boot[df_boot["outcome"] == outcome].copy()
        if d.empty:
            ax.axis("off")
            continue

        d = d.sort_values("mean_rank", ascending=True).head(9).reset_index(drop=True)
        y = np.arange(len(d))

        for i, (_, r) in enumerate(d.iterrows()):
            mr = float(r["mean_rank"])
            sr = float(r["std_rank"]) if not pd.isna(r["std_rank"]) else 0.0
            q25 = max(1.0, mr - 0.674 * sr)
            q75 = mr + 0.674 * sr
            wlo = max(1.0, mr - 1.5 * (q75 - q25))
            whi = mr + 1.5 * (q75 - q25)

            if i % 2 == 0:
                ax.axhspan(i - 0.5, i + 0.5, color=C_LIGHT, lw=0, zorder=0)

            ax.plot([wlo, whi], [i, i], color=color, lw=1.0, alpha=0.65)
            ax.add_patch(
                patches.FancyBboxPatch(
                    (q25, i - 0.20),
                    max(q75 - q25, 0.02),
                    0.40,
                    boxstyle="round,pad=0.03",
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.24,
                    zorder=2,
                )
            )
            ax.plot([mr, mr], [i - 0.24, i + 0.24], color=color, lw=1.5)
            ax.text(9.4, i, f"{100*r['pct_top3']:.0f}%", va="center", ha="right",
                    fontsize=8.5, color=color)

        ax.axvline(3.5, color=C_MUTED, ls=":", lw=0.85)
        ax.set_xlim(0.5, 9.8)
        ax.set_xticks(range(1, 10))
        ax.set_xlabel("Rank by |standardized coefficient|")
        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_yticklabels([_pretty_param(p) for p in d["parameter"]], fontsize=8.5)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle=":", color=C_GRID, lw=0.65)
        ax.text(9.4, -0.65, "% top 3", fontsize=8, color=C_MUTED, ha="right")

    fig.suptitle("Bootstrap stability of run-level portfolio ranking",
                 fontsize=14, fontweight="bold")
    savefig(fig, Path(fig_dir) / "fig_L_portfolio_bootstrap_rank_stability.png")
# ═══════════════════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default=str(repo_root / "results" / "Results_baseline_p200_etaDWC0.50_etaSCS0.50_gev15pct_gaadt011bps"),
    )
    parser.add_argument(
        "--gsa_dir",
        type=str,
        default=str(repo_root / "results" / "GSA_example"),
    )
    parser.add_argument(
        "--input_segments",
        type=str,
        default=str(repo_root / "data" / "processed" / "NHS_US_5miles_with_SCS.csv"),
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--p_param", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=6)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    gsa_dir = Path(args.gsa_dir)
    input_segments = Path(args.input_segments)

    if args.output_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = baseline_dir.parent / f"Policy_Insights_Portfolio_{ts}"
    else:
        output_dir = Path(args.output_dir)

    table_dir = output_dir / "tables"
    fig_dir = output_dir / "figures"
    mkdir(table_dir)
    mkdir(fig_dir)

    print("═" * 80)
    print("DWC + SCS Portfolio-Centered Policy Insight Analysis")
    print(f"Baseline results : {baseline_dir}")
    print(f"GSA folder       : {gsa_dir}")
    print(f"Input segments   : {input_segments}")
    print(f"Output           : {output_dir}")
    print(f"Main_plots colors imported: {COL['_imported']}")
    print("═" * 80)

    print("\n1) Building joined baseline segment table...")
    df, df_dwc, df_scs, df_meta, bin_thresholds, info = build_joined_segment_table(
        baseline_dir=baseline_dir,
        input_segments_path=input_segments,
        p_param_fallback=args.p_param,
    )

    p_param = int(info.get("p_param", args.p_param))

    df.to_csv(table_dir / "joined_baseline_segment_table.csv", index=False)
    bin_thresholds.to_csv(table_dir / "driver_bin_thresholds.csv", index=False)
    with open(table_dir / "input_join_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(json.dumps(info, indent=2))

    print("\n2) Building baseline portfolio allocation tables...")
    portfolio_tables = make_all_baseline_portfolio_tables(df)
    for name, tab in portfolio_tables.items():
        tab.to_csv(table_dir / f"portfolio_table_{name}.csv", index=False)

    comp = make_portfolio_composition_table(df)
    comp.to_csv(table_dir / "portfolio_composition_table.csv", index=False)

    policy = make_policy_readout_matrix(df)
    policy.to_csv(table_dir / "policy_readout_matrix_portfolio_centered.csv", index=False)

    print("\n3) Generating baseline portfolio figures...")
    plot_budget_allocation_heatmaps(portfolio_tables, fig_dir)
    plot_enrichment_heatmaps(portfolio_tables, fig_dir)
    plot_portfolio_composition(comp, fig_dir)
    plot_policy_readout_matrix(policy, fig_dir)
    plot_baseline_context(df, fig_dir)

    print("\n4) Building GSA run-level portfolio response table...")
    run_df, samples = build_gsa_run_level_table(gsa_dir, df, p_param=p_param)
    if run_df.empty:
        print("  WARNING: no GSA run-level selected output files found under GSA/Runs.")
    else:
        run_df.to_csv(table_dir / "gsa_run_level_portfolio_response.csv", index=False)
        print(f"  found {len(run_df)} GSA run result folders")

        coef_run = run_level_regressions(run_df, samples)
        if coef_run.empty:
            print("  WARNING: run-level regressions could not be fit.")
        else:
            top_run, code_map, coef_coded = assign_run_level_parameter_codes(
                coef_run,
                outcome="DWC_share_budget",
                top_k=args.top_k,
            )
            coef_coded.to_csv(table_dir / "gsa_run_level_regression_coefficients.csv", index=False)
            top_run.to_csv(table_dir / "gsa_run_level_parameter_key.csv", index=False)

            print("\n5) Generating GSA run-level portfolio figures...")
            plot_run_parameter_key(top_run, fig_dir)
            plot_run_level_tornado(coef_coded, code_map, fig_dir)
            plot_run_level_parameter_bin_response(run_df, code_map, fig_dir)
            # Extra Nature-style portfolio-response plots
            plot_portfolio_influence_ranking_nature(coef_coded, code_map, fig_dir)
            plot_portfolio_phase_diagram_nature(run_df, samples, code_map, fig_dir)
            plot_portfolio_marginal_sweeps_nature(run_df, samples, code_map, fig_dir)

            df_boot_portfolio = bootstrap_portfolio_rank_stability(
                run_df=run_df,
                samples=samples,
                n_boot=200,
                seed=42,
            )
            if not df_boot_portfolio.empty:
                df_boot_portfolio.to_csv(
                    table_dir / "gsa_run_level_bootstrap_rank_stability.csv",
                    index=False,
                )
                plot_portfolio_bootstrap_rank_stability(df_boot_portfolio, fig_dir)

    methods = f"""
Portfolio-centered Policy_Insight_Analysis.py
=============================================

Core interpretation
-------------------
The optimizer is treated as a fixed-budget portfolio selector. The main
denominator for baseline deployment figures is the selected portfolio, not the
full candidate universe.

Budget
------
p_param used: {p_param}
Selected DWC: {info.get("n_dwc_selected")}
Selected SCS: {info.get("n_scs_selected")}
Selected total: {info.get("n_selected_total")}
Unused or rounding budget: {info.get("unused_or_rounding_budget")}

Data construction
-----------------
Original input segment file:
{input_segments}

Electricity_Cost is derived from:
{info.get("electricity_column_used")}
Unit handling:
{info.get("electricity_unit")}

DWC selected matching:
DWC_Selected_Segments.Segment_ID == NHS_US_5miles_with_SCS.SEGMENT_ID_OG

SCS selected matching:
SCS_Selected_Stations.Host_Seg == NHS_US_5miles_with_SCS.SEGMENT_ID_OG

Main baseline quantities
------------------------
DWC budget share in a bin:
selected DWC in bin / total selected DWC

Candidate share in a bin:
candidate segments in bin / total candidate segments

Enrichment:
DWC budget share / candidate share

GSA run-level quantities
------------------------
Each GSA run is aggregated into one row:
n_DWC, n_SCS, DWC_share_budget, SCS_share_budget, selected portfolio medians.

Run-level regressions explain portfolio composition, not rare-event selection
probability over all candidate segments.
"""
    (table_dir / "methods_note_portfolio_centered.txt").write_text(methods.strip() + "\n")

    print("\nComplete.")
    print(f"Tables : {table_dir}")
    print(f"Figures: {fig_dir}")


if __name__ == "__main__":
    main()


# # cLAUDE:
# #!/usr/bin/env python3
# """
# Sensitivity_Plots.py
# ====================
# Nature Energy–grade publication figures for the DWC + SCS corridor GSA.

# Stand-alone / mock mode (no MILP needed):
#     python Sensitivity_Plots.py

# Real-data mode:
#     from Sensitivity_Plots import generate_all_plots
#     generate_all_plots('/path/to/GSA_20240618_120000/')

# Figures produced
# ----------------
# Fig 1  Parameter influence ranking (signed coefficients + cluster-robust CIs)
# Fig 2  Technology phase diagram (log relative-risk surface, top-2 parameter space)
# Fig 3  Marginal probability sweeps (3×3 small-multiples, log-probability y-axis)
# Fig 4  Bootstrap rank stability (box plots by run-level resampling)
# """

# import os
# import numpy as np
# import pandas as pd
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# import matplotlib.colors as mcolors
# import matplotlib.patches as mpatches
# from matplotlib.gridspec import GridSpec
# from matplotlib.lines import Line2D
# from scipy.stats import norm as sp_norm

# # ═══════════════════════════════════════════════════════════════════════════════
# # §1  CONSTANTS & PARAMETER METADATA
# # ═══════════════════════════════════════════════════════════════════════════════

# MM         = 1 / 25.4          # mm → inches
# DOUBLE_COL = 183 * MM          # Nature double-column width
# SINGLE_COL =  89 * MM          # Nature single-column width

# # Consistent color palette across all four figures
# C = dict(
#     dwc  = '#1A5276',   # deep navy  — DWC
#     scs  = '#922B21',   # deep red   — SCS
#     none = '#7F8C8D',   # slate gray — unselected
#     pos  = '#2166AC',   # positive coefficient
#     neg  = '#D73027',   # negative coefficient
#     ctrl = '#4A235A',   # segment-level control variables
#     band = '#F4F6F7',   # alternating row tint
#     grid = '#D5D8DC',   # grid / spine color
#     lo   = '#EBF5FB',   # light shade for heatmaps
# )

# THETA_COLS = [
#     'ETA_DWC', 'ETA_SCS_SHARE', 'g_aadt', 'g_ev',
#     'project_life', 'C_DWC_base', 'C_SCS_base_per_port',
#     'N_SCS_stalls', 'g_lcoe',
# ]
# SEG_CTRL = ['AADT_COM', 'Min_Sub_Dist']
# ALL_PRED = THETA_COLS + SEG_CTRL

# # Physical parameter ranges swept by the LHS
# BOUNDS = {
#     'ETA_DWC'            : [0.20, 0.80],
#     'ETA_SCS_SHARE'      : [0.20, 0.80],
#     'g_aadt'             : [0.008, 0.020],
#     'g_ev'               : [0.10,  0.18],
#     'project_life'       : [25,    40],
#     'C_DWC_base'         : [8e6,   12e6],
#     'C_SCS_base_per_port': [180e3, 300e3],
#     'N_SCS_stalls'       : [4,     8],
#     'g_lcoe'             : [0.005, 0.025],
# }

# # Display labels: (LaTeX symbol, plain-text description)
# LABEL = {
#     'ETA_DWC'            : (r'$\eta^{\mathrm{DWC}}$',                 'DWC capture fraction'),
#     'ETA_SCS_SHARE'      : (r'$\eta^{\mathrm{SCS}}_{\mathrm{share}}$','SCS residual capture share'),
#     'g_aadt'             : (r'$g^{\mathrm{AADT}}$',                   'Traffic CAGR'),
#     'g_ev'               : (r'$g^{\mathrm{EV}}$',                     'EV fleet growth CAGR'),
#     'project_life'       : (r'$T$',                                   'Planning horizon (yr)'),
#     'C_DWC_base'         : (r'$\mathcal{C}^{\mathrm{DWC}}$',         'DWC unit capex'),
#     'C_SCS_base_per_port': (r'$\mathcal{C}^{\mathrm{SCS}}_{\mathrm{port}}$', 'SCS capex per port'),
#     'N_SCS_stalls'       : (r'$N^{\mathrm{SCS}}$',                   'Ports per station'),
#     'g_lcoe'             : (r'$g^{\mathrm{LCOE}}$',                  'LCOE growth CAGR'),
#     'AADT_COM'           : (r'AADT',                                  'Segment AADT (control)'),
#     'Min_Sub_Dist'       : (r'$d_{\mathrm{sub}}$',                   'Substation distance (control)'),
# }

# # Axis tick formatters for marginal-effect x-axes (actual parameter scale)
# FMT = {
#     'ETA_DWC'            : lambda x: f'{x:.2f}',
#     'ETA_SCS_SHARE'      : lambda x: f'{x:.2f}',
#     'g_aadt'             : lambda x: f'{100*x:.1f}%',
#     'g_ev'               : lambda x: f'{100*x:.0f}%',
#     'project_life'       : lambda x: str(int(round(x))),
#     'C_DWC_base'         : lambda x: f'${x/1e6:.0f}M',
#     'C_SCS_base_per_port': lambda x: f'${x/1e3:.0f}k',
#     'N_SCS_stalls'       : lambda x: str(int(round(x))),
#     'g_lcoe'             : lambda x: f'{100*x:.1f}%',
# }


# # ═══════════════════════════════════════════════════════════════════════════════
# # §2  GLOBAL MATPLOTLIB STYLE  (Nature Energy aesthetic)
# # ═══════════════════════════════════════════════════════════════════════════════

# def _set_style():
#     mpl.rcParams.update({
#         'font.family'          : 'sans-serif',
#         'font.sans-serif'      : ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
#         'font.size'            : 7,
#         'axes.titlesize'       : 7.5,
#         'axes.labelsize'       : 7,
#         'xtick.labelsize'      : 6,
#         'ytick.labelsize'      : 6,
#         'legend.fontsize'      : 6,
#         'legend.framealpha'    : 0.92,
#         'legend.edgecolor'     : '#C8C8C8',
#         'legend.borderpad'     : 0.4,
#         'figure.dpi'           : 300,
#         'savefig.dpi'          : 300,
#         'savefig.bbox'         : 'tight',
#         'savefig.pad_inches'   : 0.02,
#         'axes.linewidth'       : 0.5,
#         'axes.spines.top'      : False,
#         'axes.spines.right'    : False,
#         'xtick.major.width'    : 0.5,
#         'ytick.major.width'    : 0.5,
#         'xtick.major.size'     : 2.5,
#         'ytick.major.size'     : 2.5,
#         'xtick.minor.size'     : 1.5,
#         'ytick.minor.size'     : 1.5,
#         'lines.linewidth'      : 1.0,
#         'patch.linewidth'      : 0.5,
#         'grid.linewidth'       : 0.35,
#         'grid.color'           : C['grid'],
#         'grid.alpha'           : 0.8,
#         'pdf.fonttype'         : 42,   # Type-2 fonts → embed in PDF for journal
#         'ps.fonttype'          : 42,
#         'figure.constrained_layout.use': True,
#     })


# # ═══════════════════════════════════════════════════════════════════════════════
# # §3  MNL HELPER
# # ═══════════════════════════════════════════════════════════════════════════════

# def _mnl_probs(eta_dwc, eta_scs):
#     """
#     Vectorized multinomial-logit probabilities.
#     Reference class is None (eta_none ≡ 0).
#     """
#     e_d = np.exp(np.clip(np.asarray(eta_dwc, float), -60, 60))
#     e_s = np.exp(np.clip(np.asarray(eta_scs, float), -60, 60))
#     denom = 1.0 + e_d + e_s
#     return e_d / denom, e_s / denom, 1.0 / denom


# def _sig_stars(p):
#     if p < 0.001: return '***'
#     if p < 0.01:  return '**'
#     if p < 0.05:  return '*'
#     return ''


# # ═══════════════════════════════════════════════════════════════════════════════
# # §4  MOCK DATA GENERATION
# # Reproduces the output file structure of Main_sensitivity.py so all
# # plotting functions can be developed without the MILP sweep.
# # ═══════════════════════════════════════════════════════════════════════════════

# def generate_mock_data(out_dir='mock_GSA', n_runs=1200, seed=42):
#     """
#     Write synthetic CSVs mirroring Main_sensitivity.py output.

#     Coefficient magnitudes are physically motivated:
#     • ETA_DWC, C_DWC_base dominate DWC selection (revenue and capex channels)
#     • C_SCS_base_per_port, ETA_SCS_SHARE dominate SCS selection
#     • g_ev is the third-ranked driver for both (shared demand channel)
#     • Intercepts set so baseline class balance ≈ 97.8% None / 1.5% DWC / 0.7% SCS
#       (realistic for a large national network with a tight p-param budget)
#     """
#     rng = np.random.default_rng(seed)
#     os.makedirs(out_dir, exist_ok=True)

#     # ── True standardized MNLogit coefficients ────────────────────────────────
#     # Intercepts calibrated so that at standardized-mean predictors:
#     #   P(DWC) ≈ 1.5%,  P(SCS) ≈ 0.7%,  P(None) ≈ 97.8%
#     TRUE = {
#         'DWC': dict(
#             const              = -4.18,
#             ETA_DWC            = +1.82,   # rank 1: capture fraction → revenue channel
#             C_DWC_base         = -1.51,   # rank 2: capex penalizes NPV directly
#             g_ev               = +0.93,   # rank 3: demand growth → higher lifetime yield
#             project_life       = +0.71,   # rank 4: longer Γ_rev → NPV rises
#             g_aadt             = +0.62,   # rank 5: traffic growth (correlated with g_ev)
#             g_lcoe             = -0.41,   # rank 6: energy cost erodes margin
#             ETA_SCS_SHARE      = -0.28,   # rank 7: more SCS capture → SCS preferred at margin
#             C_SCS_base_per_port= +0.19,   # rank 8: SCS costlier → budget shifts to DWC
#             N_SCS_stalls       = +0.12,   # rank 9: weak reallocation effect
#             AADT_COM           = +1.24,   # control: high traffic → DWC clearly worthwhile
#             Min_Sub_Dist       = -0.87,   # control: far from grid → feeder capex prohibitive
#         ),
#         'SCS': dict(
#             const              = -4.93,
#             C_SCS_base_per_port= -1.68,   # rank 1: symmetric capex channel for SCS
#             ETA_SCS_SHARE      = +1.44,   # rank 2: residual capture → SCS revenue
#             g_ev               = +0.88,   # rank 3: shared demand channel
#             N_SCS_stalls       = +0.72,   # rank 4: more ports → capacity + revenue both up
#             project_life       = +0.61,   # rank 5: longer Γ_rev_SCS
#             ETA_DWC            = -0.52,   # rank 6: more DWC capture → DWC outcompetes SCS
#             g_aadt             = +0.47,   # rank 7: traffic growth
#             C_DWC_base         = +0.34,   # rank 8: DWC costlier → optimizer prefers SCS
#             g_lcoe             = -0.29,   # rank 9: energy cost
#             AADT_COM           = +0.93,   # control
#             Min_Sub_Dist       = -0.61,   # control
#         ),
#     }

#     # ── mnlogit_coefficients.csv ──────────────────────────────────────────────
#     rows = [dict(
#         parameter='const', is_segment_control=False, is_intercept=True,
#         coef_DWC=TRUE['DWC']['const'],  coef_SCS=TRUE['SCS']['const'],
#         se_DWC=0.24,                    se_SCS=0.28,
#         p_DWC=sp_norm.sf(abs(TRUE['DWC']['const']/0.24))*2,
#         p_SCS=sp_norm.sf(abs(TRUE['SCS']['const']/0.28))*2,
#         abs_coef_DWC=np.nan, abs_coef_SCS=np.nan,
#         rank_DWC=np.nan,     rank_SCS=np.nan,
#     )]
#     for p in ALL_PRED:
#         c_d = TRUE['DWC'].get(p, 0.0)
#         c_s = TRUE['SCS'].get(p, 0.0)
#         # Cluster-robust SEs: ~12% of |coef| with small run-to-run jitter
#         se_d = max(0.07, abs(c_d) * 0.12) * rng.uniform(0.88, 1.12)
#         se_s = max(0.07, abs(c_s) * 0.12) * rng.uniform(0.88, 1.12)
#         rows.append(dict(
#             parameter=p, is_segment_control=(p in SEG_CTRL), is_intercept=False,
#             coef_DWC=c_d, coef_SCS=c_s,
#             se_DWC=se_d,  se_SCS=se_s,
#             p_DWC=sp_norm.sf(abs(c_d / se_d)) * 2 if se_d > 0 else 1.0,
#             p_SCS=sp_norm.sf(abs(c_s / se_s)) * 2 if se_s > 0 else 1.0,
#             abs_coef_DWC=abs(c_d), abs_coef_SCS=abs(c_s),
#             rank_DWC=np.nan, rank_SCS=np.nan,
#         ))
#     df_c = pd.DataFrame(rows)
#     tmask = ~df_c['is_segment_control'] & ~df_c['is_intercept']
#     df_c.loc[tmask, 'rank_DWC'] = df_c.loc[tmask, 'abs_coef_DWC'].rank(ascending=False)
#     df_c.loc[tmask, 'rank_SCS'] = df_c.loc[tmask, 'abs_coef_SCS'].rank(ascending=False)
#     df_c.to_csv(os.path.join(out_dir, 'mnlogit_coefficients.csv'), index=False)

#     # ── lhs_parameter_samples.csv ─────────────────────────────────────────────
#     lhs = {}
#     for param, (lo, hi) in BOUNDS.items():
#         if param in ('project_life', 'N_SCS_stalls'):
#             lhs[param] = rng.integers(int(lo), int(hi) + 1, n_runs).astype(float)
#         else:
#             u = (rng.permutation(n_runs) + rng.uniform(size=n_runs)) / n_runs
#             lhs[param] = lo + u * (hi - lo)
#     df_lhs = pd.DataFrame(lhs)
#     df_lhs.insert(0, 'run_tag', [f'gsa_{i:04d}' for i in range(n_runs)])
#     df_lhs['ETA_SCS'] = df_lhs['ETA_SCS_SHARE'] * np.maximum(0.0, 0.9 - df_lhs['ETA_DWC'])
#     df_lhs.to_csv(os.path.join(out_dir, 'lhs_parameter_samples.csv'), index=False)

#     # ── feature_standardization.csv ───────────────────────────────────────────
#     mu = {p: df_lhs[p].mean() for p in THETA_COLS}
#     sd = {p: max(df_lhs[p].std(ddof=0), 1e-12) for p in THETA_COLS}
#     mu.update({'AADT_COM': 12000.0, 'Min_Sub_Dist': 8.5})
#     sd.update({'AADT_COM':  8000.0, 'Min_Sub_Dist': 5.2})
#     pd.DataFrame({'parameter': ALL_PRED,
#                   'mean': [mu[p] for p in ALL_PRED],
#                   'std':  [sd[p] for p in ALL_PRED],
#                   }).to_csv(os.path.join(out_dir, 'feature_standardization.csv'), index=False)

#     # ── bootstrap_ranking_stability.csv ───────────────────────────────────────
#     n_boot = 30
#     boot_rows = []
#     for cls in ['DWC', 'SCS']:
#         for p in ALL_PRED:
#             base_r_series = df_c.loc[df_c['parameter'] == p, f'rank_{cls}']
#             if base_r_series.empty or pd.isna(base_r_series.values[0]):
#                 continue
#             base_r = float(base_r_series.values[0])
#             # Simulate bootstrap rank with realistic noise: top params more stable
#             noise_sd = 0.6 + 0.15 * base_r   # higher-ranked params are more stable
#             raw = rng.normal(base_r, noise_sd, n_boot)
#             ranks = np.round(np.clip(raw, 1, len(ALL_PRED))).astype(int)
#             boot_rows.append(dict(**{'class': cls, 'parameter': p},
#                                   mean_rank=ranks.mean(), std_rank=ranks.std(),
#                                   pct_top1=float((ranks == 1).mean()),
#                                   pct_top3=float((ranks <= 3).mean()),
#                                   n_valid_bootstraps=n_boot))
#     pd.DataFrame(boot_rows).to_csv(
#         os.path.join(out_dir, 'bootstrap_ranking_stability.csv'), index=False)

#     print(f"Mock data written → {out_dir}/  ({n_runs} simulated runs)")
#     return out_dir, mu, sd


# # ═══════════════════════════════════════════════════════════════════════════════
# # §5  FIG 1 — PARAMETER INFLUENCE RANKING
# # ═══════════════════════════════════════════════════════════════════════════════

# def plot_influence_ranking(df_coef, out_dir, filename='fig1_influence_ranking.pdf'):
#     """
#     Figure 1: Two-panel Cleveland dot chart (DWC | SCS).

#     • Parameters sorted by |coef_DWC|, same order in both panels so the eye
#       can track rank changes across technologies.
#     • Horizontal CI bars use 95% cluster-robust SEs.
#     • Dot color encodes sign (blue = positive, red = negative).
#     • Segment controls shown with diamond marker and purple color.
#     • Thin dashed separator between theta params and segment controls.
#     """
#     _set_style()
#     CI_Z = 1.96

#     df_theta = df_coef[~df_coef['is_intercept'] & ~df_coef['is_segment_control']].copy()
#     df_ctrl  = df_coef[~df_coef['is_intercept'] &  df_coef['is_segment_control']].copy()
#     df_theta = df_theta.sort_values('abs_coef_DWC', ascending=False).reset_index(drop=True)
#     df_ctrl  = df_ctrl.sort_values('abs_coef_DWC',  ascending=False).reset_index(drop=True)
#     df_plot  = pd.concat([df_theta, df_ctrl], ignore_index=True)
#     n_theta  = len(df_theta)
#     n_total  = len(df_plot)

#     fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 115 * MM), sharey=True)

#     for ax_idx, (ax, cls) in enumerate(zip(axes, ['DWC', 'SCS'])):
#         # ── Alternating row bands ────────────────────────────────────────────
#         for i in range(n_total):
#             if i % 2 == 0:
#                 ax.axhspan(i - 0.5, i + 0.5, color=C['band'], zorder=0, lw=0)
#         ax.axvline(0, color='#606060', lw=0.7, ls='--', zorder=1, alpha=0.8)
#         ax.xaxis.grid(True, zorder=0, lw=0.35)

#         # ── Dots + CI bars ───────────────────────────────────────────────────
#         for i, row in df_plot.iterrows():
#             coef = row[f'coef_{cls}']
#             se   = row[f'se_{cls}']
#             pval = row[f'p_{cls}']
#             lo, hi = coef - CI_Z * se, coef + CI_Z * se
#             is_ctrl = bool(row['is_segment_control'])

#             dot_color = C['ctrl'] if is_ctrl else (C['pos'] if coef >= 0 else C['neg'])
#             marker    = 'D' if is_ctrl else 'o'
#             ms        = 14 if is_ctrl else 18   # scatter s units

#             ax.plot([lo, hi], [i, i], color=dot_color, lw=1.4, zorder=3,
#                     solid_capstyle='round', alpha=0.85)
#             ax.scatter(coef, i, color=dot_color, s=ms, zorder=5,
#                        marker=marker, edgecolors='white', linewidths=0.5)

#             stars = _sig_stars(pval)
#             if stars:
#                 ax.text(hi + 0.05, i, stars, va='center', ha='left',
#                         fontsize=5, color=dot_color, zorder=6)

#         # ── Separator between theta and controls ─────────────────────────────
#         ax.axhline(n_theta - 0.5, color='#AAAAAA', lw=0.8, ls=':', zorder=2)

#         # ── Axis formatting ──────────────────────────────────────────────────
#         all_coefs = df_plot[f'coef_{cls}'].values
#         all_ses   = df_plot[f'se_{cls}'].values
#         xlo = (all_coefs - CI_Z * all_ses).min()
#         xhi = (all_coefs + CI_Z * all_ses).max()
#         pad = (xhi - xlo) * 0.18
#         ax.set_xlim(xlo - pad, xhi + pad)
#         ax.set_xlabel(r'Standardized coefficient $\hat{\beta}$', labelpad=3)

#         panel_label = chr(97 + ax_idx)  # 'a', 'b'
#         tech = 'DWC' if cls == 'DWC' else 'SCS'
#         ax.set_title(
#             f'({panel_label})  {tech} vs. unselected',
#             loc='left', pad=4, fontweight='bold', fontsize=7.5
#         )

#         # ── Spine color ──────────────────────────────────────────────────────
#         spine_col = C['dwc'] if cls == 'DWC' else C['scs']
#         ax.spines['bottom'].set_color(spine_col)
#         ax.spines['left'].set_color(spine_col)
#         ax.tick_params(colors=spine_col, which='both')
#         ax.xaxis.label.set_color(spine_col)

#     # ── Y-axis labels (left panel only; sharey so applied globally) ───────────
#     tick_labels = []
#     for _, row in df_plot.iterrows():
#         sym, desc = LABEL[row['parameter']]
#         tick_labels.append(f'{sym}  {desc}')
#     axes[0].set_yticks(range(n_total))
#     axes[0].set_yticklabels(tick_labels, fontsize=6.0)
#     axes[0].invert_yaxis()

#     # ── Shared legend (right panel) ───────────────────────────────────────────
#     legend_handles = [
#         Line2D([0], [0], marker='o', color=C['pos'], lw=1.2, ms=3.5,
#                markerfacecolor=C['pos'], markeredgecolor='white', markeredgewidth=0.4,
#                label='Positive effect'),
#         Line2D([0], [0], marker='o', color=C['neg'], lw=1.2, ms=3.5,
#                markerfacecolor=C['neg'], markeredgecolor='white', markeredgewidth=0.4,
#                label='Negative effect'),
#         Line2D([0], [0], marker='D', color=C['ctrl'], lw=1.2, ms=3.0,
#                markerfacecolor=C['ctrl'], markeredgecolor='white', markeredgewidth=0.4,
#                label='Segment control'),
#     ]
#     axes[1].legend(handles=legend_handles, loc='lower right',
#                    handlelength=1.0, handletextpad=0.4, labelspacing=0.3)

#     # ── Footnotes ─────────────────────────────────────────────────────────────
#     fig.text(0.01, -0.01,
#              r'Bars: 95% CI (cluster-robust SEs, clustered on GSA run). '
#              r'$^{***}p<0.001$  $^{**}p<0.01$  $^{*}p<0.05$. '
#              r'All predictors standardized before fitting (mean 0, SD 1).',
#              ha='left', va='top', fontsize=5, color='#555555', style='italic')

#     out_path = os.path.join(out_dir, filename)
#     fig.savefig(out_path, bbox_inches='tight')
#     print(f"  Saved: {filename}")
#     return fig


# # ═══════════════════════════════════════════════════════════════════════════════
# # §6  FIG 2 — TECHNOLOGY PHASE DIAGRAM
# # ═══════════════════════════════════════════════════════════════════════════════

# def plot_phase_diagram(df_coef, feat_std_df, out_dir,
#                        param_x='ETA_DWC', param_y='C_DWC_base',
#                        filename='fig2_phase_diagram.pdf'):
#     """
#     Figure 2: Log relative-risk surface in the space of the two most
#     important theta parameters.

#     Panel (a): log₁₀[P(DWC) / P(None)] — how much more likely a segment
#                is to receive DWC relative to the unselected baseline.
#     Panel (b): log₁₀[P(SCS) / P(None)] — same for SCS.

#     The ratio is computed analytically from the fitted MNLogit coefficients,
#     holding all other standardized predictors at zero (their mean).

#     A dashed iso-contour marks the technology-parity line P(DWC) = P(SCS);
#     above it DWC is relatively more likely, below it SCS is. The white ×
#     marks the baseline operating point (standardized mean of both axes).
#     """
#     _set_style()

#     # ── Extract needed coefficients ───────────────────────────────────────────
#     coef = df_coef.set_index('parameter')
#     get  = lambda cls, p: float(coef.loc[p, f'coef_{cls}']) if p in coef.index else 0.0

#     b0_d = get('DWC', 'const')
#     b0_s = get('SCS', 'const')
#     bx_d = get('DWC', param_x);  bx_s = get('SCS', param_x)
#     by_d = get('DWC', param_y);  by_s = get('SCS', param_y)

#     # ── Build grid in actual (un-standardized) parameter space ───────────────
#     feat = feat_std_df.set_index('parameter')
#     mu_x, sd_x = float(feat.loc[param_x, 'mean']), float(feat.loc[param_x, 'std'])
#     mu_y, sd_y = float(feat.loc[param_y, 'mean']), float(feat.loc[param_y, 'std'])

#     lo_x, hi_x = BOUNDS[param_x]
#     lo_y, hi_y = BOUNDS[param_y]
#     nx, ny = 180, 180
#     xs = np.linspace(lo_x, hi_x, nx)
#     ys = np.linspace(lo_y, hi_y, ny)
#     XX, YY = np.meshgrid(xs, ys)

#     # Standardize
#     XS = (XX - mu_x) / sd_x
#     YS = (YY - mu_y) / sd_y

#     eta_d = b0_d + bx_d * XS + by_d * YS
#     eta_s = b0_s + bx_s * XS + by_s * YS
#     p_dwc, p_scs, p_none = _mnl_probs(eta_d, eta_s)

#     # Log relative risk vs unselected baseline (avoids divide-by-zero with clipping)
#     eps = 1e-9
#     LRR_DWC = np.log10(np.maximum(p_dwc, eps) / np.maximum(p_none, eps))
#     LRR_SCS = np.log10(np.maximum(p_scs, eps) / np.maximum(p_none, eps))

#     fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 88 * MM))

#     cmaps = [
#         mcolors.LinearSegmentedColormap.from_list('blu', ['#EBF5FB', '#1A5276'], N=256),
#         mcolors.LinearSegmentedColormap.from_list('red', ['#FDEDEC', '#922B21'], N=256),
#     ]
#     data_maps  = [LRR_DWC, LRR_SCS]
#     titles     = ['(a)  DWC selection log relative risk',
#                   '(b)  SCS selection log relative risk']
#     clbr_label = [r'$\log_{10}[P(\mathrm{DWC})/P(\mathrm{None})]$',
#                   r'$\log_{10}[P(\mathrm{SCS})/P(\mathrm{None})]$']

#     for ax, data, cmap, title, clbl in zip(axes, data_maps, cmaps, titles, clbr_label):
#         vmax = np.percentile(np.abs(data), 99)
#         vmin = data.min()

#         cf = ax.contourf(XX, YY, data, levels=20, cmap=cmap,
#                          vmin=vmin, vmax=vmax, zorder=1)
#         ct = ax.contour(XX, YY, data,
#                         levels=np.linspace(vmin, vmax, 8),
#                         colors='white', linewidths=0.3, alpha=0.55, zorder=2)
#         ax.clabel(ct, inline=True, fontsize=4.5, fmt='%.1f', inline_spacing=2)

#         # Technology-parity contour P(DWC) = P(SCS)
#         parity = ax.contour(XX, YY, p_dwc - p_scs, levels=[0],
#                             colors='#F0E442', linewidths=1.0,
#                             linestyles='--', zorder=3)
#         ax.clabel(parity, fmt='DWC = SCS', fontsize=4.5, inline=True)

#         # Baseline operating point (standardized mean = actual mean)
#         ax.scatter([mu_x], [mu_y], marker='X', s=30, color='white',
#                    edgecolors='#333333', linewidths=0.6, zorder=6, label='Baseline')

#         cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
#         cbar.set_label(clbl, fontsize=5.5)
#         cbar.ax.tick_params(labelsize=5)

#         sym_x, desc_x = LABEL[param_x]
#         sym_y, desc_y = LABEL[param_y]
#         ax.set_xlabel(f'{sym_x}  ({desc_x})', labelpad=3)
#         ax.set_ylabel(f'{sym_y}  ({desc_y})', labelpad=3)
#         ax.set_title(title, loc='left', pad=4, fontweight='bold')

#         # Format axis ticks in physical units
#         ax.xaxis.set_major_formatter(mticker.FuncFormatter(
#             lambda v, _: FMT.get(param_x, lambda x: f'{x:.2g}')(v)))
#         ax.yaxis.set_major_formatter(mticker.FuncFormatter(
#             lambda v, _: FMT.get(param_y, lambda x: f'{x:.2g}')(v)))
#         ax.tick_params(axis='both', which='major', labelsize=5.5)

#     axes[0].legend(loc='upper right', markerscale=0.9, handletextpad=0.3)

#     fig.text(0.5, -0.02,
#              'Log relative-risk surfaces computed analytically from fitted MNLogit coefficients; '
#              'all other predictors held at their standardised mean (zero). '
#              r'Yellow dashed line: $P(\mathrm{DWC}) = P(\mathrm{SCS})$ boundary.',
#              ha='center', va='top', fontsize=4.8, color='#555555', style='italic')

#     out_path = os.path.join(out_dir, filename)
#     fig.savefig(out_path, bbox_inches='tight')
#     print(f"  Saved: {filename}")
#     return fig


# # ═══════════════════════════════════════════════════════════════════════════════
# # §7  FIG 3 — MARGINAL PROBABILITY SWEEPS
# # ═══════════════════════════════════════════════════════════════════════════════

# def plot_marginal_effects(df_coef, feat_std_df, out_dir,
#                           filename='fig3_marginal_effects.pdf'):
#     """
#     Figure 3: 3×3 small-multiples — one panel per swept theta parameter.

#     For each parameter:
#     • All other standardized predictors are held at zero (their mean).
#     • The focal parameter sweeps its full LHS range.
#     • Three lines: P(DWC) [blue], P(SCS) [red], P(None) [gray].

#     Y-axis is log₁₀(probability [%]) to reveal the multi-order-of-magnitude
#     variation driven by the dominant parameters, while still showing the
#     long tail of the distribution.  P(None) has a second right y-axis scaled
#     to linear [%] to keep it readable (it stays near 100% throughout).

#     This reveals both the DIRECTION and the MAGNITUDE of each parameter's
#     effect — key for identifying practical operating regimes.
#     """
#     _set_style()

#     coef    = df_coef.set_index('parameter')
#     feat    = feat_std_df.set_index('parameter')
#     b0_d    = float(coef.loc['const', 'coef_DWC'])
#     b0_s    = float(coef.loc['const', 'coef_SCS'])
#     n_pts   = 200

#     fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL, 155 * MM),
#                              sharex=False, sharey=False)
#     axes_flat = axes.flatten()

#     for idx, param in enumerate(THETA_COLS):
#         ax = axes_flat[idx]

#         mu_p = float(feat.loc[param, 'mean'])
#         sd_p = float(feat.loc[param, 'std'])
#         lo, hi = BOUNDS[param]
#         xs_actual = np.linspace(lo, hi, n_pts)
#         xs_std    = (xs_actual - mu_p) / sd_p

#         # Base linear predictors (other predictors at zero)
#         eta_d_base = b0_d
#         eta_s_base = b0_s

#         # Add the sweep contribution
#         b_d = float(coef.loc[param, 'coef_DWC']) if param in coef.index else 0.0
#         b_s = float(coef.loc[param, 'coef_SCS']) if param in coef.index else 0.0

#         eta_d = eta_d_base + b_d * xs_std
#         eta_s = eta_s_base + b_s * xs_std
#         p_d, p_s, p_n = _mnl_probs(eta_d, eta_s)

#         # Convert to % for display
#         p_d_pct = p_d * 100.0
#         p_s_pct = p_s * 100.0
#         p_n_pct = p_n * 100.0

#         # ── Primary y-axis: DWC and SCS on log scale ─────────────────────────
#         eps_pct = 1e-4   # floor at 0.0001% for log scale
#         ax.semilogy(xs_actual, np.maximum(p_d_pct, eps_pct),
#                     color=C['dwc'], lw=1.2, label='DWC', zorder=4)
#         ax.semilogy(xs_actual, np.maximum(p_s_pct, eps_pct),
#                     color=C['scs'], lw=1.2, label='SCS', linestyle='--', zorder=4)

#         # ── Secondary y-axis: P(None) on linear scale ────────────────────────
#         ax2 = ax.twinx()
#         ax2.plot(xs_actual, p_n_pct, color=C['none'], lw=0.8,
#                  linestyle=':', alpha=0.7, label='None', zorder=3)
#         ax2.set_ylim(90, 100.5)
#         ax2.tick_params(axis='y', labelsize=4.5, colors=C['none'])
#         ax2.set_ylabel('P(None) [%]', fontsize=4.5, color=C['none'], labelpad=2)
#         ax2.spines['right'].set_color(C['none'])
#         ax2.spines['right'].set_linewidth(0.4)
#         # Hide right spine of primary axis (twinx manages it)
#         ax.spines['right'].set_visible(False)

#         # ── Primary axis formatting ───────────────────────────────────────────
#         ax.set_ylabel('P(technology) [%]', fontsize=5.5, labelpad=2)
#         ax.tick_params(axis='both', labelsize=5)
#         ax.yaxis.set_major_formatter(mticker.FuncFormatter(
#             lambda v, _: f'{v:.2g}' if v >= 0.01 else f'{v:.0e}'))

#         # ── X-axis: physical units with 3 reference ticks ────────────────────
#         ax.set_xlim(lo, hi)
#         ticks_3 = np.linspace(lo, hi, 3)
#         ax.set_xticks(ticks_3)
#         fmt_fn = FMT.get(param, lambda x: f'{x:.2g}')
#         ax.set_xticklabels([fmt_fn(t) for t in ticks_3], fontsize=5)

#         # ── Panel title: symbol + short description ───────────────────────────
#         sym, desc = LABEL[param]
#         ax.set_title(f'{sym}  {desc}', fontsize=5.8, pad=3)

#         # ── Baseline (standardised mean) marker ───────────────────────────────
#         ax.axvline(mu_p, color='#888888', lw=0.5, ls=':', zorder=2, alpha=0.7)

#         # ── Gray fill for region where P(DWC) > P(SCS) ──────────────────────
#         ax.fill_between(xs_actual,
#                         np.maximum(p_d_pct, eps_pct),
#                         np.maximum(p_s_pct, eps_pct),
#                         where=(p_d_pct > p_s_pct),
#                         color=C['dwc'], alpha=0.08, zorder=1)

#         # Panel label (a)–(i)
#         ax.text(0.03, 0.97, f'({chr(97+idx)})', transform=ax.transAxes,
#                 fontsize=6, fontweight='bold', va='top', ha='left')

#     # ── Shared legend in top-left panel ──────────────────────────────────────
#     handles = [
#         Line2D([0],[0], color=C['dwc'], lw=1.2, label='DWC'),
#         Line2D([0],[0], color=C['scs'], lw=1.2, ls='--', label='SCS'),
#         Line2D([0],[0], color=C['none'], lw=0.8, ls=':', alpha=0.7, label='None (right axis)'),
#     ]
#     axes_flat[0].legend(handles=handles, loc='lower left', fontsize=5,
#                         handlelength=1.0, labelspacing=0.3, framealpha=0.9)

#     fig.text(0.5, -0.01,
#              'Each panel holds all other standardized predictors at zero (their sample mean). '
#              'Vertical dotted line: LHS mean of the focal parameter. '
#              'Blue shading: region where DWC is more likely than SCS.',
#              ha='center', va='top', fontsize=4.8, color='#555555', style='italic')

#     out_path = os.path.join(out_dir, filename)
#     fig.savefig(out_path, bbox_inches='tight')
#     print(f"  Saved: {filename}")
#     return fig


# # ═══════════════════════════════════════════════════════════════════════════════
# # §8  FIG 4 — BOOTSTRAP RANK STABILITY
# # ═══════════════════════════════════════════════════════════════════════════════

# def plot_bootstrap_stability(df_boot, df_coef, out_dir,
#                              filename='fig4_bootstrap_stability.pdf'):
#     """
#     Figure 4: Bootstrap rank stability — horizontal box plots for theta
#     parameters only (segment controls excluded from ranking).

#     • Sorted by mean rank (ascending) so the most consistently top-ranked
#       parameters appear at the top of each panel.
#     • Box width encodes pct_top3 (fraction of bootstrap resamples in which
#       the parameter ranked in the top 3).
#     • Color encodes the direction of the DWC coefficient (blue = positive
#       effect on DWC selection, red = negative).
#     • Two panels: DWC class (left), SCS class (right).

#     Interpretation: a narrow box with mean rank ≈ 1–2 indicates a robust,
#     consistently top-ranked driver. A wide box indicates rank sensitivity to
#     the particular set of runs sampled.
#     """
#     _set_style()

#     coef_dir = df_coef.set_index('parameter')

#     fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 105 * MM), sharey=False)

#     for ax, cls in zip(axes, ['DWC', 'SCS']):
#         df_cls = (df_boot[df_boot['class'] == cls]
#                   .merge(pd.DataFrame({'parameter': THETA_COLS}), on='parameter', how='inner')
#                   .sort_values('mean_rank').reset_index(drop=True))

#         n_params = len(df_cls)
#         ys = np.arange(n_params)

#         for i, row in df_cls.iterrows():
#             p    = row['parameter']
#             mr   = row['mean_rank']
#             sr   = row['std_rank']
#             p3   = row['pct_top3']   # fraction of bootstrap resamples in top-3

#             # Direction of effect on DWC (for color) — always color by DWC direction
#             coef_val = float(coef_dir.loc[p, 'coef_DWC']) if p in coef_dir.index else 0.0
#             dot_col  = C['dwc'] if coef_val >= 0 else C['neg']

#             # Alternating row background
#             if i % 2 == 0:
#                 ax.axhspan(i - 0.5, i + 0.5, color=C['band'], zorder=0, lw=0)

#             # ── Box: IQR approximation from mean±std of normally distributed ranks
#             q25 = max(1, mr - 0.674 * sr)
#             q75 = min(n_params, mr + 0.674 * sr)
#             box_h  = 0.38 * (1.0 + p3)   # wider box = more top-3 hits

#             # Box body
#             rect = mpatches.FancyBboxPatch(
#                 (q25, i - box_h / 2), q75 - q25, box_h,
#                 boxstyle='round,pad=0.05', linewidth=0.5,
#                 edgecolor=dot_col, facecolor=dot_col, alpha=0.25, zorder=2
#             )
#             ax.add_patch(rect)

#             # Median line (= mean rank for symmetric distribution)
#             ax.plot([mr, mr], [i - box_h / 2, i + box_h / 2],
#                     color=dot_col, lw=1.5, zorder=4)

#             # Whiskers to ±1.5 IQR approximation
#             w_lo = max(1.0, mr - 1.5 * (q75 - q25))
#             w_hi = min(float(n_params), mr + 1.5 * (q75 - q25))
#             ax.plot([w_lo, q25], [i, i], color=dot_col, lw=0.8, zorder=3, alpha=0.7)
#             ax.plot([q75, w_hi], [i, i], color=dot_col, lw=0.8, zorder=3, alpha=0.7)

#             # pct_top3 label on the right
#             ax.text(n_params + 0.3, i,
#                     f'{100*p3:.0f}%', va='center', ha='left',
#                     fontsize=5, color=dot_col)

#         ax.set_xlim(0.2, n_params + 1.5)
#         ax.set_xticks(range(1, n_params + 1))
#         ax.set_xticklabels([str(r) for r in range(1, n_params + 1)], fontsize=5.5)
#         ax.set_xlabel('Parameter rank by $|\\hat{\\beta}|$', labelpad=3)
#         ax.axvline(3.5, color='#AAAAAA', lw=0.6, ls=':', zorder=1, alpha=0.7)
#         ax.text(3.6, -0.55, 'Top-3 threshold', fontsize=4.5,
#                 color='#888888', va='top', ha='left')

#         ax.set_yticks(ys)
#         ylab = [f'{LABEL[p][0]}  {LABEL[p][1]}' for p in df_cls['parameter']]
#         ax.set_yticklabels(ylab, fontsize=5.8)
#         ax.invert_yaxis()

#         panel = 'a' if cls == 'DWC' else 'b'
#         tech  = 'DWC' if cls == 'DWC' else 'SCS'
#         ax.set_title(f'({panel})  {tech} class parameter ranking',
#                      loc='left', pad=4, fontweight='bold')

#         ax.xaxis.grid(True, zorder=0, lw=0.35)
#         ax.text(1.0, 1.01, '% bootstrap\nresamples in top 3',
#                 transform=ax.transAxes, fontsize=4.5, ha='right',
#                 va='bottom', color='#666666', style='italic')

#     fig.text(0.5, -0.02,
#              f'Block bootstrap resampling at the run level (N={df_boot["n_valid_bootstraps"].max()} '
#              r'resamples, sampling $\approx$70% of runs with replacement). '
#              'Box: interquartile range; whiskers: 1.5×IQR. '
#              'Box width ∝ fraction of resamples with rank ≤ 3. '
#              'Color encodes sign of DWC coefficient.',
#              ha='center', va='top', fontsize=4.8, color='#555555', style='italic')

#     out_path = os.path.join(out_dir, filename)
#     fig.savefig(out_path, bbox_inches='tight')
#     print(f"  Saved: {filename}")
#     return fig


# # ═══════════════════════════════════════════════════════════════════════════════
# # §9  ORCHESTRATOR
# # ═══════════════════════════════════════════════════════════════════════════════

# def generate_all_plots(gsa_dir):
#     """
#     Read the CSVs written by Main_sensitivity.py and produce all four figures.
#     No MILP re-solve; no import of Main_optimizer.py.

#     Parameters
#     ----------
#     gsa_dir : str  path to a GSA_<timestamp>/ directory produced by
#                    Main_sensitivity.run_full_sensitivity_study()
#     """
#     _set_style()
#     print(f"\nGenerating figures from: {gsa_dir}\n{'─'*60}")

#     df_coef  = pd.read_csv(os.path.join(gsa_dir, 'mnlogit_coefficients.csv'))
#     df_boot  = pd.read_csv(os.path.join(gsa_dir, 'bootstrap_ranking_stability.csv'))
#     feat_std = pd.read_csv(os.path.join(gsa_dir, 'feature_standardization.csv'))

#     fig1 = plot_influence_ranking(df_coef, gsa_dir)
#     fig2 = plot_phase_diagram(df_coef, feat_std, gsa_dir)
#     fig3 = plot_marginal_effects(df_coef, feat_std, gsa_dir)
#     fig4 = plot_bootstrap_stability(df_boot, df_coef, gsa_dir)

#     print(f"\nAll figures saved to: {gsa_dir}/")
#     return fig1, fig2, fig3, fig4


# # ═══════════════════════════════════════════════════════════════════════════════
# # §10  STANDALONE MOCK RUNNER
# # ═══════════════════════════════════════════════════════════════════════════════

# if __name__ == '__main__':
#     print("=" * 60)
#     print("  Sensitivity_Plots.py — MOCK DATA MODE")
#     print("  Generating synthetic GSA output and all four figures.")
#     print("=" * 60)

#     OUT = 'mock_GSA'
#     _, mu, sd = generate_mock_data(out_dir=OUT, n_runs=1200, seed=42)

#     # Generate all plots using the same orchestrator as the real pipeline
#     generate_all_plots(OUT)

#     plt.show()
#     print("\nDone.  Open mock_GSA/fig*.pdf for print-ready versions.")
