#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# Main_plots.py
# Unified publication-quality plots for Joint DWC + SCS Deployment.
#
# Merges plot_analytics.py and Main_plots_new.py into a single file.
# All plot logic is unchanged from the originals except for the two targeted
# fixes required by the C6b* / C6e* hardware-ceiling update:
#
# Everything else — plot types, layouts, colour palettes, axis labels,
# sensitivity dispatching, file names — is left exactly as it was.
# ═══════════════════════════════════════════════════════════════════════════════

import os
import sys
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

try:
    import geopandas as gpd
    _HAS_GPD = True
except ImportError:
    _HAS_GPD = False

# ══════════════════════════════════════════════════════════════════════════════
# SHARED CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
MW_TO_KW = 1_000.0

# Hourly traffic weights — must match the optimizer exactly
_HOURS       = np.arange(24)
_TRAFFIC_RAW = (
    0.4 * np.exp(-0.5 * ((_HOURS - 8)  / 2)**2) +
    0.6 * np.exp(-0.5 * ((_HOURS - 17) / 3)**2) + 0.05
)
TRAFFIC_WEIGHTS          = _TRAFFIC_RAW / _TRAFFIC_RAW.sum()
TRAFFIC_PEAK_HOUR_WEIGHT = float(np.max(TRAFFIC_WEIGHTS))

# ── plot_analytics palette ────────────────────────────────────────────────────
C_DWC_CAP  = "#1A3A5C"
C_DWC_REV  = "#5BA3D9"
C_SCS_CAP  = "#7B2D00"
C_SCS_REV  = "#F4A442"
C_REV_LINE = "#D62728"

# ── Main_plots_new palette ────────────────────────────────────────────────────
C_DWC    = "#2c3e50"
C_SCS    = "#17a589"
C_EXIST  = "#e67e22"
C_SUB    = "#000000"
C_REV    = "#27ae60"
C_CAPEX  = "#e74c3c"
C_OPEX   = "#e67e22"
C_PROFIT = "#2980b9"
C_GREY   = "#bdc3c7"
C_GOLD   = "#f39c12"
C_CYAN   = "cyan"

_MARKER_DWC   = "o"
_MARKER_SCS   = "^"
_MARKER_EXIST = "D"
_MARKER_SUB   = "."

# ── Typography (Main_plots_new style) ─────────────────────────────────────────
plt.rcParams.update({
    "font.family"       : "sans-serif",
    "font.sans-serif"   : ["DejaVu Sans", "Liberation Sans", "sans-serif"],
    "font.size"         : 13,
    "axes.labelsize"    : 13,
    "axes.titlesize"    : 14,
    "axes.titleweight"  : "bold",
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.linewidth"    : 1.0,
    "xtick.major.width" : 1.0,
    "ytick.major.width" : 1.0,
    "xtick.labelsize"   : 11,
    "ytick.labelsize"   : 11,
    "legend.fontsize"   : 11,
    "legend.frameon"    : False,
    "figure.dpi"        : 300,
    "savefig.dpi"       : 300,
    "savefig.bbox"      : "tight",
    "lines.linewidth"   : 2.0,
    "lines.markersize"  : 6,
})

# plot_analytics uses slightly different rcParams — we apply them locally
# inside each analytics function via matplotlib.rcParams.update where needed.
_ANALYTICS_RCPARAMS = {
    "font.family"       : "serif",
    "font.serif"        : ["Georgia", "DejaVu Serif", "Times New Roman"],
    "axes.titlesize"    : 12,
    "axes.labelsize"    : 10,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "figure.dpi"        : 130,
    "savefig.dpi"       : 200,
    "savefig.bbox"      : "tight",
}


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(fig, path, tight=True):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight" if tight else None,
                transparent=True)
    plt.close(fig)
    print(f"  Saved: {os.path.basename(path)}")


def _npv_factor(g, rho, T):
    """Γ(g) — lifetime PV weight for a growing annual stream."""
    r = (1.0 + g) / (1.0 + rho)
    if abs(r - 1.0) < 1e-12:
        return float(T) / (1.0 + rho)
    return (1.0 / (1.0 + rho)) * (1.0 - r ** T) / (1.0 - r)


def _label_step(n):
    if n <= 80:  return 1
    if n <= 160: return 2
    if n <= 400: return 5
    if n <= 800: return 10
    return max(1, n // 80)


def _draw_us_background(ax):
    if _HAS_GPD:
        try:
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
            usa   = world[world.name == "United States of America"]
            usa.plot(ax=ax, color="white", edgecolor="#D0D0D0",
                     linewidth=0.8, alpha=0.5, zorder=0)
        except Exception:
            pass

# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
#  Making this script independent of Main.py (Also update Main function below)
# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def load_results_from_saved_dir(results_dir):
    """Load results dict from saved Results_* directory CSVs."""
    meta_df = pd.read_csv(os.path.join(results_dir, 'run_metadata.csv'))
    meta = meta_df.iloc[0].to_dict()
    
    df_dwc = pd.read_csv(os.path.join(results_dir, 'DWC_Selected_Segments.csv'))
    df_scs = pd.read_csv(os.path.join(results_dir, 'SCS_Selected_Stations.csv'))
    df_diag = pd.read_csv(os.path.join(results_dir, 'DWC_Selection_Master_Diagnostic.csv'))
    
    df_exist = (pd.read_csv(os.path.join(results_dir, 'SCS_Existing_Stations.csv'))
                if os.path.exists(os.path.join(results_dir, 'SCS_Existing_Stations.csv'))
                else pd.DataFrame())
    df_connections = (pd.read_csv(os.path.join(results_dir, 'DWC_Network_FromTo_Table.csv'))
                      if os.path.exists(os.path.join(results_dir, 'DWC_Network_FromTo_Table.csv'))
                      else pd.DataFrame())
    df_scs_connections = (pd.read_csv(os.path.join(results_dir, 'SCS_Network_FromTo_Table.csv'))
                          if os.path.exists(os.path.join(results_dir, 'SCS_Network_FromTo_Table.csv'))
                          else pd.DataFrame())
    
    # Load coordinates
    seg_coords = {}
    if os.path.exists(os.path.join(results_dir, 'All_Segments_Coords.csv')):
        df_seg = pd.read_csv(os.path.join(results_dir, 'All_Segments_Coords.csv'))
        seg_coords = {int(r['Segment_ID']): [r['X'], r['Y']] for _, r in df_seg.iterrows()}
    
    sub_coords = {}
    if os.path.exists(os.path.join(results_dir, 'All_Substations_Coords.csv')):
        df_sub = pd.read_csv(os.path.join(results_dir, 'All_Substations_Coords.csv'))
        sub_coords = {int(r['Sub_ID']): [r['X'], r['Y']] for _, r in df_sub.iterrows()}
    
    scs_coords_cand = {}
    if not df_scs.empty and 'MID_X' in df_scs.columns:
        scs_coords_cand = {int(r['SCS_ID']): [r['MID_X'], r['MID_Y']] for _, r in df_scs.iterrows()}
    
    scs_coords_exist = {}
    if os.path.exists(os.path.join(results_dir, 'SCS_Exist_Coords.csv')):
        df_scs_exist = pd.read_csv(os.path.join(results_dir, 'SCS_Exist_Coords.csv'))
        scs_coords_exist = {int(r['SCS_ID']): [r['MID_X'], r['MID_Y']] for _, r in df_scs_exist.iterrows()}
    
    return {
        'run_tag': meta.get('run_tag', 'loaded'),
        'meta': meta,
        'params': {},
        'df_dwc': df_dwc,
        'df_scs': df_scs,
        'df_exist': df_exist,
        'df_connections': df_connections,
        'df_scs_connections': df_scs_connections,
        'df_diag': df_diag,
        'seg_coords': seg_coords,
        'sub_coords': sub_coords,
        'scs_coords_cand': scs_coords_cand,
        'scs_coords_exist': scs_coords_exist,
        'selected_substations_dwc': set(df_connections[df_connections['From_Type']=='Substation']['From_ID'].unique()) if not df_connections.empty else set(),
        'selected_substations_scs': set(df_scs_connections[df_scs_connections['From_Type']=='Substation']['From_ID'].unique()) if not df_scs_connections.empty else set(),
        'alpha': meta.get('alpha', 0.1),
        'discount_rate': meta.get('discount_rate', 0.07),
        'project_life': meta.get('project_life', 30),
    }
# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
#  SECTION A — plot_analytics functions (load, metrics, plots 1–4 and 1b–3b)
# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── Data loading (plot_analytics style: reads CSVs from a results directory) ──

def load_results_from_dir(results_dir):
    def _r(fname, req=False):
        p = os.path.join(results_dir, fname)
        if not os.path.exists(p):
            if req: raise FileNotFoundError(p)
            return pd.DataFrame()
        df = pd.read_csv(p)
        print(f"  Loaded {fname:45s} ({len(df):>4} rows)")
        return df

    print(f"\nLoading from: {results_dir}")
    df_dwc = _r("DWC_Selected_Segments.csv")
    df_scs = _r("SCS_Selected_Stations.csv")
    meta   = _r("run_metadata.csv", req=True).iloc[0].to_dict()
    return df_dwc, df_scs, meta


# ── Metric computation ────────────────────────────────────────────────────────

def _split_opex(ann_opex, lcoe, e_del):
    var = (lcoe * e_del).clip(lower=0)
    fix = (ann_opex - var).clip(lower=0)
    return var, fix


def compute_dwc_metrics(df, meta):
    if df.empty: return df
    df = df.copy()
    gamma_rev    = float(meta.get("gamma_rev_dwc", 1.0))   # DWC revenue PV factor
    gamma_var    = float(meta.get("gamma_var",     1.0))   # variable O&M PV factor
    gamma_fix_om = float(meta.get("gamma_fix_om",  1.0))   # fixed O&M PV factor
    h_op         = float(meta.get("h_op", 8760.0))
    rho_ev       = float(meta.get("rho_ev",  0.006))
    eta_dwc      = float(meta.get("eta_dwc", 0.7))

    df["installed_kw"] = df["MW_Load"] * MW_TO_KW
    df["util_rate"]    = df["E_del_DWC"] / (df["installed_kw"] * h_op)

    df["aadt_ev_dwc"] = df["AADT_COM"] * rho_ev * eta_dwc

    df["ann_var_om"], df["ann_fix_om"] = _split_opex(
        df["Ann_Opex"], df["LCOE"], df["E_del_DWC"])

    df["pv_revenue"] = df["Ann_Revenue"] * gamma_rev
    df["pv_var_om"]  = df["ann_var_om"]  * gamma_var    # variable O&M uses gamma_var
    df["pv_fix_om"]  = df["ann_fix_om"]  * gamma_fix_om
    df["pv_capex"]   = df["Capex_Total"]
    df["npv"]        = df["pv_revenue"] - df["pv_capex"] - df["pv_fix_om"] - df["pv_var_om"]

    kw = df["installed_kw"].replace(0, np.nan)
    df["pv_rev_per_kw"]  = df["pv_revenue"] / kw
    df["capex_per_kw"]   = df["pv_capex"]   / kw

    e_pv = (df["E_del_DWC"] * gamma_rev).replace(0, np.nan)
    df["pv_e_del"]           = e_pv
    df["capex_per_kwh"]      = df["pv_capex"]  / e_pv
    df["pv_fix_om_per_kwh"]  = df["pv_fix_om"] / e_pv
    df["pv_var_om_per_kwh"]  = df["pv_var_om"] / e_pv
    df["pv_rev_per_kwh"]     = df["pv_revenue"] / e_pv
    df["total_cost_per_kwh"] = (df["capex_per_kwh"] + df["pv_fix_om_per_kwh"]
                                + df["pv_var_om_per_kwh"])

    df["label"] = "DWC\n" + df["Segment_ID"].astype(str)
    df["tech"]  = "DWC"
    return df


def compute_scs_metrics(df, meta):
    if df.empty: return df
    df = df.copy()
    gamma_rev    = float(meta.get("gamma_rev_scs", 1.0))   # SCS revenue PV factor
    gamma_var    = float(meta.get("gamma_var",     1.0))   # variable O&M PV factor
    gamma_fix_om = float(meta.get("gamma_fix_om",  1.0))   # fixed O&M PV factor
    h_op         = float(meta.get("h_op", 8760.0))
    n_stalls     = float(meta.get("n_scs_stalls",      6))
    p_stall_kw   = float(meta.get("p_scs_per_stall_kw", 250.0))
    rho_ev       = float(meta.get("rho_ev",  0.006))
    eta_scs      = float(meta.get("eta_scs", 0.3))
    installed_kw = n_stalls * p_stall_kw

    df["installed_kw"] = installed_kw
    df["util_rate"]    = df["E_del_SCS"] / (installed_kw * h_op)

    if "AADT_COM" not in df.columns or df["AADT_COM"].isna().all():
        e_req = float(meta.get("e_req_kwh", 150.0))
        denom = 365.0 * rho_ev * eta_scs * e_req
        df["AADT_COM"] = (df["E_traff_SCS"] / denom) if denom > 0 else np.nan

    df["aadt_ev_scs"] = df["AADT_COM"] * rho_ev * eta_scs

    df["ann_var_om"], df["ann_fix_om"] = _split_opex(
        df["Ann_Opex"], df["LCOE"], df["E_del_SCS"])

    df["pv_revenue"] = df["Ann_Revenue"] * gamma_rev
    df["pv_var_om"]  = df["ann_var_om"]  * gamma_var    # variable O&M uses gamma_var
    df["pv_fix_om"]  = df["ann_fix_om"]  * gamma_fix_om
    df["pv_capex"]   = df["Capex_Total"]
    df["npv"]        = df["pv_revenue"] - df["pv_capex"] - df["pv_fix_om"] - df["pv_var_om"]

    df["pv_rev_per_kw"]  = df["pv_revenue"] / installed_kw
    df["capex_per_kw"]   = df["pv_capex"]   / installed_kw

    e_pv = (df["E_del_SCS"] * gamma_rev).replace(0, np.nan)
    df["pv_e_del"]           = e_pv
    df["capex_per_kwh"]      = df["pv_capex"]  / e_pv
    df["pv_fix_om_per_kwh"]  = df["pv_fix_om"] / e_pv
    df["pv_var_om_per_kwh"]  = df["pv_var_om"] / e_pv
    df["pv_rev_per_kwh"]     = df["pv_revenue"] / e_pv
    df["total_cost_per_kwh"] = (df["capex_per_kwh"] + df["pv_fix_om_per_kwh"]
                                + df["pv_var_om_per_kwh"])

    df["label"] = "SCS\n" + df["SCS_ID"].astype(str)
    df["tech"]  = "SCS"
    return df


# ── Plot 1 (analytics): Lifetime PV Revenue vs Capital Cost per kW ────────────

def plot_dollar_per_kw(df_dwc, df_scs, meta, out_dir):
    has_dwc = not df_dwc.empty
    has_scs = not df_scs.empty
    if not has_dwc and not has_scs:
        print("  [Plot 1] nothing to plot"); return

    matplotlib.rcParams.update(_ANALYTICS_RCPARAMS)

    T   = int(meta.get("project_life", 30))
    rho = float(meta.get("discount_rate", 0.07))
    g_d = float(meta.get("g_d", 0.0))

    frames = []
    for df, present in [(df_dwc, has_dwc), (df_scs, has_scs)]:
        if present:
            tmp = df[["label","tech","capex_per_kw","pv_rev_per_kw",
                       "installed_kw","AADT_COM","npv"]].copy()
            frames.append(tmp)
    combined = pd.concat(frames, ignore_index=True)
    n = len(combined)
    x = np.arange(n)
    w = 0.38

    cap_colors = [C_DWC_CAP if t=="DWC" else C_SCS_CAP for t in combined["tech"]]
    rev_colors = [C_DWC_REV if t=="DWC" else C_SCS_REV for t in combined["tech"]]

    fig, ax = plt.subplots(figsize=(max(8, n*1.6+2), 6.5))

    bars_cap = ax.bar(x - w/2, combined["capex_per_kw"].fillna(0), w,
                      color=cap_colors, edgecolor="white", linewidth=0.8, zorder=3)
    bars_rev = ax.bar(x + w/2, combined["pv_rev_per_kw"].fillna(0), w,
                      color=rev_colors, edgecolor="white", linewidth=0.8, zorder=3)

    ymax = max(combined["capex_per_kw"].fillna(0).max(),
               combined["pv_rev_per_kw"].fillna(0).max())

    def _fmt(v):
        return f"${v/1e3:.1f}k" if v >= 1000 else f"${v:.0f}"

    for bar, val in zip(bars_cap, combined["capex_per_kw"].fillna(0)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+ymax*0.01,
                _fmt(val), ha="center", va="bottom", fontsize=8, fontweight="bold")

    for bar, rev, aadt in zip(bars_rev,
                               combined["pv_rev_per_kw"].fillna(0),
                               combined["AADT_COM"].fillna(0)):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+ymax*0.01,
                f"{_fmt(rev)}\nAADT {int(aadt):,}" if aadt>0 else _fmt(rev),
                ha="center", va="bottom", fontsize=7.5, linespacing=1.35)

    ax.set_xticks(x)
    ax.set_xticklabels(combined["label"], fontsize=9)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v,_: f"${v/1e3:.1f}k" if v>=1000 else f"${v:.0f}"))
    ax.set_ylabel(f"Lifetime PV  $/kW\n(T={T} yr, ρ={rho:.0%}, g_d={g_d:.1%})")
    ax.set_title(
        "Plot 1 — Lifetime PV Revenue vs Capital Cost per Installed kW\n"
        "Both bars in the same dollars: capital (one-time) vs PV of 30-yr revenue stream",
        fontsize=11, fontweight="bold", pad=10)

    legend_els = [
        mpatches.Patch(color=C_DWC_CAP, label="DWC — Capital ($/kW, one-time)"),
        mpatches.Patch(color=C_DWC_REV, label="DWC — Lifetime PV Revenue ($/kW)"),
        mpatches.Patch(color=C_SCS_CAP, label="SCS — Capital ($/kW, one-time)"),
        mpatches.Patch(color=C_SCS_REV, label="SCS — Lifetime PV Revenue ($/kW)"),
    ]
    ax.legend(handles=legend_els, fontsize=8.5, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_ylim(bottom=0, top=ymax*1.25)

    if has_dwc and has_scs:
        ax.axvline(len(df_dwc) - 0.5, color="#AAAAAA", linewidth=1.2,
                   linestyle="--", alpha=0.7)

    plt.tight_layout()
    path = os.path.join(out_dir, "plot1_dollar_per_kw.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2 (analytics): Levelised $/kWh cost stack ────────────────────────────

def plot_dollar_per_kwh(df_dwc, df_scs, meta, out_dir):
    has_dwc = not df_dwc.empty
    has_scs = not df_scs.empty
    if not has_dwc and not has_scs:
        print("  [Plot 2] nothing to plot"); return

    matplotlib.rcParams.update(_ANALYTICS_RCPARAMS)

    T   = int(meta.get("project_life", 30))
    rho = float(meta.get("discount_rate", 0.07))

    frames = []
    for df, present in [(df_dwc, has_dwc), (df_scs, has_scs)]:
        if present:
            frames.append(df[["label","tech","capex_per_kwh",
                               "pv_fix_om_per_kwh","pv_var_om_per_kwh",
                               "pv_rev_per_kwh","AADT_COM","util_rate"]].copy())
    combined = pd.concat(frames, ignore_index=True)
    n = len(combined)
    x = np.arange(n)
    w = 0.52

    cap_vals = combined["capex_per_kwh"].fillna(0).values
    fix_vals = combined["pv_fix_om_per_kwh"].fillna(0).values
    var_vals = combined["pv_var_om_per_kwh"].fillna(0).values
    rev_vals = combined["pv_rev_per_kwh"].fillna(0).values
    total    = cap_vals + fix_vals + var_vals

    cap_clr = ["#A8C8E8" if t == "DWC" else "#F5C897" for t in combined["tech"]]
    fix_clr = ["#5A7A96" if t == "DWC" else "#C07030" for t in combined["tech"]]
    var_clr = ["#2C2C2C" if t == "DWC" else "#4A1A00" for t in combined["tech"]]

    fig, ax = plt.subplots(figsize=(max(8, n * 1.8 + 2), 6.5))

    b1 = ax.bar(x, var_vals, w, color=var_clr, edgecolor="white",
                linewidth=0.5, zorder=3)
    b2 = ax.bar(x, fix_vals, w, bottom=var_vals, color=fix_clr,
                edgecolor="white", linewidth=0.5, zorder=3)
    b3 = ax.bar(x, cap_vals, w, bottom=var_vals+fix_vals, color=cap_clr,
                edgecolor="white", linewidth=0.5, zorder=3)

    for i, tc in enumerate(total):
        ax.text(x[i], tc + max(total)*0.012, f"${tc:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    for i, (aadt, util, bot, ht) in enumerate(zip(
            combined["AADT_COM"].fillna(0),
            combined["util_rate"].fillna(0),
            var_vals+fix_vals, cap_vals)):
        if ht > max(total)*0.06:
            ax.text(x[i], bot+ht/2,
                    f"AADT {int(aadt):,}\nUtil {util:.2%}",
                    ha="center", va="center",
                    fontsize=7.5, color="#333333", linespacing=1.35)

    for i, rv in enumerate(rev_vals):
        ax.plot([x[i]-w/2, x[i]+w/2], [rv, rv],
                color=C_REV_LINE, linewidth=2.2, zorder=5, linestyle="--")

    ax.set_xticks(x)
    ax.set_xticklabels(combined["label"], fontsize=9)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v,_: f"${v:.2f}"))
    ax.set_ylabel(f"Levelised $/kWh  (T={T} yr, ρ={rho:.0%})")
    ax.set_title(
        "Plot 2 — Levelised Cost per kWh vs Retail Charging Price\n"
        "Stack below dashed line = station recovers all costs per kWh",
        fontsize=11, fontweight="bold", pad=10)

    legend_els = [
        mpatches.Patch(color="#2C2C2C", label="DWC: Energy cost (≈ grid LCOE)"),
        mpatches.Patch(color="#5A7A96", label="DWC: Fixed O&M per kWh"),
        mpatches.Patch(color="#A8C8E8", label="DWC: Capital per kWh  ← shrinks with AADT"),
        mpatches.Patch(color="#4A1A00", label="SCS: Energy cost (≈ grid LCOE)"),
        mpatches.Patch(color="#C07030", label="SCS: Fixed O&M per kWh"),
        mpatches.Patch(color="#F5C897", label="SCS: Capital per kWh  ← shrinks with AADT"),
        plt.Line2D([0],[0], color=C_REV_LINE, linewidth=2, linestyle="--",
                   label="Retail price charged to drivers ($/kWh)"),
    ]
    ax.legend(handles=legend_els, fontsize=8, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0, framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

    if has_dwc and has_scs:
        ax.axvline(len(df_dwc)-0.5, color="#AAAAAA", linewidth=1.2,
                   linestyle="--", alpha=0.7)

    plt.tight_layout()
    path = os.path.join(out_dir, "plot2_dollar_per_kwh.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3 (analytics): Traffic & utilisation summary ─────────────────────────

def plot_traffic_utilisation(df_dwc, df_scs, meta, out_dir):
    has_dwc = not df_dwc.empty
    has_scs = not df_scs.empty
    if not has_dwc and not has_scs:
        print("  [Plot 3] nothing to plot"); return

    matplotlib.rcParams.update(_ANALYTICS_RCPARAMS)

    rho_ev  = float(meta.get("rho_ev",  0.006))
    eta_dwc = float(meta.get("eta_dwc", 0.7))
    eta_scs = float(meta.get("eta_scs", 0.3))

    frames = []
    if has_dwc:
        tmp = df_dwc[["label","tech","AADT_COM","aadt_ev_dwc","util_rate"]].copy()
        tmp = tmp.rename(columns={"aadt_ev_dwc": "aadt_ev"})
        frames.append(tmp)
    if has_scs:
        tmp = df_scs[["label","tech","AADT_COM","aadt_ev_scs","util_rate"]].copy()
        tmp = tmp.rename(columns={"aadt_ev_scs": "aadt_ev"})
        frames.append(tmp)
    combined = pd.concat(frames, ignore_index=True)

    n  = len(combined)
    x  = np.arange(n)
    w  = 0.28

    fig, ax1 = plt.subplots(figsize=(max(9, n*2+2), 6))
    ax2 = ax1.twinx()

    c_aadt = [C_DWC_CAP if t=="DWC" else C_SCS_CAP for t in combined["tech"]]
    c_ev   = [C_DWC_REV if t=="DWC" else C_SCS_REV for t in combined["tech"]]

    b1 = ax1.bar(x - w, combined["AADT_COM"].fillna(0), w,
                 color=c_aadt, edgecolor="white", linewidth=0.7,
                 zorder=3, label="AADT_COM")
    b2 = ax1.bar(x,      combined["aadt_ev"].fillna(0),  w,
                 color=c_ev,   edgecolor="white", linewidth=0.7,
                 zorder=3, label="AADT × ρ_EV × η")

    util_pct = combined["util_rate"].fillna(0).values * 100
    b3 = ax2.bar(x + w, util_pct, w,
                 color="#888888", edgecolor="white", linewidth=0.7,
                 alpha=0.75, zorder=3, label="Utilisation rate")

    for bar, v in zip(b1, combined["AADT_COM"].fillna(0)):
        if v > 0:
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                     f"{int(v):,}", ha="center", va="bottom", fontsize=7.5)
    for bar, v in zip(b2, combined["aadt_ev"].fillna(0)):
        if v > 0:
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
                     f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
    for bar, v in zip(b3, util_pct):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f"{v:.2f}%", ha="center", va="bottom", fontsize=7.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(combined["label"], fontsize=9)
    ax1.set_ylabel("Vehicles per day  (base year)")
    ax2.set_ylabel("Annual capacity utilisation  (%)")
    ax2.set_ylim(0, max(util_pct.max()*1.4, 1))

    ax1.set_title(
        "Plot 3 — Traffic & Utilisation per Station  (base year)\n"
        f"ρ_EV={rho_ev:.3f}   η_DWC={eta_dwc:.2f}   η_SCS={eta_scs:.2f}",
        fontsize=11, fontweight="bold", pad=10)

    handles = [b1, b2,
               mpatches.Patch(color="#888888", alpha=0.75,
                               label="Annual utilisation rate (%) — right axis")]
    labels  = ["AADT_COM  (all commercial vehicles/day)",
               "EV vehicles/day using the charger  (AADT × ρ_EV × η)",
               "Annual utilisation rate (%) — right axis"]
    ax1.legend(handles=handles, labels=labels, fontsize=8.5,
               loc="upper right", framealpha=0.9)
    ax1.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    if has_dwc and has_scs:
        ax1.axvline(len(df_dwc)-0.5, color="#AAAAAA", linewidth=1.2,
                    linestyle="--", alpha=0.7)

    plt.tight_layout()
    path = os.path.join(out_dir, "plot3_traffic_utilisation.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 4 (analytics): Utilisation over time ─────────────────────────────────
# FIX 1 applied here: e_hw_cap uses the PHYSICAL ceiling per technology.

def _dwc_hourly_kw(row, meta):
    e_del = float(row.get("E_del_DWC", 0) or 0)
    if e_del == 0:
        return np.zeros(24)
    daily = e_del / 365.0
    return daily * TRAFFIC_WEIGHTS


def _scs_hourly_kw(row, meta):
    e_del  = float(row.get("E_del_SCS", 0) or 0)
    if e_del == 0:
        return np.zeros(24)
    daily  = e_del / 365.0
    return daily * TRAFFIC_WEIGHTS


def plot_utilisation_over_time(df_dwc, df_scs, meta, out_dir):
    has_dwc = not df_dwc.empty
    has_scs = not df_scs.empty
    if not has_dwc and not has_scs:
        print("  [Plot 4] nothing to plot"); return

    matplotlib.rcParams.update(_ANALYTICS_RCPARAMS)

    T          = int(  meta.get("project_life", 30))
    g_d        = float(meta.get("g_d",           0.0))
    h_op       = float(meta.get("h_op",       8760.0))
    n_stalls   = float(meta.get("n_scs_stalls",      6))
    p_stall_kw = float(meta.get("p_scs_per_stall_kw", 250.0))
    # FIX 1: read utilisation rate so SCS physical ceiling is correct
    util_rate_scs = float(meta.get("utilization_rate_scs", 1.0))

    years = np.arange(1, T+1)

    stations = []
    if has_dwc:
        for _, row in df_dwc.iterrows():
            inst_kw = float(row["installed_kw"])
            stations.append({
                "label"      : row["label"].replace("\n"," "),
                "tech"       : "DWC",
                "inst_kw"    : inst_kw,
                "e_del"      : float(row["E_del_DWC"]),
                "e_traff"    : float(row.get("E_traff_DWC", row["E_del_DWC"])),
                # FIX 1: DWC physical ceiling = installed_kw × h_op (no util factor)
                "e_hw_cap"   : inst_kw * h_op,
                "hourly_kwh" : _dwc_hourly_kw(row, meta),
            })
    if has_scs:
        for _, row in df_scs.iterrows():
            inst_kw = float(row["installed_kw"])
            stations.append({
                "label"      : row["label"].replace("\n"," "),
                "tech"       : "SCS",
                "inst_kw"    : inst_kw,
                "e_del"      : float(row["E_del_SCS"]),
                "e_traff"    : float(row.get("E_traff_SCS", row["E_del_SCS"])),
                # FIX 1: SCS physical ceiling includes the utilisation rate
                "e_hw_cap"   : inst_kw * h_op * util_rate_scs,
                "hourly_kwh" : _scs_hourly_kw(row, meta),
            })

    n_sta = len(stations)
    fig, axes = plt.subplots(n_sta, 3,
                             figsize=(14, 3.8 * n_sta),
                             squeeze=False)
    fig.suptitle(
        "Plot 4 — Utilisation Over Time: 24-hour Profile | Annual Energy | 30-yr Growth",
        fontsize=12, fontweight="bold", y=1.01)

    for row_idx, sta in enumerate(stations):
        tech  = sta["tech"]
        c     = C_DWC_CAP if tech == "DWC" else C_SCS_CAP
        c_lt  = C_DWC_REV if tech == "DWC" else C_SCS_REV
        label = sta["label"]

        # Panel A: 24-hour profile
        ax_a = axes[row_idx, 0]
        hourly = sta["hourly_kwh"]
        inst   = sta["inst_kw"]

        ax_a.fill_between(np.arange(24), hourly, alpha=0.7, color=c, step="mid")
        ax_a.step(np.arange(24), hourly, color=c, linewidth=1.2, where="mid")
        ax_a.axhline(inst, color="#999999", linewidth=1.2, linestyle="--",
                     label=f"Installed capacity {inst:,.0f} kW")
        ax_a.set_xlim(0, 23)
        ax_a.set_xticks([0,6,12,18,23])
        ax_a.set_xlabel("Hour of day")
        ax_a.set_ylabel("kWh delivered (base year)")
        ax_a.set_title(f"{label}\n24-hour profile (typical day, base year)",
                       fontsize=9, fontweight="bold")
        ax_a.legend(fontsize=7.5, loc="upper left")
        ax_a.grid(linestyle="--", alpha=0.3)

        # Panel B: Annual energy bar — uses physical ceiling (FIX 1)
        ax_b = axes[row_idx, 1]
        e_del   = sta["e_del"]   / 1e3
        e_traff = sta["e_traff"] / 1e3
        e_hwcap = sta["e_hw_cap"] / 1e3   # physical ceiling

        categories = ["Potential\ntraffic demand\n(E_traff)",
                      "Actual\ndelivered\n(E_del)",
                      "Physical\nhardware ceiling"]
        values     = [e_traff, e_del, e_hwcap]
        colors_b   = [c_lt, c, "#BBBBBB"]
        bars = ax_b.bar(categories, values, color=colors_b,
                        edgecolor="white", linewidth=0.7, zorder=3)
        for bar, v in zip(bars, values):
            ax_b.text(bar.get_x()+bar.get_width()/2,
                      bar.get_height() + max(values)*0.02,
                      f"{v:,.0f} MWh", ha="center", va="bottom", fontsize=8)
        ax_b.set_ylabel("MWh / year  (base year)")
        ax_b.set_title(f"{label}\nAnnual energy (base year)", fontsize=9, fontweight="bold")
        ax_b.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

        # Panel C: 30-yr cumulative energy
        ax_c = axes[row_idx, 2]
        e_del_y  = e_del  * (1 + g_d) ** (years - 1)
        cumul    = np.cumsum(e_del_y)
        ax_c.plot(years, cumul, color=c, linewidth=2.0, zorder=3,
                  label="Cumulative energy delivered")
        ax_c.fill_between(years, cumul, alpha=0.18, color=c)
        ax_c.axvline(T, color="#999999", linewidth=1, linestyle="--", alpha=0.6)
        ax_c.set_xlabel("Project year")
        ax_c.set_ylabel("Cumulative MWh delivered")
        total_mwh = cumul[-1]
        ax_c.set_title(
            f"{label}\n30-yr cumulative: {total_mwh:,.0f} MWh  (g_d={g_d:.1%}/yr)",
            fontsize=9, fontweight="bold")
        ax_c.legend(fontsize=7.5)
        ax_c.grid(linestyle="--", alpha=0.3, zorder=0)

    plt.tight_layout()
    path = os.path.join(out_dir, "plot4_utilisation_over_time.png")
    fig.savefig(path); plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 3b (analytics): Traffic & Utilisation multi-year ─────────────────────

def plot_traffic_utilisation_multiyear(df_dwc, df_scs, meta, out_dir):
    has_dwc = not df_dwc.empty
    has_scs = not df_scs.empty
    if not has_dwc and not has_scs:
        print("  [Plot 3b] nothing to plot"); return

    matplotlib.rcParams.update(_ANALYTICS_RCPARAMS)

    T       = int(  meta.get("project_life", 30))
    g_d     = float(meta.get("g_d",    0.0))
    g_aadt  = float(meta.get("g_aadt", 0.0))
    rho_ev  = float(meta.get("rho_ev",  0.006))
    eta_dwc = float(meta.get("eta_dwc", 0.7))
    eta_scs = float(meta.get("eta_scs", 0.3))

    mid_yr = T // 2 + 1
    years  = [1, mid_yr, T]
    yr_labels = [
        f"Year 1  (Base)\ng_d growth = ×1.00",
        f"Year {mid_yr}  (Mid)\ng_d growth = ×{(1+g_d)**(mid_yr-1):.2f}",
        f"Year {T}  (End)\ng_d growth = ×{(1+g_d)**(T-1):.2f}",
    ]

    frames = []
    if has_dwc:
        tmp = df_dwc[["label","tech","AADT_COM","aadt_ev_dwc",
                       "util_rate","installed_kw","E_del_DWC"]].copy()
        tmp = tmp.rename(columns={"aadt_ev_dwc": "aadt_ev",
                                   "E_del_DWC":   "e_del_base"})
        frames.append(tmp)
    if has_scs:
        tmp = df_scs[["label","tech","AADT_COM","aadt_ev_scs",
                       "util_rate","installed_kw","E_del_SCS"]].copy()
        tmp = tmp.rename(columns={"aadt_ev_scs": "aadt_ev",
                                   "E_del_SCS":   "e_del_base"})
        frames.append(tmp)
    combined = pd.concat(frames, ignore_index=True)
    n  = len(combined)
    x  = np.arange(n)
    w  = 0.28

    c_aadt = [C_DWC_CAP if t == "DWC" else C_SCS_CAP for t in combined["tech"]]
    c_ev   = [C_DWC_REV if t == "DWC" else C_SCS_REV for t in combined["tech"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f"Plot 3b — Traffic & Utilisation: Base / Mid / End Year  "
        f"(T={T} yr,  g_d={g_d:.1%}/yr,  g_aadt={g_aadt:.1%}/yr)\n"
        f"ρ_EV={rho_ev:.3f}   η_DWC={eta_dwc:.2f}   η_SCS={eta_scs:.2f}",
        fontsize=11, fontweight="bold"
    )

    max_aadt_all = 0.0
    max_util_all = 0.0
    for yr in years:
        fac_d = (1 + g_d)    ** (yr - 1)
        fac_a = (1 + g_aadt) ** (yr - 1)
        max_aadt_all = max(max_aadt_all,
                           (combined["AADT_COM"] * fac_a).max())
        h_op_v = float(meta.get("h_op", 8760.0))
        util_v = ((combined["e_del_base"] * fac_d) /
                  (combined["installed_kw"] * h_op_v)).clip(upper=1.0).max()
        max_util_all = max(max_util_all, util_v * 100)

    for col_idx, (yr, yr_label) in enumerate(zip(years, yr_labels)):
        ax1 = axes[col_idx]
        ax2 = ax1.twinx()

        fac_demand = (1 + g_d)    ** (yr - 1)
        fac_aadt   = (1 + g_aadt) ** (yr - 1)
        h_op_v     = float(meta.get("h_op", 8760.0))

        aadt_yr    = combined["AADT_COM"]   * fac_aadt
        aadt_ev_yr = combined["aadt_ev"]    * fac_demand

        util_rate_scs_val = float(meta.get("utilization_rate_scs", 1.0))
        # Physical hardware ceiling — fixed regardless of year.
        # e_del_base is the base-year value after C6b*/C6e* tightening,
        # so year-y projected delivery never exceeds the physical ceiling.
        hw_ceil_physical = np.where(
            combined["tech"] == "SCS",
            combined["installed_kw"].values * h_op_v * util_rate_scs_val,
            combined["installed_kw"].values * h_op_v
        )
        e_del_yr = np.minimum(
            (combined["e_del_base"] * fac_demand).values,
            hw_ceil_physical
        )
        util_yr = (e_del_yr / (combined["installed_kw"].values * h_op_v)).clip(0, 1.0) * 100

        b1 = ax1.bar(x - w,   aadt_yr,    w, color=c_aadt, edgecolor="white",
                     linewidth=0.7, zorder=3)
        b2 = ax1.bar(x,       aadt_ev_yr, w, color=c_ev,   edgecolor="white",
                     linewidth=0.7, zorder=3, alpha=0.85)
        b3 = ax2.bar(x + w,   util_yr,    w, color="#888888", edgecolor="white",
                     linewidth=0.7, alpha=0.75, zorder=3)

        for bar, v in zip(b1, aadt_yr):
            if v > max_aadt_all * 0.02:
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max_aadt_all * 0.01,
                         f"{int(v):,}", ha="center", va="bottom", fontsize=7)
        for bar, v in zip(b2, aadt_ev_yr):
            if v > 0:
                ax1.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + max_aadt_all * 0.01,
                         f"{v:.1f}", ha="center", va="bottom", fontsize=7)
        for bar, v in zip(b3, util_yr):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max_util_all * 0.01,
                     f"{v:.2f}%", ha="center", va="bottom", fontsize=7)

        ax1.set_xticks(x)
        ax1.set_xticklabels(combined["label"], fontsize=9)
        ax1.set_ylim(0, max_aadt_all * 1.25)
        ax2.set_ylim(0, max_util_all * 1.45)

        if col_idx == 0:
            ax1.set_ylabel("Vehicles per day")
        if col_idx == 2:
            ax2.set_ylabel("Annual capacity utilisation (%)")
        ax1.set_title(yr_label, fontsize=9.5, fontweight="bold", pad=6)
        ax1.grid(axis="y", linestyle="--", alpha=0.3)

        if has_dwc and has_scs:
            ax1.axvline(len(df_dwc) - 0.5, color="#AAAAAA",
                        linewidth=1.2, linestyle="--", alpha=0.7)

    handles = [
        mpatches.Patch(color=C_DWC_CAP, label="DWC AADT_COM"),
        mpatches.Patch(color=C_SCS_CAP, label="SCS AADT_COM"),
        mpatches.Patch(color=C_DWC_REV, label="DWC EV users/day  (AADT×ρ_EV×η, g_d growth)"),
        mpatches.Patch(color=C_SCS_REV, label="SCS EV users/day  (AADT×ρ_EV×η, g_d growth)"),
        mpatches.Patch(color="#888888", alpha=0.75, label="Annual utilisation % — right axis"),
    ]
    fig.legend(handles=handles, fontsize=8.5, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.10), framealpha=0.9)

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    path = os.path.join(out_dir, "plot3b_traffic_utilisation_multiyear.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 2b (analytics): Instantaneous $/kWh multi-year ──────────────────────

def plot_dollar_per_kwh_multiyear(df_dwc, df_scs, meta, out_dir):
    has_dwc = not df_dwc.empty
    has_scs = not df_scs.empty
    if not has_dwc and not has_scs:
        print("  [Plot 2b] nothing to plot"); return

    matplotlib.rcParams.update(_ANALYTICS_RCPARAMS)

    T     = int(  meta.get("project_life", 30))
    g_d   = float(meta.get("g_d",    0.0))
    g_aadt = float(meta.get("g_aadt", 0.0))
    rho   = float(meta.get("discount_rate", 0.07))
    alpha = float(meta.get("alpha", 0.08))
    p_charge_dwc = float(meta.get("p_charge_dwc", 0.40))
    p_charge_scs = float(meta.get("p_charge_scs", 0.40))

    mid_yr  = T // 2 + 1
    years   = [1, mid_yr, T]
    yr_labels = [
        f"Year 1  (Base)",
        f"Year {mid_yr}  (Mid)",
        f"Year {T}  (End)",
    ]

    def _make_frame(df, e_del_col, tech_name):
        out = df[["label","tech","AADT_COM","installed_kw",
                   e_del_col,"LCOE","pv_capex","ann_fix_om"]].copy()
        out = out.rename(columns={e_del_col: "e_del_base"})
        return out

    frames = []
    if has_dwc:
        frames.append(_make_frame(df_dwc, "E_del_DWC", "DWC"))
    if has_scs:
        frames.append(_make_frame(df_scs, "E_del_SCS", "SCS"))
    combined = pd.concat(frames, ignore_index=True)
    n  = len(combined)
    x  = np.arange(n)
    w  = 0.52

    max_stack = 0.0
    h_op      = float(meta.get("h_op", 8760.0))
    util_rate_scs = float(meta.get("utilization_rate_scs", 1.0))
    # Physical hardware ceiling
    hw_ceil_physical_2b = np.where(
        combined["tech"] == "SCS",
        combined["installed_kw"] * h_op * util_rate_scs,
        combined["installed_kw"] * h_op
    )
    for yr in years:
        fac = (1 + g_d) ** (yr - 1)
        e_y_raw = combined["e_del_base"] * fac
        e_y = pd.Series(np.minimum(e_y_raw.values, hw_ceil_physical_2b)).replace(0, np.nan)
        cap = alpha * combined["pv_capex"] / e_y
        fix = combined["ann_fix_om"]       / e_y
        var = combined["LCOE"]
        max_stack = max(max_stack, (cap + fix + var).max())

    cap_clr = ["#A8C8E8" if t == "DWC" else "#F5C897" for t in combined["tech"]]
    fix_clr = ["#5A7A96" if t == "DWC" else "#C07030" for t in combined["tech"]]
    var_clr = ["#2C2C2C" if t == "DWC" else "#4A1A00" for t in combined["tech"]]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True)
    fig.suptitle(
        f"Plot 2b — Instantaneous $/kWh Cost Stack: Base / Mid / End Year  "
        f"(T={T} yr,  g_d={g_d:.1%}/yr,  ρ={rho:.0%})\n"
        f"Capital annualized via CRF α={alpha:.4f}   "
        f"Stack falls as demand grows → improving annual economics",
        fontsize=11, fontweight="bold"
    )

    for col_idx, (yr, yr_label) in enumerate(zip(years, yr_labels)):
        ax = axes[col_idx]
        fac     = (1 + g_d) ** (yr - 1)
        e_y_raw = combined["e_del_base"] * fac
        e_y     = pd.Series(np.minimum(e_y_raw.values, hw_ceil_physical_2b)).replace(0, np.nan)

        cap_vals = (alpha * combined["pv_capex"] / e_y).fillna(0).values
        fix_vals = (combined["ann_fix_om"]       / e_y).fillna(0).values
        var_vals = combined["LCOE"].fillna(0).values
        total    = cap_vals + fix_vals + var_vals

        rev_vals = np.where(
            combined["tech"] == "DWC",
            p_charge_dwc, p_charge_scs
        )

        h_op_v   = float(meta.get("h_op", 8760.0))
        util_yr  = (e_y.values / (combined["installed_kw"].values * h_op_v)).clip(0, 1.0)

        ax.bar(x, var_vals, w, color=var_clr, edgecolor="white",
               linewidth=0.5, zorder=3)
        ax.bar(x, fix_vals, w, bottom=var_vals, color=fix_clr,
               edgecolor="white", linewidth=0.5, zorder=3)
        ax.bar(x, cap_vals, w, bottom=var_vals + fix_vals, color=cap_clr,
               edgecolor="white", linewidth=0.5, zorder=3)

        for i, tc in enumerate(total):
            ax.text(x[i], tc + max_stack * 0.015,
                    f"${tc:.3f}",
                    ha="center", va="bottom",
                    fontsize=9, fontweight="bold")

        for i, util_v in enumerate(util_yr):
            ax.text(x[i], total[i] + max_stack * 0.07,
                    f"Util {util_v*100:.2f}%",
                    ha="center", va="bottom",
                    fontsize=7, color="#555555")

        for i, rv in enumerate(rev_vals):
            ax.plot([x[i] - w/2, x[i] + w/2], [rv, rv],
                    color=C_REV_LINE, linewidth=2.2, zorder=5, linestyle="--")

        ax.set_xticks(x)
        ax.set_xticklabels(combined["label"], fontsize=9)
        ax.set_title(yr_label, fontsize=10, fontweight="bold", pad=6)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
        if col_idx == 0:
            ax.set_ylabel(f"$/kWh  (instantaneous, year-y basis)")
        ax.set_ylim(0, max_stack * 1.45)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

        if has_dwc and has_scs:
            ax.axvline(len(df_dwc) - 0.5, color="#AAAAAA",
                       linewidth=1.2, linestyle="--", alpha=0.7)

    legend_els = [
        mpatches.Patch(color="#2C2C2C", label="DWC: Energy cost (≈ grid LCOE, constant/kWh)"),
        mpatches.Patch(color="#5A7A96", label="DWC: Fixed O&M / E_del_y  (shrinks over time)"),
        mpatches.Patch(color="#A8C8E8", label="DWC: Annualized capital / E_del_y  (shrinks)"),
        mpatches.Patch(color="#4A1A00", label="SCS: Energy cost (≈ grid LCOE, constant/kWh)"),
        mpatches.Patch(color="#C07030", label="SCS: Fixed O&M / E_del_y  (shrinks over time)"),
        mpatches.Patch(color="#F5C897", label="SCS: Annualized capital / E_del_y  (shrinks)"),
        plt.Line2D([0], [0], color=C_REV_LINE, linewidth=2, linestyle="--",
                   label="Retail price charged to drivers ($/kWh, fixed)"),
    ]
    axes[2].legend(handles=legend_els, fontsize=7.5,
                   loc="upper left", bbox_to_anchor=(1.02, 1.0),
                   borderaxespad=0, framealpha=0.9)

    plt.tight_layout()
    path = os.path.join(out_dir, "plot2b_dollar_per_kwh_multiyear.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


# ── Plot 1b (analytics): Annualized capital vs annual revenue per kW ──────────

def plot_dollar_per_kw_multiyear(df_dwc, df_scs, meta, out_dir):
    has_dwc = not df_dwc.empty
    has_scs = not df_scs.empty
    if not has_dwc and not has_scs:
        print("  [Plot 1b] nothing to plot"); return

    matplotlib.rcParams.update(_ANALYTICS_RCPARAMS)

    T      = int(  meta.get("project_life", 30))
    g_d    = float(meta.get("g_d",    0.0))
    g_aadt = float(meta.get("g_aadt", 0.0))
    rho    = float(meta.get("discount_rate", 0.07))
    alpha  = float(meta.get("alpha", 0.08))
    p_charge_dwc = float(meta.get("p_charge_dwc", 0.40))
    p_charge_scs = float(meta.get("p_charge_scs", 0.40))
    h_op   = float(meta.get("h_op", 8760.0))

    mid_yr  = T // 2 + 1
    years   = [1, mid_yr, T]
    yr_labels = [
        f"Year 1  (Base)",
        f"Year {mid_yr}  (Mid)",
        f"Year {T}  (End)",
    ]

    def _make_frame(df, e_del_col):
        out = df[["label","tech","AADT_COM","installed_kw",
                   e_del_col,"pv_capex"]].copy()
        out = out.rename(columns={e_del_col: "e_del_base"})
        return out

    frames = []
    if has_dwc:
        frames.append(_make_frame(df_dwc, "E_del_DWC"))
    if has_scs:
        frames.append(_make_frame(df_scs, "E_del_SCS"))
    combined = pd.concat(frames, ignore_index=True)
    n  = len(combined)
    x  = np.arange(n)
    w  = 0.38

    cap_colors = [C_DWC_CAP if t == "DWC" else C_SCS_CAP for t in combined["tech"]]
    rev_colors = [C_DWC_REV if t == "DWC" else C_SCS_REV for t in combined["tech"]]

    kw_vals = combined["installed_kw"].replace(0, np.nan)
    cap_per_kw_ann = (alpha * combined["pv_capex"] / kw_vals).fillna(0).values

    global_max = cap_per_kw_ann.max()
    util_rate_scs = float(meta.get("utilization_rate_scs", 1.0))
    # Physical hardware ceiling for year-y energy cap
    hw_ceil_physical_1b = np.where(
        combined["tech"] == "SCS",
        combined["installed_kw"] * h_op * util_rate_scs,
        combined["installed_kw"] * h_op
    )
    for yr in years:
        fac   = (1 + g_d) ** (yr - 1)
        p_chg = np.where(combined["tech"] == "DWC", p_charge_dwc, p_charge_scs)
        e_y   = np.minimum((combined["e_del_base"] * fac).values, hw_ceil_physical_1b)
        rev_y = (p_chg * e_y / kw_vals.values).clip(0)
        global_max = max(global_max, rev_y.max())

    def _fmt(v):
        if abs(v) >= 1e6:   return f"${v/1e6:.2f}M"
        if abs(v) >= 1e3:   return f"${v/1e3:.1f}k"
        return f"${v:.0f}"

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    fig.suptitle(
        f"Plot 1b — Annualized Capital vs Annual Revenue per Installed kW\n"
        f"Base / Mid / End Year  (T={T} yr,  g_d={g_d:.1%}/yr,  ρ={rho:.0%},  "
        f"CRF α={alpha:.4f})\n"
        f"Revenue bar grows each year; capital bar is constant — "
        f"crossover = capital charge covered from revenue alone",
        fontsize=10, fontweight="bold"
    )

    for col_idx, (yr, yr_label) in enumerate(zip(years, yr_labels)):
        ax = axes[col_idx]
        fac = (1 + g_d) ** (yr - 1)

        p_chg   = np.where(combined["tech"] == "DWC", p_charge_dwc, p_charge_scs)
        # Reuse physical ceiling computed above
        e_del_y_raw = combined["e_del_base"] * fac
        e_del_y     = np.minimum(e_del_y_raw.values, hw_ceil_physical_1b)
        rev_y       = (p_chg * e_del_y / kw_vals.values).clip(0)
        util_yr     = (e_del_y / (combined["installed_kw"].values * h_op)).clip(0, 1.0)

        bars_cap = ax.bar(x - w/2, cap_per_kw_ann, w,
                          color=cap_colors, edgecolor="white", linewidth=0.8, zorder=3)
        bars_rev = ax.bar(x + w/2, rev_y, w,
                          color=rev_colors, edgecolor="white", linewidth=0.8, zorder=3)

        for bar, v in zip(bars_cap, cap_per_kw_ann):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + global_max * 0.01,
                    _fmt(v), ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

        for bar, rv, util_v in zip(bars_rev, rev_y, util_yr):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + global_max * 0.01,
                    f"{_fmt(rv)}\nUtil {util_v*100:.2f}%",
                    ha="center", va="bottom", fontsize=7.5, linespacing=1.35)

        ax.set_xticks(x)
        ax.set_xticklabels(combined["label"], fontsize=9)
        ax.set_title(yr_label, fontsize=10, fontweight="bold", pad=6)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: _fmt(v)))
        if col_idx == 0:
            ax.set_ylabel(f"$/kW  (annualized or annual)")
        ax.set_ylim(0, global_max * 1.28)
        ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

        if has_dwc and has_scs:
            ax.axvline(len(df_dwc) - 0.5, color="#AAAAAA",
                       linewidth=1.2, linestyle="--", alpha=0.7)

    legend_els = [
        mpatches.Patch(color=C_DWC_CAP, label="DWC — Annualized capital (α×Capex/kW, constant)"),
        mpatches.Patch(color=C_DWC_REV, label="DWC — Annual revenue (P_CHARGE×E_del_y/kW, grows)"),
        mpatches.Patch(color=C_SCS_CAP, label="SCS — Annualized capital (α×Capex/kW, constant)"),
        mpatches.Patch(color=C_SCS_REV, label="SCS — Annual revenue (P_CHARGE×E_del_y/kW, grows)"),
    ]
    axes[1].legend(handles=legend_els, fontsize=8.5,
                   loc="upper center", bbox_to_anchor=(0.5, -0.12),
                   ncol=2, framealpha=0.9)

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    path = os.path.join(out_dir, "plot1b_dollar_per_kw_multiyear.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  Saved: {path}")


# ── analytics master runner (called from generate_analytics_plots) ─────────────

def _run_analytics_plots(results_dir, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(results_dir, "Plots_analytics")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'═'*65}")
    print(f"  Analytics plots")
    print(f"  Results : {results_dir}")
    print(f"  Output  : {out_dir}")
    print(f"{'═'*65}")

    df_dwc_raw, df_scs_raw, meta = load_results_from_dir(results_dir)
    gamma_rev_dwc = float(meta.get("gamma_rev_dwc", 1.0))
    gamma_rev_scs = float(meta.get("gamma_rev_scs", 1.0))
    gamma_var     = float(meta.get("gamma_var",     1.0))
    gamma_fix_om  = float(meta.get("gamma_fix_om",  1.0))
    T   = int(  meta.get("project_life",  30))
    rho = float(meta.get("discount_rate", 0.07))
    g_d = float(meta.get("g_d",           0.0))
    p_veh_dwc = float(meta.get("p_veh_dwc_kw", 0.0))
    print(f"\n  Γ_rev_DWC={gamma_rev_dwc:.3f}  Γ_rev_SCS={gamma_rev_scs:.3f}"
          f"  Γ_var={gamma_var:.3f}  Γ_fix_om={gamma_fix_om:.3f}"
          f"  T={T}yr  ρ={rho:.0%}  g_d={g_d:.2%}"
          f"  P_veh_DWC={p_veh_dwc:.0f}kW")

    df_dwc = compute_dwc_metrics(df_dwc_raw, meta)
    df_scs = compute_scs_metrics(df_scs_raw, meta)

    def _safe_int(v):
        try:
            f = float(v); return 0 if np.isnan(f) else int(f)
        except: return 0

    print(f"\n  {'Station':<14}  {'AADT':>7}  {'Util%':>6}  {'Cap/kW':>9}"
          f"  {'PVRev/kW':>10}  {'LCOST/kWh':>10}  {'NPV':>12}")
    print(f"  {'':─<14}  {'':─>7}  {'':─>6}  {'':─>9}"
          f"  {'':─>10}  {'':─>10}  {'':─>12}")

    def _pr(df):
        for _, r in df.iterrows():
            print(f"  {r['label'].replace(chr(10),' '):<14}  "
                  f"{_safe_int(r.get('AADT_COM', 0)):>7,}  "
                  f"{r['util_rate'] * 100:>5.2f}%  "
                  f"${r['capex_per_kw']:>8,.0f}  "
                  f"${r['pv_rev_per_kw']:>9,.0f}  "
                  f"${r['total_cost_per_kwh']:>9,.4f}  "
                  f"${r['npv']:>11,.0f}")

    if not df_dwc.empty: _pr(df_dwc)
    if not df_scs.empty: _pr(df_scs)

    print("\nGenerating analytics plots ...")

    plot_dollar_per_kw              (df_dwc, df_scs, meta, out_dir)
    plot_dollar_per_kwh             (df_dwc, df_scs, meta, out_dir)
    plot_traffic_utilisation        (df_dwc, df_scs, meta, out_dir)
    plot_utilisation_over_time      (df_dwc, df_scs, meta, out_dir)
    plot_dollar_per_kw_multiyear    (df_dwc, df_scs, meta, out_dir)
    plot_dollar_per_kwh_multiyear   (df_dwc, df_scs, meta, out_dir)
    plot_traffic_utilisation_multiyear(df_dwc, df_scs, meta, out_dir)

    print(f"\n{'═'*65}")
    print(f"  7 analytics plots → {out_dir}/")
    print(f"{'═'*65}\n")
    return out_dir


def generate_analytics_plots(results, output_dir):
    """Call analytics plots using the already-saved Results folder."""
    results_dir = results["meta"]["output_dir"]
    _run_analytics_plots(results_dir,
                         out_dir=os.path.join(output_dir, "analytics"))


# ══════════════════════════════════════════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
#  SECTION B — Main_plots_new functions (_prepare, single-run plots, sensitivity)
# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def _prepare(results):
    """
    Augment df_dwc and df_scs with derived financial columns needed for plots,
    and build the project-level NPV scalars used by plot_breakeven / _summary.

    FIX 3: E_cap_DWC and E_cap_SCS columns are set to the PHYSICAL hardware
    ceiling so that the [HW] binding tag in plot_utilization compares
    base-year E_del against the correct physical reference.
    """
    meta  = results["meta"]
    alpha = results["alpha"]
    H_op  = float(meta["h_op"])
    N_scs = int(meta["n_scs_stalls"])
    p_kw  = float(meta["p_scs_per_stall_kw"])
    util  = float(meta["utilization_rate_scs"])

    # FIX 3: physical ceiling (not base-year-adjusted)
    E_cap_SCS_physical = N_scs * p_kw * H_op * util   # kWh/yr

    df_dwc = results["df_dwc"].copy() if not results["df_dwc"].empty else pd.DataFrame()
    df_scs = results["df_scs"].copy() if not results["df_scs"].empty else pd.DataFrame()

    if not df_dwc.empty:
        # FIX 3: physical DWC ceiling = MW_Load × H_op × 1000
        if "E_cap_DWC" not in df_dwc.columns:
            df_dwc["E_cap_DWC"] = df_dwc["MW_Load"] * H_op * MW_TO_KW
        df_dwc["Ann_Capex"]  = df_dwc["Capex_Total"] * alpha
        df_dwc["Ann_Profit"] = df_dwc["Ann_Revenue"] - df_dwc["Ann_Opex"] - df_dwc["Ann_Capex"]

        dwc_var_opex = df_dwc["LCOE"] * df_dwc["E_del_DWC"]
        dwc_fix_opex = df_dwc["Lambda"] * meta["c_om_base_dwc"]

        df_dwc["Payback_yr"] = (df_dwc["Capex_Total"] /
                                (df_dwc["Ann_Revenue"] - dwc_var_opex - dwc_fix_opex
                                 ).clip(lower=1.0))
        df_dwc["P_rated_kW"]         = df_dwc["MW_Load"] * MW_TO_KW
        df_dwc["DollarPerKW"]        = df_dwc["Capex_Total"] / df_dwc["P_rated_kW"].clip(lower=0.001)
        df_dwc["DollarPerKWh_Cap"]   = df_dwc["Ann_Capex"]  / df_dwc["E_del_DWC"].clip(lower=1)
        df_dwc["DollarPerKWh_Total"] = (df_dwc["Ann_Capex"] + df_dwc["Ann_Opex"]) / df_dwc["E_del_DWC"].clip(lower=1)

    if not df_scs.empty:
        # FIX 3: physical SCS ceiling
        if "E_cap_SCS" not in df_scs.columns:
            df_scs["E_cap_SCS"] = E_cap_SCS_physical
        df_scs["Ann_Capex"]          = df_scs["Capex_Total"] * alpha
        df_scs["Ann_Profit"]         = df_scs["Ann_Revenue"] - df_scs["Ann_Opex"] - df_scs["Ann_Capex"]
        scs_var_opex = df_scs["LCOE"] * df_scs["E_del_SCS"]
        scs_fix_opex = df_scs["Lambda"] * meta["c_om_base_scs"]

        df_scs["Payback_yr"] = (df_scs["Capex_Total"] /
                                (df_scs["Ann_Revenue"] - scs_var_opex - scs_fix_opex
                                 ).clip(lower=1.0))
        df_scs["P_rated_kW"]         = N_scs * p_kw
        df_scs["DollarPerKW"]        = df_scs["Capex_Total"] / (N_scs * p_kw)
        df_scs["DollarPerKWh_Cap"]   = df_scs["Ann_Capex"]  / df_scs["E_del_SCS"].clip(lower=1)
        df_scs["DollarPerKWh_Total"] = (df_scs["Ann_Capex"] + df_scs["Ann_Opex"]) / df_scs["E_del_SCS"].clip(lower=1)
        if "AADT_COM" not in df_scs.columns:
            _diag = results.get("df_diag", pd.DataFrame())
            _aadt = (_diag.set_index("Segment_ID")["AADT_COM"].to_dict()
                     if not _diag.empty else {})
            df_scs["AADT_COM"] = df_scs.get("Host_Seg", pd.Series()).map(_aadt).fillna(0)

    discount_rate = results["discount_rate"]
    project_life  = results["project_life"]

    cap_dwc     = float(df_dwc["Capex_Total"].sum()) if not df_dwc.empty else 0.0
    cap_scs     = float(df_scs["Capex_Total"].sum()) if not df_scs.empty else 0.0
    total_capex = cap_dwc + cap_scs

    g_rev_dwc = float(meta.get("g_rev_dwc", 0.0))
    g_rev_scs = float(meta.get("g_rev_scs", 0.0))
    g_var     = float(meta.get("g_var",     0.0))
    g_om      = float(meta.get("g_om",      0.0))

    base_rev_dwc = float(meta.get("total_rev_dwc", 0.0))
    base_rev_scs = float(meta.get("total_rev_scs", 0.0))
    base_rev     = base_rev_dwc + base_rev_scs

    base_var_opex = 0.0
    if not df_dwc.empty:
        base_var_opex += float((df_dwc["LCOE"] * df_dwc["E_del_DWC"]).sum())
    if not df_scs.empty:
        base_var_opex += float((df_scs["LCOE"] * df_scs["E_del_SCS"]).sum())

    base_fix_opex = 0.0
    if not df_dwc.empty:
        base_fix_opex += float((df_dwc["Lambda"] * meta["c_om_base_dwc"]).sum())
    if not df_scs.empty:
        base_fix_opex += float((df_scs["Lambda"] * meta["c_om_base_scs"]).sum())

    net_op_base = base_rev - base_var_opex - base_fix_opex

    years = np.arange(1, project_life + 1)
    disc  = 1.0 / (1.0 + discount_rate)**years

    # Each stream grows at its own rate; reconstruct year-y net operating income
    # by summing the three components separately before discounting.
    rev_stream      = (base_rev_dwc * (1.0 + g_rev_dwc)**(years - 1)
                       + base_rev_scs * (1.0 + g_rev_scs)**(years - 1))
    var_opex_stream = base_var_opex * (1.0 + g_var)**(years - 1)
    fix_opex_stream = base_fix_opex * (1.0 + g_om)**(years - 1)
    net_op_stream   = rev_stream - var_opex_stream - fix_opex_stream

    cum_npv_arr  = np.concatenate([[-total_capex],
                                   -total_capex + np.cumsum(net_op_stream * disc)])
    net_op_base  = base_rev - base_var_opex - base_fix_opex   # year-1 value
    cum_npv_flat = np.concatenate([[-total_capex],
                                   -total_capex + np.cumsum(np.full(project_life, net_op_base))])

    def _pb(cum):
        idx = np.where(cum >= 0)[0]
        if not len(idx):
            return np.nan
        i = int(idx[0])
        if i == 0:
            return 0.0
        return (i - 1) + (-cum[i - 1]) / (cum[i] - cum[i - 1])

    return df_dwc, df_scs, meta, {
        "total_capex"    : total_capex,
        "cap_dwc"        : cap_dwc,
        "cap_scs"        : cap_scs,
        "net_op"         : net_op_base,
        "base_rev"       : base_rev,
        "base_rev_dwc"   : base_rev_dwc,
        "base_rev_scs"   : base_rev_scs,
        "base_var_opex"  : base_var_opex,
        "base_fix_opex"  : base_fix_opex,
        "cum_npv_arr"    : cum_npv_arr,
        "cum_npv_flat"   : cum_npv_flat,
        "years_arr"      : np.arange(0, project_life + 1),
        "pb_disc"        : _pb(cum_npv_arr),
        "pb_simple"      : _pb(cum_npv_flat),
        "discount_rate"  : discount_rate,
        "project_life"   : project_life,
        "g_rev_dwc"      : g_rev_dwc,
        "g_rev_scs"      : g_rev_scs,
        "g_var"          : g_var,
        "g_om"           : g_om,
    }


def _summary(results):
    df_dwc, df_scs, meta, npv = _prepare(results)
    pbs = []
    if not df_dwc.empty: pbs.append(df_dwc["Payback_yr"])
    if not df_scs.empty: pbs.append(df_scs["Payback_yr"])
    pb_med = float(pd.concat(pbs).clip(0, 60).median()) if pbs else np.nan

    return {
        "run_tag"     : results["run_tag"],
        "n_dwc"       : int(meta["n_dwc_selected"]),
        "n_scs"       : int(meta["n_scs_selected"]),
        "obj_val"     : float(meta["obj_val"]),
        "rev_dwc"     : float(meta["total_rev_dwc"]),
        "rev_scs"     : float(meta["total_rev_scs"]),
        "opex_dwc"    : float(meta["total_opex_dwc"]),
        "opex_scs"    : float(meta["total_opex_scs"]),
        "cap_dwc"     : npv["cap_dwc"],
        "cap_scs"     : npv["cap_scs"],
        "total_capex" : npv["total_capex"],
        "net_op"      : npv["net_op"],
        "pb_disc"     : npv["pb_disc"],
        "pb_simple"   : npv["pb_simple"],
        "pb_unit_med" : pb_med,
        "eta_dwc"     : float(meta.get("eta_dwc", 0)),
        "eta_scs"     : float(meta.get("eta_scs", 0)),
        "rho_ev"      : float(meta.get("rho_ev", 0)),
        "g_ev"        : float(results["params"].get("g_ev",        0)),
        "g_aadt"      : float(results["params"].get("g_aadt",      0)),
        "g_price_dwc" : float(results["params"].get("g_price_dwc", 0)),
        "g_lcoe"      : float(results["params"].get("g_lcoe",      0)),
        "g_om"        : float(results["params"].get("g_om",        0)),
        "e_del_dwc"   : (float(results["df_dwc"]["E_del_DWC"].sum())
                         if not results["df_dwc"].empty else 0.0),
        "e_del_scs"   : (float(results["df_scs"]["E_del_SCS"].sum())
                         if not results["df_scs"].empty else 0.0),
    }


# ── Deployment map ─────────────────────────────────────────────────────────────

def plot_deployment_map(results, output_dir, show_existing=False, title_suffix=""):
    meta             = results["meta"]
    df_diag          = results["df_diag"]
    df_dwc           = results["df_dwc"]
    df_scs           = results["df_scs"]
    sub_coords       = results["sub_coords"]
    sel_sub_dwc      = results["selected_substations_dwc"]
    sel_sub_scs      = results["selected_substations_scs"]
    scs_coords_exist = results["scs_coords_exist"]

    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_axis_off()

    _draw_us_background(ax)

    ax.scatter(df_diag["X"], df_diag["Y"],
           c="#aab0b8", s=18, alpha=0.55, zorder=1,
           linewidths=0, rasterized=True)

    all_subs = sel_sub_dwc | sel_sub_scs
    if all_subs:
        sx = [sub_coords[s][0] for s in all_subs if s in sub_coords]
        sy = [sub_coords[s][1] for s in all_subs if s in sub_coords]
        ax.scatter(sx, sy, c="black", s=12, zorder=6,
                   linewidths=0, label="Substation (Selected)")

    if not df_dwc.empty and "MID_X" in df_dwc.columns:
        _d = df_dwc.dropna(subset=["MID_X", "MID_Y"])
        ax.scatter(_d["MID_X"], _d["MID_Y"],
                   c=C_GOLD, s=45, zorder=4,
                   edgecolors="black", linewidths=0.5,
                   label=f"Selected DWC Segment (n={len(_d)})")

    if not df_scs.empty and "MID_X" in df_scs.columns:
        _s = df_scs.dropna(subset=["MID_X", "MID_Y"])
        ax.scatter(_s["MID_X"], _s["MID_Y"],
                   marker=_MARKER_SCS, c=C_CYAN,
                   edgecolors="black", s=110, zorder=5,
                   label=f"New SCS Station (n={len(_s)})")

    if show_existing and scs_coords_exist:
        ex_xy = [(v[0], v[1]) for v in scs_coords_exist.values()
                 if v[0] is not None]
        if ex_xy:
            ax.scatter([p[0] for p in ex_xy], [p[1] for p in ex_xy],
                       marker=_MARKER_EXIST, c=C_EXIST, s=65,
                       edgecolors="black", linewidths=0.4,
                       alpha=0.60, zorder=3,
                       label=f"Existing SCS (n={len(ex_xy)})")

    ax.annotate("N", xy=(0.95, 0.08), xycoords="axes fraction",
                ha="center", va="bottom", fontsize=18, fontweight="bold")
    ax.arrow(0.95, 0.02, 0, 0.05, transform=ax.transAxes,
             length_includes_head=True, head_width=0.01, head_length=0.02,
             fc="black", ec="black")

    ax.legend(loc="lower left", fontsize=12, frameon=False,
              markerscale=1.0, handlelength=1.2,
              borderpad=0.5, labelspacing=0.4)

    fname = "map_full.png" if show_existing else "map_proposed.png"
    if title_suffix:
        fname = fname.replace(".png", f"_{title_suffix.replace(' ', '_')}.png")
    _savefig(fig, os.path.join(output_dir, fname))


# ── Traffic split ──────────────────────────────────────────────────────────────

def plot_traffic_split(results, output_dir, title_suffix=""):
    meta    = results["meta"]
    eta_dwc = float(meta.get("eta_dwc", 0))
    eta_scs = float(meta.get("eta_scs", 0))

    labels = [r"$\eta^{\mathrm{SCS}}$", r"$\eta^{\mathrm{DWC}}$"]
    values = [eta_scs * 100, eta_dwc * 100]
    colors = [C_SCS, C_DWC]

    fig, ax = plt.subplots(figsize=(5, 1.8))
    bars = ax.barh(labels, values, color=colors, height=0.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 2, bar.get_y() + bar.get_height() / 2,
                f"{width:.0f}%",
                va="center", ha="left", fontsize=12, fontweight="bold")

    ax.tick_params(axis="y", length=0, labelsize=14)
    ttl = "Traffic-Capture"
    if title_suffix:
        ttl += f"\n{title_suffix}"
    ax.set_title(ttl, fontweight="bold", fontsize=13)
    plt.tight_layout()
    fname = "traffic_split.png"
    if title_suffix:
        fname = f"traffic_split_{title_suffix.replace(' ', '_')}.png"
    _savefig(fig, os.path.join(output_dir, fname))


# ── Total units pie ────────────────────────────────────────────────────────────

def plot_total_units(results, output_dir, title_suffix=""):
    meta  = results["meta"]
    n_dwc = int(meta["n_dwc_selected"])
    n_scs = int(meta["n_scs_selected"])

    slice_labels = []
    slice_values = []
    slice_colors = []
    if n_dwc > 0:
        slice_labels.append("DWC")
        slice_values.append(n_dwc)
        slice_colors.append(C_DWC)
    if n_scs > 0:
        slice_labels.append("SCS")
        slice_values.append(n_scs)
        slice_colors.append(C_SCS)

    fig, ax = plt.subplots(figsize=(6, 6))

    if not slice_values:
        ax.pie([1], colors=[C_GREY], startangle=140)
        ax.set_title("Resulting Units\n(none selected)", fontsize=20,
                     fontweight="bold", pad=-5)
    else:
        ax.pie(
            slice_values,
            labels=None,
            autopct=lambda p: f"{round(p * sum(slice_values) / 100):.0f}",
            startangle=140,
            colors=slice_colors,
            pctdistance=0.50,
            textprops={"fontweight": "bold", "color": "white", "fontsize": 20},
        )
        ttl = "Resulting Units"
        if title_suffix:
            ttl += f"\n{title_suffix}"
        ax.set_title(ttl, fontsize=20, fontweight="bold", pad=-5)

    legend_handles = []
    if n_dwc > 0:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, color=C_DWC, label=f"DWC (n={n_dwc})"))
    if n_scs > 0:
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, color=C_SCS, label=f"SCS (n={n_scs})"))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower center",
                  bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=12,
                  frameon=False)

    plt.tight_layout()
    fname = "total_units.png"
    if title_suffix:
        fname = f"total_units_{title_suffix.replace(' ', '_')}.png"
    _savefig(fig, os.path.join(output_dir, fname))


# ── Capex / revenue / profit ───────────────────────────────────────────────────

def plot_capex_revenue_profit(results, output_dir, title_suffix=""):
    df_dwc, df_scs, meta, npv = _prepare(results)
    alpha       = results["alpha"]
    have_dwc    = not df_dwc.empty
    have_scs    = not df_scs.empty
    have_any    = have_dwc or have_scs

    p_charge_dwc = float(meta.get("p_charge_dwc", 0.40))
    p_charge_scs = float(meta.get("p_charge_scs", 0.42))

    if not have_any:
        print("  Skipping capex_revenue_profit.png (no assets selected)")
        return

    def _build_rec(df_a, e_del_col, capex_col, opex_col, rev_col,
                   p_rated_col, ann_capex_col, retail, tech):
        if len(df_a) == 0:
            return None
        e   = df_a[e_del_col].mean()
        rev = df_a[rev_col].mean()
        ac  = df_a[ann_capex_col].mean()
        ao  = df_a[opex_col].mean()
        cap = df_a[capex_col].mean()
        pkw = df_a[p_rated_col].mean()
        return {
            "label"        : tech,
            "n"            : len(df_a),
            "E_del_kWh"    : e,
            "Ann_Rev"      : rev,
            "Ann_Capex"    : ac,
            "Ann_Opex"     : ao,
            "Ann_Profit"   : rev - ac - ao,
            "Capex_total"  : cap,
            "P_rated_kW"   : pkw,
            "DolPerKW"     : cap / max(pkw, 1),
            "DolPerKWh_cap": ac  / max(e, 1),
            "DolPerKWh_op" : ao  / max(e, 1),
            "DolPerKWh_rev": rev / max(e, 1),
            "Retail"       : retail,
            "Margin_kWh"   : retail - (ac + ao) / max(e, 1),
        }

    records = []
    if have_dwc:
        records.append(_build_rec(df_dwc, "E_del_DWC", "Capex_Total", "Ann_Opex",
                                  "Ann_Revenue", "P_rated_kW", "Ann_Capex",
                                  p_charge_dwc, "DWC"))
    if have_scs:
        records.append(_build_rec(df_scs, "E_del_SCS", "Capex_Total", "Ann_Opex",
                                  "Ann_Revenue", "P_rated_kW", "Ann_Capex",
                                  p_charge_scs, "SCS"))
    records = [r for r in records if r is not None]
    if not records:
        return

    colors  = {"DWC": C_DWC, "SCS": C_SCS}
    n_tech  = len(records)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    ax = axes[0]
    x_pos = np.arange(n_tech)
    vals  = [r["DolPerKW"] / 1e3 for r in records]
    bars  = ax.bar(x_pos, vals,
                   color=[colors[r["label"]] for r in records],
                   width=0.45, alpha=0.85, edgecolor="none")
    for b, v, r in zip(bars, vals, records):
        ax.text(b.get_x() + b.get_width() / 2,
                v + max(vals) * 0.02,
                f"${v:.1f}K/kW\n({r['P_rated_kW']:.0f} kW)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([r["label"] for r in records], fontsize=13)
    ax.set_ylabel("Capex per kW  ($K/kW)")
    ax.set_title("(a)  Capex Intensity\n($/kW rated)", fontweight="bold")
    ax.tick_params(axis="x", length=0)
    ax.axhline(1.0, color="grey", lw=1.2, ls=":", alpha=0.6,
               label="$1K/kW ref (DCFC low)")
    ax.axhline(4.0, color="grey", lw=1.2, ls="--", alpha=0.6,
               label="$4K/kW ref (DCFC high)")
    ax.legend(fontsize=9, frameon=False)
    ax.set_ylim(0, max(max(vals) * 1.40, 5.0))

    ax = axes[1]
    x_pos2 = np.arange(n_tech)
    w      = 0.28

    for idx, rec in enumerate(records):
        col   = colors[rec["label"]]
        x_cap = idx - w
        x_rev = idx + w
        ax.bar(x_cap, rec["DolPerKWh_cap"], width=w * 1.8,
               color=col, alpha=0.90, edgecolor="none",
               label="Ann. Capex/kWh" if idx == 0 else "")
        ax.bar(x_cap, rec["DolPerKWh_op"], width=w * 1.8,
               bottom=rec["DolPerKWh_cap"],
               color=col, alpha=0.40, edgecolor="none",
               label="Opex/kWh" if idx == 0 else "")
        ax.bar(x_rev, rec["DolPerKWh_rev"], width=w * 1.8,
               color=C_REV, alpha=0.85, edgecolor="none",
               label="Revenue/kWh" if idx == 0 else "")

        total_cost = rec["DolPerKWh_cap"] + rec["DolPerKWh_op"]
        margin     = rec["DolPerKWh_rev"] - total_cost
        mid_y      = max(total_cost, rec["DolPerKWh_rev"]) + 0.008
        sign       = "+" if margin >= 0 else ""
        ax.annotate(
            f"{sign}${margin:.3f}/kWh\nmargin",
            xy=(idx, mid_y), ha="center", va="bottom",
            fontsize=10, fontweight="bold",
            color=C_REV if margin >= 0 else C_CAPEX,
        )
        ax.text(x_cap, total_cost + 0.003,
                f"${total_cost:.3f}", ha="center", va="bottom",
                fontsize=9, color="white" if total_cost > 0.05 else "black")
        ax.text(x_rev, rec["DolPerKWh_rev"] + 0.003,
                f"${rec['DolPerKWh_rev']:.3f}", ha="center", va="bottom",
                fontsize=9, color=C_REV)

    ax.set_xticks(x_pos2)
    ax.set_xticklabels(
        [f"{r['label']}\n(n={r['n']}, {r['E_del_kWh']/1e3:.0f} MWh/yr/unit)"
         for r in records],
        fontsize=11
    )
    for rec in records:
        ax.axhline(rec["Retail"], color=colors[rec["label"]],
                   lw=1.5, ls="--", alpha=0.55,
                   label=f"{rec['label']} retail ${rec['Retail']:.2f}/kWh")
    ax.set_ylabel("$/kWh delivered")
    ax.set_title("(b)  Cost vs Revenue per kWh\n"
                 "Left=costs  Right=retail revenue", fontweight="bold")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_ylim(0, max(
        max(r["DolPerKWh_rev"] for r in records),
        max(r["DolPerKWh_cap"] + r["DolPerKWh_op"] for r in records)
    ) * 1.45)

    ax = axes[2]
    bar_labels = ["Ann.\nRevenue", "Ann.\nCapex", "Ann.\nOpex", "Ann.\nProfit"]
    x_grp      = np.arange(len(bar_labels))
    bar_w      = 0.80 / max(n_tech, 1)

    for idx, rec in enumerate(records):
        col    = colors[rec["label"]]
        vals_k = [rec["Ann_Rev"] / 1e3, rec["Ann_Capex"] / 1e3,
                  rec["Ann_Opex"] / 1e3, rec["Ann_Profit"] / 1e3]
        bar_cols = [C_REV, C_CAPEX, C_OPEX,
                    C_REV if rec["Ann_Profit"] >= 0 else C_CAPEX]
        x_offset = (idx - (n_tech - 1) / 2) * bar_w
        for xi, (v, bc) in enumerate(zip(vals_k, bar_cols)):
            ax.bar(x_grp[xi] + x_offset, v, width=bar_w * 0.85,
                   color=bc, alpha=0.80, edgecolor="white", linewidth=0.5)
            fmt = f"${v:.0f}K" if abs(v) >= 1 else f"${v:.2f}K"
            ax.text(x_grp[xi] + x_offset,
                    v + (abs(max(vals_k)) * 0.025) * np.sign(v) if v != 0 else 0.5,
                    fmt, ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=9, fontweight="bold", color=bc)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x_grp)
    ax.set_xticklabels(bar_labels, fontsize=11)
    ax.set_ylabel("Annual $ per Unit ($K/yr)")
    ax.set_title("(c)  Annual Financial Summary per Unit\n"
                 "(per segment for DWC, per station for SCS)", fontweight="bold")
    ax.tick_params(axis="x", length=0)
    _tech_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[r["label"]], alpha=0.80,
                       label=f"{r['label']} (n={r['n']})")
        for r in records
    ]
    ax.legend(handles=_tech_handles, fontsize=9, frameon=False, loc="upper right")

    _summary_lines = []
    for rec in records:
        pb = rec["Capex_total"] / max(rec["Ann_Rev"] - rec["Ann_Opex"], 1)
        _summary_lines.append(
            f"{rec['label']}: capex ${rec['Capex_total']/1e6:.2f}M  |  "
            f"margin ${rec['Margin_kWh']:.3f}/kWh  |  simple payback {pb:.1f} yr"
        )
    fig.text(0.5, -0.02, "\n".join(_summary_lines),
             ha="center", va="top", fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="lightyellow", edgecolor="grey", alpha=0.9))

    _ttl = "Per-Unit Revenue vs Cost Analysis — DWC & SCS"
    if title_suffix:
        _ttl += f"\n{title_suffix}"
    fig.suptitle(_ttl, fontweight="bold", y=1.02)
    plt.tight_layout()
    fname = "capex_revenue_profit.png"
    if title_suffix:
        fname = f"capex_revenue_profit_{title_suffix.replace(' ', '_')}.png"
    _savefig(fig, os.path.join(output_dir, fname))


# ── Breakeven ─────────────────────────────────────────────────────────────────

def plot_breakeven(results, output_dir, title_suffix=""):
    df_dwc, df_scs, meta, npv = _prepare(results)

    project_life  = npv["project_life"]
    discount_rate = npv["discount_rate"]
    total_capex   = npv["total_capex"]
    g_rev_dwc = npv["g_rev_dwc"]
    g_rev_scs = npv["g_rev_scs"]
    g_var     = npv["g_var"]
    g_om      = npv["g_om"]
    # For display label on the chart use the DWC revenue rate as representative
    g_rev     = g_rev_dwc

    base_rev      = npv["base_rev"]
    base_rev_dwc  = npv["base_rev_dwc"]
    base_rev_scs  = npv["base_rev_scs"]
    base_var_opex = npv["base_var_opex"]
    base_fix_opex = npv["base_fix_opex"]

    years_ext = np.arange(0, project_life + 1)
    yrs       = np.arange(1, project_life + 1)
    disc      = 1.0 / (1.0 + discount_rate)**yrs

    growing_annual = (
        base_rev_dwc * (1.0 + g_rev_dwc)**(yrs - 1)
        + base_rev_scs * (1.0 + g_rev_scs)**(yrs - 1)
        - base_var_opex * (1.0 + g_var)**(yrs - 1)
        - base_fix_opex * (1.0 + g_om)**(yrs - 1)
    )
    cum_npv = np.concatenate([[-total_capex],
                               -total_capex + np.cumsum(growing_annual * disc)])

    def _find_pb(cum_arr):
        idx = np.where(cum_arr >= 0)[0]
        if len(idx) == 0:
            return np.nan
        i = int(idx[0])
        if i == 0:
            return 0.0
        y0, y1 = cum_arr[i - 1], cum_arr[i]
        return (i - 1) + (-y0) / (y1 - y0)

    pb_proj = _find_pb(cum_npv)

    unit_labels, unit_pb_disc, unit_pb_simple, unit_colors, unit_techs = [], [], [], [], []

    if not df_dwc.empty:
        for _, row in df_dwc.iterrows():
            capex   = float(row["Capex_Total"])
            ann_rev = float(row["Ann_Revenue"])
            var_om  = float(row["LCOE"]) * float(row["E_del_DWC"])
            fix_om  = float(row["Lambda"]) * float(meta["c_om_base_dwc"])
            net_y   = (ann_rev * (1.0 + g_rev_dwc)**(yrs - 1)
                       - var_om * (1.0 + g_var)**(yrs - 1)
                       - fix_om * (1.0 + g_om)**(yrs - 1))
            cum_d   = np.concatenate([[-capex], -capex + np.cumsum(net_y * disc)])
            net_f   = ann_rev - var_om - fix_om
            cum_f   = np.concatenate([[-capex], -capex + np.cumsum(np.full(project_life, net_f))])
            unit_labels.append(f"DWC\n{row['Segment_ID']}")
            unit_pb_disc.append(_find_pb(cum_d))
            unit_pb_simple.append(_find_pb(cum_f))
            unit_colors.append(C_DWC)
            unit_techs.append("DWC")

    if not df_scs.empty:
        for _, row in df_scs.iterrows():
            capex   = float(row["Capex_Total"])
            ann_rev = float(row["Ann_Revenue"])
            var_om  = float(row["LCOE"]) * float(row["E_del_SCS"])
            fix_om  = float(row["Lambda"]) * float(meta["c_om_base_scs"])
            net_y   = (ann_rev * (1.0 + g_rev_scs)**(yrs - 1)
                       - var_om * (1.0 + g_var)**(yrs - 1)
                       - fix_om * (1.0 + g_om)**(yrs - 1))
            cum_d   = np.concatenate([[-capex], -capex + np.cumsum(net_y * disc)])
            net_f   = ann_rev - var_om - fix_om
            cum_f   = np.concatenate([[-capex], -capex + np.cumsum(np.full(project_life, net_f))])
            unit_labels.append(f"SCS\n{row['SCS_ID']}")
            unit_pb_disc.append(_find_pb(cum_d))
            unit_pb_simple.append(_find_pb(cum_f))
            unit_colors.append(C_SCS)
            unit_techs.append("SCS")

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(14, 5.5), facecolor="white",
        gridspec_kw={"wspace": 0.38}
    )

    ax = ax_left
    ax.set_facecolor("white")
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)

    pb_str = f"{pb_proj:.1f} yr" if not np.isnan(pb_proj) and pb_proj <= project_life else ">horizon"
    label = (f"NPV cash flow  (g={g_rev:.1%}, r={discount_rate:.0%})\n"
             f"Payback: {pb_str}")
    ax.plot(years_ext, cum_npv / 1e6, color=C_DWC, lw=2.0, label=label)

    if not np.isnan(pb_proj) and pb_proj <= project_life:
        ax.axvline(pb_proj, color=C_DWC, lw=1.0, ls=":", alpha=0.7)
        ax.scatter([pb_proj], [0], color=C_DWC, s=60, zorder=5,
                   edgecolors="white", linewidths=0.8)

    ax.fill_between(years_ext, cum_npv / 1e6, 0,
                    where=(cum_npv >= 0), color=C_REV, alpha=0.07, interpolate=True)

    ax.text(0.02, 0.04,
            f"CAPEX  ${total_capex/1e6:.1f} M",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, family="monospace", color="#333333",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9f9f9",
                      edgecolor="#dddddd", alpha=0.95))

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Cumulative cash flow  ($M)", fontsize=11)
    ax.set_xlim(0, project_life)
    ax.set_title("Project-Level Cumulative NPV", fontweight="bold", fontsize=12, pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = ax_right
    ax.set_facecolor("white")

    n_units = len(unit_labels)
    x_pos   = np.arange(n_units)
    bar_w   = 0.35

    pb_d_plot = [min(v, project_life * 1.05) if not np.isnan(v) else project_life * 1.05
                 for v in unit_pb_disc]
    pb_s_plot = [min(v, project_life * 1.05) if not np.isnan(v) else project_life * 1.05
                 for v in unit_pb_simple]

    bars_d = ax.bar(x_pos - bar_w/2, pb_d_plot, bar_w,
                    color=unit_colors, alpha=0.85, edgecolor="white", linewidth=0.5,
                    label="Discounted payback")
    bars_s = ax.bar(x_pos + bar_w/2, pb_s_plot, bar_w,
                    color=unit_colors, alpha=0.40, edgecolor="white", linewidth=0.5,
                    label="Simple payback")

    for xi, (pd_v, ps_v) in enumerate(zip(unit_pb_disc, unit_pb_simple)):
        pd_str = f"{pd_v:.1f}" if not np.isnan(pd_v) and pd_v <= project_life else ">"
        ps_str = f"{ps_v:.1f}" if not np.isnan(ps_v) and ps_v <= project_life else ">"
        ax.text(xi - bar_w/2, pb_d_plot[xi] + 0.3, pd_str,
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=unit_colors[xi])
        ax.text(xi + bar_w/2, pb_s_plot[xi] + 0.3, ps_str,
                ha="center", va="bottom", fontsize=9,
                color=unit_colors[xi], alpha=0.7)

    ax.axhline(project_life, color="grey", lw=1.2, ls="--", alpha=0.7,
               label=f"Project horizon  {project_life} yr")

    legend_handles = []
    if any(t == "DWC" for t in unit_techs):
        legend_handles.append(
            plt.Rectangle((0,0),1,1, color=C_DWC, alpha=0.85, label="DWC — discounted PB"))
        legend_handles.append(
            plt.Rectangle((0,0),1,1, color=C_DWC, alpha=0.40, label="DWC — simple PB"))
    if any(t == "SCS" for t in unit_techs):
        legend_handles.append(
            plt.Rectangle((0,0),1,1, color=C_SCS, alpha=0.85, label="SCS — discounted PB"))
        legend_handles.append(
            plt.Rectangle((0,0),1,1, color=C_SCS, alpha=0.40, label="SCS — simple PB"))
    legend_handles.append(
        Line2D([0],[0], color="grey", lw=1.2, ls="--",
               label=f"Project horizon  {project_life} yr"))
    ax.legend(handles=legend_handles, fontsize=8, frameon=False, loc="upper left")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(unit_labels, fontsize=10)
    ax.set_ylabel("Payback year", fontsize=11)
    ax.set_ylim(0, project_life * 1.15)
    ax.set_title("Per-Unit Payback Year\nDark = discounted  |  Light = simple (no growth, no discount)",
                 fontweight="bold", fontsize=12, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if title_suffix:
        fig.suptitle(title_suffix, fontsize=11, color="#555555", y=1.01)

    fname = "breakeven.png"
    if title_suffix:
        fname = f"breakeven_{title_suffix.replace(' ', '_')}.png"
    _savefig(fig, os.path.join(output_dir, fname))


# ── Utilization ───────────────────────────────────────────────────────────────

def plot_utilization(results, output_dir, title_suffix=""):
    df_dwc, df_scs, meta, _ = _prepare(results)
    have_dwc = not df_dwc.empty
    have_scs = not df_scs.empty

    if not (have_dwc or have_scs):
        print("  Skipping utilization.png (no assets selected)")
        return

    panels = []
    if have_dwc:
        panels.append(("DWC", df_dwc, "E_traff_DWC", "E_del_DWC", "E_cap_DWC",
                        "Segment_ID", C_DWC, int(meta["n_dwc_selected"])))
    if have_scs:
        panels.append(("SCS", df_scs, "E_traff_SCS", "E_del_SCS", "E_cap_SCS",
                        "SCS_ID", C_SCS, int(meta["n_scs_selected"])))

    ncols = len(panels)
    n_max = max(len(df_dwc) if have_dwc else 0,
                len(df_scs) if have_scs else 0)
    fig_h = max(8, min(n_max * 0.22 + 2, 60))
    fig, axes = plt.subplots(1, ncols, figsize=(16 * ncols, fig_h))
    if ncols == 1:
        axes = [axes]

    BH = 0.22
    for ax, (tech, df, col_tr, col_del, col_cap, col_id, color_d, n_lbl) in \
            zip(axes, panels):
        df_s = df.sort_values(col_tr, ascending=True).reset_index(drop=True)
        n    = len(df_s)
        y    = np.arange(n)
        step = _label_step(n)
        fs   = max(5.0, min(8.5, 400 / max(n, 1)))

        ax.barh(y + 0.26, df_s[col_cap] / 1e6,  BH,
                color="#d5d8dc", edgecolor="none", label="Hardware ceiling")
        ax.barh(y,        df_s[col_tr]  / 1e6,  BH,
                color=color_d, alpha=0.50, edgecolor="none", label="Traffic demand")
        ax.barh(y - 0.26, df_s[col_del] / 1e6,  BH,
                color=C_REV, alpha=0.85, edgecolor="none", label="Delivered")

        if n <= 120:
            x_max = float(df_s[col_cap].max()) / 1e6
            for k in range(n):
                util    = float(df_s[col_del].iloc[k]) / max(float(df_s[col_tr].iloc[k]), 1) * 100
                hw_bind = float(df_s[col_del].iloc[k]) >= 0.99 * float(df_s[col_cap].iloc[k])
                tr_bind = float(df_s[col_del].iloc[k]) >= 0.99 * float(df_s[col_tr].iloc[k])
                tag     = " [HW]" if hw_bind else (" [Tr]" if tr_bind else "")
                ax.text(float(df_s[col_del].iloc[k]) / 1e6 + x_max * 0.01,
                        y[k] - 0.26,
                        f"{util:.0f}%{tag}",
                        va="center", fontsize=max(4, fs - 1.5), color="#444444")

        ax.set_yticks(y[::step])
        ax.set_yticklabels(df_s[col_id].iloc[::step].astype(str), fontsize=fs)
        ax.set_ylabel(f"{col_id}  (sorted by traffic demand ↑)")
        ax.set_xlabel("Annual energy  (GWh / yr)")

        u_arr      = df_s[col_del] / df_s[col_tr].clip(lower=1)
        n_hw       = (df_s[col_del] >= 0.99 * df_s[col_cap]).sum()
        n_tr       = (df_s[col_del] >= 0.99 * df_s[col_tr]).sum()
        total_del  = df_s[col_del].sum()
        total_traf = df_s[col_tr].sum()

        ax.set_title(
            f"{tech} Utilisation  (n = {n_lbl})\n"
            f"Delivered {total_del/1e6:.1f} GWh / Demand {total_traf/1e6:.1f} GWh"
            f" = {total_del/max(total_traf,1)*100:.1f}% aggregate\n"
            "[HW]=hardware-bound  |  [Tr]=traffic-bound",
            fontweight="bold"
        )
        ax.text(0.98, 0.03,
                f"Traffic util: median {u_arr.median()*100:.1f}%  "
                f"mean {u_arr.mean()*100:.1f}%\n"
                f"HW-bound: {n_hw}/{n}   Traffic-bound: {n_tr}/{n}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                          edgecolor="grey", alpha=0.9))
        ax.legend(fontsize=9, frameon=False, loc="lower right")

    _ttl = "DWC & SCS Energy Utilisation"
    if title_suffix:
        _ttl += f" — {title_suffix}"
    fig.suptitle(_ttl, fontweight="bold", y=1.01)
    plt.tight_layout()
    fname = "utilization.png"
    if title_suffix:
        fname = f"utilization_{title_suffix.replace(' ', '_')}.png"
    _savefig(fig, os.path.join(output_dir, fname))


# ── Heatmaps ──────────────────────────────────────────────────────────────────

def plot_heatmaps(results, output_dir, title_suffix=""):
    df_diag          = results["df_diag"]
    df_dwc           = results["df_dwc"]
    df_scs           = results["df_scs"]
    scs_coords_exist = results["scs_coords_exist"]
    meta             = results["meta"]
    use_existing_scs = meta.get("use_existing_scs", False)

    if "Is_DWC" not in df_diag.columns:
        df_diag = df_diag.copy()
        df_diag["Is_DWC"] = 0

    def _overlay(ax, df_dwc_=df_dwc, df_scs_=df_scs):
        sel = df_diag[df_diag["Is_DWC"] == 1]
        if not sel.empty:
            ax.scatter(sel["X"], sel["Y"],
                       facecolors="none", edgecolors=C_DWC,
                       linewidths=1.2, s=120, zorder=4,
                       label="Selected DWC")
        if not df_scs_.empty and "MID_X" in df_scs_.columns:
            _sp = df_scs_.dropna(subset=["MID_X", "MID_Y"])
            if not _sp.empty:
                ax.scatter(_sp["MID_X"], _sp["MID_Y"],
                           marker=_MARKER_SCS, c=C_CYAN,
                           edgecolors="black", s=80, zorder=5,
                           label="New SCS")
        if use_existing_scs and scs_coords_exist:
            ex_xy = [(v[0], v[1]) for v in scs_coords_exist.values()
                     if v[0] is not None]
            if ex_xy:
                ax.scatter([p[0] for p in ex_xy], [p[1] for p in ex_xy],
                           marker=_MARKER_EXIST, c=C_EXIST,
                           edgecolors="black", s=40, alpha=0.60, zorder=3,
                           label="Existing SCS")

    def _heatmap(col, title, label, cmap, filename, log_scale=True):
        fig_, ax_ = plt.subplots(figsize=(13, 10))
        _draw_us_background(ax_)

        vals = df_diag[col][df_diag[col] > 0] if log_scale else df_diag[col]
        _col_max = df_diag[col].max()
        _col_min = df_diag[col].min()
        if log_scale and len(vals) > 0 and vals.min() > 0 and _col_max > 0:
            norm = mcolors.LogNorm(vmin=vals.min(), vmax=_col_max)
        else:
            _vmin = _col_min if _col_min != _col_max else 0
            _vmax = _col_max if _col_max != _col_min else 1
            norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)

        sc = ax_.scatter(df_diag["X"], df_diag["Y"],
                         c=df_diag[col], cmap=cmap,
                         s=55, alpha=0.75, norm=norm,
                         edgecolors="none", zorder=2)
        _overlay(ax_)

        cbar = plt.colorbar(sc, ax=ax_, pad=0.02)
        cbar.set_label(label, rotation=270, labelpad=18, fontsize=13)
        cbar.ax.tick_params(labelsize=11)
        ax_.set_title(title + (f" — {title_suffix}" if title_suffix else ""),
                      fontsize=16, fontweight="bold", pad=10)
        for spine in ax_.spines.values():
            spine.set_visible(False)
        ax_.set_xticks([]); ax_.set_yticks([])
        ax_.legend(fontsize=11, loc="upper left", frameon=False)
        plt.tight_layout()
        fpath = os.path.join(output_dir,
                             filename if not title_suffix else
                             filename.replace(".png",
                                             f"_{title_suffix.replace(' ','_')}.png"))
        _savefig(fig_, fpath)

    _heatmap("Total_Revenue",
             "Spatial Revenue Potential (DWC)",
             "Annual Revenue Potential ($)", "YlOrRd",
             "heatmap_revenue.png", log_scale=True)
    _heatmap("Rev_Dist_Ratio",
             "Revenue / Substation Distance Ratio",
             "Rev / Distance Index", "viridis",
             "heatmap_rev_dist.png", log_scale=True)
    _heatmap("AADT_COM",
             "Commercial Traffic Demand (AADT)",
             "Commercial AADT", "YlGn",
             "heatmap_traffic.png", log_scale=True)
    _heatmap("Lambda",
             r"Locational Cost Multiplier ($\lambda$)",
             r"$\lambda$ Value", "Purples",
             "heatmap_lambda.png", log_scale=False)

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
    for ax_, col, title, label, cmap, log in [
        (ax1, "AADT_COM", "Commercial Traffic Demand", "AADT", "YlGn", True),
        (ax2, "Lambda",   r"Locational Multiplier ($\lambda$)", r"$\lambda$", "Purples", False),
    ]:
        _draw_us_background(ax_)
        _dvals = df_diag[col][df_diag[col] > 0]
        _dmax  = df_diag[col].max()
        _dmin  = df_diag[col].min()
        if log and len(_dvals) > 0 and _dvals.min() > 0 and _dmax > 0:
            norm = mcolors.LogNorm(vmin=_dvals.min(), vmax=_dmax)
        else:
            norm = mcolors.Normalize(
                vmin=_dmin if _dmin != _dmax else 0,
                vmax=_dmax if _dmax != _dmin else 1
            )
        sc = ax_.scatter(df_diag["X"], df_diag["Y"],
                         c=df_diag[col], cmap=cmap,
                         s=50, alpha=0.75, norm=norm, edgecolors="none")
        _overlay(ax_)
        cbar = plt.colorbar(sc, ax=ax_, pad=0.02)
        cbar.set_label(label, rotation=270, labelpad=18, fontsize=13)
        ax_.set_title(title, fontsize=16, fontweight="bold", pad=10)
        for spine in ax_.spines.values():
            spine.set_visible(False)
        ax_.set_xticks([]); ax_.set_yticks([])
        ax_.legend(fontsize=10, frameon=False)

    plt.tight_layout()
    fname2 = "heatmap_demand_vs_lambda.png"
    if title_suffix:
        fname2 = f"heatmap_demand_vs_lambda_{title_suffix.replace(' ','_')}.png"
    _savefig(fig2, os.path.join(output_dir, fname2))


# ── Financial summary multi-year ───────────────────────────────────────────────
# FIX 2 applied here: year-y energy is capped at the physical hardware ceiling.

def plot_financial_summary_multiyear(results, output_dir, title_suffix=""):
    df_dwc, df_scs, meta, _ = _prepare(results)
    have_dwc = not df_dwc.empty
    have_scs = not df_scs.empty
    if not have_dwc and not have_scs:
        print("  Skipping financial_summary_multiyear.png (no assets selected)")
        return

    T      = results["project_life"]
    g_d       = float(meta.get("g_d",       0.0))
    g_price_u = None   # set per-unit below from tech type
    g_var_m   = float(meta.get("g_var",     0.0))
    g_om_m    = float(meta.get("g_om",      0.0))
    g_rev_dwc_m = float(meta.get("g_rev_dwc", 0.0))
    g_rev_scs_m = float(meta.get("g_rev_scs", 0.0))
    alpha  = results["alpha"]
    p_charge_dwc = float(meta.get("p_charge_dwc", 0.40))
    p_charge_scs = float(meta.get("p_charge_scs", 0.40))

    # Physical ceiling values needed for FIX 2
    H_op_plot  = float(meta.get("h_op", 8760.0))
    util_scs   = float(meta.get("utilization_rate_scs", 1.0))
    n_stalls_p = float(meta.get("n_scs_stalls", 6))
    p_stall_p  = float(meta.get("p_scs_per_stall_kw", 250.0))

    mid_yr = T // 2 + 1
    years  = [1, mid_yr, T]
    yr_labels = [f"Year 1\n(Base)", f"Year {mid_yr}\n(Mid)", f"Year {T}\n(End)"]

    units = []
    if have_dwc:
        for _, row in df_dwc.iterrows():
            units.append({
                "label"            : f"DWC {row['Segment_ID']}",
                "tech"             : "DWC",
                "e_del_base"       : float(row["E_del_DWC"]),
                "fix_om"           : float(row["Lambda"]) * float(meta["c_om_base_dwc"]),
                "lcoe"             : float(row["LCOE"]),
                "capex"            : float(row["Capex_Total"]),
                 "p_charge"         : p_charge_dwc,
                "g_rev_unit"       : g_rev_dwc_m,
                "g_var_unit"       : g_var_m,
                "g_om_unit"        : g_om_m,
                # FIX 2: physical ceiling for DWC = MW_Load × H_op × 1000
                "hw_ceil_physical" : float(row["MW_Load"]) * H_op_plot * MW_TO_KW,
            })
    if have_scs:
        for _, row in df_scs.iterrows():
            units.append({
                "label"            : f"SCS {row['SCS_ID']}",
                "tech"             : "SCS",
                "e_del_base"       : float(row["E_del_SCS"]),
                "fix_om"           : float(row["Lambda"]) * float(meta["c_om_base_scs"]),
                "lcoe"             : float(row["LCOE"]),
                "capex"            : float(row["Capex_Total"]),
                "p_charge"         : p_charge_scs,
                "g_rev_unit"       : g_rev_scs_m,
                "g_var_unit"       : g_var_m,
                "g_om_unit"        : g_om_m,
                # FIX 2: physical ceiling for SCS includes utilisation rate
                "hw_ceil_physical" : n_stalls_p * p_stall_p * H_op_plot * util_scs,
            })

    n_units = len(units)
    fig, axes = plt.subplots(
        n_units, 3,
        figsize=(16, max(4.5 * n_units, 6)),
        squeeze=False
    )
    fig.suptitle(
        "Per-Unit Annual Financial Summary — Year 1 / Mid / End\n"
        "(instantaneous annual $, not discounted; energy grows at g_d each year)",
        fontweight="bold", fontsize=13, y=1.01
    )

    bar_labels = ["Revenue", "Opex\n(excl. capex)", "Op. Income\n(pre-capex)"]

    for row_idx, unit in enumerate(units):
        tech    = unit["tech"]
        c_main  = C_DWC if tech == "DWC" else C_SCS
        c_rev   = C_REV
        c_opex  = C_OPEX
        c_prof  = C_PROFIT
        cap_ann = alpha * unit["capex"]

        # Pre-compute axis range across all years
        y_max, y_min = 0.0, 0.0
        for yr in years:
            fac      = (1 + g_d) ** (yr - 1)
            # FIX 2: cap at physical ceiling
            e_y      = min(unit["e_del_base"] * fac, unit["hw_ceil_physical"])
            rev      = unit["p_charge"] * e_y
            var_om   = unit["lcoe"] * e_y
            opex     = unit["fix_om"] + var_om
            profit   = rev - opex
            y_max    = max(y_max, rev, opex, profit, cap_ann)
            y_min    = min(y_min, profit)

        y_range = y_max - y_min
        y_lo = y_min - y_range * 0.12 if y_min < 0 else 0.0
        y_hi = y_max + y_range * 0.20

        for col_idx, (yr, yr_label) in enumerate(zip(years, yr_labels)):
            ax = axes[row_idx, col_idx]

            fac_demand = (1 + g_d)                ** (yr - 1)
            fac_price  = (1 + unit["g_rev_unit"]) ** (yr - 1)
            fac_lcoe   = (1 + unit["g_var_unit"]) ** (yr - 1)
            fac_om     = (1 + unit["g_om_unit"])  ** (yr - 1)

            # Energy volume grows with demand; price per kWh grows separately
            e_y    = min(unit["e_del_base"] * fac_demand, unit["hw_ceil_physical"])
            rev    = unit["p_charge"] * fac_price * e_y
            var_om = unit["lcoe"]    * fac_lcoe   * e_y
            opex   = unit["fix_om"]  * fac_om + var_om
            profit = rev - opex

            vals   = [rev, opex, profit]
            colors = [c_rev, c_opex, c_prof if profit >= 0 else C_CAPEX]

            x_pos = np.arange(3)
            bars  = ax.bar(x_pos, vals, color=colors, width=0.55,
                           alpha=0.85, edgecolor="white", linewidth=0.6)

            for bar, v in zip(bars, vals):
                sign = 1 if v >= 0 else -1
                va   = "bottom" if v >= 0 else "top"
                offset = y_range * 0.03 * sign
                fmt = f"${v/1e3:.1f}K" if abs(v) >= 1000 else f"${v:.0f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + offset,
                    fmt, ha="center", va=va,
                    fontsize=9, fontweight="bold",
                    color=colors[list(vals).index(v)]
                )

            ax.axhline(cap_ann, color=C_CAPEX, lw=1.4, ls="--", alpha=0.75,
                       label=f"Ann. capex ${cap_ann/1e3:.1f}K/yr")
            ax.axhline(0, color="black", lw=0.7)

            ax.text(0.97, 0.97,
                    f"E_del: {e_y/1e3:.0f} MWh\n×{fac_demand:.2f} base",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, color="#555555",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                              edgecolor="#cccccc", alpha=0.9))

            ax.set_xticks(x_pos)
            ax.set_xticklabels(bar_labels, fontsize=10)
            ax.set_ylim(y_lo, y_hi)
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda v, _: f"${v/1e3:.0f}K" if abs(v) >= 1000 else f"${v:.0f}"
                )
            )
            ax.tick_params(axis="x", length=0)

            if row_idx == 0:
                ax.set_title(yr_label, fontweight="bold", fontsize=11)

            if col_idx == 0:
                ax.set_ylabel(unit["label"], fontweight="bold",
                              fontsize=10, color=c_main)

            if row_idx == 0 and col_idx == 2:
                ax.legend(fontsize=8, frameon=False, loc="upper left")

            ax.yaxis.grid(True, alpha=0.20, ls="--", zorder=0)

    plt.tight_layout()
    fname = "financial_summary_multiyear.png"
    if title_suffix:
        fname = f"financial_summary_multiyear_{title_suffix.replace(' ', '_')}.png"
    _savefig(fig, os.path.join(output_dir, fname))


# ══════════════════════════════════════════════════════════════════════════════
# MASTER SINGLE-RUN WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_plots(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n── Single-run plots → {output_dir}/")
    plot_deployment_map(results, output_dir, show_existing=False)
    # plot_deployment_map(results, output_dir, show_existing=True)
    # plot_traffic_split(results, output_dir)
    plot_total_units(results, output_dir)
    # plot_capex_revenue_profit(results, output_dir)
    # plot_breakeven(results, output_dir)
    # plot_utilization(results, output_dir)
    # plot_financial_summary_multiyear(results, output_dir)
    # plot_heatmaps(results, output_dir)
    # generate_analytics_plots(results, output_dir)  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    print(f"  Requested single-run plots complete.")


# ══════════════════════════════════════════════════════════════════════════════
# SENSITIVITY PLOTS — all unchanged from Main_plots_new.py
# ══════════════════════════════════════════════════════════════════════════════

def generate_eta_split_plots(results_list, param_values, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n── eta_split sensitivity plots → {output_dir}/")
    if not results_list:
        print("  No results — skipping.")
        return

    summs = [_summary(r) for r in results_list]
    eta_d = [s["eta_dwc"] for s in summs]
    eta_s = [s["eta_scs"] for s in summs]
    xlbls = [f"{d:.2f}\n(η$^S$={s:.2f})" for d, s in zip(eta_d, eta_s)]
    x     = np.arange(len(results_list))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(eta_d, [s["n_dwc"] for s in summs], "o-", color=C_DWC,
            lw=2, ms=7, label="DWC segments")
    ax.plot(eta_d, [s["n_scs"] for s in summs], "s--", color=C_SCS,
            lw=2, ms=7, label="SCS stations")
    ax.set_xlabel(r"$\eta^{DWC}$")
    ax.set_ylabel("Units selected")
    ax.set_title("Infrastructure Selection vs Traffic-Capture Split", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "eta_units_selected.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rev_dwc_M = [s["rev_dwc"] / 1e6 for s in summs]
    rev_scs_M = [s["rev_scs"] / 1e6 for s in summs]
    rev_tot_M = [a + b for a, b in zip(rev_dwc_M, rev_scs_M)]
    bw = 0.35
    ax.bar(x - bw / 2, rev_dwc_M, bw, color=C_DWC, alpha=0.80, label="DWC")
    ax.bar(x + bw / 2, rev_scs_M, bw, color=C_SCS, alpha=0.80, label="SCS")
    ax.plot(x, rev_tot_M, "o-", color=C_REV, lw=2, ms=6, label="Total", zorder=5)
    ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_xlabel(r"$\eta^{DWC}$  ($\eta^{SCS}$)")
    ax.set_ylabel("Annual revenue  ($M / yr, base-year)")
    ax.set_title("Revenue vs Traffic-Capture Split", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "eta_revenue.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    alpha0   = results_list[0]["alpha"]
    prof_dwc = [(s["rev_dwc"] - s["opex_dwc"] - s["cap_dwc"] * alpha0) / 1e6 for s in summs]
    prof_scs = [(s["rev_scs"] - s["opex_scs"] - s["cap_scs"] * alpha0) / 1e6 for s in summs]
    prof_tot = [a + b for a, b in zip(prof_dwc, prof_scs)]
    ax.bar(x - bw / 2, prof_dwc, bw, color=C_DWC, alpha=0.80, label="DWC")
    ax.bar(x + bw / 2, prof_scs, bw, color=C_SCS, alpha=0.80, label="SCS")
    ax.plot(x, prof_tot, "o-", color=C_PROFIT, lw=2, ms=6, label="Total", zorder=5)
    ax.axhline(0, color="black", lw=0.8, ls=":")
    for xi, nd, ns, pt in zip(x,
                               [s["n_dwc"] for s in summs],
                               [s["n_scs"] for s in summs],
                               prof_tot):
        ax.text(xi, pt + abs(max(prof_tot) - min(prof_tot)) * 0.03,
                f"D:{nd} S:{ns}", ha="center", fontsize=8, color="dimgray")
    ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=9)
    ax.set_xlabel(r"$\eta^{DWC}$  ($\eta^{SCS}$)")
    ax.set_ylabel("Annual profit  ($M / yr, base-year)")
    ax.set_title("Profit vs Traffic-Capture Split", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "eta_profit.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    proj_life = results_list[0]["project_life"]
    _cap      = proj_life * 1.1
    pbs_d = [s["pb_disc"]   for s in summs]
    pbs_s = [s["pb_simple"] for s in summs]
    ax.plot(eta_d, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_d],
            "o-",  color=C_CAPEX, lw=2, ms=7, label="Discounted payback")
    ax.plot(eta_d, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_s],
            "s--", color=C_OPEX,  lw=2, ms=7, label="Simple payback")
    ax.axhline(proj_life, color="grey", lw=1.0, ls=":", label=f"Project life {proj_life} yr")
    ax.fill_between(eta_d, proj_life, _cap, alpha=0.07, color=C_CAPEX)
    ax.set_xlabel(r"$\eta^{DWC}$")
    ax.set_ylabel("Payback year")
    ax.set_ylim(0, _cap * 1.05)
    ax.set_title("Breakeven vs Traffic-Capture Efficiency", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "eta_breakeven.png"))

    n_sc = len(results_list)
    ncols_pie = min(n_sc, 4)
    nrows_pie = (n_sc + ncols_pie - 1) // ncols_pie
    fig_pie, axes_pie = plt.subplots(nrows_pie, ncols_pie,
                                     figsize=(5 * ncols_pie, 5 * nrows_pie))
    axes_pie = np.array(axes_pie).flatten()
    for idx, (res, ed, es) in enumerate(zip(results_list, eta_d, eta_s)):
        ax_p = axes_pie[idx]
        nd   = int(res["meta"]["n_dwc_selected"])
        ns   = int(res["meta"]["n_scs_selected"])
        sv, sc_ = [], []
        if nd > 0: sv.append(nd); sc_.append(C_DWC)
        if ns > 0: sv.append(ns); sc_.append(C_SCS)
        if sv:
            ax_p.pie(sv, colors=sc_, startangle=140,
                     autopct=lambda p: f"{round(p*sum(sv)/100)}",
                     pctdistance=0.5,
                     textprops={"fontweight": "bold", "color": "white", "fontsize": 14})
        ax_p.set_title(f"η$^D$={ed:.2f} η$^S$={es:.2f}\n"
                       f"DWC={nd} SCS={ns}", fontsize=11, fontweight="bold")
    for idx in range(n_sc, len(axes_pie)):
        axes_pie[idx].set_visible(False)
    fig_pie.suptitle("Units Selected — η Split Scenarios", fontweight="bold")
    plt.tight_layout()
    _savefig(fig_pie, os.path.join(output_dir, "eta_units_pie.png"))

    for res, ed, es in zip(results_list, eta_d, eta_s):
        suffix = f"etaDWC{ed:.2f}_etaSCS{es:.2f}"
        plot_deployment_map(res, output_dir, show_existing=False, title_suffix=suffix)

    print(f"  eta_split plots complete.")


def generate_ev_growth_plots(results_list, param_values, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n── ev_growth sensitivity plots → {output_dir}/")
    if not results_list:
        return

    summs  = [_summary(r) for r in results_list]
    g_evs  = param_values
    scenario_labels = [f"g_ev = {v:.0%}" for v in g_evs]
    meta0      = results_list[0]["meta"]
    rho_ev     = float(meta0.get("rho_ev", 0.02))
    proj_life  = results_list[0]["project_life"]
    colors3    = [C_DWC, C_SCS, C_CAPEX, C_OPEX, C_PROFIT]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    years_t = np.arange(0, proj_life + 1)
    for i, (g_ev, lbl) in enumerate(zip(g_evs, scenario_labels)):
        ev_share = rho_ev * (1 + g_ev)**years_t * 100
        ax.plot(years_t, ev_share, color=colors3[i % len(colors3)], lw=2, label=lbl)
    ax.set_xlabel("Year")
    ax.set_ylabel("EV share of traffic  (%)")
    ax.set_title("EV Penetration Trajectories", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "ev_penetration_curves.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(g_evs, [s["n_dwc"] for s in summs], "o-", color=C_DWC,
            lw=2, ms=7, label="DWC segments")
    ax.plot(g_evs, [s["n_scs"] for s in summs], "s--", color=C_SCS,
            lw=2, ms=7, label="SCS stations")
    ax.set_xlabel(r"EV growth rate  $g_{ev}$")
    ax.set_ylabel("Units selected")
    ax.set_title("Infrastructure Selection vs EV Growth Scenario", fontweight="bold")
    ax.legend(); ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "ev_units_selected.png"))

    fig, axes_ = plt.subplots(1, 2, figsize=(13, 5))
    for i, (res, g_ev, lbl) in enumerate(zip(results_list, g_evs, scenario_labels)):
        s_     = summs[i]
        meta_  = res["meta"]
        g_rev_ = float(meta_.get("g_rev", 0.0))
        yrs    = np.arange(1, proj_life + 1)
        df_d   = res.get("df_dwc", pd.DataFrame())
        df_s   = res.get("df_scs", pd.DataFrame())

        base_var = 0.0
        if not df_d.empty:
            base_var += float((df_d["LCOE"] * df_d["E_del_DWC"]).sum())
        if not df_s.empty:
            base_var += float((df_s["LCOE"] * df_s["E_del_SCS"]).sum())

        base_fix = 0.0
        if not df_d.empty:
            base_fix += float((df_d["Lambda"] * meta_["c_om_base_dwc"]).sum())
        if not df_s.empty:
            base_fix += float((df_s["Lambda"] * meta_["c_om_base_scs"]).sum())

        base_rev = s_["rev_dwc"] + s_["rev_scs"]
        rev_yr   = base_rev * (1 + g_rev_)**(yrs - 1) / 1e6
        var_yr   = base_var * (1 + g_rev_)**(yrs - 1) / 1e6
        fix_yr   = base_fix / 1e6
        net_yr   = rev_yr - var_yr - fix_yr
        c = colors3[i % len(colors3)]
        axes_[0].plot(yrs, rev_yr, color=c, lw=2, label=lbl)
        axes_[1].plot(yrs, net_yr, color=c, lw=2, label=lbl)

    axes_[0].set_xlabel("Year")
    axes_[0].set_ylabel("Annual revenue  ($M / yr)")
    axes_[0].set_title("(a)  Revenue over project life", fontweight="bold")
    axes_[0].legend(fontsize=9); axes_[0].yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    axes_[1].set_xlabel("Year")
    axes_[1].set_ylabel("Net operating income  ($M / yr)")
    axes_[1].set_title("(b)  Net op. income over project life", fontweight="bold")
    axes_[1].axhline(0, color="black", lw=0.8, ls=":")
    axes_[1].legend(fontsize=9); axes_[1].yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    fig.suptitle("Year-by-Year Revenue & Profit by EV Growth Scenario", fontweight="bold")
    plt.tight_layout()
    _savefig(fig, os.path.join(output_dir, "ev_revenue_profit_timeseries.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _cap  = proj_life * 1.1
    pbs_d = [s["pb_disc"]   for s in summs]
    pbs_s = [s["pb_simple"] for s in summs]
    ax.plot(g_evs, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_d],
            "o-",  color=C_CAPEX, lw=2, ms=7, label="Discounted payback")
    ax.plot(g_evs, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_s],
            "s--", color=C_OPEX,  lw=2, ms=7, label="Simple payback")
    ax.axhline(proj_life, color="grey", lw=1.0, ls=":", label=f"Project life {proj_life} yr")
    ax.fill_between(g_evs, proj_life, _cap, alpha=0.07, color=C_CAPEX)
    ax.set_xlabel(r"EV growth rate  $g_{ev}$")
    ax.set_ylabel("Payback year")
    ax.set_ylim(0, _cap * 1.05)
    ax.set_title("Breakeven vs EV Growth Scenario", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "ev_breakeven.png"))

    for res, g_ev, lbl in zip(results_list, g_evs, scenario_labels):
        suffix = lbl.replace(" ", "_").replace("=", "").replace("%", "pct")
        plot_deployment_map(res, output_dir, show_existing=False, title_suffix=suffix)

    print(f"  ev_growth plots complete.")


def generate_traffic_growth_plots(results_list, param_values, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n── traffic_growth sensitivity plots → {output_dir}/")
    if not results_list:
        return

    summs   = [_summary(r) for r in results_list]
    g_aadts = param_values
    scenario_labels = [f"g_aadt = {v:.1%}" for v in g_aadts]
    proj_life = results_list[0]["project_life"]
    colors3   = [C_DWC, C_SCS, C_CAPEX, C_OPEX, C_PROFIT]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    years_t = np.arange(0, proj_life + 1)
    for i, (g, lbl) in enumerate(zip(g_aadts, scenario_labels)):
        ax.plot(years_t, (1 + g)**years_t,
                color=colors3[i % len(colors3)], lw=2, label=lbl)
    ax.set_xlabel("Year")
    ax.set_ylabel("AADT growth index  (base = 1)")
    ax.set_title("AADT Growth Trajectories", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "traffic_growth_curves.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(g_aadts, [s["n_dwc"] for s in summs], "o-", color=C_DWC,
            lw=2, ms=7, label="DWC segments")
    ax.plot(g_aadts, [s["n_scs"] for s in summs], "s--", color=C_SCS,
            lw=2, ms=7, label="SCS stations")
    ax.set_xlabel(r"AADT growth rate  $g_{AADT}$")
    ax.set_ylabel("Units selected")
    ax.set_title("Infrastructure Selection vs AADT Growth", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "traffic_units_selected.png"))

    fig, axes_ = plt.subplots(1, 2, figsize=(13, 5))
    for i, (res, g_aadt, lbl) in enumerate(zip(results_list, g_aadts, scenario_labels)):
        s_     = summs[i]
        meta_  = res["meta"]
        g_rev_ = float(meta_.get("g_rev", 0.0))
        yrs    = np.arange(1, proj_life + 1)
        df_d   = res.get("df_dwc", pd.DataFrame())
        df_s   = res.get("df_scs", pd.DataFrame())

        base_var = 0.0
        if not df_d.empty:
            base_var += float((df_d["LCOE"] * df_d["E_del_DWC"]).sum())
        if not df_s.empty:
            base_var += float((df_s["LCOE"] * df_s["E_del_SCS"]).sum())

        base_fix = 0.0
        if not df_d.empty:
            base_fix += float((df_d["Lambda"] * meta_["c_om_base_dwc"]).sum())
        if not df_s.empty:
            base_fix += float((df_s["Lambda"] * meta_["c_om_base_scs"]).sum())

        base_rev = s_["rev_dwc"] + s_["rev_scs"]
        rev_yr   = base_rev * (1 + g_rev_)**(yrs - 1) / 1e6
        var_yr   = base_var * (1 + g_rev_)**(yrs - 1) / 1e6
        fix_yr   = base_fix / 1e6
        net_yr   = rev_yr - var_yr - fix_yr
        c = colors3[i % len(colors3)]
        axes_[0].plot(yrs, rev_yr, color=c, lw=2, label=lbl)
        axes_[1].plot(yrs, net_yr, color=c, lw=2, label=lbl)

    axes_[0].set_xlabel("Year")
    axes_[0].set_ylabel("Annual revenue  ($M / yr)")
    axes_[0].set_title("(a)  Revenue over project life", fontweight="bold")
    axes_[0].legend(fontsize=9); axes_[0].yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    axes_[1].set_xlabel("Year")
    axes_[1].set_ylabel("Net operating income  ($M / yr)")
    axes_[1].set_title("(b)  Net operating income", fontweight="bold")
    axes_[1].axhline(0, color="black", lw=0.8, ls=":")
    axes_[1].legend(fontsize=9); axes_[1].yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    fig.suptitle("Year-by-Year Revenue & Profit by AADT Growth Scenario", fontweight="bold")
    plt.tight_layout()
    _savefig(fig, os.path.join(output_dir, "traffic_revenue_profit_timeseries.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _cap  = proj_life * 1.1
    pbs_d = [s["pb_disc"]   for s in summs]
    pbs_s = [s["pb_simple"] for s in summs]
    ax.plot(g_aadts, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_d],
            "o-",  color=C_CAPEX, lw=2, ms=7, label="Discounted payback")
    ax.plot(g_aadts, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_s],
            "s--", color=C_OPEX,  lw=2, ms=7, label="Simple payback")
    ax.axhline(proj_life, color="grey", lw=1.0, ls=":", label=f"Project life {proj_life} yr")
    ax.fill_between(g_aadts, proj_life, _cap, alpha=0.07, color=C_CAPEX)
    ax.set_xlabel(r"AADT growth rate  $g_{AADT}$")
    ax.set_ylabel("Payback year")
    ax.set_ylim(0, _cap * 1.05)
    ax.set_title("Breakeven vs AADT Growth Scenario", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "traffic_breakeven.png"))

    for res, g, lbl in zip(results_list, g_aadts, scenario_labels):
        suffix = lbl.replace(" ", "_").replace("=", "").replace("%", "pct")
        plot_deployment_map(res, output_dir, show_existing=False, title_suffix=suffix)

    print(f"  traffic_growth plots complete.")


def generate_dwc_cost_plots(results_list, param_values, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n── dwc_cost sensitivity plots → {output_dir}/")
    if not results_list:
        return

    summs     = [_summary(r) for r in results_list]
    cost_vals = param_values
    cost_M    = [cv / 1e6 for cv in cost_vals]
    proj_life = results_list[0]["project_life"]
    alpha0    = results_list[0]["alpha"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    c_dwc_M = [float(r["params"]["C_DWC_base"]) / 1e6 for r in results_list]
    ax.plot(cost_M, c_dwc_M, "o-", color=C_CAPEX, lw=2, ms=7)
    ax.set_xlabel("DWC base cost  ($M / 5-mile segment)")
    ax.set_ylabel(r"$C^{DWC}_{base}$  ($M)")
    ax.set_title("DWC Segment Cost vs Sweep Level", fontweight="bold")
    ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "dwc_cost_per_segment.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cost_M, [s["n_dwc"] for s in summs], "o-", color=C_DWC,
            lw=2, ms=7, label="DWC segments")
    ax.plot(cost_M, [s["n_scs"] for s in summs], "s--", color=C_SCS,
            lw=2, ms=7, label="SCS stations")
    ax.set_xlabel("DWC base cost  ($M / 5-mi)")
    ax.set_ylabel("Units selected")
    ax.set_title("Infrastructure Selection vs DWC Capital Cost", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "dwc_cost_units.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    rev_tot_M = [(s["rev_dwc"] + s["rev_scs"]) / 1e6 for s in summs]
    rev_dwc_M = [s["rev_dwc"] / 1e6 for s in summs]
    rev_scs_M = [s["rev_scs"] / 1e6 for s in summs]
    prof_dwc  = [(s["rev_dwc"] - s["opex_dwc"] - s["cap_dwc"] * alpha0) / 1e6 for s in summs]
    prof_scs  = [(s["rev_scs"] - s["opex_scs"] - s["cap_scs"] * alpha0) / 1e6 for s in summs]
    prof_tot  = [a + b for a, b in zip(prof_dwc, prof_scs)]

    ax.plot(cost_M, rev_tot_M,  "o-",  color=C_REV,    lw=2, ms=7, label="Total revenue")
    ax.plot(cost_M, rev_dwc_M,  "o--", color=C_DWC,    lw=1.5, alpha=0.7, label="DWC revenue")
    ax.plot(cost_M, rev_scs_M,  "o--", color=C_SCS,    lw=1.5, alpha=0.7, label="SCS revenue")
    ax.plot(cost_M, prof_tot,   "s-",  color=C_PROFIT,  lw=2, ms=7, label="Total profit")
    ax.plot(cost_M, prof_dwc,   "s--", color=C_DWC,    lw=1.0, alpha=0.5, label="DWC profit")
    ax.plot(cost_M, prof_scs,   "s--", color=C_SCS,    lw=1.0, alpha=0.5, label="SCS profit")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    for xv, nd, ns, pt in zip(cost_M,
                               [s["n_dwc"] for s in summs],
                               [s["n_scs"] for s in summs],
                               prof_tot):
        ax.text(xv, pt, f" D:{nd}\n S:{ns}", fontsize=8, color="dimgray", va="center")
    ax.set_xlabel("DWC base cost  ($M / 5-mi)")
    ax.set_ylabel("$M / yr  (base-year)")
    ax.set_title("Revenue & Profit vs DWC Capital Cost", fontweight="bold")
    ax.legend(fontsize=9, ncol=2); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "dwc_cost_revenue_profit.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _cap  = proj_life * 1.1
    pbs_d = [s["pb_disc"]   for s in summs]
    pbs_s = [s["pb_simple"] for s in summs]
    ax.plot(cost_M, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_d],
            "o-",  color=C_CAPEX, lw=2, ms=7, label="Discounted payback")
    ax.plot(cost_M, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_s],
            "s--", color=C_OPEX,  lw=2, ms=7, label="Simple payback")
    ax.axhline(proj_life, color="grey", lw=1.0, ls=":", label=f"Project life {proj_life} yr")
    ax.fill_between(cost_M, proj_life, _cap, alpha=0.07, color=C_CAPEX)
    ax.set_xlabel("DWC base cost  ($M / 5-mi)")
    ax.set_ylabel("Payback year")
    ax.set_ylim(0, _cap * 1.05)
    ax.set_title("Breakeven vs DWC Capital Cost", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "dwc_cost_breakeven.png"))

    for res, cv_m in zip(results_list, cost_M):
        suffix = f"cost_{cv_m:.1f}M"
        plot_deployment_map(res, output_dir, show_existing=False, title_suffix=suffix)

    print(f"  dwc_cost plots complete.")


def generate_custom_plots(results_list, param_name, param_values, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n── custom sensitivity plots → {output_dir}/")
    if not results_list:
        return

    summs     = [_summary(r) for r in results_list]
    proj_life = results_list[0]["project_life"]
    alpha0    = results_list[0]["alpha"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(param_values, [s["n_dwc"] for s in summs], "o-", color=C_DWC,
            lw=2, ms=7, label="DWC segments")
    ax.plot(param_values, [s["n_scs"] for s in summs], "s--", color=C_SCS,
            lw=2, ms=7, label="SCS stations")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Units selected")
    ax.set_title(f"Infrastructure Selection vs {param_name}", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "custom_units.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    rev_M  = [(s["rev_dwc"] + s["rev_scs"]) / 1e6 for s in summs]
    prof_M = [(s["rev_dwc"] + s["rev_scs"] - s["opex_dwc"] - s["opex_scs"]
               - (s["cap_dwc"] + s["cap_scs"]) * alpha0) / 1e6
              for s in summs]
    ax.plot(param_values, rev_M,  "o-",  color=C_REV,    lw=2, ms=7, label="Revenue")
    ax.plot(param_values, prof_M, "s--", color=C_PROFIT, lw=2, ms=7, label="Profit")
    ax.axhline(0, color="black", lw=0.8, ls=":")
    ax.set_xlabel(param_name)
    ax.set_ylabel("$M / yr  (base-year)")
    ax.set_title(f"Revenue & Profit vs {param_name}", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "custom_revenue_profit.png"))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    _cap  = proj_life * 1.1
    pbs_d = [s["pb_disc"]   for s in summs]
    pbs_s = [s["pb_simple"] for s in summs]
    ax.plot(param_values, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_d],
            "o-",  color=C_CAPEX, lw=2, ms=7, label="Discounted payback")
    ax.plot(param_values, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_s],
            "s--", color=C_OPEX,  lw=2, ms=7, label="Simple payback")
    ax.axhline(proj_life, color="grey", lw=1.0, ls=":")
    ax.fill_between(param_values, proj_life, _cap, alpha=0.07, color=C_CAPEX)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Payback year")
    ax.set_ylim(0, _cap * 1.05)
    ax.set_title(f"Breakeven vs {param_name}", fontweight="bold")
    ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
    _savefig(fig, os.path.join(output_dir, "custom_breakeven.png"))

    safe_pname = param_name.replace(" ", "_")[:20]
    for res, pv in zip(results_list, param_values):
        suffix = f"{safe_pname}_{pv}"
        plot_deployment_map(res, output_dir, show_existing=False, title_suffix=suffix)

    print(f"  custom plots complete.")


# ── Master sensitivity dispatcher ─────────────────────────────────────────────

def generate_sensitivity_plots(results_list, param_name, param_values, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not results_list:
        print("  generate_sensitivity_plots: no results — skipping.")
        return

    pn = param_name.lower()
    if "eta_dwc" in pn or "traffic-capture" in pn or "eta_split" in pn:
        generate_eta_split_plots(results_list, param_values, output_dir)
    elif "ev penetration" in pn or "g_ev" in pn or "ev_growth" in pn:
        generate_ev_growth_plots(results_list, param_values, output_dir)
    elif "aadt" in pn or "g_aadt" in pn or "traffic_growth" in pn:
        generate_traffic_growth_plots(results_list, param_values, output_dir)
    elif ("dwc" in pn and "cost" in pn) or "capital cost" in pn:
        generate_dwc_cost_plots(results_list, param_values, output_dir)
    else:
        generate_custom_plots(results_list, param_name, param_values, output_dir)

    rows = []
    for r, pv in zip(results_list, param_values):
        s = _summary(r)
        s["param_value"] = pv
        rows.append(s)
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, "SA_summary.csv"), index=False
    )
    print(f"  SA_summary.csv saved.")


# ── Compatibility shims ────────────────────────────────────────────────────────

def plot_sensitivity_eta_split(results_list, output_dir):
    eta_d_vals = [float(r["meta"].get("eta_dwc", 0)) for r in results_list]
    generate_eta_split_plots(results_list, eta_d_vals, output_dir)


def _compute_run_summary(results):
    return _summary(results)


# ══════════════════════════════════════════════════════════════════════════════
# CLI — mirrors the original plot_analytics.py CLI for standalone use
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified plots from optimizer CSV outputs.")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--out_dir",     default=None)
    args = parser.parse_args()

    rd = args.results_dir
    if rd is None:
        cands = sorted(glob.glob("Results_*/"), key=os.path.getmtime, reverse=True)
        if not cands:
            print("ERROR: no Results_*/ directory found."); sys.exit(1)
        rd = cands[0].rstrip("/")
        print(f"Auto-discovered: {rd}")

    _run_analytics_plots(rd, args.out_dir)

# If running this code independently of Main.py:
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Generate plots from saved optimizer results.")
#     parser.add_argument("--results_dir", default=None)
#     parser.add_argument("--out_dir",     default=None)
#     parser.add_argument("--analytics_only", action="store_true", 
#                         help="Generate only analytics plots (legacy)")
#     args = parser.parse_args()

#     rd = args.results_dir
#     if rd is None:
#         cands = sorted(glob.glob("Results_*/"), key=os.path.getmtime, reverse=True)
#         if not cands:
#             print("ERROR: no Results_*/ directory found."); sys.exit(1)
#         rd = cands[0].rstrip("/")
#         print(f"Auto-discovered: {rd}")

#     if args.analytics_only:
#         _run_analytics_plots(rd, args.out_dir)
#     else:
#         results = load_results_from_saved_dir(rd)
#         if args.out_dir is None:
#             args.out_dir = f"Plots_{results['run_tag']}"
#         generate_all_plots(results, args.out_dir)





# #!/usr/bin/env python3
# # ═══════════════════════════════════════════════════════════════════════════════
# # Main_plots.py
# # Unified publication-quality plots for Joint DWC + SCS Deployment.
# # Updated visual style: clean Nature-style typography, muted colors, black bar outlines,
# # light y-grid, white exports. Data inputs, plot logic, wrapper names, and filenames are unchanged.
# #
# # Merges plot_analytics.py and Main_plots_new.py into a single file.
# # All plot logic is unchanged from the originals except for the two targeted
# # fixes required by the C6b* / C6e* hardware-ceiling update:
# #
# # Everything else — plot types, layouts, colour palettes, axis labels,
# # sensitivity dispatching, file names — is left exactly as it was.
# # ═══════════════════════════════════════════════════════════════════════════════

# import os
# import sys
# import glob
# import argparse
# import warnings
# import numpy as np
# import pandas as pd
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import matplotlib.ticker as mticker
# import matplotlib.patches as mpatches
# import matplotlib.patches as patches
# from matplotlib.lines import Line2D

# warnings.filterwarnings("ignore", category=FutureWarning)
# warnings.filterwarnings("ignore")

# try:
#     import geopandas as gpd
#     _HAS_GPD = True
# except ImportError:
#     _HAS_GPD = False

# # ══════════════════════════════════════════════════════════════════════════════
# # SHARED CONSTANTS
# # ══════════════════════════════════════════════════════════════════════════════
# MW_TO_KW = 1_000.0

# # Hourly traffic weights — must match the optimizer exactly
# _HOURS       = np.arange(24)
# _TRAFFIC_RAW = (
#     0.4 * np.exp(-0.5 * ((_HOURS - 8)  / 2)**2) +
#     0.6 * np.exp(-0.5 * ((_HOURS - 17) / 3)**2) + 0.05
# )
# TRAFFIC_WEIGHTS          = _TRAFFIC_RAW / _TRAFFIC_RAW.sum()
# TRAFFIC_PEAK_HOUR_WEIGHT = float(np.max(TRAFFIC_WEIGHTS))

# # ── Publication palette and plot style ────────────────────────────────────────
# # The numerical logic and output file names are unchanged.  This block only
# # standardizes typography, colors, line weights, grid treatment, and export style
# # so that all plots have a consistent manuscript-quality appearance.

# # Core technology colors: muted, high-contrast, colorblind-tolerant.
# C_DWC_CAP  = "#1F4E79"   # deep blue
# C_DWC_REV  = "#8CB6D9"   # light blue
# C_SCS_CAP  = "#7A4A20"   # muted brown/orange
# C_SCS_REV  = "#E8A35A"   # light orange
# C_REV_LINE = "#111111"   # black revenue/reference lines

# C_DWC    = "#1F4E79"     # selected DWC / DWC bars
# C_SCS    = "#6AA84F"     # selected SCS / SCS bars
# C_EXIST  = "#B46A3C"     # existing SCS
# C_SUB    = "#3A3A3A"     # substations
# C_REV    = "#2F6F4E"     # revenue
# C_CAPEX  = "#C9463D"     # capital / negative values
# C_OPEX   = "#E08E45"     # opex
# C_PROFIT = "#3E75A6"     # profit
# C_GREY   = "#B8BFC7"
# C_GOLD   = "#DDAA33"
# C_CYAN   = "#2DA8C7"

# PAPER_EDGE   = "#111111"
# PAPER_GRID   = "#D9DEE4"
# PAPER_TEXT   = "#111111"
# PAPER_MUTED  = "#5F666D"
# PAPER_BG     = "#FFFFFF"

# _MARKER_DWC   = "o"
# _MARKER_SCS   = "^"
# _MARKER_EXIST = "D"
# _MARKER_SUB   = "."

# # ── Typography and axes: compact, clear, manuscript-oriented ─────────────────
# _PAPER_RCPARAMS = {
#     "font.family"       : "sans-serif",
#     "font.sans-serif"   : ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"],
#     "font.size"         : 9,
#     "axes.labelsize"    : 9.5,
#     "axes.titlesize"    : 10,
#     "axes.titleweight"  : "bold",
#     "axes.labelcolor"   : PAPER_TEXT,
#     "axes.edgecolor"    : PAPER_EDGE,
#     "axes.linewidth"    : 0.8,
#     "axes.spines.top"   : False,
#     "axes.spines.right" : False,
#     "xtick.labelsize"   : 8.5,
#     "ytick.labelsize"   : 8.5,
#     "xtick.major.width" : 0.8,
#     "ytick.major.width" : 0.8,
#     "xtick.major.size"  : 3.5,
#     "ytick.major.size"  : 3.5,
#     "xtick.color"       : PAPER_TEXT,
#     "ytick.color"       : PAPER_TEXT,
#     "legend.fontsize"   : 8.5,
#     "legend.frameon"    : False,
#     "figure.dpi"        : 300,
#     "savefig.dpi"       : 300,
#     "savefig.bbox"      : "tight",
#     "savefig.facecolor" : PAPER_BG,
#     "figure.facecolor"  : PAPER_BG,
#     "axes.facecolor"    : PAPER_BG,
#     "lines.linewidth"   : 1.8,
#     "lines.markersize"  : 5,
#     "patch.linewidth"   : 0.7,
#     "pdf.fonttype"      : 42,
#     "ps.fonttype"       : 42,
# }

# plt.rcParams.update(_PAPER_RCPARAMS)

# # Analytics plots used to switch to a separate serif style.  Keep the same
# # variable name so all existing functions remain unchanged, but point it to the
# # unified paper style.
# _ANALYTICS_RCPARAMS = dict(_PAPER_RCPARAMS)


# # ══════════════════════════════════════════════════════════════════════════════
# # SHARED HELPERS
# # ══════════════════════════════════════════════════════════════════════════════

# def _apply_publication_style(fig):
#     """Apply final visual cleanup without altering any plotted data."""
#     fig.patch.set_facecolor(PAPER_BG)

#     for ax in fig.get_axes():
#         if not ax.get_visible():
#             continue

#         # Maps and intentionally axis-free panels are left axis-free.
#         if not ax.axison:
#             ax.set_facecolor(PAPER_BG)
#             continue

#         ax.set_facecolor(PAPER_BG)
#         ax.set_axisbelow(True)

#         # Clean journal-style axes.  Keep right spine for twinx axes.
#         for side in ("left", "bottom"):
#             ax.spines[side].set_visible(True)
#             ax.spines[side].set_color(PAPER_EDGE)
#             ax.spines[side].set_linewidth(0.8)
#         ax.spines["top"].set_visible(False)
#         if ax.yaxis.get_label_position() == "right":
#             ax.spines["right"].set_visible(True)
#             ax.spines["right"].set_color(PAPER_EDGE)
#             ax.spines["right"].set_linewidth(0.8)
#         else:
#             ax.spines["right"].set_visible(False)

#         ax.tick_params(axis="both", which="major", labelsize=8.5,
#                        width=0.8, length=3.5, color=PAPER_EDGE,
#                        labelcolor=PAPER_TEXT, direction="out")
#         ax.tick_params(axis="both", which="minor", width=0.6,
#                        length=2.0, color=PAPER_EDGE, direction="out")

#         # Light y-grid only.  This preserves existing axes while reducing clutter.
#         ax.grid(False)
#         ax.yaxis.grid(True, color=PAPER_GRID, linewidth=0.55, linestyle="-", alpha=0.65)

#         # Make bars resemble the reference examples: clean fills with black edges.
#         for p in ax.patches:
#             if isinstance(p, patches.Rectangle) and p is not ax.patch:
#                 # Avoid turning background rectangles into data marks.
#                 if p.get_width() != 0 and p.get_height() != 0:
#                     p.set_edgecolor(PAPER_EDGE)
#                     p.set_linewidth(0.7)

#         # Clean legends.  If a function explicitly uses frameon=False, keep it;
#         # otherwise use a white frame with a faint border.
#         leg = ax.get_legend()
#         if leg is not None:
#             leg.get_frame().set_facecolor(PAPER_BG)
#             leg.get_frame().set_edgecolor("#D0D4D8")
#             leg.get_frame().set_linewidth(0.6)
#             for t in leg.get_texts():
#                 t.set_color(PAPER_TEXT)
#                 t.set_fontsize(8.5)

#         ax.title.set_color(PAPER_TEXT)
#         ax.xaxis.label.set_color(PAPER_TEXT)
#         ax.yaxis.label.set_color(PAPER_TEXT)

#     return fig


# def _savefig(fig, path, tight=True):
#     os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
#     _apply_publication_style(fig)
#     fig.savefig(path, dpi=300, bbox_inches="tight" if tight else None,
#                 transparent=False, facecolor=PAPER_BG, edgecolor="none")
#     plt.close(fig)
#     print(f"  Saved: {os.path.basename(path)}")


# def _npv_factor(g, rho, T):
#     """Γ(g) — lifetime PV weight for a growing annual stream."""
#     r = (1.0 + g) / (1.0 + rho)
#     if abs(r - 1.0) < 1e-12:
#         return float(T) / (1.0 + rho)
#     return (1.0 / (1.0 + rho)) * (1.0 - r ** T) / (1.0 - r)


# def _label_step(n):
#     if n <= 80:  return 1
#     if n <= 160: return 2
#     if n <= 400: return 5
#     if n <= 800: return 10
#     return max(1, n // 80)


# def _draw_us_background(ax):
#     if _HAS_GPD:
#         try:
#             world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
#             usa   = world[world.name == "United States of America"]
#             usa.plot(ax=ax, color="white", edgecolor="#D0D0D0",
#                      linewidth=0.8, alpha=0.5, zorder=0)
#         except Exception:
#             pass

# # ══════════════════════════════════════════════════════════════════════════════
# # ─────────────────────────────────────────────────────────────────────────────
# #  Making this script independent of Main.py (Also update Main function below)
# # ─────────────────────────────────────────────────────────────────────────────
# # ══════════════════════════════════════════════════════════════════════════════

# def load_results_from_saved_dir(results_dir):
#     """Load results dict from saved Results_* directory CSVs."""
#     meta_df = pd.read_csv(os.path.join(results_dir, 'run_metadata.csv'))
#     meta = meta_df.iloc[0].to_dict()
    
#     df_dwc = pd.read_csv(os.path.join(results_dir, 'DWC_Selected_Segments.csv'))
#     df_scs = pd.read_csv(os.path.join(results_dir, 'SCS_Selected_Stations.csv'))
#     df_diag = pd.read_csv(os.path.join(results_dir, 'DWC_Selection_Master_Diagnostic.csv'))
    
#     df_exist = (pd.read_csv(os.path.join(results_dir, 'SCS_Existing_Stations.csv'))
#                 if os.path.exists(os.path.join(results_dir, 'SCS_Existing_Stations.csv'))
#                 else pd.DataFrame())
#     df_connections = (pd.read_csv(os.path.join(results_dir, 'DWC_Network_FromTo_Table.csv'))
#                       if os.path.exists(os.path.join(results_dir, 'DWC_Network_FromTo_Table.csv'))
#                       else pd.DataFrame())
#     df_scs_connections = (pd.read_csv(os.path.join(results_dir, 'SCS_Network_FromTo_Table.csv'))
#                           if os.path.exists(os.path.join(results_dir, 'SCS_Network_FromTo_Table.csv'))
#                           else pd.DataFrame())
    
#     # Load coordinates
#     seg_coords = {}
#     if os.path.exists(os.path.join(results_dir, 'All_Segments_Coords.csv')):
#         df_seg = pd.read_csv(os.path.join(results_dir, 'All_Segments_Coords.csv'))
#         seg_coords = {int(r['Segment_ID']): [r['X'], r['Y']] for _, r in df_seg.iterrows()}
    
#     sub_coords = {}
#     if os.path.exists(os.path.join(results_dir, 'All_Substations_Coords.csv')):
#         df_sub = pd.read_csv(os.path.join(results_dir, 'All_Substations_Coords.csv'))
#         sub_coords = {int(r['Sub_ID']): [r['X'], r['Y']] for _, r in df_sub.iterrows()}
    
#     scs_coords_cand = {}
#     if not df_scs.empty and 'MID_X' in df_scs.columns:
#         scs_coords_cand = {int(r['SCS_ID']): [r['MID_X'], r['MID_Y']] for _, r in df_scs.iterrows()}
    
#     scs_coords_exist = {}
#     if os.path.exists(os.path.join(results_dir, 'SCS_Exist_Coords.csv')):
#         df_scs_exist = pd.read_csv(os.path.join(results_dir, 'SCS_Exist_Coords.csv'))
#         scs_coords_exist = {int(r['SCS_ID']): [r['MID_X'], r['MID_Y']] for _, r in df_scs_exist.iterrows()}
    
#     return {
#         'run_tag': meta.get('run_tag', 'loaded'),
#         'meta': meta,
#         'params': {},
#         'df_dwc': df_dwc,
#         'df_scs': df_scs,
#         'df_exist': df_exist,
#         'df_connections': df_connections,
#         'df_scs_connections': df_scs_connections,
#         'df_diag': df_diag,
#         'seg_coords': seg_coords,
#         'sub_coords': sub_coords,
#         'scs_coords_cand': scs_coords_cand,
#         'scs_coords_exist': scs_coords_exist,
#         'selected_substations_dwc': set(df_connections[df_connections['From_Type']=='Substation']['From_ID'].unique()) if not df_connections.empty else set(),
#         'selected_substations_scs': set(df_scs_connections[df_scs_connections['From_Type']=='Substation']['From_ID'].unique()) if not df_scs_connections.empty else set(),
#         'alpha': meta.get('alpha', 0.1),
#         'discount_rate': meta.get('discount_rate', 0.07),
#         'project_life': meta.get('project_life', 30),
#     }
# # ══════════════════════════════════════════════════════════════════════════════
# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION A — plot_analytics functions (load, metrics, plots 1–4 and 1b–3b)
# # ─────────────────────────────────────────────────────────────────────────────
# # ══════════════════════════════════════════════════════════════════════════════

# # ── Data loading (plot_analytics style: reads CSVs from a results directory) ──

# def load_results_from_dir(results_dir):
#     def _r(fname, req=False):
#         p = os.path.join(results_dir, fname)
#         if not os.path.exists(p):
#             if req: raise FileNotFoundError(p)
#             return pd.DataFrame()
#         df = pd.read_csv(p)
#         print(f"  Loaded {fname:45s} ({len(df):>4} rows)")
#         return df

#     print(f"\nLoading from: {results_dir}")
#     df_dwc = _r("DWC_Selected_Segments.csv")
#     df_scs = _r("SCS_Selected_Stations.csv")
#     meta   = _r("run_metadata.csv", req=True).iloc[0].to_dict()
#     return df_dwc, df_scs, meta


# # ── Metric computation ────────────────────────────────────────────────────────

# def _split_opex(ann_opex, lcoe, e_del):
#     var = (lcoe * e_del).clip(lower=0)
#     fix = (ann_opex - var).clip(lower=0)
#     return var, fix


# def compute_dwc_metrics(df, meta):
#     if df.empty: return df
#     df = df.copy()
#     gamma_rev    = float(meta.get("gamma_rev_dwc", 1.0))   # DWC revenue PV factor
#     gamma_var    = float(meta.get("gamma_var",     1.0))   # variable O&M PV factor
#     gamma_fix_om = float(meta.get("gamma_fix_om",  1.0))   # fixed O&M PV factor
#     h_op         = float(meta.get("h_op", 8760.0))
#     rho_ev       = float(meta.get("rho_ev",  0.006))
#     eta_dwc      = float(meta.get("eta_dwc", 0.7))

#     df["installed_kw"] = df["MW_Load"] * MW_TO_KW
#     df["util_rate"]    = df["E_del_DWC"] / (df["installed_kw"] * h_op)

#     df["aadt_ev_dwc"] = df["AADT_COM"] * rho_ev * eta_dwc

#     df["ann_var_om"], df["ann_fix_om"] = _split_opex(
#         df["Ann_Opex"], df["LCOE"], df["E_del_DWC"])

#     df["pv_revenue"] = df["Ann_Revenue"] * gamma_rev
#     df["pv_var_om"]  = df["ann_var_om"]  * gamma_var    # variable O&M uses gamma_var
#     df["pv_fix_om"]  = df["ann_fix_om"]  * gamma_fix_om
#     df["pv_capex"]   = df["Capex_Total"]
#     df["npv"]        = df["pv_revenue"] - df["pv_capex"] - df["pv_fix_om"] - df["pv_var_om"]

#     kw = df["installed_kw"].replace(0, np.nan)
#     df["pv_rev_per_kw"]  = df["pv_revenue"] / kw
#     df["capex_per_kw"]   = df["pv_capex"]   / kw

#     e_pv = (df["E_del_DWC"] * gamma_rev).replace(0, np.nan)
#     df["pv_e_del"]           = e_pv
#     df["capex_per_kwh"]      = df["pv_capex"]  / e_pv
#     df["pv_fix_om_per_kwh"]  = df["pv_fix_om"] / e_pv
#     df["pv_var_om_per_kwh"]  = df["pv_var_om"] / e_pv
#     df["pv_rev_per_kwh"]     = df["pv_revenue"] / e_pv
#     df["total_cost_per_kwh"] = (df["capex_per_kwh"] + df["pv_fix_om_per_kwh"]
#                                 + df["pv_var_om_per_kwh"])

#     df["label"] = "DWC\n" + df["Segment_ID"].astype(str)
#     df["tech"]  = "DWC"
#     return df


# def compute_scs_metrics(df, meta):
#     if df.empty: return df
#     df = df.copy()
#     gamma_rev    = float(meta.get("gamma_rev_scs", 1.0))   # SCS revenue PV factor
#     gamma_var    = float(meta.get("gamma_var",     1.0))   # variable O&M PV factor
#     gamma_fix_om = float(meta.get("gamma_fix_om",  1.0))   # fixed O&M PV factor
#     h_op         = float(meta.get("h_op", 8760.0))
#     n_stalls     = float(meta.get("n_scs_stalls",      6))
#     p_stall_kw   = float(meta.get("p_scs_per_stall_kw", 250.0))
#     rho_ev       = float(meta.get("rho_ev",  0.006))
#     eta_scs      = float(meta.get("eta_scs", 0.3))
#     installed_kw = n_stalls * p_stall_kw

#     df["installed_kw"] = installed_kw
#     df["util_rate"]    = df["E_del_SCS"] / (installed_kw * h_op)

#     if "AADT_COM" not in df.columns or df["AADT_COM"].isna().all():
#         e_req = float(meta.get("e_req_kwh", 150.0))
#         denom = 365.0 * rho_ev * eta_scs * e_req
#         df["AADT_COM"] = (df["E_traff_SCS"] / denom) if denom > 0 else np.nan

#     df["aadt_ev_scs"] = df["AADT_COM"] * rho_ev * eta_scs

#     df["ann_var_om"], df["ann_fix_om"] = _split_opex(
#         df["Ann_Opex"], df["LCOE"], df["E_del_SCS"])

#     df["pv_revenue"] = df["Ann_Revenue"] * gamma_rev
#     df["pv_var_om"]  = df["ann_var_om"]  * gamma_var    # variable O&M uses gamma_var
#     df["pv_fix_om"]  = df["ann_fix_om"]  * gamma_fix_om
#     df["pv_capex"]   = df["Capex_Total"]
#     df["npv"]        = df["pv_revenue"] - df["pv_capex"] - df["pv_fix_om"] - df["pv_var_om"]

#     df["pv_rev_per_kw"]  = df["pv_revenue"] / installed_kw
#     df["capex_per_kw"]   = df["pv_capex"]   / installed_kw

#     e_pv = (df["E_del_SCS"] * gamma_rev).replace(0, np.nan)
#     df["pv_e_del"]           = e_pv
#     df["capex_per_kwh"]      = df["pv_capex"]  / e_pv
#     df["pv_fix_om_per_kwh"]  = df["pv_fix_om"] / e_pv
#     df["pv_var_om_per_kwh"]  = df["pv_var_om"] / e_pv
#     df["pv_rev_per_kwh"]     = df["pv_revenue"] / e_pv
#     df["total_cost_per_kwh"] = (df["capex_per_kwh"] + df["pv_fix_om_per_kwh"]
#                                 + df["pv_var_om_per_kwh"])

#     df["label"] = "SCS\n" + df["SCS_ID"].astype(str)
#     df["tech"]  = "SCS"
#     return df


# # ── Plot 1 (analytics): Lifetime PV Revenue vs Capital Cost per kW ────────────

# def plot_dollar_per_kw(df_dwc, df_scs, meta, out_dir):
#     has_dwc = not df_dwc.empty
#     has_scs = not df_scs.empty
#     if not has_dwc and not has_scs:
#         print("  [Plot 1] nothing to plot"); return

#     matplotlib.rcParams.update(_ANALYTICS_RCPARAMS)

#     T   = int(meta.get("project_life", 30))
#     rho = float(meta.get("discount_rate", 0.07))
#     g_d = float(meta.get("g_d", 0.0))

#     frames = []
#     for df, present in [(df_dwc, has_dwc), (df_scs, has_scs)]:
#         if present:
#             tmp = df[["label","tech","capex_per_kw","pv_rev_per_kw",
#                        "installed_kw","AADT_COM","npv"]].copy()
#             frames.append(tmp)
#     combined = pd.concat(frames, ignore_index=True)
#     n = len(combined)
#     x = np.arange(n)
#     w = 0.38

#     cap_colors = [C_DWC_CAP if t=="DWC" else C_SCS_CAP for t in combined["tech"]]
#     rev_colors = [C_DWC_REV if t=="DWC" else C_SCS_REV for t in combined["tech"]]

#     fig, ax = plt.subplots(figsize=(max(8, n*1.6+2), 6.5))

#     bars_cap = ax.bar(x - w/2, combined["capex_per_kw"].fillna(0), w,
#                       color=cap_colors, edgecolor="white", linewidth=0.8, zorder=3)
#     bars_rev = ax.bar(x + w/2, combined["pv_rev_per_kw"].fillna(0), w,
#                       color=rev_colors, edgecolor="white", linewidth=0.8, zorder=3)

#     ymax = max(combined["capex_per_kw"].fillna(0).max(),
#                combined["pv_rev_per_kw"].fillna(0).max())

#     def _fmt(v):
#         return f"${v/1e3:.1f}k" if v >= 1000 else f"${v:.0f}"

#     for bar, val in zip(bars_cap, combined["capex_per_kw"].fillna(0)):
#         ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+ymax*0.01,
#                 _fmt(val), ha="center", va="bottom", fontsize=8, fontweight="bold")

#     for bar, rev, aadt in zip(bars_rev,
#                                combined["pv_rev_per_kw"].fillna(0),
#                                combined["AADT_COM"].fillna(0)):
#         ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+ymax*0.01,
#                 f"{_fmt(rev)}\nAADT {int(aadt):,}" if aadt>0 else _fmt(rev),
#                 ha="center", va="bottom", fontsize=7.5, linespacing=1.35)

#     ax.set_xticks(x)
#     ax.set_xticklabels(combined["label"], fontsize=9)
#     ax.yaxis.set_major_formatter(
#         mticker.FuncFormatter(lambda v,_: f"${v/1e3:.1f}k" if v>=1000 else f"${v:.0f}"))
#     ax.set_ylabel(f"Lifetime PV  $/kW\n(T={T} yr, ρ={rho:.0%}, g_d={g_d:.1%})")
#     ax.set_title(
#         "Plot 1 — Lifetime PV Revenue vs Capital Cost per Installed kW\n"
#         "Both bars in the same dollars: capital (one-time) vs PV of 30-yr revenue stream",
#         fontsize=11, fontweight="bold", pad=10)

#     legend_els = [
#         mpatches.Patch(color=C_DWC_CAP, label="DWC — Capital ($/kW, one-time)"),
#         mpatches.Patch(color=C_DWC_REV, label="DWC — Lifetime PV Revenue ($/kW)"),
#         mpatches.Patch(color=C_SCS_CAP, label="SCS — Capital ($/kW, one-time)"),
#         mpatches.Patch(color=C_SCS_REV, label="SCS — Lifetime PV Revenue ($/kW)"),
#     ]
#     ax.legend(handles=legend_els, fontsize=8.5, loc="upper right", framealpha=0.9)
#     ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
#     ax.set_ylim(bottom=0, top=ymax*1.25)

#     if has_dwc and has_scs:
#         ax.axvline(len(df_dwc) - 0.5, color="#AAAAAA", linewidth=1.2,
#                    linestyle="--", alpha=0.7)

#     plt.tight_layout()
#     path = os.path.join(out_dir, "plot1_dollar_per_kw.png")
#     _savefig(fig, path)
#     print(f"  Saved: {path}")


# # ── Plot 2 (analytics): Levelised $/kWh cost stack ────────────────────────────

# def plot_dollar_per_kwh(df_dwc, df_scs, meta, out_dir):
#     has_dwc = not df_dwc.empty
#     has_scs = not df_scs.empty
#     if not has_dwc and not has_scs:
#         print("  [Plot 2] nothing to plot"); return

#     matplotlib.rcParams.update(_ANALYTICS_RCPARAMS)

#     T   = int(meta.get("project_life", 30))
#     rho = float(meta.get("discount_rate", 0.07))

#     frames = []
#     for df, present in [(df_dwc, has_dwc), (df_scs, has_scs)]:
#         if present:
#             frames.append(df[["label","tech","capex_per_kwh",
#                                "pv_fix_om_per_kwh","pv_var_om_per_kwh",
#                                "pv_rev_per_kwh","AADT_COM","util_rate"]].copy())
#     combined = pd.concat(frames, ignore_index=True)
#     n = len(combined)
#     x = np.arange(n)
#     w = 0.52

#     cap_vals = combined["capex_per_kwh"].fillna(0).values
#     fix_vals = combined["pv_fix_om_per_kwh"].fillna(0).values
#     var_vals = combined["pv_var_om_per_kwh"].fillna(0).values
#     rev_vals = combined["pv_rev_per_kwh"].fillna(0).values
#     total    = cap_vals + fix_vals + var_vals

#     cap_clr = ["#A8C8E8" if t == "DWC" else "#F5C897" for t in combined["tech"]]
#     fix_clr = ["#5A7A96" if t == "DWC" else "#C07030" for t in combined["tech"]]
#     var_clr = ["#2C2C2C" if t == "DWC" else "#4A1A00" for t in combined["tech"]]

#     fig, ax = plt.subplots(figsize=(max(8, n * 1.8 + 2), 6.5))

#     b1 = ax.bar(x, var_vals, w, color=var_clr, edgecolor="white",
#                 linewidth=0.5, zorder=3)
#     b2 = ax.bar(x, fix_vals, w, bottom=var_vals, color=fix_clr,
#                 edgecolor="white", linewidth=0.5, zorder=3)
#     b3 = ax.bar(x, cap_vals, w, bottom=var_vals+fix_vals, color=cap_clr,
#                 edgecolor="white", linewidth=0.5, zorder=3)

#     for i, tc in enumerate(total):
#         ax.text(x[i], tc + max(total)*0.012, f"${tc:.3f}",
#                 ha="center", va="bottom", fontsize=9, fontweight="bold")

#     for i, (aadt, util, bot, ht) in enumerate(zip(
#             combined["AADT_COM"].fillna(0),
#             combined["util_rate"].fillna(0),
#             var_vals+fix_vals, cap_vals)):
#         if ht > max(total)*0.06:
#             ax.text(x[i], bot+ht/2,
#                     f"AADT {int(aadt):,}\nUtil {util:.2%}",
#                     ha="center", va="center",
#                     fontsize=7.5, color="#333333", linespacing=1.35)

#     for i, rv in enumerate(rev_vals):
#         ax.plot([x[i]-w/2, x[i]+w/2], [rv, rv],
#                 color=C_REV_LINE, linewidth=2.2, zorder=5, linestyle="--")

#     ax.set_xticks(x)
#     ax.set_xticklabels(combined["label"], fontsize=9)
#     ax.yaxis.set_major_formatter(
#         mticker.FuncFormatter(lambda v,_: f"${v:.2f}"))
#     ax.set_ylabel(f"Levelised $/kWh  (T={T} yr, ρ={rho:.0%})")
#     ax.set_title(
#         "Plot 2 — Levelised Cost per kWh vs Retail Charging Price\n"
#         "Stack below dashed line = station recovers all costs per kWh",
#         fontsize=11, fontweight="bold", pad=10)

#     legend_els = [
#         mpatches.Patch(color="#2C2C2C", label="DWC: Energy cost (≈ grid LCOE)"),
#         mpatches.Patch(color="#5A7A96", label="DWC: Fixed O&M per kWh"),
#         mpatches.Patch(color="#A8C8E8", label="DWC: Capital per kWh  ← shrinks with AADT"),
#         mpatches.Patch(color="#4A1A00", label="SCS: Energy cost (≈ grid LCOE)"),
#         mpatches.Patch(color="#C07030", label="SCS: Fixed O&M per kWh"),
#         mpatches.Patch(color="#F5C897", label="SCS: Capital per kWh  ← shrinks with AADT"),
#         plt.Line2D([0],[0], color=C_REV_LINE, linewidth=2, linestyle="--",
#                    label="Retail price charged to drivers ($/kWh)"),
#     ]
#     ax.legend(handles=legend_els, fontsize=8, loc="upper left",
#               bbox_to_anchor=(1.01, 1.0), borderaxespad=0, framealpha=0.9)
#     ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

#     if has_dwc and has_scs:
#         ax.axvline(len(df_dwc)-0.5, color="#AAAAAA", linewidth=1.2,
#                    linestyle="--", alpha=0.7)

#     plt.tight_layout()
#     path = os.path.join(out_dir, "plot2_dollar_per_kwh.png")
#     _savefig(fig, path)
#     print(f"  Saved: {path}")


# # ── Plot 3 (analytics): Traffic & utilisation summary ─────────────────────────

# def plot_traffic_utilisation(df_dwc, df_scs, meta, out_dir):
#     has_dwc = not df_dwc.empty
#     has_scs = not df_scs.empty
#     if not has_dwc and not has_scs:
#         print("  [Plot 3] nothing to plot"); return

#     matplotlib.rcParams.update(_ANALYTICS_RCPARAMS)

#     rho_ev  = float(meta.get("rho_ev",  0.006))
#     eta_dwc = float(meta.get("eta_dwc", 0.7))
#     eta_scs = float(meta.get("eta_scs", 0.3))

#     frames = []
#     if has_dwc:
#         tmp = df_dwc[["label","tech","AADT_COM","aadt_ev_dwc","util_rate"]].copy()
#         tmp = tmp.rename(columns={"aadt_ev_dwc": "aadt_ev"})
#         frames.append(tmp)
#     if has_scs:
#         tmp = df_scs[["label","tech","AADT_COM","aadt_ev_scs","util_rate"]].copy()
#         tmp = tmp.rename(columns={"aadt_ev_scs": "aadt_ev"})
#         frames.append(tmp)
#     combined = pd.concat(frames, ignore_index=True)

#     n  = len(combined)
#     x  = np.arange(n)
#     w  = 0.28

#     fig, ax1 = plt.subplots(figsize=(max(9, n*2+2), 6))
#     ax2 = ax1.twinx()

#     c_aadt = [C_DWC_CAP if t=="DWC" else C_SCS_CAP for t in combined["tech"]]
#     c_ev   = [C_DWC_REV if t=="DWC" else C_SCS_REV for t in combined["tech"]]

#     b1 = ax1.bar(x - w, combined["AADT_COM"].fillna(0), w,
#                  color=c_aadt, edgecolor="white", linewidth=0.7,
#                  zorder=3, label="AADT_COM")
#     b2 = ax1.bar(x,      combined["aadt_ev"].fillna(0),  w,
#                  color=c_ev,   edgecolor="white", linewidth=0.7,
#                  zorder=3, label="AADT × ρ_EV × η")

#     util_pct = combined["util_rate"].fillna(0).values * 100
#     b3 = ax2.bar(x + w, util_pct, w,
#                  color="#888888", edgecolor="white", linewidth=0.7,
#                  alpha=0.75, zorder=3, label="Utilisation rate")

#     for bar, v in zip(b1, combined["AADT_COM"].fillna(0)):
#         if v > 0:
#             ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
#                      f"{int(v):,}", ha="center", va="bottom", fontsize=7.5)
#     for bar, v in zip(b2, combined["aadt_ev"].fillna(0)):
#         if v > 0:
#             ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50,
#                      f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
#     for bar, v in zip(b3, util_pct):
#         ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
#                  f"{v:.2f}%", ha="center", va="bottom", fontsize=7.5)

#     ax1.set_xticks(x)
#     ax1.set_xticklabels(combined["label"], fontsize=9)
#     ax1.set_ylabel("Vehicles per day  (base year)")
#     ax2.set_ylabel("Annual capacity utilisation  (%)")
#     ax2.set_ylim(0, max(util_pct.max()*1.4, 1))

#     ax1.set_title(
#         "Plot 3 — Traffic & Utilisation per Station  (base year)\n"
#         f"ρ_EV={rho_ev:.3f}   η_DWC={eta_dwc:.2f}   η_SCS={eta_scs:.2f}",
#         fontsize=11, fontweight="bold", pad=10)

#     handles = [b1, b2,
#                mpatches.Patch(color="#888888", alpha=0.75,
#                                label="Annual utilisation rate (%) — right axis")]
#     labels  = ["AADT_COM  (all commercial vehicles/day)",
#                "EV vehicles/day using the charger  (AADT × ρ_EV × η)",
#                "Annual utilisation rate (%) — right axis"]
#     ax1.legend(handles=handles, labels=labels, fontsize=8.5,
#                loc="upper right", framealpha=0.9)
#     ax1.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

#     if has_dwc and has_scs:
#         ax1.axvline(len(df_dwc)-0.5, color="#AAAAAA", linewidth=1.2,
#                     linestyle="--", alpha=0.7)

#     plt.tight_layout()
#     path = os.path.join(out_dir, "plot3_traffic_utilisation.png")
#     _savefig(fig, path)
#     print(f"  Saved: {path}")


# # ── Plot 4 (analytics): Utilisation over time ─────────────────────────────────
# # FIX 1 applied here: e_hw_cap uses the PHYSICAL ceiling per technology.

# def _dwc_hourly_kw(row, meta):
#     e_del = float(row.get("E_del_DWC", 0) or 0)
#     if e_del == 0:
#         return np.zeros(24)
#     daily = e_del / 365.0
#     return daily * TRAFFIC_WEIGHTS


# def _scs_hourly_kw(row, meta):
#     e_del  = float(row.get("E_del_SCS", 0) or 0)
#     if e_del == 0:
#         return np.zeros(24)
#     daily  = e_del / 365.0
#     return daily * TRAFFIC_WEIGHTS


# def plot_utilisation_over_time(df_dwc, df_scs, meta, out_dir):
#     has_dwc = not df_dwc.empty
#     has_scs = not df_scs.empty
#     if not has_dwc and not has_scs:
#         print("  [Plot 4] nothing to plot"); return

#     matplotlib.rcParams.update(_ANALYTICS_RCPARAMS)

#     T          = int(  meta.get("project_life", 30))
#     g_d        = float(meta.get("g_d",           0.0))
#     h_op       = float(meta.get("h_op",       8760.0))
#     n_stalls   = float(meta.get("n_scs_stalls",      6))
#     p_stall_kw = float(meta.get("p_scs_per_stall_kw", 250.0))
#     # FIX 1: read utilisation rate so SCS physical ceiling is correct
#     util_rate_scs = float(meta.get("utilization_rate_scs", 1.0))

#     years = np.arange(1, T+1)

#     stations = []
#     if has_dwc:
#         for _, row in df_dwc.iterrows():
#             inst_kw = float(row["installed_kw"])
#             stations.append({
#                 "label"      : row["label"].replace("\n"," "),
#                 "tech"       : "DWC",
#                 "inst_kw"    : inst_kw,
#                 "e_del"      : float(row["E_del_DWC"]),
#                 "e_traff"    : float(row.get("E_traff_DWC", row["E_del_DWC"])),
#                 # FIX 1: DWC physical ceiling = installed_kw × h_op (no util factor)
#                 "e_hw_cap"   : inst_kw * h_op,
#                 "hourly_kwh" : _dwc_hourly_kw(row, meta),
#             })
#     if has_scs:
#         for _, row in df_scs.iterrows():
#             inst_kw = float(row["installed_kw"])
#             stations.append({
#                 "label"      : row["label"].replace("\n"," "),
#                 "tech"       : "SCS",
#                 "inst_kw"    : inst_kw,
#                 "e_del"      : float(row["E_del_SCS"]),
#                 "e_traff"    : float(row.get("E_traff_SCS", row["E_del_SCS"])),
#                 # FIX 1: SCS physical ceiling includes the utilisation rate
#                 "e_hw_cap"   : inst_kw * h_op * util_rate_scs,
#                 "hourly_kwh" : _scs_hourly_kw(row, meta),
#             })

#     n_sta = len(stations)
#     fig, axes = plt.subplots(n_sta, 3,
#                              figsize=(14, 3.8 * n_sta),
#                              squeeze=False)
#     fig.suptitle(
#         "Plot 4 — Utilisation Over Time: 24-hour Profile | Annual Energy | 30-yr Growth",
#         fontsize=12, fontweight="bold", y=1.01)

#     for row_idx, sta in enumerate(stations):
#         tech  = sta["tech"]
#         c     = C_DWC_CAP if tech == "DWC" else C_SCS_CAP
#         c_lt  = C_DWC_REV if tech == "DWC" else C_SCS_REV
#         label = sta["label"]

#         # Panel A: 24-hour profile
#         ax_a = axes[row_idx, 0]
#         hourly = sta["hourly_kwh"]
#         inst   = sta["inst_kw"]

#         ax_a.fill_between(np.arange(24), hourly, alpha=0.7, color=c, step="mid")
#         ax_a.step(np.arange(24), hourly, color=c, linewidth=1.2, where="mid")
#         ax_a.axhline(inst, color="#999999", linewidth=1.2, linestyle="--",
#                      label=f"Installed capacity {inst:,.0f} kW")
#         ax_a.set_xlim(0, 23)
#         ax_a.set_xticks([0,6,12,18,23])
#         ax_a.set_xlabel("Hour of day")
#         ax_a.set_ylabel("kWh delivered (base year)")
#         ax_a.set_title(f"{label}\n24-hour profile (typical day, base year)",
#                        fontsize=9, fontweight="bold")
#         ax_a.legend(fontsize=7.5, loc="upper left")
#         ax_a.grid(linestyle="--", alpha=0.3)

#         # Panel B: Annual energy bar — uses physical ceiling (FIX 1)
#         ax_b = axes[row_idx, 1]
#         e_del   = sta["e_del"]   / 1e3
#         e_traff = sta["e_traff"] / 1e3
#         e_hwcap = sta["e_hw_cap"] / 1e3   # physical ceiling

#         categories = ["Potential\ntraffic demand\n(E_traff)",
#                       "Actual\ndelivered\n(E_del)",
#                       "Physical\nhardware ceiling"]
#         values     = [e_traff, e_del, e_hwcap]
#         colors_b   = [c_lt, c, "#BBBBBB"]
#         bars = ax_b.bar(categories, values, color=colors_b,
#                         edgecolor="white", linewidth=0.7, zorder=3)
#         for bar, v in zip(bars, values):
#             ax_b.text(bar.get_x()+bar.get_width()/2,
#                       bar.get_height() + max(values)*0.02,
#                       f"{v:,.0f} MWh", ha="center", va="bottom", fontsize=8)
#         ax_b.set_ylabel("MWh / year  (base year)")
#         ax_b.set_title(f"{label}\nAnnual energy (base year)", fontsize=9, fontweight="bold")
#         ax_b.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

#         # Panel C: 30-yr cumulative energy
#         ax_c = axes[row_idx, 2]
#         e_del_y  = e_del  * (1 + g_d) ** (years - 1)
#         cumul    = np.cumsum(e_del_y)
#         ax_c.plot(years, cumul, color=c, linewidth=2.0, zorder=3,
#                   label="Cumulative energy delivered")
#         ax_c.fill_between(years, cumul, alpha=0.18, color=c)
#         ax_c.axvline(T, color="#999999", linewidth=1, linestyle="--", alpha=0.6)
#         ax_c.set_xlabel("Project year")
#         ax_c.set_ylabel("Cumulative MWh delivered")
#         total_mwh = cumul[-1]
#         ax_c.set_title(
#             f"{label}\n30-yr cumulative: {total_mwh:,.0f} MWh  (g_d={g_d:.1%}/yr)",
#             fontsize=9, fontweight="bold")
#         ax_c.legend(fontsize=7.5)
#         ax_c.grid(linestyle="--", alpha=0.3, zorder=0)

#     plt.tight_layout()
#     path = os.path.join(out_dir, "plot4_utilisation_over_time.png")
#     _savefig(fig, path)
#     print(f"  Saved: {path}")


# # ── Plot 3b (analytics): Traffic & Utilisation multi-year ─────────────────────

# def plot_traffic_utilisation_multiyear(df_dwc, df_scs, meta, out_dir):
#     has_dwc = not df_dwc.empty
#     has_scs = not df_scs.empty
#     if not has_dwc and not has_scs:
#         print("  [Plot 3b] nothing to plot"); return

#     matplotlib.rcParams.update(_ANALYTICS_RCPARAMS)

#     T       = int(  meta.get("project_life", 30))
#     g_d     = float(meta.get("g_d",    0.0))
#     g_aadt  = float(meta.get("g_aadt", 0.0))
#     rho_ev  = float(meta.get("rho_ev",  0.006))
#     eta_dwc = float(meta.get("eta_dwc", 0.7))
#     eta_scs = float(meta.get("eta_scs", 0.3))

#     mid_yr = T // 2 + 1
#     years  = [1, mid_yr, T]
#     yr_labels = [
#         f"Year 1  (Base)\ng_d growth = ×1.00",
#         f"Year {mid_yr}  (Mid)\ng_d growth = ×{(1+g_d)**(mid_yr-1):.2f}",
#         f"Year {T}  (End)\ng_d growth = ×{(1+g_d)**(T-1):.2f}",
#     ]

#     frames = []
#     if has_dwc:
#         tmp = df_dwc[["label","tech","AADT_COM","aadt_ev_dwc",
#                        "util_rate","installed_kw","E_del_DWC"]].copy()
#         tmp = tmp.rename(columns={"aadt_ev_dwc": "aadt_ev",
#                                    "E_del_DWC":   "e_del_base"})
#         frames.append(tmp)
#     if has_scs:
#         tmp = df_scs[["label","tech","AADT_COM","aadt_ev_scs",
#                        "util_rate","installed_kw","E_del_SCS"]].copy()
#         tmp = tmp.rename(columns={"aadt_ev_scs": "aadt_ev",
#                                    "E_del_SCS":   "e_del_base"})
#         frames.append(tmp)
#     combined = pd.concat(frames, ignore_index=True)
#     n  = len(combined)
#     x  = np.arange(n)
#     w  = 0.28

#     c_aadt = [C_DWC_CAP if t == "DWC" else C_SCS_CAP for t in combined["tech"]]
#     c_ev   = [C_DWC_REV if t == "DWC" else C_SCS_REV for t in combined["tech"]]

#     fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#     fig.suptitle(
#         f"Plot 3b — Traffic & Utilisation: Base / Mid / End Year  "
#         f"(T={T} yr,  g_d={g_d:.1%}/yr,  g_aadt={g_aadt:.1%}/yr)\n"
#         f"ρ_EV={rho_ev:.3f}   η_DWC={eta_dwc:.2f}   η_SCS={eta_scs:.2f}",
#         fontsize=11, fontweight="bold"
#     )

#     max_aadt_all = 0.0
#     max_util_all = 0.0
#     for yr in years:
#         fac_d = (1 + g_d)    ** (yr - 1)
#         fac_a = (1 + g_aadt) ** (yr - 1)
#         max_aadt_all = max(max_aadt_all,
#                            (combined["AADT_COM"] * fac_a).max())
#         h_op_v = float(meta.get("h_op", 8760.0))
#         util_v = ((combined["e_del_base"] * fac_d) /
#                   (combined["installed_kw"] * h_op_v)).clip(upper=1.0).max()
#         max_util_all = max(max_util_all, util_v * 100)

#     for col_idx, (yr, yr_label) in enumerate(zip(years, yr_labels)):
#         ax1 = axes[col_idx]
#         ax2 = ax1.twinx()

#         fac_demand = (1 + g_d)    ** (yr - 1)
#         fac_aadt   = (1 + g_aadt) ** (yr - 1)
#         h_op_v     = float(meta.get("h_op", 8760.0))

#         aadt_yr    = combined["AADT_COM"]   * fac_aadt
#         aadt_ev_yr = combined["aadt_ev"]    * fac_demand

#         util_rate_scs_val = float(meta.get("utilization_rate_scs", 1.0))
#         # Physical hardware ceiling — fixed regardless of year.
#         # e_del_base is the base-year value after C6b*/C6e* tightening,
#         # so year-y projected delivery never exceeds the physical ceiling.
#         hw_ceil_physical = np.where(
#             combined["tech"] == "SCS",
#             combined["installed_kw"].values * h_op_v * util_rate_scs_val,
#             combined["installed_kw"].values * h_op_v
#         )
#         e_del_yr = np.minimum(
#             (combined["e_del_base"] * fac_demand).values,
#             hw_ceil_physical
#         )
#         util_yr = (e_del_yr / (combined["installed_kw"].values * h_op_v)).clip(0, 1.0) * 100

#         b1 = ax1.bar(x - w,   aadt_yr,    w, color=c_aadt, edgecolor="white",
#                      linewidth=0.7, zorder=3)
#         b2 = ax1.bar(x,       aadt_ev_yr, w, color=c_ev,   edgecolor="white",
#                      linewidth=0.7, zorder=3, alpha=0.85)
#         b3 = ax2.bar(x + w,   util_yr,    w, color="#888888", edgecolor="white",
#                      linewidth=0.7, alpha=0.75, zorder=3)

#         for bar, v in zip(b1, aadt_yr):
#             if v > max_aadt_all * 0.02:
#                 ax1.text(bar.get_x() + bar.get_width() / 2,
#                          bar.get_height() + max_aadt_all * 0.01,
#                          f"{int(v):,}", ha="center", va="bottom", fontsize=7)
#         for bar, v in zip(b2, aadt_ev_yr):
#             if v > 0:
#                 ax1.text(bar.get_x() + bar.get_width() / 2,
#                          bar.get_height() + max_aadt_all * 0.01,
#                          f"{v:.1f}", ha="center", va="bottom", fontsize=7)
#         for bar, v in zip(b3, util_yr):
#             ax2.text(bar.get_x() + bar.get_width() / 2,
#                      bar.get_height() + max_util_all * 0.01,
#                      f"{v:.2f}%", ha="center", va="bottom", fontsize=7)

#         ax1.set_xticks(x)
#         ax1.set_xticklabels(combined["label"], fontsize=9)
#         ax1.set_ylim(0, max_aadt_all * 1.25)
#         ax2.set_ylim(0, max_util_all * 1.45)

#         if col_idx == 0:
#             ax1.set_ylabel("Vehicles per day")
#         if col_idx == 2:
#             ax2.set_ylabel("Annual capacity utilisation (%)")
#         ax1.set_title(yr_label, fontsize=9.5, fontweight="bold", pad=6)
#         ax1.grid(axis="y", linestyle="--", alpha=0.3)

#         if has_dwc and has_scs:
#             ax1.axvline(len(df_dwc) - 0.5, color="#AAAAAA",
#                         linewidth=1.2, linestyle="--", alpha=0.7)

#     handles = [
#         mpatches.Patch(color=C_DWC_CAP, label="DWC AADT_COM"),
#         mpatches.Patch(color=C_SCS_CAP, label="SCS AADT_COM"),
#         mpatches.Patch(color=C_DWC_REV, label="DWC EV users/day  (AADT×ρ_EV×η, g_d growth)"),
#         mpatches.Patch(color=C_SCS_REV, label="SCS EV users/day  (AADT×ρ_EV×η, g_d growth)"),
#         mpatches.Patch(color="#888888", alpha=0.75, label="Annual utilisation % — right axis"),
#     ]
#     fig.legend(handles=handles, fontsize=8.5, loc="lower center",
#                ncol=3, bbox_to_anchor=(0.5, -0.10), framealpha=0.9)

#     plt.tight_layout(rect=[0, 0.07, 1, 1])
#     path = os.path.join(out_dir, "plot3b_traffic_utilisation_multiyear.png")
#     _savefig(fig, path)
#     print(f"  Saved: {path}")


# # ── Plot 2b (analytics): Instantaneous $/kWh multi-year ──────────────────────

# def plot_dollar_per_kwh_multiyear(df_dwc, df_scs, meta, out_dir):
#     has_dwc = not df_dwc.empty
#     has_scs = not df_scs.empty
#     if not has_dwc and not has_scs:
#         print("  [Plot 2b] nothing to plot"); return

#     matplotlib.rcParams.update(_ANALYTICS_RCPARAMS)

#     T     = int(  meta.get("project_life", 30))
#     g_d   = float(meta.get("g_d",    0.0))
#     g_aadt = float(meta.get("g_aadt", 0.0))
#     rho   = float(meta.get("discount_rate", 0.07))
#     alpha = float(meta.get("alpha", 0.08))
#     p_charge_dwc = float(meta.get("p_charge_dwc", 0.40))
#     p_charge_scs = float(meta.get("p_charge_scs", 0.40))

#     mid_yr  = T // 2 + 1
#     years   = [1, mid_yr, T]
#     yr_labels = [
#         f"Year 1  (Base)",
#         f"Year {mid_yr}  (Mid)",
#         f"Year {T}  (End)",
#     ]

#     def _make_frame(df, e_del_col, tech_name):
#         out = df[["label","tech","AADT_COM","installed_kw",
#                    e_del_col,"LCOE","pv_capex","ann_fix_om"]].copy()
#         out = out.rename(columns={e_del_col: "e_del_base"})
#         return out

#     frames = []
#     if has_dwc:
#         frames.append(_make_frame(df_dwc, "E_del_DWC", "DWC"))
#     if has_scs:
#         frames.append(_make_frame(df_scs, "E_del_SCS", "SCS"))
#     combined = pd.concat(frames, ignore_index=True)
#     n  = len(combined)
#     x  = np.arange(n)
#     w  = 0.52

#     max_stack = 0.0
#     h_op      = float(meta.get("h_op", 8760.0))
#     util_rate_scs = float(meta.get("utilization_rate_scs", 1.0))
#     # Physical hardware ceiling
#     hw_ceil_physical_2b = np.where(
#         combined["tech"] == "SCS",
#         combined["installed_kw"] * h_op * util_rate_scs,
#         combined["installed_kw"] * h_op
#     )
#     for yr in years:
#         fac = (1 + g_d) ** (yr - 1)
#         e_y_raw = combined["e_del_base"] * fac
#         e_y = pd.Series(np.minimum(e_y_raw.values, hw_ceil_physical_2b)).replace(0, np.nan)
#         cap = alpha * combined["pv_capex"] / e_y
#         fix = combined["ann_fix_om"]       / e_y
#         var = combined["LCOE"]
#         max_stack = max(max_stack, (cap + fix + var).max())

#     cap_clr = ["#A8C8E8" if t == "DWC" else "#F5C897" for t in combined["tech"]]
#     fix_clr = ["#5A7A96" if t == "DWC" else "#C07030" for t in combined["tech"]]
#     var_clr = ["#2C2C2C" if t == "DWC" else "#4A1A00" for t in combined["tech"]]

#     fig, axes = plt.subplots(1, 3, figsize=(18, 6.5), sharey=True)
#     fig.suptitle(
#         f"Plot 2b — Instantaneous $/kWh Cost Stack: Base / Mid / End Year  "
#         f"(T={T} yr,  g_d={g_d:.1%}/yr,  ρ={rho:.0%})\n"
#         f"Capital annualized via CRF α={alpha:.4f}   "
#         f"Stack falls as demand grows → improving annual economics",
#         fontsize=11, fontweight="bold"
#     )

#     for col_idx, (yr, yr_label) in enumerate(zip(years, yr_labels)):
#         ax = axes[col_idx]
#         fac     = (1 + g_d) ** (yr - 1)
#         e_y_raw = combined["e_del_base"] * fac
#         e_y     = pd.Series(np.minimum(e_y_raw.values, hw_ceil_physical_2b)).replace(0, np.nan)

#         cap_vals = (alpha * combined["pv_capex"] / e_y).fillna(0).values
#         fix_vals = (combined["ann_fix_om"]       / e_y).fillna(0).values
#         var_vals = combined["LCOE"].fillna(0).values
#         total    = cap_vals + fix_vals + var_vals

#         rev_vals = np.where(
#             combined["tech"] == "DWC",
#             p_charge_dwc, p_charge_scs
#         )

#         h_op_v   = float(meta.get("h_op", 8760.0))
#         util_yr  = (e_y.values / (combined["installed_kw"].values * h_op_v)).clip(0, 1.0)

#         ax.bar(x, var_vals, w, color=var_clr, edgecolor="white",
#                linewidth=0.5, zorder=3)
#         ax.bar(x, fix_vals, w, bottom=var_vals, color=fix_clr,
#                edgecolor="white", linewidth=0.5, zorder=3)
#         ax.bar(x, cap_vals, w, bottom=var_vals + fix_vals, color=cap_clr,
#                edgecolor="white", linewidth=0.5, zorder=3)

#         for i, tc in enumerate(total):
#             ax.text(x[i], tc + max_stack * 0.015,
#                     f"${tc:.3f}",
#                     ha="center", va="bottom",
#                     fontsize=9, fontweight="bold")

#         for i, util_v in enumerate(util_yr):
#             ax.text(x[i], total[i] + max_stack * 0.07,
#                     f"Util {util_v*100:.2f}%",
#                     ha="center", va="bottom",
#                     fontsize=7, color="#555555")

#         for i, rv in enumerate(rev_vals):
#             ax.plot([x[i] - w/2, x[i] + w/2], [rv, rv],
#                     color=C_REV_LINE, linewidth=2.2, zorder=5, linestyle="--")

#         ax.set_xticks(x)
#         ax.set_xticklabels(combined["label"], fontsize=9)
#         ax.set_title(yr_label, fontsize=10, fontweight="bold", pad=6)
#         ax.yaxis.set_major_formatter(
#             mticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
#         if col_idx == 0:
#             ax.set_ylabel(f"$/kWh  (instantaneous, year-y basis)")
#         ax.set_ylim(0, max_stack * 1.45)
#         ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

#         if has_dwc and has_scs:
#             ax.axvline(len(df_dwc) - 0.5, color="#AAAAAA",
#                        linewidth=1.2, linestyle="--", alpha=0.7)

#     legend_els = [
#         mpatches.Patch(color="#2C2C2C", label="DWC: Energy cost (≈ grid LCOE, constant/kWh)"),
#         mpatches.Patch(color="#5A7A96", label="DWC: Fixed O&M / E_del_y  (shrinks over time)"),
#         mpatches.Patch(color="#A8C8E8", label="DWC: Annualized capital / E_del_y  (shrinks)"),
#         mpatches.Patch(color="#4A1A00", label="SCS: Energy cost (≈ grid LCOE, constant/kWh)"),
#         mpatches.Patch(color="#C07030", label="SCS: Fixed O&M / E_del_y  (shrinks over time)"),
#         mpatches.Patch(color="#F5C897", label="SCS: Annualized capital / E_del_y  (shrinks)"),
#         plt.Line2D([0], [0], color=C_REV_LINE, linewidth=2, linestyle="--",
#                    label="Retail price charged to drivers ($/kWh, fixed)"),
#     ]
#     axes[2].legend(handles=legend_els, fontsize=7.5,
#                    loc="upper left", bbox_to_anchor=(1.02, 1.0),
#                    borderaxespad=0, framealpha=0.9)

#     plt.tight_layout()
#     path = os.path.join(out_dir, "plot2b_dollar_per_kwh_multiyear.png")
#     _savefig(fig, path)
#     print(f"  Saved: {path}")


# # ── Plot 1b (analytics): Annualized capital vs annual revenue per kW ──────────

# def plot_dollar_per_kw_multiyear(df_dwc, df_scs, meta, out_dir):
#     has_dwc = not df_dwc.empty
#     has_scs = not df_scs.empty
#     if not has_dwc and not has_scs:
#         print("  [Plot 1b] nothing to plot"); return

#     matplotlib.rcParams.update(_ANALYTICS_RCPARAMS)

#     T      = int(  meta.get("project_life", 30))
#     g_d    = float(meta.get("g_d",    0.0))
#     g_aadt = float(meta.get("g_aadt", 0.0))
#     rho    = float(meta.get("discount_rate", 0.07))
#     alpha  = float(meta.get("alpha", 0.08))
#     p_charge_dwc = float(meta.get("p_charge_dwc", 0.40))
#     p_charge_scs = float(meta.get("p_charge_scs", 0.40))
#     h_op   = float(meta.get("h_op", 8760.0))

#     mid_yr  = T // 2 + 1
#     years   = [1, mid_yr, T]
#     yr_labels = [
#         f"Year 1  (Base)",
#         f"Year {mid_yr}  (Mid)",
#         f"Year {T}  (End)",
#     ]

#     def _make_frame(df, e_del_col):
#         out = df[["label","tech","AADT_COM","installed_kw",
#                    e_del_col,"pv_capex"]].copy()
#         out = out.rename(columns={e_del_col: "e_del_base"})
#         return out

#     frames = []
#     if has_dwc:
#         frames.append(_make_frame(df_dwc, "E_del_DWC"))
#     if has_scs:
#         frames.append(_make_frame(df_scs, "E_del_SCS"))
#     combined = pd.concat(frames, ignore_index=True)
#     n  = len(combined)
#     x  = np.arange(n)
#     w  = 0.38

#     cap_colors = [C_DWC_CAP if t == "DWC" else C_SCS_CAP for t in combined["tech"]]
#     rev_colors = [C_DWC_REV if t == "DWC" else C_SCS_REV for t in combined["tech"]]

#     kw_vals = combined["installed_kw"].replace(0, np.nan)
#     cap_per_kw_ann = (alpha * combined["pv_capex"] / kw_vals).fillna(0).values

#     global_max = cap_per_kw_ann.max()
#     util_rate_scs = float(meta.get("utilization_rate_scs", 1.0))
#     # Physical hardware ceiling for year-y energy cap
#     hw_ceil_physical_1b = np.where(
#         combined["tech"] == "SCS",
#         combined["installed_kw"] * h_op * util_rate_scs,
#         combined["installed_kw"] * h_op
#     )
#     for yr in years:
#         fac   = (1 + g_d) ** (yr - 1)
#         p_chg = np.where(combined["tech"] == "DWC", p_charge_dwc, p_charge_scs)
#         e_y   = np.minimum((combined["e_del_base"] * fac).values, hw_ceil_physical_1b)
#         rev_y = (p_chg * e_y / kw_vals.values).clip(0)
#         global_max = max(global_max, rev_y.max())

#     def _fmt(v):
#         if abs(v) >= 1e6:   return f"${v/1e6:.2f}M"
#         if abs(v) >= 1e3:   return f"${v/1e3:.1f}k"
#         return f"${v:.0f}"

#     fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
#     fig.suptitle(
#         f"Plot 1b — Annualized Capital vs Annual Revenue per Installed kW\n"
#         f"Base / Mid / End Year  (T={T} yr,  g_d={g_d:.1%}/yr,  ρ={rho:.0%},  "
#         f"CRF α={alpha:.4f})\n"
#         f"Revenue bar grows each year; capital bar is constant — "
#         f"crossover = capital charge covered from revenue alone",
#         fontsize=10, fontweight="bold"
#     )

#     for col_idx, (yr, yr_label) in enumerate(zip(years, yr_labels)):
#         ax = axes[col_idx]
#         fac = (1 + g_d) ** (yr - 1)

#         p_chg   = np.where(combined["tech"] == "DWC", p_charge_dwc, p_charge_scs)
#         # Reuse physical ceiling computed above
#         e_del_y_raw = combined["e_del_base"] * fac
#         e_del_y     = np.minimum(e_del_y_raw.values, hw_ceil_physical_1b)
#         rev_y       = (p_chg * e_del_y / kw_vals.values).clip(0)
#         util_yr     = (e_del_y / (combined["installed_kw"].values * h_op)).clip(0, 1.0)

#         bars_cap = ax.bar(x - w/2, cap_per_kw_ann, w,
#                           color=cap_colors, edgecolor="white", linewidth=0.8, zorder=3)
#         bars_rev = ax.bar(x + w/2, rev_y, w,
#                           color=rev_colors, edgecolor="white", linewidth=0.8, zorder=3)

#         for bar, v in zip(bars_cap, cap_per_kw_ann):
#             ax.text(bar.get_x() + bar.get_width() / 2,
#                     bar.get_height() + global_max * 0.01,
#                     _fmt(v), ha="center", va="bottom",
#                     fontsize=8, fontweight="bold")

#         for bar, rv, util_v in zip(bars_rev, rev_y, util_yr):
#             ax.text(bar.get_x() + bar.get_width() / 2,
#                     bar.get_height() + global_max * 0.01,
#                     f"{_fmt(rv)}\nUtil {util_v*100:.2f}%",
#                     ha="center", va="bottom", fontsize=7.5, linespacing=1.35)

#         ax.set_xticks(x)
#         ax.set_xticklabels(combined["label"], fontsize=9)
#         ax.set_title(yr_label, fontsize=10, fontweight="bold", pad=6)
#         ax.yaxis.set_major_formatter(
#             mticker.FuncFormatter(lambda v, _: _fmt(v)))
#         if col_idx == 0:
#             ax.set_ylabel(f"$/kW  (annualized or annual)")
#         ax.set_ylim(0, global_max * 1.28)
#         ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)

#         if has_dwc and has_scs:
#             ax.axvline(len(df_dwc) - 0.5, color="#AAAAAA",
#                        linewidth=1.2, linestyle="--", alpha=0.7)

#     legend_els = [
#         mpatches.Patch(color=C_DWC_CAP, label="DWC — Annualized capital (α×Capex/kW, constant)"),
#         mpatches.Patch(color=C_DWC_REV, label="DWC — Annual revenue (P_CHARGE×E_del_y/kW, grows)"),
#         mpatches.Patch(color=C_SCS_CAP, label="SCS — Annualized capital (α×Capex/kW, constant)"),
#         mpatches.Patch(color=C_SCS_REV, label="SCS — Annual revenue (P_CHARGE×E_del_y/kW, grows)"),
#     ]
#     axes[1].legend(handles=legend_els, fontsize=8.5,
#                    loc="upper center", bbox_to_anchor=(0.5, -0.12),
#                    ncol=2, framealpha=0.9)

#     plt.tight_layout(rect=[0, 0.10, 1, 1])
#     path = os.path.join(out_dir, "plot1b_dollar_per_kw_multiyear.png")
#     _savefig(fig, path)
#     print(f"  Saved: {path}")


# # ── analytics master runner (called from generate_analytics_plots) ─────────────

# def _run_analytics_plots(results_dir, out_dir=None):
#     if out_dir is None:
#         out_dir = os.path.join(results_dir, "Plots_analytics")
#     os.makedirs(out_dir, exist_ok=True)

#     print(f"\n{'═'*65}")
#     print(f"  Analytics plots")
#     print(f"  Results : {results_dir}")
#     print(f"  Output  : {out_dir}")
#     print(f"{'═'*65}")

#     df_dwc_raw, df_scs_raw, meta = load_results_from_dir(results_dir)
#     gamma_rev_dwc = float(meta.get("gamma_rev_dwc", 1.0))
#     gamma_rev_scs = float(meta.get("gamma_rev_scs", 1.0))
#     gamma_var     = float(meta.get("gamma_var",     1.0))
#     gamma_fix_om  = float(meta.get("gamma_fix_om",  1.0))
#     T   = int(  meta.get("project_life",  30))
#     rho = float(meta.get("discount_rate", 0.07))
#     g_d = float(meta.get("g_d",           0.0))
#     p_veh_dwc = float(meta.get("p_veh_dwc_kw", 0.0))
#     print(f"\n  Γ_rev_DWC={gamma_rev_dwc:.3f}  Γ_rev_SCS={gamma_rev_scs:.3f}"
#           f"  Γ_var={gamma_var:.3f}  Γ_fix_om={gamma_fix_om:.3f}"
#           f"  T={T}yr  ρ={rho:.0%}  g_d={g_d:.2%}"
#           f"  P_veh_DWC={p_veh_dwc:.0f}kW")

#     df_dwc = compute_dwc_metrics(df_dwc_raw, meta)
#     df_scs = compute_scs_metrics(df_scs_raw, meta)

#     def _safe_int(v):
#         try:
#             f = float(v); return 0 if np.isnan(f) else int(f)
#         except: return 0

#     print(f"\n  {'Station':<14}  {'AADT':>7}  {'Util%':>6}  {'Cap/kW':>9}"
#           f"  {'PVRev/kW':>10}  {'LCOST/kWh':>10}  {'NPV':>12}")
#     print(f"  {'':─<14}  {'':─>7}  {'':─>6}  {'':─>9}"
#           f"  {'':─>10}  {'':─>10}  {'':─>12}")

#     def _pr(df):
#         for _, r in df.iterrows():
#             print(f"  {r['label'].replace(chr(10),' '):<14}  "
#                   f"{_safe_int(r.get('AADT_COM', 0)):>7,}  "
#                   f"{r['util_rate'] * 100:>5.2f}%  "
#                   f"${r['capex_per_kw']:>8,.0f}  "
#                   f"${r['pv_rev_per_kw']:>9,.0f}  "
#                   f"${r['total_cost_per_kwh']:>9,.4f}  "
#                   f"${r['npv']:>11,.0f}")

#     if not df_dwc.empty: _pr(df_dwc)
#     if not df_scs.empty: _pr(df_scs)

#     print("\nGenerating analytics plots ...")

#     plot_dollar_per_kw              (df_dwc, df_scs, meta, out_dir)
#     plot_dollar_per_kwh             (df_dwc, df_scs, meta, out_dir)
#     plot_traffic_utilisation        (df_dwc, df_scs, meta, out_dir)
#     plot_utilisation_over_time      (df_dwc, df_scs, meta, out_dir)
#     plot_dollar_per_kw_multiyear    (df_dwc, df_scs, meta, out_dir)
#     plot_dollar_per_kwh_multiyear   (df_dwc, df_scs, meta, out_dir)
#     plot_traffic_utilisation_multiyear(df_dwc, df_scs, meta, out_dir)

#     print(f"\n{'═'*65}")
#     print(f"  7 analytics plots → {out_dir}/")
#     print(f"{'═'*65}\n")
#     return out_dir


# def generate_analytics_plots(results, output_dir):
#     """Call analytics plots using the already-saved Results folder."""
#     results_dir = results["meta"]["output_dir"]
#     _run_analytics_plots(results_dir,
#                          out_dir=os.path.join(output_dir, "analytics"))


# # ══════════════════════════════════════════════════════════════════════════════
# # ─────────────────────────────────────────────────────────────────────────────
# #  SECTION B — Main_plots_new functions (_prepare, single-run plots, sensitivity)
# # ─────────────────────────────────────────────────────────────────────────────
# # ══════════════════════════════════════════════════════════════════════════════

# def _prepare(results):
#     """
#     Augment df_dwc and df_scs with derived financial columns needed for plots,
#     and build the project-level NPV scalars used by plot_breakeven / _summary.

#     FIX 3: E_cap_DWC and E_cap_SCS columns are set to the PHYSICAL hardware
#     ceiling so that the [HW] binding tag in plot_utilization compares
#     base-year E_del against the correct physical reference.
#     """
#     meta  = results["meta"]
#     alpha = results["alpha"]
#     H_op  = float(meta["h_op"])
#     N_scs = int(meta["n_scs_stalls"])
#     p_kw  = float(meta["p_scs_per_stall_kw"])
#     util  = float(meta["utilization_rate_scs"])

#     # FIX 3: physical ceiling (not base-year-adjusted)
#     E_cap_SCS_physical = N_scs * p_kw * H_op * util   # kWh/yr

#     df_dwc = results["df_dwc"].copy() if not results["df_dwc"].empty else pd.DataFrame()
#     df_scs = results["df_scs"].copy() if not results["df_scs"].empty else pd.DataFrame()

#     if not df_dwc.empty:
#         # FIX 3: physical DWC ceiling = MW_Load × H_op × 1000
#         if "E_cap_DWC" not in df_dwc.columns:
#             df_dwc["E_cap_DWC"] = df_dwc["MW_Load"] * H_op * MW_TO_KW
#         df_dwc["Ann_Capex"]  = df_dwc["Capex_Total"] * alpha
#         df_dwc["Ann_Profit"] = df_dwc["Ann_Revenue"] - df_dwc["Ann_Opex"] - df_dwc["Ann_Capex"]

#         dwc_var_opex = df_dwc["LCOE"] * df_dwc["E_del_DWC"]
#         dwc_fix_opex = df_dwc["Lambda"] * meta["c_om_base_dwc"]

#         df_dwc["Payback_yr"] = (df_dwc["Capex_Total"] /
#                                 (df_dwc["Ann_Revenue"] - dwc_var_opex - dwc_fix_opex
#                                  ).clip(lower=1.0))
#         df_dwc["P_rated_kW"]         = df_dwc["MW_Load"] * MW_TO_KW
#         df_dwc["DollarPerKW"]        = df_dwc["Capex_Total"] / df_dwc["P_rated_kW"].clip(lower=0.001)
#         df_dwc["DollarPerKWh_Cap"]   = df_dwc["Ann_Capex"]  / df_dwc["E_del_DWC"].clip(lower=1)
#         df_dwc["DollarPerKWh_Total"] = (df_dwc["Ann_Capex"] + df_dwc["Ann_Opex"]) / df_dwc["E_del_DWC"].clip(lower=1)

#     if not df_scs.empty:
#         # FIX 3: physical SCS ceiling
#         if "E_cap_SCS" not in df_scs.columns:
#             df_scs["E_cap_SCS"] = E_cap_SCS_physical
#         df_scs["Ann_Capex"]          = df_scs["Capex_Total"] * alpha
#         df_scs["Ann_Profit"]         = df_scs["Ann_Revenue"] - df_scs["Ann_Opex"] - df_scs["Ann_Capex"]
#         scs_var_opex = df_scs["LCOE"] * df_scs["E_del_SCS"]
#         scs_fix_opex = df_scs["Lambda"] * meta["c_om_base_scs"]

#         df_scs["Payback_yr"] = (df_scs["Capex_Total"] /
#                                 (df_scs["Ann_Revenue"] - scs_var_opex - scs_fix_opex
#                                  ).clip(lower=1.0))
#         df_scs["P_rated_kW"]         = N_scs * p_kw
#         df_scs["DollarPerKW"]        = df_scs["Capex_Total"] / (N_scs * p_kw)
#         df_scs["DollarPerKWh_Cap"]   = df_scs["Ann_Capex"]  / df_scs["E_del_SCS"].clip(lower=1)
#         df_scs["DollarPerKWh_Total"] = (df_scs["Ann_Capex"] + df_scs["Ann_Opex"]) / df_scs["E_del_SCS"].clip(lower=1)
#         if "AADT_COM" not in df_scs.columns:
#             _diag = results.get("df_diag", pd.DataFrame())
#             _aadt = (_diag.set_index("Segment_ID")["AADT_COM"].to_dict()
#                      if not _diag.empty else {})
#             df_scs["AADT_COM"] = df_scs.get("Host_Seg", pd.Series()).map(_aadt).fillna(0)

#     discount_rate = results["discount_rate"]
#     project_life  = results["project_life"]

#     cap_dwc     = float(df_dwc["Capex_Total"].sum()) if not df_dwc.empty else 0.0
#     cap_scs     = float(df_scs["Capex_Total"].sum()) if not df_scs.empty else 0.0
#     total_capex = cap_dwc + cap_scs

#     g_rev_dwc = float(meta.get("g_rev_dwc", 0.0))
#     g_rev_scs = float(meta.get("g_rev_scs", 0.0))
#     g_var     = float(meta.get("g_var",     0.0))
#     g_om      = float(meta.get("g_om",      0.0))

#     base_rev_dwc = float(meta.get("total_rev_dwc", 0.0))
#     base_rev_scs = float(meta.get("total_rev_scs", 0.0))
#     base_rev     = base_rev_dwc + base_rev_scs

#     base_var_opex = 0.0
#     if not df_dwc.empty:
#         base_var_opex += float((df_dwc["LCOE"] * df_dwc["E_del_DWC"]).sum())
#     if not df_scs.empty:
#         base_var_opex += float((df_scs["LCOE"] * df_scs["E_del_SCS"]).sum())

#     base_fix_opex = 0.0
#     if not df_dwc.empty:
#         base_fix_opex += float((df_dwc["Lambda"] * meta["c_om_base_dwc"]).sum())
#     if not df_scs.empty:
#         base_fix_opex += float((df_scs["Lambda"] * meta["c_om_base_scs"]).sum())

#     net_op_base = base_rev - base_var_opex - base_fix_opex

#     years = np.arange(1, project_life + 1)
#     disc  = 1.0 / (1.0 + discount_rate)**years

#     # Each stream grows at its own rate; reconstruct year-y net operating income
#     # by summing the three components separately before discounting.
#     rev_stream      = (base_rev_dwc * (1.0 + g_rev_dwc)**(years - 1)
#                        + base_rev_scs * (1.0 + g_rev_scs)**(years - 1))
#     var_opex_stream = base_var_opex * (1.0 + g_var)**(years - 1)
#     fix_opex_stream = base_fix_opex * (1.0 + g_om)**(years - 1)
#     net_op_stream   = rev_stream - var_opex_stream - fix_opex_stream

#     cum_npv_arr  = np.concatenate([[-total_capex],
#                                    -total_capex + np.cumsum(net_op_stream * disc)])
#     net_op_base  = base_rev - base_var_opex - base_fix_opex   # year-1 value
#     cum_npv_flat = np.concatenate([[-total_capex],
#                                    -total_capex + np.cumsum(np.full(project_life, net_op_base))])

#     def _pb(cum):
#         idx = np.where(cum >= 0)[0]
#         if not len(idx):
#             return np.nan
#         i = int(idx[0])
#         if i == 0:
#             return 0.0
#         return (i - 1) + (-cum[i - 1]) / (cum[i] - cum[i - 1])

#     return df_dwc, df_scs, meta, {
#         "total_capex"    : total_capex,
#         "cap_dwc"        : cap_dwc,
#         "cap_scs"        : cap_scs,
#         "net_op"         : net_op_base,
#         "base_rev"       : base_rev,
#         "base_rev_dwc"   : base_rev_dwc,
#         "base_rev_scs"   : base_rev_scs,
#         "base_var_opex"  : base_var_opex,
#         "base_fix_opex"  : base_fix_opex,
#         "cum_npv_arr"    : cum_npv_arr,
#         "cum_npv_flat"   : cum_npv_flat,
#         "years_arr"      : np.arange(0, project_life + 1),
#         "pb_disc"        : _pb(cum_npv_arr),
#         "pb_simple"      : _pb(cum_npv_flat),
#         "discount_rate"  : discount_rate,
#         "project_life"   : project_life,
#         "g_rev_dwc"      : g_rev_dwc,
#         "g_rev_scs"      : g_rev_scs,
#         "g_var"          : g_var,
#         "g_om"           : g_om,
#     }


# def _summary(results):
#     df_dwc, df_scs, meta, npv = _prepare(results)
#     pbs = []
#     if not df_dwc.empty: pbs.append(df_dwc["Payback_yr"])
#     if not df_scs.empty: pbs.append(df_scs["Payback_yr"])
#     pb_med = float(pd.concat(pbs).clip(0, 60).median()) if pbs else np.nan

#     return {
#         "run_tag"     : results["run_tag"],
#         "n_dwc"       : int(meta["n_dwc_selected"]),
#         "n_scs"       : int(meta["n_scs_selected"]),
#         "obj_val"     : float(meta["obj_val"]),
#         "rev_dwc"     : float(meta["total_rev_dwc"]),
#         "rev_scs"     : float(meta["total_rev_scs"]),
#         "opex_dwc"    : float(meta["total_opex_dwc"]),
#         "opex_scs"    : float(meta["total_opex_scs"]),
#         "cap_dwc"     : npv["cap_dwc"],
#         "cap_scs"     : npv["cap_scs"],
#         "total_capex" : npv["total_capex"],
#         "net_op"      : npv["net_op"],
#         "pb_disc"     : npv["pb_disc"],
#         "pb_simple"   : npv["pb_simple"],
#         "pb_unit_med" : pb_med,
#         "eta_dwc"     : float(meta.get("eta_dwc", 0)),
#         "eta_scs"     : float(meta.get("eta_scs", 0)),
#         "rho_ev"      : float(meta.get("rho_ev", 0)),
#         "g_ev"        : float(results["params"].get("g_ev",        0)),
#         "g_aadt"      : float(results["params"].get("g_aadt",      0)),
#         "g_price_dwc" : float(results["params"].get("g_price_dwc", 0)),
#         "g_lcoe"      : float(results["params"].get("g_lcoe",      0)),
#         "g_om"        : float(results["params"].get("g_om",        0)),
#         "e_del_dwc"   : (float(results["df_dwc"]["E_del_DWC"].sum())
#                          if not results["df_dwc"].empty else 0.0),
#         "e_del_scs"   : (float(results["df_scs"]["E_del_SCS"].sum())
#                          if not results["df_scs"].empty else 0.0),
#     }


# # ── Deployment map ─────────────────────────────────────────────────────────────

# def plot_deployment_map(results, output_dir, show_existing=False, title_suffix=""):
#     meta             = results["meta"]
#     df_diag          = results["df_diag"]
#     df_dwc           = results["df_dwc"]
#     df_scs           = results["df_scs"]
#     sub_coords       = results["sub_coords"]
#     sel_sub_dwc      = results["selected_substations_dwc"]
#     sel_sub_scs      = results["selected_substations_scs"]
#     scs_coords_exist = results["scs_coords_exist"]

#     fig, ax = plt.subplots(figsize=(16, 11))
#     ax.set_axis_off()

#     _draw_us_background(ax)

#     ax.scatter(df_diag["X"], df_diag["Y"],
#            c="#aab0b8", s=18, alpha=0.55, zorder=1,
#            linewidths=0, rasterized=True)

#     all_subs = sel_sub_dwc | sel_sub_scs
#     if all_subs:
#         sx = [sub_coords[s][0] for s in all_subs if s in sub_coords]
#         sy = [sub_coords[s][1] for s in all_subs if s in sub_coords]
#         ax.scatter(sx, sy, c="black", s=12, zorder=6,
#                    linewidths=0, label="Substation (Selected)")

#     if not df_dwc.empty and "MID_X" in df_dwc.columns:
#         _d = df_dwc.dropna(subset=["MID_X", "MID_Y"])
#         ax.scatter(_d["MID_X"], _d["MID_Y"],
#                    c=C_GOLD, s=45, zorder=4,
#                    edgecolors="black", linewidths=0.5,
#                    label=f"Selected DWC Segment (n={len(_d)})")

#     if not df_scs.empty and "MID_X" in df_scs.columns:
#         _s = df_scs.dropna(subset=["MID_X", "MID_Y"])
#         ax.scatter(_s["MID_X"], _s["MID_Y"],
#                    marker=_MARKER_SCS, c=C_CYAN,
#                    edgecolors="black", s=110, zorder=5,
#                    label=f"New SCS Station (n={len(_s)})")

#     if show_existing and scs_coords_exist:
#         ex_xy = [(v[0], v[1]) for v in scs_coords_exist.values()
#                  if v[0] is not None]
#         if ex_xy:
#             ax.scatter([p[0] for p in ex_xy], [p[1] for p in ex_xy],
#                        marker=_MARKER_EXIST, c=C_EXIST, s=65,
#                        edgecolors="black", linewidths=0.4,
#                        alpha=0.60, zorder=3,
#                        label=f"Existing SCS (n={len(ex_xy)})")

#     ax.annotate("N", xy=(0.95, 0.08), xycoords="axes fraction",
#                 ha="center", va="bottom", fontsize=18, fontweight="bold")
#     ax.arrow(0.95, 0.02, 0, 0.05, transform=ax.transAxes,
#              length_includes_head=True, head_width=0.01, head_length=0.02,
#              fc="black", ec="black")

#     ax.legend(loc="lower left", fontsize=12, frameon=False,
#               markerscale=1.0, handlelength=1.2,
#               borderpad=0.5, labelspacing=0.4)

#     fname = "map_full.png" if show_existing else "map_proposed.png"
#     if title_suffix:
#         fname = fname.replace(".png", f"_{title_suffix.replace(' ', '_')}.png")
#     _savefig(fig, os.path.join(output_dir, fname))


# # ── Traffic split ──────────────────────────────────────────────────────────────

# def plot_traffic_split(results, output_dir, title_suffix=""):
#     meta    = results["meta"]
#     eta_dwc = float(meta.get("eta_dwc", 0))
#     eta_scs = float(meta.get("eta_scs", 0))

#     labels = [r"$\eta^{\mathrm{SCS}}$", r"$\eta^{\mathrm{DWC}}$"]
#     values = [eta_scs * 100, eta_dwc * 100]
#     colors = [C_SCS, C_DWC]

#     fig, ax = plt.subplots(figsize=(5, 1.8))
#     bars = ax.barh(labels, values, color=colors, height=0.6)

#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.spines["bottom"].set_visible(False)
#     ax.set_xticks([])

#     for bar in bars:
#         width = bar.get_width()
#         ax.text(width + 2, bar.get_y() + bar.get_height() / 2,
#                 f"{width:.0f}%",
#                 va="center", ha="left", fontsize=12, fontweight="bold")

#     ax.tick_params(axis="y", length=0, labelsize=14)
#     ttl = "Traffic-Capture"
#     if title_suffix:
#         ttl += f"\n{title_suffix}"
#     ax.set_title(ttl, fontweight="bold", fontsize=13)
#     plt.tight_layout()
#     fname = "traffic_split.png"
#     if title_suffix:
#         fname = f"traffic_split_{title_suffix.replace(' ', '_')}.png"
#     _savefig(fig, os.path.join(output_dir, fname))


# # ── Total units pie ────────────────────────────────────────────────────────────

# def plot_total_units(results, output_dir, title_suffix=""):
#     meta  = results["meta"]
#     n_dwc = int(meta["n_dwc_selected"])
#     n_scs = int(meta["n_scs_selected"])

#     slice_labels = []
#     slice_values = []
#     slice_colors = []
#     if n_dwc > 0:
#         slice_labels.append("DWC")
#         slice_values.append(n_dwc)
#         slice_colors.append(C_DWC)
#     if n_scs > 0:
#         slice_labels.append("SCS")
#         slice_values.append(n_scs)
#         slice_colors.append(C_SCS)

#     fig, ax = plt.subplots(figsize=(6, 6))

#     if not slice_values:
#         ax.pie([1], colors=[C_GREY], startangle=140)
#         ax.set_title("Resulting Units\n(none selected)", fontsize=20,
#                      fontweight="bold", pad=-5)
#     else:
#         ax.pie(
#             slice_values,
#             labels=None,
#             autopct=lambda p: f"{round(p * sum(slice_values) / 100):.0f}",
#             startangle=140,
#             colors=slice_colors,
#             pctdistance=0.50,
#             textprops={"fontweight": "bold", "color": "white", "fontsize": 20},
#         )
#         ttl = "Resulting Units"
#         if title_suffix:
#             ttl += f"\n{title_suffix}"
#         ax.set_title(ttl, fontsize=20, fontweight="bold", pad=-5)

#     legend_handles = []
#     if n_dwc > 0:
#         legend_handles.append(
#             plt.Rectangle((0, 0), 1, 1, color=C_DWC, label=f"DWC (n={n_dwc})"))
#     if n_scs > 0:
#         legend_handles.append(
#             plt.Rectangle((0, 0), 1, 1, color=C_SCS, label=f"SCS (n={n_scs})"))
#     if legend_handles:
#         ax.legend(handles=legend_handles, loc="lower center",
#                   bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=12,
#                   frameon=False)

#     plt.tight_layout()
#     fname = "total_units.png"
#     if title_suffix:
#         fname = f"total_units_{title_suffix.replace(' ', '_')}.png"
#     _savefig(fig, os.path.join(output_dir, fname))


# # ── Capex / revenue / profit ───────────────────────────────────────────────────

# def plot_capex_revenue_profit(results, output_dir, title_suffix=""):
#     df_dwc, df_scs, meta, npv = _prepare(results)
#     alpha       = results["alpha"]
#     have_dwc    = not df_dwc.empty
#     have_scs    = not df_scs.empty
#     have_any    = have_dwc or have_scs

#     p_charge_dwc = float(meta.get("p_charge_dwc", 0.40))
#     p_charge_scs = float(meta.get("p_charge_scs", 0.42))

#     if not have_any:
#         print("  Skipping capex_revenue_profit.png (no assets selected)")
#         return

#     def _build_rec(df_a, e_del_col, capex_col, opex_col, rev_col,
#                    p_rated_col, ann_capex_col, retail, tech):
#         if len(df_a) == 0:
#             return None
#         e   = df_a[e_del_col].mean()
#         rev = df_a[rev_col].mean()
#         ac  = df_a[ann_capex_col].mean()
#         ao  = df_a[opex_col].mean()
#         cap = df_a[capex_col].mean()
#         pkw = df_a[p_rated_col].mean()
#         return {
#             "label"        : tech,
#             "n"            : len(df_a),
#             "E_del_kWh"    : e,
#             "Ann_Rev"      : rev,
#             "Ann_Capex"    : ac,
#             "Ann_Opex"     : ao,
#             "Ann_Profit"   : rev - ac - ao,
#             "Capex_total"  : cap,
#             "P_rated_kW"   : pkw,
#             "DolPerKW"     : cap / max(pkw, 1),
#             "DolPerKWh_cap": ac  / max(e, 1),
#             "DolPerKWh_op" : ao  / max(e, 1),
#             "DolPerKWh_rev": rev / max(e, 1),
#             "Retail"       : retail,
#             "Margin_kWh"   : retail - (ac + ao) / max(e, 1),
#         }

#     records = []
#     if have_dwc:
#         records.append(_build_rec(df_dwc, "E_del_DWC", "Capex_Total", "Ann_Opex",
#                                   "Ann_Revenue", "P_rated_kW", "Ann_Capex",
#                                   p_charge_dwc, "DWC"))
#     if have_scs:
#         records.append(_build_rec(df_scs, "E_del_SCS", "Capex_Total", "Ann_Opex",
#                                   "Ann_Revenue", "P_rated_kW", "Ann_Capex",
#                                   p_charge_scs, "SCS"))
#     records = [r for r in records if r is not None]
#     if not records:
#         return

#     colors  = {"DWC": C_DWC, "SCS": C_SCS}
#     n_tech  = len(records)

#     fig, axes = plt.subplots(1, 3, figsize=(18, 7))

#     ax = axes[0]
#     x_pos = np.arange(n_tech)
#     vals  = [r["DolPerKW"] / 1e3 for r in records]
#     bars  = ax.bar(x_pos, vals,
#                    color=[colors[r["label"]] for r in records],
#                    width=0.45, alpha=0.85, edgecolor="none")
#     for b, v, r in zip(bars, vals, records):
#         ax.text(b.get_x() + b.get_width() / 2,
#                 v + max(vals) * 0.02,
#                 f"${v:.1f}K/kW\n({r['P_rated_kW']:.0f} kW)",
#                 ha="center", va="bottom", fontsize=10, fontweight="bold")
#     ax.set_xticks(x_pos)
#     ax.set_xticklabels([r["label"] for r in records], fontsize=13)
#     ax.set_ylabel("Capex per kW  ($K/kW)")
#     ax.set_title("(a)  Capex Intensity\n($/kW rated)", fontweight="bold")
#     ax.tick_params(axis="x", length=0)
#     ax.axhline(1.0, color="grey", lw=1.2, ls=":", alpha=0.6,
#                label="$1K/kW ref (DCFC low)")
#     ax.axhline(4.0, color="grey", lw=1.2, ls="--", alpha=0.6,
#                label="$4K/kW ref (DCFC high)")
#     ax.legend(fontsize=9, frameon=False)
#     ax.set_ylim(0, max(max(vals) * 1.40, 5.0))

#     ax = axes[1]
#     x_pos2 = np.arange(n_tech)
#     w      = 0.28

#     for idx, rec in enumerate(records):
#         col   = colors[rec["label"]]
#         x_cap = idx - w
#         x_rev = idx + w
#         ax.bar(x_cap, rec["DolPerKWh_cap"], width=w * 1.8,
#                color=col, alpha=0.90, edgecolor="none",
#                label="Ann. Capex/kWh" if idx == 0 else "")
#         ax.bar(x_cap, rec["DolPerKWh_op"], width=w * 1.8,
#                bottom=rec["DolPerKWh_cap"],
#                color=col, alpha=0.40, edgecolor="none",
#                label="Opex/kWh" if idx == 0 else "")
#         ax.bar(x_rev, rec["DolPerKWh_rev"], width=w * 1.8,
#                color=C_REV, alpha=0.85, edgecolor="none",
#                label="Revenue/kWh" if idx == 0 else "")

#         total_cost = rec["DolPerKWh_cap"] + rec["DolPerKWh_op"]
#         margin     = rec["DolPerKWh_rev"] - total_cost
#         mid_y      = max(total_cost, rec["DolPerKWh_rev"]) + 0.008
#         sign       = "+" if margin >= 0 else ""
#         ax.annotate(
#             f"{sign}${margin:.3f}/kWh\nmargin",
#             xy=(idx, mid_y), ha="center", va="bottom",
#             fontsize=10, fontweight="bold",
#             color=C_REV if margin >= 0 else C_CAPEX,
#         )
#         ax.text(x_cap, total_cost + 0.003,
#                 f"${total_cost:.3f}", ha="center", va="bottom",
#                 fontsize=9, color="white" if total_cost > 0.05 else "black")
#         ax.text(x_rev, rec["DolPerKWh_rev"] + 0.003,
#                 f"${rec['DolPerKWh_rev']:.3f}", ha="center", va="bottom",
#                 fontsize=9, color=C_REV)

#     ax.set_xticks(x_pos2)
#     ax.set_xticklabels(
#         [f"{r['label']}\n(n={r['n']}, {r['E_del_kWh']/1e3:.0f} MWh/yr/unit)"
#          for r in records],
#         fontsize=11
#     )
#     for rec in records:
#         ax.axhline(rec["Retail"], color=colors[rec["label"]],
#                    lw=1.5, ls="--", alpha=0.55,
#                    label=f"{rec['label']} retail ${rec['Retail']:.2f}/kWh")
#     ax.set_ylabel("$/kWh delivered")
#     ax.set_title("(b)  Cost vs Revenue per kWh\n"
#                  "Left=costs  Right=retail revenue", fontweight="bold")
#     ax.legend(fontsize=8, frameon=False, loc="upper left")
#     ax.set_ylim(0, max(
#         max(r["DolPerKWh_rev"] for r in records),
#         max(r["DolPerKWh_cap"] + r["DolPerKWh_op"] for r in records)
#     ) * 1.45)

#     ax = axes[2]
#     bar_labels = ["Ann.\nRevenue", "Ann.\nCapex", "Ann.\nOpex", "Ann.\nProfit"]
#     x_grp      = np.arange(len(bar_labels))
#     bar_w      = 0.80 / max(n_tech, 1)

#     for idx, rec in enumerate(records):
#         col    = colors[rec["label"]]
#         vals_k = [rec["Ann_Rev"] / 1e3, rec["Ann_Capex"] / 1e3,
#                   rec["Ann_Opex"] / 1e3, rec["Ann_Profit"] / 1e3]
#         bar_cols = [C_REV, C_CAPEX, C_OPEX,
#                     C_REV if rec["Ann_Profit"] >= 0 else C_CAPEX]
#         x_offset = (idx - (n_tech - 1) / 2) * bar_w
#         for xi, (v, bc) in enumerate(zip(vals_k, bar_cols)):
#             ax.bar(x_grp[xi] + x_offset, v, width=bar_w * 0.85,
#                    color=bc, alpha=0.80, edgecolor="white", linewidth=0.5)
#             fmt = f"${v:.0f}K" if abs(v) >= 1 else f"${v:.2f}K"
#             ax.text(x_grp[xi] + x_offset,
#                     v + (abs(max(vals_k)) * 0.025) * np.sign(v) if v != 0 else 0.5,
#                     fmt, ha="center",
#                     va="bottom" if v >= 0 else "top",
#                     fontsize=9, fontweight="bold", color=bc)

#     ax.axhline(0, color="black", lw=0.8)
#     ax.set_xticks(x_grp)
#     ax.set_xticklabels(bar_labels, fontsize=11)
#     ax.set_ylabel("Annual $ per Unit ($K/yr)")
#     ax.set_title("(c)  Annual Financial Summary per Unit\n"
#                  "(per segment for DWC, per station for SCS)", fontweight="bold")
#     ax.tick_params(axis="x", length=0)
#     _tech_handles = [
#         plt.Rectangle((0, 0), 1, 1, color=colors[r["label"]], alpha=0.80,
#                        label=f"{r['label']} (n={r['n']})")
#         for r in records
#     ]
#     ax.legend(handles=_tech_handles, fontsize=9, frameon=False, loc="upper right")

#     _summary_lines = []
#     for rec in records:
#         pb = rec["Capex_total"] / max(rec["Ann_Rev"] - rec["Ann_Opex"], 1)
#         _summary_lines.append(
#             f"{rec['label']}: capex ${rec['Capex_total']/1e6:.2f}M  |  "
#             f"margin ${rec['Margin_kWh']:.3f}/kWh  |  simple payback {pb:.1f} yr"
#         )
#     fig.text(0.5, -0.02, "\n".join(_summary_lines),
#              ha="center", va="top", fontsize=10,
#              bbox=dict(boxstyle="round,pad=0.5",
#                        facecolor="lightyellow", edgecolor="grey", alpha=0.9))

#     _ttl = "Per-Unit Revenue vs Cost Analysis — DWC & SCS"
#     if title_suffix:
#         _ttl += f"\n{title_suffix}"
#     fig.suptitle(_ttl, fontweight="bold", y=1.02)
#     plt.tight_layout()
#     fname = "capex_revenue_profit.png"
#     if title_suffix:
#         fname = f"capex_revenue_profit_{title_suffix.replace(' ', '_')}.png"
#     _savefig(fig, os.path.join(output_dir, fname))


# # ── Breakeven ─────────────────────────────────────────────────────────────────

# def plot_breakeven(results, output_dir, title_suffix=""):
#     df_dwc, df_scs, meta, npv = _prepare(results)

#     project_life  = npv["project_life"]
#     discount_rate = npv["discount_rate"]
#     total_capex   = npv["total_capex"]
#     g_rev_dwc = npv["g_rev_dwc"]
#     g_rev_scs = npv["g_rev_scs"]
#     g_var     = npv["g_var"]
#     g_om      = npv["g_om"]
#     # For display label on the chart use the DWC revenue rate as representative
#     g_rev     = g_rev_dwc

#     base_rev      = npv["base_rev"]
#     base_rev_dwc  = npv["base_rev_dwc"]
#     base_rev_scs  = npv["base_rev_scs"]
#     base_var_opex = npv["base_var_opex"]
#     base_fix_opex = npv["base_fix_opex"]

#     years_ext = np.arange(0, project_life + 1)
#     yrs       = np.arange(1, project_life + 1)
#     disc      = 1.0 / (1.0 + discount_rate)**yrs

#     growing_annual = (
#         base_rev_dwc * (1.0 + g_rev_dwc)**(yrs - 1)
#         + base_rev_scs * (1.0 + g_rev_scs)**(yrs - 1)
#         - base_var_opex * (1.0 + g_var)**(yrs - 1)
#         - base_fix_opex * (1.0 + g_om)**(yrs - 1)
#     )
#     cum_npv = np.concatenate([[-total_capex],
#                                -total_capex + np.cumsum(growing_annual * disc)])

#     def _find_pb(cum_arr):
#         idx = np.where(cum_arr >= 0)[0]
#         if len(idx) == 0:
#             return np.nan
#         i = int(idx[0])
#         if i == 0:
#             return 0.0
#         y0, y1 = cum_arr[i - 1], cum_arr[i]
#         return (i - 1) + (-y0) / (y1 - y0)

#     pb_proj = _find_pb(cum_npv)

#     unit_labels, unit_pb_disc, unit_pb_simple, unit_colors, unit_techs = [], [], [], [], []

#     if not df_dwc.empty:
#         for _, row in df_dwc.iterrows():
#             capex   = float(row["Capex_Total"])
#             ann_rev = float(row["Ann_Revenue"])
#             var_om  = float(row["LCOE"]) * float(row["E_del_DWC"])
#             fix_om  = float(row["Lambda"]) * float(meta["c_om_base_dwc"])
#             net_y   = (ann_rev * (1.0 + g_rev_dwc)**(yrs - 1)
#                        - var_om * (1.0 + g_var)**(yrs - 1)
#                        - fix_om * (1.0 + g_om)**(yrs - 1))
#             cum_d   = np.concatenate([[-capex], -capex + np.cumsum(net_y * disc)])
#             net_f   = ann_rev - var_om - fix_om
#             cum_f   = np.concatenate([[-capex], -capex + np.cumsum(np.full(project_life, net_f))])
#             unit_labels.append(f"DWC\n{row['Segment_ID']}")
#             unit_pb_disc.append(_find_pb(cum_d))
#             unit_pb_simple.append(_find_pb(cum_f))
#             unit_colors.append(C_DWC)
#             unit_techs.append("DWC")

#     if not df_scs.empty:
#         for _, row in df_scs.iterrows():
#             capex   = float(row["Capex_Total"])
#             ann_rev = float(row["Ann_Revenue"])
#             var_om  = float(row["LCOE"]) * float(row["E_del_SCS"])
#             fix_om  = float(row["Lambda"]) * float(meta["c_om_base_scs"])
#             net_y   = (ann_rev * (1.0 + g_rev_scs)**(yrs - 1)
#                        - var_om * (1.0 + g_var)**(yrs - 1)
#                        - fix_om * (1.0 + g_om)**(yrs - 1))
#             cum_d   = np.concatenate([[-capex], -capex + np.cumsum(net_y * disc)])
#             net_f   = ann_rev - var_om - fix_om
#             cum_f   = np.concatenate([[-capex], -capex + np.cumsum(np.full(project_life, net_f))])
#             unit_labels.append(f"SCS\n{row['SCS_ID']}")
#             unit_pb_disc.append(_find_pb(cum_d))
#             unit_pb_simple.append(_find_pb(cum_f))
#             unit_colors.append(C_SCS)
#             unit_techs.append("SCS")

#     fig, (ax_left, ax_right) = plt.subplots(
#         1, 2, figsize=(14, 5.5), facecolor="white",
#         gridspec_kw={"wspace": 0.38}
#     )

#     ax = ax_left
#     ax.set_facecolor("white")
#     ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)

#     pb_str = f"{pb_proj:.1f} yr" if not np.isnan(pb_proj) and pb_proj <= project_life else ">horizon"
#     label = (f"NPV cash flow  (g={g_rev:.1%}, r={discount_rate:.0%})\n"
#              f"Payback: {pb_str}")
#     ax.plot(years_ext, cum_npv / 1e6, color=C_DWC, lw=2.0, label=label)

#     if not np.isnan(pb_proj) and pb_proj <= project_life:
#         ax.axvline(pb_proj, color=C_DWC, lw=1.0, ls=":", alpha=0.7)
#         ax.scatter([pb_proj], [0], color=C_DWC, s=60, zorder=5,
#                    edgecolors="white", linewidths=0.8)

#     ax.fill_between(years_ext, cum_npv / 1e6, 0,
#                     where=(cum_npv >= 0), color=C_REV, alpha=0.07, interpolate=True)

#     ax.text(0.02, 0.04,
#             f"CAPEX  ${total_capex/1e6:.1f} M",
#             transform=ax.transAxes, ha="left", va="bottom",
#             fontsize=9, family="monospace", color="#333333",
#             bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9f9f9",
#                       edgecolor="#dddddd", alpha=0.95))

#     ax.set_xlabel("Year", fontsize=11)
#     ax.set_ylabel("Cumulative cash flow  ($M)", fontsize=11)
#     ax.set_xlim(0, project_life)
#     ax.set_title("Project-Level Cumulative NPV", fontweight="bold", fontsize=12, pad=10)
#     ax.legend(fontsize=9, frameon=False, loc="upper left")
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

#     ax = ax_right
#     ax.set_facecolor("white")

#     n_units = len(unit_labels)
#     x_pos   = np.arange(n_units)
#     bar_w   = 0.35

#     pb_d_plot = [min(v, project_life * 1.05) if not np.isnan(v) else project_life * 1.05
#                  for v in unit_pb_disc]
#     pb_s_plot = [min(v, project_life * 1.05) if not np.isnan(v) else project_life * 1.05
#                  for v in unit_pb_simple]

#     bars_d = ax.bar(x_pos - bar_w/2, pb_d_plot, bar_w,
#                     color=unit_colors, alpha=0.85, edgecolor="white", linewidth=0.5,
#                     label="Discounted payback")
#     bars_s = ax.bar(x_pos + bar_w/2, pb_s_plot, bar_w,
#                     color=unit_colors, alpha=0.40, edgecolor="white", linewidth=0.5,
#                     label="Simple payback")

#     for xi, (pd_v, ps_v) in enumerate(zip(unit_pb_disc, unit_pb_simple)):
#         pd_str = f"{pd_v:.1f}" if not np.isnan(pd_v) and pd_v <= project_life else ">"
#         ps_str = f"{ps_v:.1f}" if not np.isnan(ps_v) and ps_v <= project_life else ">"
#         ax.text(xi - bar_w/2, pb_d_plot[xi] + 0.3, pd_str,
#                 ha="center", va="bottom", fontsize=9, fontweight="bold",
#                 color=unit_colors[xi])
#         ax.text(xi + bar_w/2, pb_s_plot[xi] + 0.3, ps_str,
#                 ha="center", va="bottom", fontsize=9,
#                 color=unit_colors[xi], alpha=0.7)

#     ax.axhline(project_life, color="grey", lw=1.2, ls="--", alpha=0.7,
#                label=f"Project horizon  {project_life} yr")

#     legend_handles = []
#     if any(t == "DWC" for t in unit_techs):
#         legend_handles.append(
#             plt.Rectangle((0,0),1,1, color=C_DWC, alpha=0.85, label="DWC — discounted PB"))
#         legend_handles.append(
#             plt.Rectangle((0,0),1,1, color=C_DWC, alpha=0.40, label="DWC — simple PB"))
#     if any(t == "SCS" for t in unit_techs):
#         legend_handles.append(
#             plt.Rectangle((0,0),1,1, color=C_SCS, alpha=0.85, label="SCS — discounted PB"))
#         legend_handles.append(
#             plt.Rectangle((0,0),1,1, color=C_SCS, alpha=0.40, label="SCS — simple PB"))
#     legend_handles.append(
#         Line2D([0],[0], color="grey", lw=1.2, ls="--",
#                label=f"Project horizon  {project_life} yr"))
#     ax.legend(handles=legend_handles, fontsize=8, frameon=False, loc="upper left")

#     ax.set_xticks(x_pos)
#     ax.set_xticklabels(unit_labels, fontsize=10)
#     ax.set_ylabel("Payback year", fontsize=11)
#     ax.set_ylim(0, project_life * 1.15)
#     ax.set_title("Per-Unit Payback Year\nDark = discounted  |  Light = simple (no growth, no discount)",
#                  fontweight="bold", fontsize=12, pad=10)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

#     if title_suffix:
#         fig.suptitle(title_suffix, fontsize=11, color="#555555", y=1.01)

#     fname = "breakeven.png"
#     if title_suffix:
#         fname = f"breakeven_{title_suffix.replace(' ', '_')}.png"
#     _savefig(fig, os.path.join(output_dir, fname))


# # ── Utilization ───────────────────────────────────────────────────────────────

# def plot_utilization(results, output_dir, title_suffix=""):
#     df_dwc, df_scs, meta, _ = _prepare(results)
#     have_dwc = not df_dwc.empty
#     have_scs = not df_scs.empty

#     if not (have_dwc or have_scs):
#         print("  Skipping utilization.png (no assets selected)")
#         return

#     panels = []
#     if have_dwc:
#         panels.append(("DWC", df_dwc, "E_traff_DWC", "E_del_DWC", "E_cap_DWC",
#                         "Segment_ID", C_DWC, int(meta["n_dwc_selected"])))
#     if have_scs:
#         panels.append(("SCS", df_scs, "E_traff_SCS", "E_del_SCS", "E_cap_SCS",
#                         "SCS_ID", C_SCS, int(meta["n_scs_selected"])))

#     ncols = len(panels)
#     n_max = max(len(df_dwc) if have_dwc else 0,
#                 len(df_scs) if have_scs else 0)
#     fig_h = max(8, min(n_max * 0.22 + 2, 60))
#     fig, axes = plt.subplots(1, ncols, figsize=(16 * ncols, fig_h))
#     if ncols == 1:
#         axes = [axes]

#     BH = 0.22
#     for ax, (tech, df, col_tr, col_del, col_cap, col_id, color_d, n_lbl) in \
#             zip(axes, panels):
#         df_s = df.sort_values(col_tr, ascending=True).reset_index(drop=True)
#         n    = len(df_s)
#         y    = np.arange(n)
#         step = _label_step(n)
#         fs   = max(5.0, min(8.5, 400 / max(n, 1)))

#         ax.barh(y + 0.26, df_s[col_cap] / 1e6,  BH,
#                 color="#d5d8dc", edgecolor="none", label="Hardware ceiling")
#         ax.barh(y,        df_s[col_tr]  / 1e6,  BH,
#                 color=color_d, alpha=0.50, edgecolor="none", label="Traffic demand")
#         ax.barh(y - 0.26, df_s[col_del] / 1e6,  BH,
#                 color=C_REV, alpha=0.85, edgecolor="none", label="Delivered")

#         if n <= 120:
#             x_max = float(df_s[col_cap].max()) / 1e6
#             for k in range(n):
#                 util    = float(df_s[col_del].iloc[k]) / max(float(df_s[col_tr].iloc[k]), 1) * 100
#                 hw_bind = float(df_s[col_del].iloc[k]) >= 0.99 * float(df_s[col_cap].iloc[k])
#                 tr_bind = float(df_s[col_del].iloc[k]) >= 0.99 * float(df_s[col_tr].iloc[k])
#                 tag     = " [HW]" if hw_bind else (" [Tr]" if tr_bind else "")
#                 ax.text(float(df_s[col_del].iloc[k]) / 1e6 + x_max * 0.01,
#                         y[k] - 0.26,
#                         f"{util:.0f}%{tag}",
#                         va="center", fontsize=max(4, fs - 1.5), color="#444444")

#         ax.set_yticks(y[::step])
#         ax.set_yticklabels(df_s[col_id].iloc[::step].astype(str), fontsize=fs)
#         ax.set_ylabel(f"{col_id}  (sorted by traffic demand ↑)")
#         ax.set_xlabel("Annual energy  (GWh / yr)")

#         u_arr      = df_s[col_del] / df_s[col_tr].clip(lower=1)
#         n_hw       = (df_s[col_del] >= 0.99 * df_s[col_cap]).sum()
#         n_tr       = (df_s[col_del] >= 0.99 * df_s[col_tr]).sum()
#         total_del  = df_s[col_del].sum()
#         total_traf = df_s[col_tr].sum()

#         ax.set_title(
#             f"{tech} Utilisation  (n = {n_lbl})\n"
#             f"Delivered {total_del/1e6:.1f} GWh / Demand {total_traf/1e6:.1f} GWh"
#             f" = {total_del/max(total_traf,1)*100:.1f}% aggregate\n"
#             "[HW]=hardware-bound  |  [Tr]=traffic-bound",
#             fontweight="bold"
#         )
#         ax.text(0.98, 0.03,
#                 f"Traffic util: median {u_arr.median()*100:.1f}%  "
#                 f"mean {u_arr.mean()*100:.1f}%\n"
#                 f"HW-bound: {n_hw}/{n}   Traffic-bound: {n_tr}/{n}",
#                 transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
#                 bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
#                           edgecolor="grey", alpha=0.9))
#         ax.legend(fontsize=9, frameon=False, loc="lower right")

#     _ttl = "DWC & SCS Energy Utilisation"
#     if title_suffix:
#         _ttl += f" — {title_suffix}"
#     fig.suptitle(_ttl, fontweight="bold", y=1.01)
#     plt.tight_layout()
#     fname = "utilization.png"
#     if title_suffix:
#         fname = f"utilization_{title_suffix.replace(' ', '_')}.png"
#     _savefig(fig, os.path.join(output_dir, fname))


# # ── Heatmaps ──────────────────────────────────────────────────────────────────

# def plot_heatmaps(results, output_dir, title_suffix=""):
#     df_diag          = results["df_diag"]
#     df_dwc           = results["df_dwc"]
#     df_scs           = results["df_scs"]
#     scs_coords_exist = results["scs_coords_exist"]
#     meta             = results["meta"]
#     use_existing_scs = meta.get("use_existing_scs", False)

#     if "Is_DWC" not in df_diag.columns:
#         df_diag = df_diag.copy()
#         df_diag["Is_DWC"] = 0

#     def _overlay(ax, df_dwc_=df_dwc, df_scs_=df_scs):
#         sel = df_diag[df_diag["Is_DWC"] == 1]
#         if not sel.empty:
#             ax.scatter(sel["X"], sel["Y"],
#                        facecolors="none", edgecolors=C_DWC,
#                        linewidths=1.2, s=120, zorder=4,
#                        label="Selected DWC")
#         if not df_scs_.empty and "MID_X" in df_scs_.columns:
#             _sp = df_scs_.dropna(subset=["MID_X", "MID_Y"])
#             if not _sp.empty:
#                 ax.scatter(_sp["MID_X"], _sp["MID_Y"],
#                            marker=_MARKER_SCS, c=C_CYAN,
#                            edgecolors="black", s=80, zorder=5,
#                            label="New SCS")
#         if use_existing_scs and scs_coords_exist:
#             ex_xy = [(v[0], v[1]) for v in scs_coords_exist.values()
#                      if v[0] is not None]
#             if ex_xy:
#                 ax.scatter([p[0] for p in ex_xy], [p[1] for p in ex_xy],
#                            marker=_MARKER_EXIST, c=C_EXIST,
#                            edgecolors="black", s=40, alpha=0.60, zorder=3,
#                            label="Existing SCS")

#     def _heatmap(col, title, label, cmap, filename, log_scale=True):
#         fig_, ax_ = plt.subplots(figsize=(13, 10))
#         _draw_us_background(ax_)

#         vals = df_diag[col][df_diag[col] > 0] if log_scale else df_diag[col]
#         _col_max = df_diag[col].max()
#         _col_min = df_diag[col].min()
#         if log_scale and len(vals) > 0 and vals.min() > 0 and _col_max > 0:
#             norm = mcolors.LogNorm(vmin=vals.min(), vmax=_col_max)
#         else:
#             _vmin = _col_min if _col_min != _col_max else 0
#             _vmax = _col_max if _col_max != _col_min else 1
#             norm = mcolors.Normalize(vmin=_vmin, vmax=_vmax)

#         sc = ax_.scatter(df_diag["X"], df_diag["Y"],
#                          c=df_diag[col], cmap=cmap,
#                          s=55, alpha=0.75, norm=norm,
#                          edgecolors="none", zorder=2)
#         _overlay(ax_)

#         cbar = plt.colorbar(sc, ax=ax_, pad=0.02)
#         cbar.set_label(label, rotation=270, labelpad=18, fontsize=13)
#         cbar.ax.tick_params(labelsize=11)
#         ax_.set_title(title + (f" — {title_suffix}" if title_suffix else ""),
#                       fontsize=16, fontweight="bold", pad=10)
#         for spine in ax_.spines.values():
#             spine.set_visible(False)
#         ax_.set_xticks([]); ax_.set_yticks([])
#         ax_.legend(fontsize=11, loc="upper left", frameon=False)
#         plt.tight_layout()
#         fpath = os.path.join(output_dir,
#                              filename if not title_suffix else
#                              filename.replace(".png",
#                                              f"_{title_suffix.replace(' ','_')}.png"))
#         _savefig(fig_, fpath)

#     _heatmap("Total_Revenue",
#              "Spatial Revenue Potential (DWC)",
#              "Annual Revenue Potential ($)", "YlOrRd",
#              "heatmap_revenue.png", log_scale=True)
#     _heatmap("Rev_Dist_Ratio",
#              "Revenue / Substation Distance Ratio",
#              "Rev / Distance Index", "viridis",
#              "heatmap_rev_dist.png", log_scale=True)
#     _heatmap("AADT_COM",
#              "Commercial Traffic Demand (AADT)",
#              "Commercial AADT", "YlGn",
#              "heatmap_traffic.png", log_scale=True)
#     _heatmap("Lambda",
#              r"Locational Cost Multiplier ($\lambda$)",
#              r"$\lambda$ Value", "Purples",
#              "heatmap_lambda.png", log_scale=False)

#     fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 9))
#     for ax_, col, title, label, cmap, log in [
#         (ax1, "AADT_COM", "Commercial Traffic Demand", "AADT", "YlGn", True),
#         (ax2, "Lambda",   r"Locational Multiplier ($\lambda$)", r"$\lambda$", "Purples", False),
#     ]:
#         _draw_us_background(ax_)
#         _dvals = df_diag[col][df_diag[col] > 0]
#         _dmax  = df_diag[col].max()
#         _dmin  = df_diag[col].min()
#         if log and len(_dvals) > 0 and _dvals.min() > 0 and _dmax > 0:
#             norm = mcolors.LogNorm(vmin=_dvals.min(), vmax=_dmax)
#         else:
#             norm = mcolors.Normalize(
#                 vmin=_dmin if _dmin != _dmax else 0,
#                 vmax=_dmax if _dmax != _dmin else 1
#             )
#         sc = ax_.scatter(df_diag["X"], df_diag["Y"],
#                          c=df_diag[col], cmap=cmap,
#                          s=50, alpha=0.75, norm=norm, edgecolors="none")
#         _overlay(ax_)
#         cbar = plt.colorbar(sc, ax=ax_, pad=0.02)
#         cbar.set_label(label, rotation=270, labelpad=18, fontsize=13)
#         ax_.set_title(title, fontsize=16, fontweight="bold", pad=10)
#         for spine in ax_.spines.values():
#             spine.set_visible(False)
#         ax_.set_xticks([]); ax_.set_yticks([])
#         ax_.legend(fontsize=10, frameon=False)

#     plt.tight_layout()
#     fname2 = "heatmap_demand_vs_lambda.png"
#     if title_suffix:
#         fname2 = f"heatmap_demand_vs_lambda_{title_suffix.replace(' ','_')}.png"
#     _savefig(fig2, os.path.join(output_dir, fname2))


# # ── Financial summary multi-year ───────────────────────────────────────────────
# # FIX 2 applied here: year-y energy is capped at the physical hardware ceiling.

# def plot_financial_summary_multiyear(results, output_dir, title_suffix=""):
#     df_dwc, df_scs, meta, _ = _prepare(results)
#     have_dwc = not df_dwc.empty
#     have_scs = not df_scs.empty
#     if not have_dwc and not have_scs:
#         print("  Skipping financial_summary_multiyear.png (no assets selected)")
#         return

#     T      = results["project_life"]
#     g_d       = float(meta.get("g_d",       0.0))
#     g_price_u = None   # set per-unit below from tech type
#     g_var_m   = float(meta.get("g_var",     0.0))
#     g_om_m    = float(meta.get("g_om",      0.0))
#     g_rev_dwc_m = float(meta.get("g_rev_dwc", 0.0))
#     g_rev_scs_m = float(meta.get("g_rev_scs", 0.0))
#     alpha  = results["alpha"]
#     p_charge_dwc = float(meta.get("p_charge_dwc", 0.40))
#     p_charge_scs = float(meta.get("p_charge_scs", 0.40))

#     # Physical ceiling values needed for FIX 2
#     H_op_plot  = float(meta.get("h_op", 8760.0))
#     util_scs   = float(meta.get("utilization_rate_scs", 1.0))
#     n_stalls_p = float(meta.get("n_scs_stalls", 6))
#     p_stall_p  = float(meta.get("p_scs_per_stall_kw", 250.0))

#     mid_yr = T // 2 + 1
#     years  = [1, mid_yr, T]
#     yr_labels = [f"Year 1\n(Base)", f"Year {mid_yr}\n(Mid)", f"Year {T}\n(End)"]

#     units = []
#     if have_dwc:
#         for _, row in df_dwc.iterrows():
#             units.append({
#                 "label"            : f"DWC {row['Segment_ID']}",
#                 "tech"             : "DWC",
#                 "e_del_base"       : float(row["E_del_DWC"]),
#                 "fix_om"           : float(row["Lambda"]) * float(meta["c_om_base_dwc"]),
#                 "lcoe"             : float(row["LCOE"]),
#                 "capex"            : float(row["Capex_Total"]),
#                  "p_charge"         : p_charge_dwc,
#                 "g_rev_unit"       : g_rev_dwc_m,
#                 "g_var_unit"       : g_var_m,
#                 "g_om_unit"        : g_om_m,
#                 # FIX 2: physical ceiling for DWC = MW_Load × H_op × 1000
#                 "hw_ceil_physical" : float(row["MW_Load"]) * H_op_plot * MW_TO_KW,
#             })
#     if have_scs:
#         for _, row in df_scs.iterrows():
#             units.append({
#                 "label"            : f"SCS {row['SCS_ID']}",
#                 "tech"             : "SCS",
#                 "e_del_base"       : float(row["E_del_SCS"]),
#                 "fix_om"           : float(row["Lambda"]) * float(meta["c_om_base_scs"]),
#                 "lcoe"             : float(row["LCOE"]),
#                 "capex"            : float(row["Capex_Total"]),
#                 "p_charge"         : p_charge_scs,
#                 "g_rev_unit"       : g_rev_scs_m,
#                 "g_var_unit"       : g_var_m,
#                 "g_om_unit"        : g_om_m,
#                 # FIX 2: physical ceiling for SCS includes utilisation rate
#                 "hw_ceil_physical" : n_stalls_p * p_stall_p * H_op_plot * util_scs,
#             })

#     n_units = len(units)
#     fig, axes = plt.subplots(
#         n_units, 3,
#         figsize=(16, max(4.5 * n_units, 6)),
#         squeeze=False
#     )
#     fig.suptitle(
#         "Per-Unit Annual Financial Summary — Year 1 / Mid / End\n"
#         "(instantaneous annual $, not discounted; energy grows at g_d each year)",
#         fontweight="bold", fontsize=13, y=1.01
#     )

#     bar_labels = ["Revenue", "Opex\n(excl. capex)", "Op. Income\n(pre-capex)"]

#     for row_idx, unit in enumerate(units):
#         tech    = unit["tech"]
#         c_main  = C_DWC if tech == "DWC" else C_SCS
#         c_rev   = C_REV
#         c_opex  = C_OPEX
#         c_prof  = C_PROFIT
#         cap_ann = alpha * unit["capex"]

#         # Pre-compute axis range across all years
#         y_max, y_min = 0.0, 0.0
#         for yr in years:
#             fac      = (1 + g_d) ** (yr - 1)
#             # FIX 2: cap at physical ceiling
#             e_y      = min(unit["e_del_base"] * fac, unit["hw_ceil_physical"])
#             rev      = unit["p_charge"] * e_y
#             var_om   = unit["lcoe"] * e_y
#             opex     = unit["fix_om"] + var_om
#             profit   = rev - opex
#             y_max    = max(y_max, rev, opex, profit, cap_ann)
#             y_min    = min(y_min, profit)

#         y_range = y_max - y_min
#         y_lo = y_min - y_range * 0.12 if y_min < 0 else 0.0
#         y_hi = y_max + y_range * 0.20

#         for col_idx, (yr, yr_label) in enumerate(zip(years, yr_labels)):
#             ax = axes[row_idx, col_idx]

#             fac_demand = (1 + g_d)                ** (yr - 1)
#             fac_price  = (1 + unit["g_rev_unit"]) ** (yr - 1)
#             fac_lcoe   = (1 + unit["g_var_unit"]) ** (yr - 1)
#             fac_om     = (1 + unit["g_om_unit"])  ** (yr - 1)

#             # Energy volume grows with demand; price per kWh grows separately
#             e_y    = min(unit["e_del_base"] * fac_demand, unit["hw_ceil_physical"])
#             rev    = unit["p_charge"] * fac_price * e_y
#             var_om = unit["lcoe"]    * fac_lcoe   * e_y
#             opex   = unit["fix_om"]  * fac_om + var_om
#             profit = rev - opex

#             vals   = [rev, opex, profit]
#             colors = [c_rev, c_opex, c_prof if profit >= 0 else C_CAPEX]

#             x_pos = np.arange(3)
#             bars  = ax.bar(x_pos, vals, color=colors, width=0.55,
#                            alpha=0.85, edgecolor="white", linewidth=0.6)

#             for bar, v in zip(bars, vals):
#                 sign = 1 if v >= 0 else -1
#                 va   = "bottom" if v >= 0 else "top"
#                 offset = y_range * 0.03 * sign
#                 fmt = f"${v/1e3:.1f}K" if abs(v) >= 1000 else f"${v:.0f}"
#                 ax.text(
#                     bar.get_x() + bar.get_width() / 2,
#                     v + offset,
#                     fmt, ha="center", va=va,
#                     fontsize=9, fontweight="bold",
#                     color=colors[list(vals).index(v)]
#                 )

#             ax.axhline(cap_ann, color=C_CAPEX, lw=1.4, ls="--", alpha=0.75,
#                        label=f"Ann. capex ${cap_ann/1e3:.1f}K/yr")
#             ax.axhline(0, color="black", lw=0.7)

#             ax.text(0.97, 0.97,
#                     f"E_del: {e_y/1e3:.0f} MWh\n×{fac_demand:.2f} base",
#                     transform=ax.transAxes, ha="right", va="top",
#                     fontsize=8, color="#555555",
#                     bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
#                               edgecolor="#cccccc", alpha=0.9))

#             ax.set_xticks(x_pos)
#             ax.set_xticklabels(bar_labels, fontsize=10)
#             ax.set_ylim(y_lo, y_hi)
#             ax.yaxis.set_major_formatter(
#                 mticker.FuncFormatter(
#                     lambda v, _: f"${v/1e3:.0f}K" if abs(v) >= 1000 else f"${v:.0f}"
#                 )
#             )
#             ax.tick_params(axis="x", length=0)

#             if row_idx == 0:
#                 ax.set_title(yr_label, fontweight="bold", fontsize=11)

#             if col_idx == 0:
#                 ax.set_ylabel(unit["label"], fontweight="bold",
#                               fontsize=10, color=c_main)

#             if row_idx == 0 and col_idx == 2:
#                 ax.legend(fontsize=8, frameon=False, loc="upper left")

#             ax.yaxis.grid(True, alpha=0.20, ls="--", zorder=0)

#     plt.tight_layout()
#     fname = "financial_summary_multiyear.png"
#     if title_suffix:
#         fname = f"financial_summary_multiyear_{title_suffix.replace(' ', '_')}.png"
#     _savefig(fig, os.path.join(output_dir, fname))


# # ══════════════════════════════════════════════════════════════════════════════
# # MASTER SINGLE-RUN WRAPPER
# # ══════════════════════════════════════════════════════════════════════════════

# def generate_all_plots(results, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"\n── Single-run plots → {output_dir}/")
#     plot_deployment_map(results, output_dir, show_existing=False)
#     plot_deployment_map(results, output_dir, show_existing=True)
#     plot_traffic_split(results, output_dir)
#     plot_total_units(results, output_dir)
#     plot_capex_revenue_profit(results, output_dir)
#     plot_breakeven(results, output_dir)
#     plot_utilization(results, output_dir)
#     plot_financial_summary_multiyear(results, output_dir)
#     plot_heatmaps(results, output_dir)
#     generate_analytics_plots(results, output_dir)  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#     print(f"  All single-run plots complete.")


# # ══════════════════════════════════════════════════════════════════════════════
# # SENSITIVITY PLOTS — all unchanged from Main_plots_new.py
# # ══════════════════════════════════════════════════════════════════════════════

# def generate_eta_split_plots(results_list, param_values, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"\n── eta_split sensitivity plots → {output_dir}/")
#     if not results_list:
#         print("  No results — skipping.")
#         return

#     summs = [_summary(r) for r in results_list]
#     eta_d = [s["eta_dwc"] for s in summs]
#     eta_s = [s["eta_scs"] for s in summs]
#     xlbls = [f"{d:.2f}\n(η$^S$={s:.2f})" for d, s in zip(eta_d, eta_s)]
#     x     = np.arange(len(results_list))

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     ax.plot(eta_d, [s["n_dwc"] for s in summs], "o-", color=C_DWC,
#             lw=2, ms=7, label="DWC segments")
#     ax.plot(eta_d, [s["n_scs"] for s in summs], "s--", color=C_SCS,
#             lw=2, ms=7, label="SCS stations")
#     ax.set_xlabel(r"$\eta^{DWC}$")
#     ax.set_ylabel("Units selected")
#     ax.set_title("Infrastructure Selection vs Traffic-Capture Split", fontweight="bold")
#     ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "eta_units_selected.png"))

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     rev_dwc_M = [s["rev_dwc"] / 1e6 for s in summs]
#     rev_scs_M = [s["rev_scs"] / 1e6 for s in summs]
#     rev_tot_M = [a + b for a, b in zip(rev_dwc_M, rev_scs_M)]
#     bw = 0.35
#     ax.bar(x - bw / 2, rev_dwc_M, bw, color=C_DWC, alpha=0.80, label="DWC")
#     ax.bar(x + bw / 2, rev_scs_M, bw, color=C_SCS, alpha=0.80, label="SCS")
#     ax.plot(x, rev_tot_M, "o-", color=C_REV, lw=2, ms=6, label="Total", zorder=5)
#     ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=9)
#     ax.set_xlabel(r"$\eta^{DWC}$  ($\eta^{SCS}$)")
#     ax.set_ylabel("Annual revenue  ($M / yr, base-year)")
#     ax.set_title("Revenue vs Traffic-Capture Split", fontweight="bold")
#     ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "eta_revenue.png"))

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     alpha0   = results_list[0]["alpha"]
#     prof_dwc = [(s["rev_dwc"] - s["opex_dwc"] - s["cap_dwc"] * alpha0) / 1e6 for s in summs]
#     prof_scs = [(s["rev_scs"] - s["opex_scs"] - s["cap_scs"] * alpha0) / 1e6 for s in summs]
#     prof_tot = [a + b for a, b in zip(prof_dwc, prof_scs)]
#     ax.bar(x - bw / 2, prof_dwc, bw, color=C_DWC, alpha=0.80, label="DWC")
#     ax.bar(x + bw / 2, prof_scs, bw, color=C_SCS, alpha=0.80, label="SCS")
#     ax.plot(x, prof_tot, "o-", color=C_PROFIT, lw=2, ms=6, label="Total", zorder=5)
#     ax.axhline(0, color="black", lw=0.8, ls=":")
#     for xi, nd, ns, pt in zip(x,
#                                [s["n_dwc"] for s in summs],
#                                [s["n_scs"] for s in summs],
#                                prof_tot):
#         ax.text(xi, pt + abs(max(prof_tot) - min(prof_tot)) * 0.03,
#                 f"D:{nd} S:{ns}", ha="center", fontsize=8, color="dimgray")
#     ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=9)
#     ax.set_xlabel(r"$\eta^{DWC}$  ($\eta^{SCS}$)")
#     ax.set_ylabel("Annual profit  ($M / yr, base-year)")
#     ax.set_title("Profit vs Traffic-Capture Split", fontweight="bold")
#     ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "eta_profit.png"))

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     proj_life = results_list[0]["project_life"]
#     _cap      = proj_life * 1.1
#     pbs_d = [s["pb_disc"]   for s in summs]
#     pbs_s = [s["pb_simple"] for s in summs]
#     ax.plot(eta_d, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_d],
#             "o-",  color=C_CAPEX, lw=2, ms=7, label="Discounted payback")
#     ax.plot(eta_d, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_s],
#             "s--", color=C_OPEX,  lw=2, ms=7, label="Simple payback")
#     ax.axhline(proj_life, color="grey", lw=1.0, ls=":", label=f"Project life {proj_life} yr")
#     ax.fill_between(eta_d, proj_life, _cap, alpha=0.07, color=C_CAPEX)
#     ax.set_xlabel(r"$\eta^{DWC}$")
#     ax.set_ylabel("Payback year")
#     ax.set_ylim(0, _cap * 1.05)
#     ax.set_title("Breakeven vs Traffic-Capture Efficiency", fontweight="bold")
#     ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "eta_breakeven.png"))

#     n_sc = len(results_list)
#     ncols_pie = min(n_sc, 4)
#     nrows_pie = (n_sc + ncols_pie - 1) // ncols_pie
#     fig_pie, axes_pie = plt.subplots(nrows_pie, ncols_pie,
#                                      figsize=(5 * ncols_pie, 5 * nrows_pie))
#     axes_pie = np.array(axes_pie).flatten()
#     for idx, (res, ed, es) in enumerate(zip(results_list, eta_d, eta_s)):
#         ax_p = axes_pie[idx]
#         nd   = int(res["meta"]["n_dwc_selected"])
#         ns   = int(res["meta"]["n_scs_selected"])
#         sv, sc_ = [], []
#         if nd > 0: sv.append(nd); sc_.append(C_DWC)
#         if ns > 0: sv.append(ns); sc_.append(C_SCS)
#         if sv:
#             ax_p.pie(sv, colors=sc_, startangle=140,
#                      autopct=lambda p: f"{round(p*sum(sv)/100)}",
#                      pctdistance=0.5,
#                      textprops={"fontweight": "bold", "color": "white", "fontsize": 14})
#         ax_p.set_title(f"η$^D$={ed:.2f} η$^S$={es:.2f}\n"
#                        f"DWC={nd} SCS={ns}", fontsize=11, fontweight="bold")
#     for idx in range(n_sc, len(axes_pie)):
#         axes_pie[idx].set_visible(False)
#     fig_pie.suptitle("Units Selected — η Split Scenarios", fontweight="bold")
#     plt.tight_layout()
#     _savefig(fig_pie, os.path.join(output_dir, "eta_units_pie.png"))

#     for res, ed, es in zip(results_list, eta_d, eta_s):
#         suffix = f"etaDWC{ed:.2f}_etaSCS{es:.2f}"
#         plot_deployment_map(res, output_dir, show_existing=False, title_suffix=suffix)

#     print(f"  eta_split plots complete.")


# def generate_ev_growth_plots(results_list, param_values, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"\n── ev_growth sensitivity plots → {output_dir}/")
#     if not results_list:
#         return

#     summs  = [_summary(r) for r in results_list]
#     g_evs  = param_values
#     scenario_labels = [f"g_ev = {v:.0%}" for v in g_evs]
#     meta0      = results_list[0]["meta"]
#     rho_ev     = float(meta0.get("rho_ev", 0.02))
#     proj_life  = results_list[0]["project_life"]
#     colors3    = [C_DWC, C_SCS, C_CAPEX, C_OPEX, C_PROFIT]

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     years_t = np.arange(0, proj_life + 1)
#     for i, (g_ev, lbl) in enumerate(zip(g_evs, scenario_labels)):
#         ev_share = rho_ev * (1 + g_ev)**years_t * 100
#         ax.plot(years_t, ev_share, color=colors3[i % len(colors3)], lw=2, label=lbl)
#     ax.set_xlabel("Year")
#     ax.set_ylabel("EV share of traffic  (%)")
#     ax.set_title("EV Penetration Trajectories", fontweight="bold")
#     ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "ev_penetration_curves.png"))

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     ax.plot(g_evs, [s["n_dwc"] for s in summs], "o-", color=C_DWC,
#             lw=2, ms=7, label="DWC segments")
#     ax.plot(g_evs, [s["n_scs"] for s in summs], "s--", color=C_SCS,
#             lw=2, ms=7, label="SCS stations")
#     ax.set_xlabel(r"EV growth rate  $g_{ev}$")
#     ax.set_ylabel("Units selected")
#     ax.set_title("Infrastructure Selection vs EV Growth Scenario", fontweight="bold")
#     ax.legend(); ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
#     ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "ev_units_selected.png"))

#     fig, axes_ = plt.subplots(1, 2, figsize=(13, 5))
#     for i, (res, g_ev, lbl) in enumerate(zip(results_list, g_evs, scenario_labels)):
#         s_     = summs[i]
#         meta_  = res["meta"]
#         g_rev_ = float(meta_.get("g_rev", 0.0))
#         yrs    = np.arange(1, proj_life + 1)
#         df_d   = res.get("df_dwc", pd.DataFrame())
#         df_s   = res.get("df_scs", pd.DataFrame())

#         base_var = 0.0
#         if not df_d.empty:
#             base_var += float((df_d["LCOE"] * df_d["E_del_DWC"]).sum())
#         if not df_s.empty:
#             base_var += float((df_s["LCOE"] * df_s["E_del_SCS"]).sum())

#         base_fix = 0.0
#         if not df_d.empty:
#             base_fix += float((df_d["Lambda"] * meta_["c_om_base_dwc"]).sum())
#         if not df_s.empty:
#             base_fix += float((df_s["Lambda"] * meta_["c_om_base_scs"]).sum())

#         base_rev = s_["rev_dwc"] + s_["rev_scs"]
#         rev_yr   = base_rev * (1 + g_rev_)**(yrs - 1) / 1e6
#         var_yr   = base_var * (1 + g_rev_)**(yrs - 1) / 1e6
#         fix_yr   = base_fix / 1e6
#         net_yr   = rev_yr - var_yr - fix_yr
#         c = colors3[i % len(colors3)]
#         axes_[0].plot(yrs, rev_yr, color=c, lw=2, label=lbl)
#         axes_[1].plot(yrs, net_yr, color=c, lw=2, label=lbl)

#     axes_[0].set_xlabel("Year")
#     axes_[0].set_ylabel("Annual revenue  ($M / yr)")
#     axes_[0].set_title("(a)  Revenue over project life", fontweight="bold")
#     axes_[0].legend(fontsize=9); axes_[0].yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     axes_[1].set_xlabel("Year")
#     axes_[1].set_ylabel("Net operating income  ($M / yr)")
#     axes_[1].set_title("(b)  Net op. income over project life", fontweight="bold")
#     axes_[1].axhline(0, color="black", lw=0.8, ls=":")
#     axes_[1].legend(fontsize=9); axes_[1].yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     fig.suptitle("Year-by-Year Revenue & Profit by EV Growth Scenario", fontweight="bold")
#     plt.tight_layout()
#     _savefig(fig, os.path.join(output_dir, "ev_revenue_profit_timeseries.png"))

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     _cap  = proj_life * 1.1
#     pbs_d = [s["pb_disc"]   for s in summs]
#     pbs_s = [s["pb_simple"] for s in summs]
#     ax.plot(g_evs, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_d],
#             "o-",  color=C_CAPEX, lw=2, ms=7, label="Discounted payback")
#     ax.plot(g_evs, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_s],
#             "s--", color=C_OPEX,  lw=2, ms=7, label="Simple payback")
#     ax.axhline(proj_life, color="grey", lw=1.0, ls=":", label=f"Project life {proj_life} yr")
#     ax.fill_between(g_evs, proj_life, _cap, alpha=0.07, color=C_CAPEX)
#     ax.set_xlabel(r"EV growth rate  $g_{ev}$")
#     ax.set_ylabel("Payback year")
#     ax.set_ylim(0, _cap * 1.05)
#     ax.set_title("Breakeven vs EV Growth Scenario", fontweight="bold")
#     ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
#     ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "ev_breakeven.png"))

#     for res, g_ev, lbl in zip(results_list, g_evs, scenario_labels):
#         suffix = lbl.replace(" ", "_").replace("=", "").replace("%", "pct")
#         plot_deployment_map(res, output_dir, show_existing=False, title_suffix=suffix)

#     print(f"  ev_growth plots complete.")


# def generate_traffic_growth_plots(results_list, param_values, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"\n── traffic_growth sensitivity plots → {output_dir}/")
#     if not results_list:
#         return

#     summs   = [_summary(r) for r in results_list]
#     g_aadts = param_values
#     scenario_labels = [f"g_aadt = {v:.1%}" for v in g_aadts]
#     proj_life = results_list[0]["project_life"]
#     colors3   = [C_DWC, C_SCS, C_CAPEX, C_OPEX, C_PROFIT]

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     years_t = np.arange(0, proj_life + 1)
#     for i, (g, lbl) in enumerate(zip(g_aadts, scenario_labels)):
#         ax.plot(years_t, (1 + g)**years_t,
#                 color=colors3[i % len(colors3)], lw=2, label=lbl)
#     ax.set_xlabel("Year")
#     ax.set_ylabel("AADT growth index  (base = 1)")
#     ax.set_title("AADT Growth Trajectories", fontweight="bold")
#     ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "traffic_growth_curves.png"))

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     ax.plot(g_aadts, [s["n_dwc"] for s in summs], "o-", color=C_DWC,
#             lw=2, ms=7, label="DWC segments")
#     ax.plot(g_aadts, [s["n_scs"] for s in summs], "s--", color=C_SCS,
#             lw=2, ms=7, label="SCS stations")
#     ax.set_xlabel(r"AADT growth rate  $g_{AADT}$")
#     ax.set_ylabel("Units selected")
#     ax.set_title("Infrastructure Selection vs AADT Growth", fontweight="bold")
#     ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
#     ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "traffic_units_selected.png"))

#     fig, axes_ = plt.subplots(1, 2, figsize=(13, 5))
#     for i, (res, g_aadt, lbl) in enumerate(zip(results_list, g_aadts, scenario_labels)):
#         s_     = summs[i]
#         meta_  = res["meta"]
#         g_rev_ = float(meta_.get("g_rev", 0.0))
#         yrs    = np.arange(1, proj_life + 1)
#         df_d   = res.get("df_dwc", pd.DataFrame())
#         df_s   = res.get("df_scs", pd.DataFrame())

#         base_var = 0.0
#         if not df_d.empty:
#             base_var += float((df_d["LCOE"] * df_d["E_del_DWC"]).sum())
#         if not df_s.empty:
#             base_var += float((df_s["LCOE"] * df_s["E_del_SCS"]).sum())

#         base_fix = 0.0
#         if not df_d.empty:
#             base_fix += float((df_d["Lambda"] * meta_["c_om_base_dwc"]).sum())
#         if not df_s.empty:
#             base_fix += float((df_s["Lambda"] * meta_["c_om_base_scs"]).sum())

#         base_rev = s_["rev_dwc"] + s_["rev_scs"]
#         rev_yr   = base_rev * (1 + g_rev_)**(yrs - 1) / 1e6
#         var_yr   = base_var * (1 + g_rev_)**(yrs - 1) / 1e6
#         fix_yr   = base_fix / 1e6
#         net_yr   = rev_yr - var_yr - fix_yr
#         c = colors3[i % len(colors3)]
#         axes_[0].plot(yrs, rev_yr, color=c, lw=2, label=lbl)
#         axes_[1].plot(yrs, net_yr, color=c, lw=2, label=lbl)

#     axes_[0].set_xlabel("Year")
#     axes_[0].set_ylabel("Annual revenue  ($M / yr)")
#     axes_[0].set_title("(a)  Revenue over project life", fontweight="bold")
#     axes_[0].legend(fontsize=9); axes_[0].yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     axes_[1].set_xlabel("Year")
#     axes_[1].set_ylabel("Net operating income  ($M / yr)")
#     axes_[1].set_title("(b)  Net operating income", fontweight="bold")
#     axes_[1].axhline(0, color="black", lw=0.8, ls=":")
#     axes_[1].legend(fontsize=9); axes_[1].yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     fig.suptitle("Year-by-Year Revenue & Profit by AADT Growth Scenario", fontweight="bold")
#     plt.tight_layout()
#     _savefig(fig, os.path.join(output_dir, "traffic_revenue_profit_timeseries.png"))

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     _cap  = proj_life * 1.1
#     pbs_d = [s["pb_disc"]   for s in summs]
#     pbs_s = [s["pb_simple"] for s in summs]
#     ax.plot(g_aadts, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_d],
#             "o-",  color=C_CAPEX, lw=2, ms=7, label="Discounted payback")
#     ax.plot(g_aadts, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_s],
#             "s--", color=C_OPEX,  lw=2, ms=7, label="Simple payback")
#     ax.axhline(proj_life, color="grey", lw=1.0, ls=":", label=f"Project life {proj_life} yr")
#     ax.fill_between(g_aadts, proj_life, _cap, alpha=0.07, color=C_CAPEX)
#     ax.set_xlabel(r"AADT growth rate  $g_{AADT}$")
#     ax.set_ylabel("Payback year")
#     ax.set_ylim(0, _cap * 1.05)
#     ax.set_title("Breakeven vs AADT Growth Scenario", fontweight="bold")
#     ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
#     ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "traffic_breakeven.png"))

#     for res, g, lbl in zip(results_list, g_aadts, scenario_labels):
#         suffix = lbl.replace(" ", "_").replace("=", "").replace("%", "pct")
#         plot_deployment_map(res, output_dir, show_existing=False, title_suffix=suffix)

#     print(f"  traffic_growth plots complete.")


# def generate_dwc_cost_plots(results_list, param_values, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"\n── dwc_cost sensitivity plots → {output_dir}/")
#     if not results_list:
#         return

#     summs     = [_summary(r) for r in results_list]
#     cost_vals = param_values
#     cost_M    = [cv / 1e6 for cv in cost_vals]
#     proj_life = results_list[0]["project_life"]
#     alpha0    = results_list[0]["alpha"]

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     c_dwc_M = [float(r["params"]["C_DWC_base"]) / 1e6 for r in results_list]
#     ax.plot(cost_M, c_dwc_M, "o-", color=C_CAPEX, lw=2, ms=7)
#     ax.set_xlabel("DWC base cost  ($M / 5-mile segment)")
#     ax.set_ylabel(r"$C^{DWC}_{base}$  ($M)")
#     ax.set_title("DWC Segment Cost vs Sweep Level", fontweight="bold")
#     ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "dwc_cost_per_segment.png"))

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     ax.plot(cost_M, [s["n_dwc"] for s in summs], "o-", color=C_DWC,
#             lw=2, ms=7, label="DWC segments")
#     ax.plot(cost_M, [s["n_scs"] for s in summs], "s--", color=C_SCS,
#             lw=2, ms=7, label="SCS stations")
#     ax.set_xlabel("DWC base cost  ($M / 5-mi)")
#     ax.set_ylabel("Units selected")
#     ax.set_title("Infrastructure Selection vs DWC Capital Cost", fontweight="bold")
#     ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "dwc_cost_units.png"))

#     fig, ax = plt.subplots(figsize=(8, 5))
#     rev_tot_M = [(s["rev_dwc"] + s["rev_scs"]) / 1e6 for s in summs]
#     rev_dwc_M = [s["rev_dwc"] / 1e6 for s in summs]
#     rev_scs_M = [s["rev_scs"] / 1e6 for s in summs]
#     prof_dwc  = [(s["rev_dwc"] - s["opex_dwc"] - s["cap_dwc"] * alpha0) / 1e6 for s in summs]
#     prof_scs  = [(s["rev_scs"] - s["opex_scs"] - s["cap_scs"] * alpha0) / 1e6 for s in summs]
#     prof_tot  = [a + b for a, b in zip(prof_dwc, prof_scs)]

#     ax.plot(cost_M, rev_tot_M,  "o-",  color=C_REV,    lw=2, ms=7, label="Total revenue")
#     ax.plot(cost_M, rev_dwc_M,  "o--", color=C_DWC,    lw=1.5, alpha=0.7, label="DWC revenue")
#     ax.plot(cost_M, rev_scs_M,  "o--", color=C_SCS,    lw=1.5, alpha=0.7, label="SCS revenue")
#     ax.plot(cost_M, prof_tot,   "s-",  color=C_PROFIT,  lw=2, ms=7, label="Total profit")
#     ax.plot(cost_M, prof_dwc,   "s--", color=C_DWC,    lw=1.0, alpha=0.5, label="DWC profit")
#     ax.plot(cost_M, prof_scs,   "s--", color=C_SCS,    lw=1.0, alpha=0.5, label="SCS profit")
#     ax.axhline(0, color="black", lw=0.8, ls=":")
#     for xv, nd, ns, pt in zip(cost_M,
#                                [s["n_dwc"] for s in summs],
#                                [s["n_scs"] for s in summs],
#                                prof_tot):
#         ax.text(xv, pt, f" D:{nd}\n S:{ns}", fontsize=8, color="dimgray", va="center")
#     ax.set_xlabel("DWC base cost  ($M / 5-mi)")
#     ax.set_ylabel("$M / yr  (base-year)")
#     ax.set_title("Revenue & Profit vs DWC Capital Cost", fontweight="bold")
#     ax.legend(fontsize=9, ncol=2); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "dwc_cost_revenue_profit.png"))

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     _cap  = proj_life * 1.1
#     pbs_d = [s["pb_disc"]   for s in summs]
#     pbs_s = [s["pb_simple"] for s in summs]
#     ax.plot(cost_M, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_d],
#             "o-",  color=C_CAPEX, lw=2, ms=7, label="Discounted payback")
#     ax.plot(cost_M, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_s],
#             "s--", color=C_OPEX,  lw=2, ms=7, label="Simple payback")
#     ax.axhline(proj_life, color="grey", lw=1.0, ls=":", label=f"Project life {proj_life} yr")
#     ax.fill_between(cost_M, proj_life, _cap, alpha=0.07, color=C_CAPEX)
#     ax.set_xlabel("DWC base cost  ($M / 5-mi)")
#     ax.set_ylabel("Payback year")
#     ax.set_ylim(0, _cap * 1.05)
#     ax.set_title("Breakeven vs DWC Capital Cost", fontweight="bold")
#     ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "dwc_cost_breakeven.png"))

#     for res, cv_m in zip(results_list, cost_M):
#         suffix = f"cost_{cv_m:.1f}M"
#         plot_deployment_map(res, output_dir, show_existing=False, title_suffix=suffix)

#     print(f"  dwc_cost plots complete.")


# def generate_custom_plots(results_list, param_name, param_values, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"\n── custom sensitivity plots → {output_dir}/")
#     if not results_list:
#         return

#     summs     = [_summary(r) for r in results_list]
#     proj_life = results_list[0]["project_life"]
#     alpha0    = results_list[0]["alpha"]

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     ax.plot(param_values, [s["n_dwc"] for s in summs], "o-", color=C_DWC,
#             lw=2, ms=7, label="DWC segments")
#     ax.plot(param_values, [s["n_scs"] for s in summs], "s--", color=C_SCS,
#             lw=2, ms=7, label="SCS stations")
#     ax.set_xlabel(param_name)
#     ax.set_ylabel("Units selected")
#     ax.set_title(f"Infrastructure Selection vs {param_name}", fontweight="bold")
#     ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "custom_units.png"))

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     rev_M  = [(s["rev_dwc"] + s["rev_scs"]) / 1e6 for s in summs]
#     prof_M = [(s["rev_dwc"] + s["rev_scs"] - s["opex_dwc"] - s["opex_scs"]
#                - (s["cap_dwc"] + s["cap_scs"]) * alpha0) / 1e6
#               for s in summs]
#     ax.plot(param_values, rev_M,  "o-",  color=C_REV,    lw=2, ms=7, label="Revenue")
#     ax.plot(param_values, prof_M, "s--", color=C_PROFIT, lw=2, ms=7, label="Profit")
#     ax.axhline(0, color="black", lw=0.8, ls=":")
#     ax.set_xlabel(param_name)
#     ax.set_ylabel("$M / yr  (base-year)")
#     ax.set_title(f"Revenue & Profit vs {param_name}", fontweight="bold")
#     ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "custom_revenue_profit.png"))

#     fig, ax = plt.subplots(figsize=(7, 4.5))
#     _cap  = proj_life * 1.1
#     pbs_d = [s["pb_disc"]   for s in summs]
#     pbs_s = [s["pb_simple"] for s in summs]
#     ax.plot(param_values, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_d],
#             "o-",  color=C_CAPEX, lw=2, ms=7, label="Discounted payback")
#     ax.plot(param_values, [min(v, _cap) if not np.isnan(v) else _cap for v in pbs_s],
#             "s--", color=C_OPEX,  lw=2, ms=7, label="Simple payback")
#     ax.axhline(proj_life, color="grey", lw=1.0, ls=":")
#     ax.fill_between(param_values, proj_life, _cap, alpha=0.07, color=C_CAPEX)
#     ax.set_xlabel(param_name)
#     ax.set_ylabel("Payback year")
#     ax.set_ylim(0, _cap * 1.05)
#     ax.set_title(f"Breakeven vs {param_name}", fontweight="bold")
#     ax.legend(); ax.yaxis.grid(True, alpha=0.25, ls="--", zorder=0)
#     _savefig(fig, os.path.join(output_dir, "custom_breakeven.png"))

#     safe_pname = param_name.replace(" ", "_")[:20]
#     for res, pv in zip(results_list, param_values):
#         suffix = f"{safe_pname}_{pv}"
#         plot_deployment_map(res, output_dir, show_existing=False, title_suffix=suffix)

#     print(f"  custom plots complete.")


# # ── Master sensitivity dispatcher ─────────────────────────────────────────────

# def generate_sensitivity_plots(results_list, param_name, param_values, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     if not results_list:
#         print("  generate_sensitivity_plots: no results — skipping.")
#         return

#     pn = param_name.lower()
#     if "eta_dwc" in pn or "traffic-capture" in pn or "eta_split" in pn:
#         generate_eta_split_plots(results_list, param_values, output_dir)
#     elif "ev penetration" in pn or "g_ev" in pn or "ev_growth" in pn:
#         generate_ev_growth_plots(results_list, param_values, output_dir)
#     elif "aadt" in pn or "g_aadt" in pn or "traffic_growth" in pn:
#         generate_traffic_growth_plots(results_list, param_values, output_dir)
#     elif ("dwc" in pn and "cost" in pn) or "capital cost" in pn:
#         generate_dwc_cost_plots(results_list, param_values, output_dir)
#     else:
#         generate_custom_plots(results_list, param_name, param_values, output_dir)

#     rows = []
#     for r, pv in zip(results_list, param_values):
#         s = _summary(r)
#         s["param_value"] = pv
#         rows.append(s)
#     pd.DataFrame(rows).to_csv(
#         os.path.join(output_dir, "SA_summary.csv"), index=False
#     )
#     print(f"  SA_summary.csv saved.")


# # ── Compatibility shims ────────────────────────────────────────────────────────

# def plot_sensitivity_eta_split(results_list, output_dir):
#     eta_d_vals = [float(r["meta"].get("eta_dwc", 0)) for r in results_list]
#     generate_eta_split_plots(results_list, eta_d_vals, output_dir)


# def _compute_run_summary(results):
#     return _summary(results)


# # ══════════════════════════════════════════════════════════════════════════════
# # CLI — mirrors the original plot_analytics.py CLI for standalone use
# # ══════════════════════════════════════════════════════════════════════════════

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Unified plots from optimizer CSV outputs.")
#     parser.add_argument("--results_dir", default=None)
#     parser.add_argument("--out_dir",     default=None)
#     args = parser.parse_args()

#     rd = args.results_dir
#     if rd is None:
#         cands = sorted(glob.glob("Results_*/"), key=os.path.getmtime, reverse=True)
#         if not cands:
#             print("ERROR: no Results_*/ directory found."); sys.exit(1)
#         rd = cands[0].rstrip("/")
#         print(f"Auto-discovered: {rd}")

#     _run_analytics_plots(rd, args.out_dir)

# # If running this code independently of Main.py:
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser(
# #         description="Generate plots from saved optimizer results.")
# #     parser.add_argument("--results_dir", default=None)
# #     parser.add_argument("--out_dir",     default=None)
# #     parser.add_argument("--analytics_only", action="store_true", 
# #                         help="Generate only analytics plots (legacy)")
# #     args = parser.parse_args()

# #     rd = args.results_dir
# #     if rd is None:
# #         cands = sorted(glob.glob("Results_*/"), key=os.path.getmtime, reverse=True)
# #         if not cands:
# #             print("ERROR: no Results_*/ directory found."); sys.exit(1)
# #         rd = cands[0].rstrip("/")
# #         print(f"Auto-discovered: {rd}")

# #     if args.analytics_only:
# #         _run_analytics_plots(rd, args.out_dir)
# #     else:
# #         results = load_results_from_saved_dir(rd)
# #         if args.out_dir is None:
# #             args.out_dir = f"Plots_{results['run_tag']}"
# #         generate_all_plots(results, args.out_dir)
