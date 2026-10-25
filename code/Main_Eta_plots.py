#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot diagnostics for the endogenous mixed DWC/SCS optimization.

The script reads the two existing exclusive-technology result folders and the
new endogenous mixed result folder. It does not rerun the optimization.

Outputs:
    Figure6c_delivered_cost_endogenous
    Figure6d_mixed_deployment_composition
    Figure6e_npv_gain_decomposition
    Figure6f_discounted_delivered_energy
    Figure6g_segment_traffic_allocation
    Figure6h_eta_dwc_by_segment

Each figure is saved as PNG and SVG. Supporting CSV summaries are also written.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import font_manager


# =============================================================================
# INPUTS AND OUTPUT
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = Path(
    os.environ.get(
        "DWC_UNCONSTRAINED_RESULTS_DIR",
        REPO_ROOT / "Results_Base_Run_Unconstrained",
    )
)

ALL_DWC_RESULTS_DIR = (
    RESULTS_ROOT
    / "Results_baseline_p500000_etaDWC1.00_etaSCS0.00_gev15pct_gaadt011bps"
)
ALL_SCS_RESULTS_DIR = (
    RESULTS_ROOT
    / "Results_baseline_p500000_etaDWC0.00_etaSCS1.00_gev15pct_gaadt011bps"
)
MIXED_RESULTS_DIR = (
    RESULTS_ROOT
    / "Results_baseline_p5000000_endogenous_segment_traffic_gev15pct_gaadt011bps"
)

RUNS = [
    ("DWC only", ALL_DWC_RESULTS_DIR),
    ("SCS only", ALL_SCS_RESULTS_DIR),
    ("Mixed", MIXED_RESULTS_DIR),
]

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR

# A technology share must exceed this threshold to count as a meaningful use of
# that technology on a segment. This avoids classifying solver-scale numerical
# residuals as a genuine fractional traffic split.
MEANINGFUL_SPLIT_THRESHOLD = 0.01


# =============================================================================
# FIGURE STYLE
# =============================================================================

FONT_FAMILY = "Arial"
FONT_SIZE_TICK = 5.5
FONT_SIZE_LEGEND = 5.5
FONT_SIZE_LABEL = 6.5
FONT_SIZE_OTHER = 5.5

FIG_FULL_WIDTH_IN = 7.2
FIG_PANEL_WIDTH_IN = FIG_FULL_WIDTH_IN / 3.0
FIG_HEIGHT_IN = 2.45
DPI = 600

C_DWC = "#075C47"
C_DWC_LIGHT = "#65A987"
C_SCS = "#CAA540"
C_SCS_LIGHT = "#E8D48A"
C_ALL = "#9E9E9E"
C_EDGE = "#111111"
C_GRID = "#B8B8B8"
C_CAPEX = "#075C47"
C_FIXED = "#65A987"
C_VARIABLE = "#D3E5CC"
C_POSITIVE = "#075C47"
C_NEGATIVE = "#CAA540"
C_UNSERVED = "#B8B8B8"


def configure_fonts() -> None:
    font_dirs = [
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "fonts"),
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "share", "fonts"),
        os.path.expanduser("~/.local/share/fonts"),
        os.path.expanduser("~/.fonts"),
    ]
    font_dirs.extend(
        str(path) for path in Path.home().glob("micromamba/pkgs/mscorefonts-*/fonts")
    )

    for font_dir in font_dirs:
        if not os.path.isdir(font_dir):
            continue
        for root, _, files in os.walk(font_dir):
            for filename in files:
                if filename.lower().endswith((".ttf", ".otf")):
                    font_manager.fontManager.addfont(os.path.join(root, filename))

    available = {font.name for font in font_manager.fontManager.ttflist}
    active_family = FONT_FAMILY if FONT_FAMILY in available else "DejaVu Sans"
    if active_family != FONT_FAMILY:
        print(f"WARNING: {FONT_FAMILY} not found; using {active_family}.")

    plt.rcParams.update(
        {
            "font.family": active_family,
            "font.sans-serif": [active_family],
            "font.size": FONT_SIZE_LABEL,
            "axes.titlesize": FONT_SIZE_LABEL,
            "axes.labelsize": FONT_SIZE_LABEL,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "legend.fontsize": FONT_SIZE_LEGEND,
            "axes.linewidth": 0.45,
            "xtick.major.width": 0.40,
            "ytick.major.width": 0.40,
            "xtick.major.size": 1.8,
            "ytick.major.size": 1.8,
            "axes.edgecolor": C_EDGE,
            "axes.labelcolor": C_EDGE,
            "xtick.color": C_EDGE,
            "ytick.color": C_EDGE,
            "text.color": C_EDGE,
            "figure.dpi": 150,
            "savefig.dpi": DPI,
            "savefig.facecolor": "none",
            "figure.facecolor": "none",
            "axes.facecolor": "none",
            "savefig.transparent": True,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": False,
            "hatch.linewidth": 0.18,
        }
    )


def style_axis(ax: plt.Axes, y_grid: bool = True) -> None:
    for spine in ("left", "right", "top", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.40)
        ax.spines[spine].set_color(C_EDGE)

    if y_grid:
        ax.yaxis.grid(
            True,
            color=C_GRID,
            linestyle=":",
            linewidth=0.45,
            alpha=0.95,
            zorder=0,
        )
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", pad=1.0)


def save_figure(fig: plt.Figure, base_name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = OUTPUT_DIR / f"{base_name}.{suffix}"
        fig.savefig(
            path,
            dpi=DPI,
            bbox_inches=None,
            transparent=True,
            facecolor="none",
            edgecolor="none",
        )
        print(f"Saved: {path}")
    plt.close(fig)


# =============================================================================
# DATA LOADING AND PORTFOLIO CALCULATIONS
# =============================================================================


def read_optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def numeric_column(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(0.0)


def meta_float(meta: dict, key: str, default: float = 0.0) -> float:
    value = meta.get(key, default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if not np.isfinite(value) else value


def npv_factor(growth_rate: float, discount_rate: float, project_life: int) -> float:
    ratio = (1.0 + growth_rate) / (1.0 + discount_rate)
    if abs(ratio - 1.0) < 1e-12:
        return float(project_life) / (1.0 + discount_rate)
    return (
        (1.0 / (1.0 + discount_rate))
        * (1.0 - ratio ** project_life)
        / (1.0 - ratio)
    )


def load_run(label: str, directory: Path) -> dict:
    if not directory.is_dir():
        raise FileNotFoundError(f"Results directory not found: {directory}")

    metadata_path = directory / "run_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Required metadata not found: {metadata_path}")

    return {
        "label": label,
        "directory": directory,
        "meta": pd.read_csv(metadata_path).iloc[0].to_dict(),
        "dwc": read_optional_csv(directory / "DWC_Selected_Segments.csv"),
        "scs": read_optional_csv(directory / "SCS_Selected_Stations.csv"),
    }


def technology_components(df: pd.DataFrame, tech: str, meta: dict) -> dict:
    if tech == "DWC":
        energy_column = "E_del_DWC"
        fixed_key = "c_om_base_dwc"
        gamma_revenue_key = "gamma_rev_dwc"
    elif tech == "SCS":
        energy_column = "E_del_SCS"
        fixed_key = "c_om_base_scs"
        gamma_revenue_key = "gamma_rev_scs"
    else:
        raise ValueError("tech must be 'DWC' or 'SCS'")

    if df.empty:
        return {
            "n": 0,
            "revenue": 0.0,
            "capex": 0.0,
            "fixed": 0.0,
            "variable": 0.0,
            "base_energy": 0.0,
        }

    gamma_revenue = meta_float(meta, gamma_revenue_key, 1.0)
    gamma_variable = meta_float(meta, "gamma_var", 1.0)
    gamma_fixed = meta_float(meta, "gamma_fix_om", 1.0)

    energy = numeric_column(df, energy_column)
    variable_base = numeric_column(df, "LCOE") * energy

    fixed_coefficient = meta_float(meta, fixed_key, np.nan)
    if np.isfinite(fixed_coefficient):
        fixed_base = numeric_column(df, "Lambda") * fixed_coefficient
    else:
        fixed_base = (
            numeric_column(df, "Ann_Opex") - variable_base
        ).clip(lower=0.0)

    return {
        "n": int(len(df)),
        "revenue": float((numeric_column(df, "Ann_Revenue") * gamma_revenue).sum()),
        "capex": float(numeric_column(df, "Capex_Total").sum()),
        "fixed": float((fixed_base * gamma_fixed).sum()),
        "variable": float((variable_base * gamma_variable).sum()),
        "base_energy": float(energy.sum()),
    }


def summarize_run(run: dict) -> dict:
    meta = run["meta"]
    dwc = technology_components(run["dwc"], "DWC", meta)
    scs = technology_components(run["scs"], "SCS", meta)

    discount_rate = meta_float(meta, "discount_rate", 0.07)
    project_life = int(round(meta_float(meta, "project_life", 30)))

    g_d = meta_float(meta, "g_d", np.nan)
    if not np.isfinite(g_d):
        g_aadt = meta_float(meta, "g_aadt", 0.0)
        g_ev = meta_float(meta, "g_ev", 0.0)
        g_d = (1.0 + g_aadt) * (1.0 + g_ev) - 1.0

    gamma_energy = npv_factor(g_d, discount_rate, project_life)
    base_energy = dwc["base_energy"] + scs["base_energy"]
    discounted_energy = base_energy * gamma_energy

    revenue = dwc["revenue"] + scs["revenue"]
    capex = dwc["capex"] + scs["capex"]
    fixed = dwc["fixed"] + scs["fixed"]
    variable = dwc["variable"] + scs["variable"]
    lifetime_cost = capex + fixed + variable
    npv = revenue - lifetime_cost

    return {
        "label": run["label"],
        "directory": str(run["directory"]),
        "n_dwc": dwc["n"],
        "n_scs": scs["n"],
        "revenue": revenue,
        "capex": capex,
        "fixed": fixed,
        "variable": variable,
        "lifetime_cost": lifetime_cost,
        "base_energy_kwh": base_energy,
        "discounted_energy_kwh": discounted_energy,
        "lcde_usd_per_kwh": (
            lifetime_cost / discounted_energy if discounted_energy > 0 else np.nan
        ),
        "npv": npv,
        "saved_objective": meta_float(meta, "obj_val", np.nan),
    }


def load_traffic_split() -> pd.DataFrame:
    path = MIXED_RESULTS_DIR / "Endogenous_Traffic_Split_By_Segment.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required traffic-split file not found: {path}")

    df = pd.read_csv(path)
    required = {
        "Segment_ID",
        "Annual_EV_Traffic_Pool",
        "ETA_DWC_Endogenous",
        "ETA_SCS_Endogenous",
        "ETA_Unserved",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Traffic-split file is missing columns: {missing}")

    for column in (
        "Annual_EV_Traffic_Pool",
        "ETA_DWC_Endogenous",
        "ETA_SCS_Endogenous",
        "ETA_Unserved",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    return df


# =============================================================================
# PLOTS
# =============================================================================


def plot_delivered_cost(summary: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(FIG_PANEL_WIDTH_IN, FIG_HEIGHT_IN))
    x = np.arange(len(summary))
    width = 0.54

    def cents(component: str) -> np.ndarray:
        return np.array(
            [
                100.0 * row[component] / row["discounted_energy_kwh"]
                if row["discounted_energy_kwh"] > 0
                else np.nan
                for row in summary
            ],
            dtype=float,
        )

    capex = cents("capex")
    fixed = cents("fixed")
    variable = cents("variable")
    total = capex + fixed + variable

    ax.bar(
        x,
        capex,
        width=width,
        color=C_CAPEX,
        edgecolor=C_EDGE,
        linewidth=0.30,
        label="Capital cost",
        zorder=3,
    )
    ax.bar(
        x,
        fixed,
        bottom=capex,
        width=width,
        color=C_FIXED,
        edgecolor=C_EDGE,
        linewidth=0.30,
        label="Fixed O&M",
        zorder=3,
    )
    ax.bar(
        x,
        variable,
        bottom=capex + fixed,
        width=width,
        color=C_VARIABLE,
        edgecolor=C_EDGE,
        linewidth=0.30,
        label="Electricity / variable O&M",
        zorder=3,
    )

    ymax = float(np.nanmax(total))
    offset = max(0.025 * ymax, 0.05)
    for xi, value in zip(x, total):
        ax.text(
            xi,
            value + offset,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_OTHER,
            fontweight="bold",
        )

    ax.set_xticks(x, [row["label"] for row in summary])
    ax.set_ylabel("USD cents per delivered kWh", labelpad=1.0)
    ax.set_ylim(0.0, ymax * 1.22)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.16),
        frameon=False,
        ncol=2,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.70,
        borderaxespad=0.0,
    )
    style_axis(ax)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.96, bottom=0.30)
    save_figure(fig, "Figure6c_delivered_cost_endogenous")


def plot_mixed_deployment_composition(mixed: dict) -> None:
    n_dwc = int(mixed["n_dwc"])
    n_scs = int(mixed["n_scs"])
    total = n_dwc + n_scs
    if total <= 0:
        raise ValueError("The mixed solution contains no selected assets.")

    share_dwc = 100.0 * n_dwc / total
    share_scs = 100.0 * n_scs / total

    fig, ax = plt.subplots(figsize=(4.15, 1.55))
    ax.barh(
        [0],
        [share_dwc],
        color=C_DWC,
        edgecolor=C_EDGE,
        linewidth=0.35,
        height=0.42,
        label="DWC",
        zorder=3,
    )
    ax.barh(
        [0],
        [share_scs],
        left=[share_dwc],
        color=C_SCS,
        edgecolor=C_EDGE,
        linewidth=0.35,
        height=0.42,
        label="SCS",
        zorder=3,
    )

    ax.text(
        share_dwc / 2.0,
        0,
        f"DWC\n{n_dwc:,} ({share_dwc:.1f}%)",
        ha="center",
        va="center",
        color="white",
        fontsize=FONT_SIZE_OTHER,
        fontweight="bold",
    )

    if share_scs >= 12.0:
        ax.text(
            share_dwc + share_scs / 2.0,
            0,
            f"SCS\n{n_scs:,} ({share_scs:.1f}%)",
            ha="center",
            va="center",
            color=C_EDGE,
            fontsize=FONT_SIZE_OTHER,
            fontweight="bold",
        )
    else:
        ax.annotate(
            f"SCS: {n_scs:,} ({share_scs:.1f}%)",
            xy=(share_dwc + share_scs / 2.0, 0.0),
            xytext=(min(share_dwc, 78.0), 0.42),
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_OTHER,
            fontweight="bold",
            arrowprops={"arrowstyle": "-", "linewidth": 0.45, "color": C_EDGE},
        )

    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(-0.60, 0.72)
    ax.set_yticks([])
    ax.set_xlabel("Share of selected infrastructure units (%)", labelpad=2.0)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.50, -0.28),
        frameon=False,
        ncol=2,
        handlelength=0.9,
        handletextpad=0.35,
        columnspacing=0.80,
        borderaxespad=0.0,
    )
    style_axis(ax, y_grid=False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.36)
    save_figure(fig, "Figure6d_mixed_deployment_composition")


def plot_npv_gain_decomposition(summary: list[dict]) -> pd.DataFrame:
    exclusive = max(summary[:2], key=lambda row: row["npv"])
    mixed = summary[2]

    components = [
        ("Revenue", mixed["revenue"] - exclusive["revenue"]),
        ("Capital cost", -(mixed["capex"] - exclusive["capex"])),
        ("Fixed O&M", -(mixed["fixed"] - exclusive["fixed"])),
        ("Variable O&M", -(mixed["variable"] - exclusive["variable"])),
    ]
    total_gain = sum(value for _, value in components)

    rows = [
        {"Component": name, "Contribution_USD": value}
        for name, value in components
    ]
    rows.append({"Component": "Net NPV gain", "Contribution_USD": total_gain})
    df_decomposition = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(4.45, FIG_HEIGHT_IN))
    running = 0.0
    x = np.arange(len(components) + 1)
    values_billion = [value / 1e9 for _, value in components]

    for idx, value in enumerate(values_billion):
        next_value = running + value
        bottom = min(running, next_value)
        height = abs(value)
        color = C_POSITIVE if value >= 0 else C_NEGATIVE

        ax.bar(
            idx,
            height,
            bottom=bottom,
            width=0.58,
            color=color,
            edgecolor=C_EDGE,
            linewidth=0.35,
            zorder=3,
        )
        ax.plot(
            [idx + 0.29, idx + 0.71],
            [next_value, next_value],
            color=C_EDGE,
            linewidth=0.40,
            zorder=4,
        )
        ax.text(
            idx,
            next_value + (0.05 if value >= 0 else -0.05),
            f"{value:+.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=FONT_SIZE_OTHER,
            fontweight="bold",
        )
        running = next_value

    total_billion = total_gain / 1e9
    ax.bar(
        len(components),
        abs(total_billion),
        bottom=min(0.0, total_billion),
        width=0.58,
        color=C_ALL,
        edgecolor=C_EDGE,
        linewidth=0.35,
        zorder=3,
    )
    ax.text(
        len(components),
        total_billion + (0.05 if total_billion >= 0 else -0.05),
        f"{total_billion:+.2f}",
        ha="center",
        va="bottom" if total_billion >= 0 else "top",
        fontsize=FONT_SIZE_OTHER,
        fontweight="bold",
    )

    ax.axhline(0.0, color=C_EDGE, linewidth=0.45, zorder=2)
    ax.set_xticks(
        x,
        ["Revenue", "Capital\ncost", "Fixed\nO&M", "Variable\nO&M", "Net gain"],
    )
    ax.set_ylabel(
        f"Contribution to mixed minus {exclusive['label']} NPV\n(billion USD)",
        labelpad=1.0,
    )
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    style_axis(ax)
    fig.subplots_adjust(left=0.19, right=0.98, top=0.95, bottom=0.19)
    save_figure(fig, "Figure6e_npv_gain_decomposition")

    return df_decomposition


def plot_discounted_delivered_energy(summary: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(FIG_PANEL_WIDTH_IN, FIG_HEIGHT_IN))
    x = np.arange(len(summary))
    values = np.array(
        [row["discounted_energy_kwh"] / 1e12 for row in summary], dtype=float
    )
    colors = [C_DWC, C_SCS, C_ALL]

    bars = ax.bar(
        x,
        values,
        width=0.54,
        color=colors,
        edgecolor=C_EDGE,
        linewidth=0.35,
        zorder=3,
    )

    offset = max(float(values.max()) * 0.025, 0.01)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + offset,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_OTHER,
            fontweight="bold",
        )

    exclusive = max(summary[:2], key=lambda row: row["npv"])
    mixed = summary[2]
    energy_change = (
        mixed["discounted_energy_kwh"] - exclusive["discounted_energy_kwh"]
    ) / 1e12
    ax.text(
        0.98,
        0.96,
        f"Mixed − {exclusive['label']}: {energy_change:+.1f} TWh",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_SIZE_OTHER,
    )

    ax.set_xticks(x, [row["label"] for row in summary])
    ax.set_ylabel("Discounted lifetime delivered energy (TWh)", labelpad=1.0)
    ax.set_ylim(0.0, values.max() * 1.18)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    style_axis(ax)
    fig.subplots_adjust(left=0.23, right=0.97, top=0.96, bottom=0.16)
    save_figure(fig, "Figure6f_discounted_delivered_energy")


def classify_segment_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eta_dwc = df["ETA_DWC_Endogenous"].clip(lower=0.0, upper=1.0)
    eta_scs = df["ETA_SCS_Endogenous"].clip(lower=0.0, upper=1.0)
    total_served = eta_dwc + eta_scs
    threshold = MEANINGFUL_SPLIT_THRESHOLD

    categories = np.select(
        [
            total_served < threshold,
            (eta_dwc >= threshold) & (eta_scs >= threshold),
            eta_dwc >= eta_scs,
        ],
        [
            "Unserved",
            "DWC + SCS split",
            "DWC only",
        ],
        default="SCS only",
    )

    classified = df.copy()
    classified["Traffic_Allocation_Class"] = categories

    order = ["DWC only", "SCS only", "DWC + SCS split", "Unserved"]
    counts = classified["Traffic_Allocation_Class"].value_counts().reindex(
        order, fill_value=0
    )
    summary = pd.DataFrame(
        {
            "Traffic_Allocation_Class": order,
            "Number_of_Segments": counts.values.astype(int),
        }
    )
    summary["Share_of_Segments"] = (
        summary["Number_of_Segments"] / len(classified)
    )

    return classified, summary


def plot_segment_split_categories(split_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(4.55, FIG_HEIGHT_IN))
    labels = split_summary["Traffic_Allocation_Class"].tolist()
    counts = split_summary["Number_of_Segments"].to_numpy(dtype=float)
    shares = 100.0 * split_summary["Share_of_Segments"].to_numpy(dtype=float)
    colors = [C_DWC, C_SCS, C_DWC_LIGHT, C_UNSERVED]
    x = np.arange(len(labels))

    bars = ax.bar(
        x,
        counts,
        width=0.58,
        color=colors,
        edgecolor=C_EDGE,
        linewidth=0.35,
        zorder=3,
    )

    offset = max(float(counts.max()) * 0.025, 1.0)
    for bar, count, share in zip(bars, counts, shares):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            count + offset,
            f"{int(count):,}\n({share:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=FONT_SIZE_OTHER,
            fontweight="bold",
        )

    ax.set_xticks(
        x,
        ["DWC only", "SCS only", "DWC + SCS\nsplit", "Unserved"],
    )
    ax.set_ylabel("Number of road segments", labelpad=1.0)
    ax.set_ylim(0.0, counts.max() * 1.19 if counts.max() > 0 else 1.0)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    style_axis(ax)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.95, bottom=0.20)
    save_figure(fig, "Figure6g_segment_traffic_allocation")


def plot_eta_dwc_by_segment(df: pd.DataFrame) -> pd.DataFrame:
    eta = df["ETA_DWC_Endogenous"].clip(lower=0.0, upper=1.0)
    sorted_df = df[["Segment_ID", "Annual_EV_Traffic_Pool"]].copy()
    sorted_df["ETA_DWC_Endogenous"] = eta
    sorted_df = sorted_df.sort_values(
        ["ETA_DWC_Endogenous", "Segment_ID"]
    ).reset_index(drop=True)
    sorted_df.insert(0, "Segment_Rank", np.arange(1, len(sorted_df) + 1))

    mean_eta = float(eta.mean())
    median_eta = float(eta.median())
    weights = df["Annual_EV_Traffic_Pool"].clip(lower=0.0)
    weighted_mean = (
        float(np.average(eta, weights=weights)) if weights.sum() > 0 else np.nan
    )

    fig, ax = plt.subplots(figsize=(FIG_FULL_WIDTH_IN, 2.75))
    x = sorted_df["Segment_Rank"].to_numpy(dtype=float)
    y = sorted_df["ETA_DWC_Endogenous"].to_numpy(dtype=float)

    ax.plot(x, y, color=C_DWC, linewidth=0.70, zorder=3)
    ax.scatter(
        x,
        y,
        s=1.2,
        color=C_DWC_LIGHT,
        edgecolor="none",
        alpha=0.80,
        zorder=4,
    )
    ax.axhline(
        mean_eta,
        color=C_SCS,
        linestyle="--",
        linewidth=0.95,
        label=fr"Mean $\eta_{{DWC}}$ = {mean_eta:.3f}",
        zorder=5,
    )
    ax.axhline(
        median_eta,
        color=C_ALL,
        linestyle=":",
        linewidth=1.10,
        label=fr"Median $\eta_{{DWC}}$ = {median_eta:.3f}",
        zorder=5,
    )

    ax.set_xlim(1, len(sorted_df))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"Road segments ranked by endogenous $\eta_{DWC}$", labelpad=2.0)
    ax.set_ylabel(r"Endogenous segment-level $\eta_{DWC}$", labelpad=1.0)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.20))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, integer=True))
    ax.legend(
        loc="upper left",
        frameon=False,
        handlelength=1.4,
        handletextpad=0.45,
        borderaxespad=0.30,
    )
    style_axis(ax)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.96, bottom=0.20)
    save_figure(fig, "Figure6h_eta_dwc_by_segment")

    statistics = pd.DataFrame(
        [
            {
                "Number_of_Segments": len(df),
                "Mean_ETA_DWC": mean_eta,
                "Median_ETA_DWC": median_eta,
                "Traffic_Weighted_Mean_ETA_DWC": weighted_mean,
            }
        ]
    )
    statistics.to_csv(
        OUTPUT_DIR / "Endogenous_Mixed_ETA_DWC_Statistics.csv", index=False
    )
    sorted_df.to_csv(
        OUTPUT_DIR / "Endogenous_Mixed_ETA_DWC_Sorted_By_Segment.csv", index=False
    )

    print(
        f"ETA_DWC statistics: mean={mean_eta:.6f}, median={median_eta:.6f}, "
        f"traffic-weighted mean={weighted_mean:.6f}"
    )
    return statistics


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    configure_fonts()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loaded_runs = [load_run(label, directory) for label, directory in RUNS]
    summary = [summarize_run(run) for run in loaded_runs]
    summary_df = pd.DataFrame(summary)

    print("\nPortfolio summary:")
    print(
        summary_df[
            [
                "label",
                "n_dwc",
                "n_scs",
                "revenue",
                "capex",
                "fixed",
                "variable",
                "discounted_energy_kwh",
                "lcde_usd_per_kwh",
                "npv",
                "saved_objective",
            ]
        ].to_string(index=False)
    )

    summary_df.to_csv(
        OUTPUT_DIR / "Endogenous_Mixed_Portfolio_Diagnostics_Summary.csv",
        index=False,
    )

    plot_delivered_cost(summary)
    plot_mixed_deployment_composition(summary[2])
    decomposition = plot_npv_gain_decomposition(summary)
    decomposition.to_csv(
        OUTPUT_DIR / "Endogenous_Mixed_NPV_Gain_Decomposition.csv", index=False
    )
    plot_discounted_delivered_energy(summary)

    traffic_split = load_traffic_split()
    classified, split_summary = classify_segment_splits(traffic_split)
    classified.to_csv(
        OUTPUT_DIR / "Endogenous_Mixed_Segment_Split_Classification.csv",
        index=False,
    )
    split_summary.to_csv(
        OUTPUT_DIR / "Endogenous_Mixed_Segment_Split_Summary.csv",
        index=False,
    )

    print("\nSegment traffic-allocation summary:")
    print(split_summary.to_string(index=False))
    plot_segment_split_categories(split_summary)
    plot_eta_dwc_by_segment(traffic_split)

    print("\nDone.")


if __name__ == "__main__":
    main()
