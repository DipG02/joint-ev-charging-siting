#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# Sensitivity_Plots.py  (revised per reviewer critique)
#
# Pure READ-ONLY plotting layer for the parameter-influence study produced by
# Main_sensitivity.py. This module NEVER imports Main_optimizer, NEVER touches
# Gurobi, and NEVER re-solves or re-fits anything. Every function here takes a
# path to a GSA_<timestamp> directory and reads only the CSVs already saved
# there by Main_sensitivity.py.
#
# ── REVISION NOTES ───────────────────────────────────────────────────────────
# Restructured around decision insights rather than statistical machinery:
#
#   MAIN PAPER SET (5 figures)
#   ─────────────────────────
#   Fig 1  plot_parameter_importance()
#            Simplified tornado: absolute standardised effect, one panel,
#            both DWC and SCS effects side-by-side on a single axis.
#            Answers: "Which parameters matter most?"
#
#   Fig 2  plot_marginal_effect_curves()
#            P(DWC) / P(SCS) / P(None) vs each top-k parameter across its
#            sampled range. Answers: "How much does each parameter matter?"
#
#   Fig 3  plot_phase_diagram()
#            Clean two-axis phase map (DWC capex vs EV growth) showing which
#            regime dominates.  Answers: "When does DWC beat SCS?"
#
#   Fig 4  plot_breakeven_frontier()
#            For each EV growth rate, the maximum DWC capex that still yields
#            DWC deployment. Answers: "What are the economic thresholds?"
#
#   Fig 5  plot_corridor_deployment_frequency()
#            Selection frequency by segment across the GSA ensemble.
#            Answers: "Which segments are robustly selected?"
#
#   SUPPLEMENTARY SET (kept, demoted)
#   ──────────────────────────────────
#   Supp A  plot_bootstrap_rank_stability()   (unchanged logic, kept for appendix)
#   Supp B  plot_input_correlation_structure() (pure diagnostic, appendix only)
#   Supp C  plot_parameter_space_coverage()   (methods check, appendix only)
#   Supp D  plot_class_balance()              (diagnostic)
#   Supp E  plot_dwc_vs_scs_win_rate()        (context)
#   Supp F  plot_dwc_scs_share_vs_parameter() (per-parameter scatter)
#
# ── STYLE / OUTPUT CONTRACT ──────────────────────────────────────────────────
#   * Visual language inherited from Main_plots.py (DWC/SCS colors identical).
#   * Every figure is saved as its own PNG file (300 dpi). No SVG export.
#   * Font family: Arial everywhere (falls back to DejaVu Sans).
#   * Input/output interface unchanged: generate_all_plots(gsa_dir) still works.
# ═══════════════════════════════════════════════════════════════════════════════

import os
import json
import glob
import numpy as np
import pandas as pd
from scipy import stats as sstats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — INHERIT STYLE + DWC/SCS PALETTE FROM Main_plots.py
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from Main_plots import (
        C_DWC, C_SCS, C_EXIST, C_REV, C_CAPEX, C_OPEX, C_PROFIT,
        C_GREY, C_GOLD, C_CYAN,
    )
except Exception:
    C_DWC    = "#2c3e50"
    C_SCS    = "#17a589"
    C_EXIST  = "#e67e22"
    C_REV    = "#27ae60"
    C_CAPEX  = "#e74c3c"
    C_OPEX   = "#e67e22"
    C_PROFIT = "#2980b9"
    C_GREY   = "#bdc3c7"
    C_GOLD   = "#f39c12"
    C_CYAN   = "cyan"

COLOR_DWC      = C_DWC
COLOR_SCS      = C_SCS
COLOR_NONE     = C_GREY
COLOR_CI       = "#2D2D2D"
COLOR_BASELINE = "#111111"
COLOR_POS      = C_DWC
COLOR_NEG      = C_EXIST

# ── Typography ────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family"       : "sans-serif",
    "font.sans-serif"   : ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size"         : 10,
    "axes.titlesize"    : 11,
    "axes.titleweight"  : "bold",
    "axes.labelsize"    : 10,
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "legend.fontsize"   : 9,
    "legend.frameon"    : False,
    "axes.spines.top"   : False,
    "axes.spines.right" : False,
    "axes.linewidth"    : 0.9,
    "xtick.major.width" : 0.9,
    "ytick.major.width" : 0.9,
    "xtick.major.size"  : 3,
    "ytick.major.size"  : 3,
    "figure.dpi"        : 300,
    "savefig.dpi"       : 300,
    "savefig.bbox"      : "tight",
    "axes.grid"         : False,
})

_DEFAULT_DISPLAY_NAMES = {
    'ETA_DWC'              : r'$\eta^{DWC}$ (DWC traffic allocation)',
    'ETA_SCS_SHARE'        : r'$\eta^{SCS}_{share}$ (SCS share of residual capture)',
    'ETA_SCS'              : r'$\eta^{SCS}$ (SCS traffic allocation)',
    'g_aadt'               : r'$g^{AADT}$ (traffic growth)',
    'g_ev'                 : r'$g^{EV}$ (EV growth)',
    'project_life'         : r'$T$ (project life, yr)',
    'C_DWC_base'           : r'$C^{DWC}_{base}$ (DWC capex)',
    'C_SCS_base_per_port'  : r'$C^{SCS}_{port}$ (SCS capex/port)',
    'N_SCS_stalls'         : r'$N^{SCS}_{stalls}$ (ports per station)',
    'g_lcoe'               : r'$g^{LCOE}$ (LCOE growth)',
    'AADT_COM'             : 'AADT (segment control)',
    'Min_Sub_Dist'         : 'Substation distance (segment control)',
    'const'                : 'Intercept (baseline log-odds)',
}

_DEFAULT_SHORT_NAMES = {
    'ETA_DWC'              : r'$\eta^{DWC}$',
    'ETA_SCS_SHARE'        : r'$\eta^{SCS}_{share}$',
    'ETA_SCS'              : r'$\eta^{SCS}$',
    'g_aadt'               : r'$g^{AADT}$',
    'g_ev'                 : r'$g^{EV}$',
    'project_life'         : r'$T$',
    'C_DWC_base'           : r'$C^{DWC}_{base}$',
    'C_SCS_base_per_port'  : r'$C^{SCS}_{port}$',
    'N_SCS_stalls'         : r'$N^{SCS}_{stalls}$',
    'g_lcoe'               : r'$g^{LCOE}$',
    'AADT_COM'             : 'AADT',
    'Min_Sub_Dist'         : 'Sub. dist.',
}


def _sig_stars(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def _ensure_fig_dir(gsa_dir):
    fig_dir = os.path.join(gsa_dir, "Figures")
    os.makedirs(fig_dir, exist_ok=True)
    return fig_dir


def _load_manifest(gsa_dir):
    manifest_path = os.path.join(gsa_dir, 'gsa_run_manifest.json')
    if not os.path.exists(manifest_path):
        return {}
    with open(manifest_path, 'r') as f:
        return json.load(f)


def _white_to_color_cmap(hex_color):
    return LinearSegmentedColormap.from_list('w2c', ['#FFFFFF', hex_color])


def _diverging_cmap(neg_hex, pos_hex):
    return LinearSegmentedColormap.from_list('div', [neg_hex, '#FFFFFF', pos_hex])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0b — SAVE HELPER (PNG only)
# ═══════════════════════════════════════════════════════════════════════════════
def _save_fig(fig, fig_dir, basename):
    path = os.path.join(fig_dir, f"{basename}.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved {basename}.png")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LOAD + CLEAN THE COEFFICIENT TABLE
# ═══════════════════════════════════════════════════════════════════════════════
def load_coefficient_table(gsa_dir):
    path = os.path.join(gsa_dir, 'mnlogit_coefficients.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run Main_sensitivity.run_full_sensitivity_study "
            f"(or at least fit_parameter_influence_model) first."
        )
    return pd.read_csv(path)


def build_readable_influence_table(gsa_dir, param_display_names=None):
    df = load_coefficient_table(gsa_dir)
    df = df[df['parameter'] != 'const'].copy()
    display_names = param_display_names or _DEFAULT_DISPLAY_NAMES

    class_cols = [c.replace('coef_', '') for c in df.columns if c.startswith('coef_')]

    rows = []
    for _, r in df.iterrows():
        row = {
            'parameter': r['parameter'],
            'display_name': display_names.get(r['parameter'], r['parameter']),
            'role': 'segment control' if r.get('is_segment_control', False) else 'swept parameter',
        }
        for cls in class_cols:
            coef = r[f'coef_{cls}']
            p = r[f'p_{cls}']
            direction = '↑ more' if coef > 0 else '↓ less'
            row[f'{cls}_effect'] = f"{direction} {cls} {_sig_stars(p)}".strip()
            row[f'{cls}_coef'] = coef
            row[f'{cls}_pvalue'] = p
            row[f'{cls}_rank'] = r.get(f'rank_{cls}', np.nan)
        rows.append(row)

    df_readable = pd.DataFrame(rows)
    primary_rank_col = 'DWC_rank' if 'DWC_rank' in df_readable.columns else f'{class_cols[0]}_rank'
    df_readable = df_readable.sort_values(primary_rank_col).reset_index(drop=True)

    out_path = os.path.join(gsa_dir, 'parameter_influence_table.csv')
    df_readable.to_csv(out_path, index=False)
    print(f"Readable influence table saved -> {out_path}")
    return df_readable


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PROBABILITY RECONSTRUCTION FROM SAVED COEFFICIENTS
# ═══════════════════════════════════════════════════════════════════════════════
def _load_feature_standardization(gsa_dir):
    path = os.path.join(gsa_dir, 'feature_standardization.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path).set_index('parameter')
    return df['mean'].to_dict(), df['std'].to_dict()


def _std(raw, mean, std):
    s = std if (std not in (0, None) and not pd.isna(std)) else 1.0
    return (raw - (mean or 0.0)) / s


def _coef_lookup(df_params, cls):
    return df_params.set_index('parameter')[f'coef_{cls}']


def _predict_probs(df_params, intercepts, class_cols, contributions):
    z = {}
    for cls in class_cols:
        coefs = _coef_lookup(df_params, cls)
        val = intercepts.get(cls, 0.0)
        for p, v in contributions.items():
            if p in coefs.index:
                val = val + coefs[p] * v
        z[cls] = val

    shape = None
    for v in z.values():
        if hasattr(v, 'shape'):
            shape = v.shape
            break
    zero = np.zeros(shape) if shape is not None else 0.0

    exp_vals = {'None': np.exp(zero)}
    for cls, v in z.items():
        exp_vals[cls] = np.exp(v)
    total = sum(exp_vals.values())
    return {k: v / total for k, v in exp_vals.items()}


def _split_const_and_params(df_coef_full):
    class_cols = [c.replace('coef_', '') for c in df_coef_full.columns if c.startswith('coef_')]
    intercepts = {cls: 0.0 for cls in class_cols}
    if 'const' in df_coef_full['parameter'].values:
        crow = df_coef_full[df_coef_full['parameter'] == 'const'].iloc[0]
        intercepts = {cls: float(crow.get(f'coef_{cls}', 0.0)) for cls in class_cols}
    df_params = df_coef_full[df_coef_full['parameter'] != 'const'].reset_index(drop=True)
    return df_params, intercepts, class_cols


def _load_segment_population(gsa_dir):
    runs_dir = os.path.join(gsa_dir, 'Runs')
    if not os.path.isdir(runs_dir):
        raise FileNotFoundError(f"{runs_dir} not found.")
    for d in sorted(os.listdir(runs_dir)):
        cand = os.path.join(runs_dir, d, 'DWC_Selection_Master_Diagnostic.csv')
        if os.path.exists(cand):
            return pd.read_csv(cand, usecols=['Segment_ID', 'AADT_COM', 'Min_Sub_Dist'])
    raise FileNotFoundError("No DWC_Selection_Master_Diagnostic.csv found under Runs/.")


def _sample_pooled_csv(gsa_dir, n_target=15000, seed=42, chunksize=300_000, size_threshold_mb=80):
    path = os.path.join(gsa_dir, 'segment_level_pooled.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- run the GSA sweep first.")
    usecols = ['Segment_ID', 'AADT_COM', 'Min_Sub_Dist', 'y']
    size_mb = os.path.getsize(path) / (1024 ** 2)

    if size_mb <= size_threshold_mb:
        df = pd.read_csv(path, usecols=usecols)
    else:
        rng = np.random.default_rng(seed)
        pieces = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            take_n = max(1, int(len(chunk) * 0.03))
            idx = rng.choice(len(chunk), size=min(take_n, len(chunk)), replace=False)
            pieces.append(chunk.iloc[idx])
        df = pd.concat(pieces, ignore_index=True)

    if len(df) > n_target:
        df = df.sample(n=n_target, random_state=seed)
    return df.reset_index(drop=True)


def _build_ctx(gsa_dir, param_display_names=None, n_sample=15000):
    df_coef_full = load_coefficient_table(gsa_dir)
    df_params, intercepts, class_cols = _split_const_and_params(df_coef_full)
    means, stds = _load_feature_standardization(gsa_dir)
    df_seg = _load_segment_population(gsa_dir)
    df_sample = _sample_pooled_csv(gsa_dir, n_target=n_sample)
    manifest = _load_manifest(gsa_dir)
    param_bounds = manifest.get('param_bounds', {})
    display_names = param_display_names or _DEFAULT_DISPLAY_NAMES

    return dict(
        df_params=df_params, intercepts=intercepts, class_cols=class_cols,
        means=means, stds=stds, df_seg=df_seg, df_sample=df_sample,
        param_bounds=param_bounds, display_names=display_names,
        cmap_dwc=_white_to_color_cmap(COLOR_DWC),
        cmap_scs=_white_to_color_cmap(COLOR_SCS),
        manifest=manifest,
    )


def _top_theta_params(ctx, k=4):
    df_params = ctx['df_params']
    class_cols = ctx['class_cols']
    bounds = ctx['param_bounds']

    theta_df = df_params[~df_params['is_segment_control']].copy()
    theta_df = theta_df[theta_df['parameter'].isin(bounds.keys())]
    abs_cols = [f'abs_coef_{c}' for c in class_cols if f'abs_coef_{c}' in theta_df.columns]
    if abs_cols:
        theta_df['_imp'] = theta_df[abs_cols].max(axis=1)
    else:
        # fallback: compute from coef columns
        theta_df['_imp'] = theta_df[[f'coef_{c}' for c in class_cols
                                      if f'coef_{c}' in theta_df.columns]].abs().max(axis=1)
    theta_df = theta_df.sort_values('_imp', ascending=False)
    return theta_df['parameter'].tolist()[:k]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — FIG 1: PARAMETER IMPORTANCE (simplified tornado)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_parameter_importance(gsa_dir, ctx=None, top_n=None):
    """
    Single-panel horizontal bar chart of absolute standardised MNLogit
    coefficients for DWC and SCS shown side-by-side on one axis.

    Design principles (per reviewer critique):
    - One ranking per figure, not one panel per class.
    - Parameters sorted by DWC |coef| descending (primary ordering signal).
    - DWC bars left-anchored (solid), SCS bars offset (hatched), so the
      comparison is instant without reading two separate panels.
    - No error bars on the primary ranking figure — those live in the
      supplementary bootstrap stability plot.
    - Significance stars retained as lightweight annotation.

    Answers: "Which parameters matter most?"
    """
    if ctx is None:
        ctx = _build_ctx(gsa_dir)
    df_all = ctx['df_params'].copy()
    df_all = df_all[~df_all['is_segment_control']]
    class_cols = [c for c in ['DWC', 'SCS'] if c in ctx['class_cols']]
    if not class_cols:
        class_cols = ctx['class_cols']
    short_names = _DEFAULT_SHORT_NAMES

    # Sort by DWC absolute coefficient (primary class), or first available
    sort_cls = class_cols[0]
    abs_col = f'abs_coef_{sort_cls}'
    if abs_col not in df_all.columns:
        df_all[abs_col] = df_all[f'coef_{sort_cls}'].abs()
    df_all = df_all.sort_values(abs_col, ascending=True)
    if top_n:
        df_all = df_all.tail(top_n)

    y_pos = np.arange(len(df_all))
    names = [short_names.get(p, p) for p in df_all['parameter']]
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=(6.0, max(3.5, 0.45 * len(df_all) + 1.2)))

    color_map = {'DWC': COLOR_DWC, 'SCS': COLOR_SCS}
    offsets   = {'DWC': bar_h * 0.5, 'SCS': -bar_h * 0.5}
    hatch_map = {'DWC': '', 'SCS': '///'}

    for cls in class_cols:
        abs_c = df_all[f'coef_{cls}'].abs() if f'abs_coef_{cls}' not in df_all.columns \
                else df_all[f'abs_coef_{cls}']
        p_vals = df_all[f'p_{cls}']
        yp = y_pos + offsets[cls]

        bars = ax.barh(yp, abs_c, height=bar_h,
                       color=color_map.get(cls, C_GREY),
                       hatch=hatch_map.get(cls, ''),
                       edgecolor='white', linewidth=0.5, zorder=2,
                       label=cls)

        for b, pv in zip(bars, p_vals):
            stars = _sig_stars(pv)
            if stars:
                ax.text(b.get_width() + ax.get_xlim()[1] * 0.01,
                        b.get_y() + b.get_height() / 2,
                        stars, va='center', fontsize=7.5,
                        color=COLOR_CI, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel(r'Absolute standardised coefficient  $|\beta|$  (log-odds vs. None)')
    ax.set_title('Parameter importance ranking', loc='left', fontsize=12)
    ax.margins(y=0.04)
    ax.legend(loc='lower right', title='Technology', title_fontsize=8)

    fig.suptitle('Which parameters drive deployment decisions?',
                 x=0.01, ha='left', fontsize=11, style='italic', y=1.01)
    fig.tight_layout()

    fig_dir = _ensure_fig_dir(gsa_dir)
    out = _save_fig(fig, fig_dir, 'fig_parameter_importance')
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — FIG 2: MARGINAL EFFECT CURVES  (top drivers — unchanged logic)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_marginal_effect_curves(gsa_dir, ctx=None, top_k=3, n_points=120):
    """
    For the top_k most influential swept parameters, vary that parameter
    across its sampled range (holding everything else at its mean /
    standardised 0) and plot smooth P(DWC) / P(SCS) / P(None) curves.

    Answers: "How much does each parameter matter?"
    """
    if ctx is None:
        ctx = _build_ctx(gsa_dir)
    top_params = _top_theta_params(ctx, k=top_k)
    if not top_params:
        print("  Skipping marginal effect curves — no swept parameters with bounds found.")
        return None

    bounds = ctx['param_bounds']
    means, stds = ctx['means'], ctx['stds']
    display_names = ctx['display_names']

    fig, axes = plt.subplots(1, len(top_params), figsize=(4.6 * len(top_params), 4.0), sharey=True)
    if len(top_params) == 1:
        axes = [axes]

    for ax, p in zip(axes, top_params):
        lo, hi = bounds[p]
        grid = np.linspace(lo, hi, n_points)
        contributions = {p: _std(grid, means.get(p), stds.get(p))}
        probs = _predict_probs(ctx['df_params'], ctx['intercepts'], ctx['class_cols'], contributions)

        if 'DWC' in probs:
            ax.plot(grid, probs['DWC'], color=COLOR_DWC, linewidth=2.4, label='P(DWC)')
        if 'SCS' in probs:
            ax.plot(grid, probs['SCS'], color=COLOR_SCS, linewidth=2.4, label='P(SCS)')
        if 'None' in probs:
            ax.plot(grid, probs['None'], color=COLOR_NONE, linewidth=1.6,
                    linestyle='--', label='P(None)')

        # Baseline marker
        base_val = ctx['manifest'].get('base_params_used', {}).get(p)
        if base_val is not None:
            try:
                ax.axvline(float(base_val), color=COLOR_BASELINE,
                           linewidth=1.0, linestyle=':', label='baseline')
            except (TypeError, ValueError):
                pass

        ax.set_xlabel(display_names.get(p, p), fontsize=9.5)
        ax.set_ylim(0, 1.0)
        ax.set_title(_DEFAULT_SHORT_NAMES.get(p, p), loc='left', fontsize=10)

    axes[0].set_ylabel('Predicted deployment probability')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.10), frameon=False)
    fig.suptitle('How much does each parameter matter?',
                 x=0.01, ha='left', fontsize=12, fontweight='bold', y=1.04)
    fig.tight_layout()

    fig_dir = _ensure_fig_dir(gsa_dir)
    out = _save_fig(fig, fig_dir, 'fig_marginal_effect_curves')
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FIG 3: PHASE DIAGRAM  (DWC capex × EV growth)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_phase_diagram(gsa_dir, ctx=None, n_grid=200,
                        x_param='C_DWC_base', y_param='g_ev'):
    """
    Clean two-axis phase map showing which deployment regime dominates as
    DWC capex and EV growth vary, with all other parameters held at their
    sampled mean.

    Three coloured regions: DWC dominant (blue), SCS dominant (teal),
    Neither viable (grey).  The baseline operating point is marked.
    A thin contour marks the DWC=SCS indifference boundary.

    Answers: "When does DWC beat SCS?"
    """
    if ctx is None:
        ctx = _build_ctx(gsa_dir)
    bounds = ctx['param_bounds']
    means, stds = ctx['means'], ctx['stds']
    manifest = ctx['manifest']
    base_used = manifest.get('base_params_used', {})

    if x_param not in bounds or y_param not in bounds:
        print(f"  Skipping phase diagram — {x_param} or {y_param} not in param_bounds.")
        return None

    x_lo, x_hi = bounds[x_param]
    y_lo, y_hi = bounds[y_param]
    x_grid = np.linspace(x_lo, x_hi, n_grid)
    y_grid = np.linspace(y_lo, y_hi, n_grid)
    XX, YY = np.meshgrid(x_grid, y_grid)

    contributions = {
        x_param: _std(XX, means.get(x_param), stds.get(x_param)),
        y_param: _std(YY, means.get(y_param),  stds.get(y_param)),
    }
    probs = _predict_probs(ctx['df_params'], ctx['intercepts'], ctx['class_cols'], contributions)

    class_order = ['None', 'SCS', 'DWC']
    class_order = [c for c in class_order if c in probs] + \
                  [c for c in probs if c not in class_order]
    stack = np.stack([probs[c] for c in class_order], axis=0)
    regime_idx = np.argmax(stack, axis=0)

    color_lookup = {'DWC': COLOR_DWC, 'SCS': COLOR_SCS, 'None': COLOR_NONE}
    colors_list = [color_lookup.get(c, C_GREY) for c in class_order]
    cmap_regime = ListedColormap(colors_list)
    bounds_n = np.arange(len(class_order) + 1) - 0.5
    norm_regime = BoundaryNorm(bounds_n, cmap_regime.N)

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    ax.pcolormesh(XX, YY, regime_idx, cmap=cmap_regime, norm=norm_regime,
                  shading='auto', zorder=1, alpha=0.85)

    # Contour marking DWC=SCS indifference boundary
    if 'DWC' in probs and 'SCS' in probs:
        diff = probs['DWC'] - probs['SCS']
        try:
            ax.contour(XX, YY, diff, levels=[0.0],
                       colors=[COLOR_CI], linewidths=1.4, linestyles='--', zorder=3)
        except Exception:
            pass

    # Baseline operating point
    bx = base_used.get(x_param)
    by = base_used.get(y_param)
    if bx is not None and by is not None:
        try:
            ax.scatter([float(bx)], [float(by)], s=80, color='white',
                       edgecolor=COLOR_CI, linewidths=1.5, zorder=5,
                       marker='D', label='Baseline')
            ax.legend(loc='upper right', handletextpad=0.4)
        except (TypeError, ValueError):
            pass

    # Axis labels — use display names if available, else tidy the param name
    x_label = ctx['display_names'].get(x_param, x_param)
    y_label = ctx['display_names'].get(y_param, y_param)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # X-axis: show values in $M for cost params
    if 'C_DWC' in x_param or 'C_SCS' in x_param or 'cost' in x_param.lower():
        xt = ax.get_xticks()
        ax.set_xticklabels([f'${v/1e6:.0f}M' for v in xt], fontsize=8)

    ax.set_title('Deployment regime phase diagram', loc='left')

    legend_handles = [Patch(facecolor=color_lookup.get(c, C_GREY),
                             edgecolor='white', label=c)
                      for c in class_order]
    legend_handles.append(Line2D([0], [0], color=COLOR_CI, linewidth=1.4,
                                  linestyle='--', label='DWC = SCS boundary'))
    ax.legend(handles=legend_handles, loc='upper right',
              framealpha=0.92, fontsize=8.5)

    fig.suptitle('When does DWC beat SCS?',
                 x=0.01, ha='left', fontsize=11, style='italic', y=1.02)
    fig.tight_layout()

    fig_dir = _ensure_fig_dir(gsa_dir)
    fname = f'fig_phase_diagram_{x_param}_vs_{y_param}'
    out = _save_fig(fig, fig_dir, fname)
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FIG 4: ECONOMIC BREAK-EVEN FRONTIER  (new figure)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_breakeven_frontier(gsa_dir, ctx=None, n_ev_grid=60, n_cost_grid=200,
                             cost_param='C_DWC_base', ev_param='g_ev'):
    """
    For each EV growth rate, find the maximum DWC capex at which DWC still
    has the highest predicted deployment probability (P(DWC) > P(SCS) and
    P(DWC) > P(None)), all other parameters held at their sampled mean.

    The resulting curve is the "break-even" cost frontier:
      - Points below the line → DWC is likely deployed
      - Points above the line → SCS or Nothing dominates

    Also shades the DWC-viable region and marks the baseline operating point.

    Answers: "What are the economic thresholds planners need to care about?"
    """
    if ctx is None:
        ctx = _build_ctx(gsa_dir)
    bounds = ctx['param_bounds']
    means, stds = ctx['means'], ctx['stds']
    manifest = ctx['manifest']
    base_used = manifest.get('base_params_used', {})

    if cost_param not in bounds or ev_param not in bounds:
        print(f"  Skipping break-even frontier — {cost_param} or {ev_param} not in param_bounds.")
        return None

    cost_lo, cost_hi = bounds[cost_param]
    ev_lo,   ev_hi   = bounds[ev_param]

    ev_vals   = np.linspace(ev_lo, ev_hi, n_ev_grid)
    cost_vals = np.linspace(cost_lo, cost_hi, n_cost_grid)

    breakeven_costs = []
    for ev_val in ev_vals:
        ev_std = _std(ev_val, means.get(ev_param), stds.get(ev_param))
        dwc_probs = []
        for c_val in cost_vals:
            c_std = _std(c_val, means.get(cost_param), stds.get(cost_param))
            contributions = {
                ev_param  : ev_std,
                cost_param: c_std,
            }
            probs = _predict_probs(ctx['df_params'], ctx['intercepts'],
                                   ctx['class_cols'], contributions)
            p_dwc = float(probs.get('DWC', 0.0))
            dwc_probs.append(p_dwc)

        # Find the cost at which P(DWC) drops below max(P(SCS), P(None))
        # i.e. the last cost where DWC is still the modal outcome
        dwc_arr = np.array(dwc_probs)
        # DWC is dominant when its probability is the highest
        contributions_scs = {ev_param: ev_std}
        probs_base = _predict_probs(ctx['df_params'], ctx['intercepts'],
                                     ctx['class_cols'], contributions_scs)
        # sweep cost upward; find first crossing where DWC no longer leads
        be_cost = np.nan
        for c_val, p_dwc_val in zip(cost_vals, dwc_probs):
            c_std = _std(c_val, means.get(cost_param), stds.get(cost_param))
            contribs = {ev_param: ev_std, cost_param: c_std}
            probs_c = _predict_probs(ctx['df_params'], ctx['intercepts'],
                                      ctx['class_cols'], contribs)
            p_dwc_c = float(probs_c.get('DWC', 0.0))
            p_scs_c = float(probs_c.get('SCS', 0.0))
            p_none_c = float(probs_c.get('None', 0.0))
            if p_dwc_c < max(p_scs_c, p_none_c):
                be_cost = c_val
                break
        if np.isnan(be_cost):
            be_cost = cost_hi  # DWC dominates across the entire cost range
        breakeven_costs.append(be_cost)

    breakeven_costs = np.array(breakeven_costs)
    valid = ~np.isnan(breakeven_costs)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))

    # Fill DWC-viable region
    ax.fill_betweenx(ev_vals[valid], cost_lo, breakeven_costs[valid],
                     color=COLOR_DWC, alpha=0.18, label='DWC likely deployed', zorder=1)
    ax.fill_betweenx(ev_vals[valid], breakeven_costs[valid], cost_hi,
                     color=COLOR_SCS, alpha=0.12, label='SCS / None likely', zorder=1)

    # Break-even curve
    ax.plot(breakeven_costs[valid], ev_vals[valid],
            color=COLOR_DWC, linewidth=2.4, zorder=3, label='Break-even frontier')

    # Baseline operating point
    bx = base_used.get(cost_param)
    by = base_used.get(ev_param)
    if bx is not None and by is not None:
        try:
            ax.scatter([float(bx)], [float(by)], s=80, color='white',
                       edgecolor=COLOR_CI, linewidths=1.5, zorder=5,
                       marker='D', label='Baseline')
        except (TypeError, ValueError):
            pass

    ax.set_xlabel(ctx['display_names'].get(cost_param, cost_param))
    ax.set_ylabel(ctx['display_names'].get(ev_param, ev_param))
    ax.set_xlim(cost_lo, cost_hi)
    ax.set_ylim(ev_lo, ev_hi)

    # Format x-axis as $M for cost params
    if 'C_DWC' in cost_param or 'C_SCS' in cost_param:
        xt = np.linspace(cost_lo, cost_hi, 5)
        ax.set_xticks(xt)
        ax.set_xticklabels([f'${v/1e6:.0f}M' for v in xt], fontsize=8)

    ax.set_title('Economic break-even frontier', loc='left')
    ax.legend(loc='lower right', fontsize=8.5)

    fig.suptitle('What DWC capex is too expensive given EV growth?',
                 x=0.01, ha='left', fontsize=11, style='italic', y=1.02)
    fig.tight_layout()

    fig_dir = _ensure_fig_dir(gsa_dir)
    out = _save_fig(fig, fig_dir, 'fig_breakeven_frontier')
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FIG 5: CORRIDOR DEPLOYMENT FREQUENCY
# ═══════════════════════════════════════════════════════════════════════════════
def plot_corridor_deployment_frequency(gsa_dir, ctx=None):
    """
    For every segment, the fraction of GSA runs in which it was selected as
    DWC vs SCS, plotted as stacked area ribbons against segment index.

    Unlike the old 'spatial deployment structure' figure which sorted by
    substation distance, this figure preserves the natural corridor order
    (sorted by Segment_ID) so readers can see which contiguous corridor
    sections are robustly selected vs borderline.  A second sort (by
    selection frequency) is also provided as a separate subplot to make
    the "always selected" vs "rarely selected" distinction explicit.

    Answers: "Which segments are robustly selected regardless of uncertainty?"
    """
    path = os.path.join(gsa_dir, 'segment_level_pooled.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    if ctx is None:
        ctx = _build_ctx(gsa_dir)

    usecols = ['Segment_ID', 'y']
    df_pooled = pd.read_csv(path, usecols=usecols)
    freq = df_pooled.groupby('Segment_ID')['y'].value_counts(normalize=True).unstack(fill_value=0)
    for col in ['DWC', 'SCS', 'None']:
        if col not in freq.columns:
            freq[col] = 0.0
    freq = freq.reset_index().sort_values('Segment_ID').reset_index(drop=True)

    # Corridor order (natural segment ID order)
    x_corr = np.arange(len(freq))
    # Frequency-sorted order
    freq_sort = freq.copy().sort_values('DWC', ascending=True).reset_index(drop=True)
    x_sort = np.arange(len(freq_sort))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.2))

    # ── Panel 1: corridor order ───────────────────────────────────────────────
    ax1.fill_between(x_corr, 0, freq['DWC'],
                     color=COLOR_DWC, alpha=0.9, label='P(DWC selected)', linewidth=0)
    ax1.fill_between(x_corr, freq['DWC'], freq['DWC'] + freq['SCS'],
                     color=COLOR_SCS, alpha=0.9, label='P(SCS selected)', linewidth=0)
    ax1.set_xlim(0, len(freq) - 1)
    ax1.set_ylim(0, 1.02)
    ax1.set_xlabel('Segment index (corridor order by Segment_ID)')
    ax1.set_ylabel('Selection frequency across GSA runs')
    ax1.set_title('Corridor order', loc='left')
    ax1.legend(loc='upper right', fontsize=8.5)

    # ── Panel 2: sorted by DWC frequency ─────────────────────────────────────
    ax2.fill_between(x_sort, 0, freq_sort['DWC'],
                     color=COLOR_DWC, alpha=0.9, label='P(DWC selected)', linewidth=0)
    ax2.fill_between(x_sort, freq_sort['DWC'], freq_sort['DWC'] + freq_sort['SCS'],
                     color=COLOR_SCS, alpha=0.9, label='P(SCS selected)', linewidth=0)

    # Mark "robust" threshold at 80%
    n_robust = int((freq_sort['DWC'] >= 0.8).sum())
    if n_robust > 0:
        ax2.axvline(len(freq_sort) - n_robust, color=COLOR_CI,
                    linewidth=1.0, linestyle=':', alpha=0.8)
        ax2.text(len(freq_sort) - n_robust + 1, 0.82,
                 f'≥80% robust\n({n_robust} segs)', fontsize=7.5,
                 color=COLOR_CI, va='bottom')
    ax2.set_xlim(0, len(freq_sort) - 1)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel('Segment index (sorted by DWC selection frequency ↑)')
    ax2.set_ylabel('')
    ax2.set_title('Sorted by robustness', loc='left')

    fig.suptitle('Which segments are robustly selected across the GSA ensemble?',
                 x=0.01, ha='left', fontsize=12, fontweight='bold', y=1.03)
    fig.tight_layout()

    fig_dir = _ensure_fig_dir(gsa_dir)
    out = _save_fig(fig, fig_dir, 'fig_corridor_deployment_frequency')
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SUPPLEMENTARY A: BOOTSTRAP RANK STABILITY
# ═══════════════════════════════════════════════════════════════════════════════
def plot_bootstrap_rank_stability(gsa_dir, ctx=None):
    """
    [SUPPLEMENTARY] Dot-and-whisker plot of mean rank (1 = most influential)
    with std_rank as the uncertainty band, dot size mapped to pct_top3,
    faceted DWC | SCS.  Demoted from main paper per reviewer.
    """
    path = os.path.join(gsa_dir, 'bootstrap_ranking_stability.csv')
    if not os.path.exists(path):
        print(f"  Skipping bootstrap stability plot — {path} not found.")
        return None
    df = pd.read_csv(path)
    if ctx is None:
        ctx = _build_ctx(gsa_dir)
    short_names = _DEFAULT_SHORT_NAMES

    classes = [c for c in ['DWC', 'SCS'] if c in df['class'].unique()] or sorted(df['class'].unique())
    color_map = {'DWC': COLOR_DWC, 'SCS': COLOR_SCS}

    fig, axes = plt.subplots(1, len(classes), figsize=(5.0 * len(classes), 4.4), sharex=False)
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        sub = df[df['class'] == cls].copy().sort_values('mean_rank', ascending=True)
        y_pos = np.arange(len(sub))
        names = [short_names.get(p, p) for p in sub['parameter']]
        c = color_map.get(cls, C_GREY)

        ax.errorbar(sub['mean_rank'], y_pos, xerr=sub['std_rank'], fmt='none',
                    ecolor=c, elinewidth=1.3, capsize=3, alpha=0.55, zorder=2)
        sizes = 40 + 260 * sub['pct_top3'].clip(0, 1)
        ax.scatter(sub['mean_rank'], y_pos, s=sizes, color=c,
                   edgecolor='white', linewidth=0.8, zorder=3)

        for yp, mr, p1 in zip(y_pos, sub['mean_rank'], sub['pct_top1']):
            ax.text(mr, yp + 0.32, f"{p1*100:.0f}% top-1", ha='center', va='bottom',
                    fontsize=7, color=COLOR_CI)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9.5)
        ax.invert_xaxis()
        ax.set_xlabel('Mean rank across run-resamples (1 = most influential)')
        ax.set_title(f'{cls}', loc='left', fontsize=12)
        ax.margins(y=0.12)

    legend_el = [Line2D([0], [0], marker='o', color='w', markerfacecolor=C_GREY,
                         markersize=11, label='larger = more often in top-3')]
    fig.legend(handles=legend_el, loc='lower center', ncol=1,
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle('[Supplementary] Ranking stability under run-level block bootstrap',
                 x=0.01, ha='left', fontsize=12, fontweight='bold', y=1.02)
    fig.tight_layout()

    fig_dir = _ensure_fig_dir(gsa_dir)
    out = _save_fig(fig, fig_dir, 'supp_bootstrap_rank_stability')
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — SUPPLEMENTARY B: INPUT CORRELATION STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════
def plot_input_correlation_structure(gsa_dir, ctx=None):
    """
    [SUPPLEMENTARY] Pearson + Spearman correlation heatmap among swept
    parameters. Confirms LHS near-orthogonality. Appendix only.
    """
    path = os.path.join(gsa_dir, 'lhs_parameter_samples.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df_lhs = pd.read_csv(path)
    if ctx is None:
        ctx = _build_ctx(gsa_dir)
    bounds = ctx['param_bounds']
    short_names = _DEFAULT_SHORT_NAMES

    params = [p for p in bounds.keys() if p in df_lhs.columns]
    X = df_lhs[params].astype(float)
    pearson = X.corr(method='pearson').values
    spearman = X.corr(method='spearman').values

    combo = np.full_like(pearson, np.nan)
    n = len(params)
    for i in range(n):
        for j in range(n):
            if i < j:
                combo[i, j] = pearson[i, j]
            elif i > j:
                combo[i, j] = spearman[i, j]
            else:
                combo[i, j] = 1.0

    names = [short_names.get(p, p) for p in params]
    cmap = _diverging_cmap(COLOR_SCS, COLOR_DWC)
    norm = TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    fig, ax = plt.subplots(figsize=(0.55 * n + 2.2, 0.55 * n + 2.0))
    im = ax.imshow(combo, cmap=cmap, norm=norm)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val = combo[i, j]
            txt_color = 'white' if abs(val) > 0.55 else COLOR_CI
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=6.6, color=txt_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', length=0)
    ax.set_title('[Supplementary] Input correlation  (upper: Pearson · lower: Spearman)',
                 loc='left', fontsize=10.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('correlation')
    fig.tight_layout()

    fig_dir = _ensure_fig_dir(gsa_dir)
    out = _save_fig(fig, fig_dir, 'supp_input_correlation_structure')
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — SUPPLEMENTARY C: PARAMETER SPACE COVERAGE
# ═══════════════════════════════════════════════════════════════════════════════
def plot_parameter_space_coverage(gsa_dir, ctx=None):
    """
    [SUPPLEMENTARY] Parallel-coordinates plot of LHS draws. Methods-check
    figure confirming joint space coverage. Appendix only.
    """
    path = os.path.join(gsa_dir, 'lhs_parameter_samples.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    df_lhs = pd.read_csv(path)
    if ctx is None:
        ctx = _build_ctx(gsa_dir)
    bounds = ctx['param_bounds']
    short_names = _DEFAULT_SHORT_NAMES
    manifest = ctx['manifest']
    base_used = manifest.get('base_params_used', {})

    params = [p for p in bounds.keys() if p in df_lhs.columns]
    n = len(params)
    X = df_lhs[params].astype(float)
    X_norm = (X - X.min()) / (X.max() - X.min()).replace(0, 1.0)

    fig, ax = plt.subplots(figsize=(1.05 * n + 1.5, 4.6))
    x_axis = np.arange(n)

    for _, row in X_norm.iterrows():
        ax.plot(x_axis, row.values, color=COLOR_DWC, alpha=0.025, linewidth=0.7,
                solid_capstyle='round')

    for i, p in enumerate(params):
        base_val = base_used.get(p)
        if base_val is not None:
            try:
                bv = float(base_val)
                lo, hi = X[p].min(), X[p].max()
                bv_norm = (bv - lo) / (hi - lo) if hi > lo else 0.5
                ax.scatter([i], [np.clip(bv_norm, 0, 1)], color=C_GOLD,
                           s=46, zorder=5, edgecolor='white', linewidth=0.8, marker='D')
            except (TypeError, ValueError):
                pass

    ax.set_xticks(x_axis)
    ax.set_xticklabels([short_names.get(p, p) for p in params], rotation=30, ha='right', fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['min', 'max'], fontsize=8.5)
    ax.set_ylim(-0.05, 1.05)
    for i in x_axis:
        ax.axvline(i, color=C_GREY, linewidth=0.7, alpha=0.6, zorder=0)
    ax.set_ylabel('Normalized sampled range')
    ax.set_title('[Supplementary] Joint parameter space coverage  (each line = one LHS draw)',
                 loc='left', fontsize=11)

    legend_handles = [
        Line2D([0], [0], color=COLOR_DWC, alpha=0.6, linewidth=1.5, label='sampled run'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=C_GOLD,
               markeredgecolor='white', markersize=8, label='baseline value'),
    ]
    ax.legend(handles=legend_handles, loc='upper center', ncol=2,
              bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()

    fig_dir = _ensure_fig_dir(gsa_dir)
    out = _save_fig(fig, fig_dir, 'supp_parameter_space_coverage')
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — SUPPLEMENTARY D: CLASS BALANCE
# ═══════════════════════════════════════════════════════════════════════════════
def plot_class_balance(gsa_dir):
    """[SUPPLEMENTARY] Pooled outcome class balance — pure diagnostic."""
    path = os.path.join(gsa_dir, 'class_balance.csv')
    if not os.path.exists(path):
        print(f"  Skipping class balance plot — {path} not found.")
        return None
    df = pd.read_csv(path)
    fig_dir = _ensure_fig_dir(gsa_dir)

    color_map = {'DWC': COLOR_DWC, 'SCS': COLOR_SCS, 'None': COLOR_NONE}
    class_labels = df['class'].astype(str).tolist()
    colors = [color_map.get(c, C_GREY) for c in class_labels]

    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    ax.bar(range(len(df)), df['fraction'], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(class_labels)
    for i, (cls, frac, cnt) in enumerate(zip(class_labels, df['fraction'], df['count'])):
        ax.text(i, frac + 0.01, f"{frac*100:.1f}%\n(n={int(cnt):,})",
                ha='center', va='bottom', fontsize=7.5)
    ax.set_ylabel('Fraction of (run × segment) rows')
    ax.set_title('[Supplementary] Pooled outcome class balance', loc='left')
    ax.set_ylim(0, min(1.05, df['fraction'].max() * 1.35 + 0.05))

    fig.tight_layout()
    out = _save_fig(fig, fig_dir, 'supp_class_balance')
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — SUPPLEMENTARY E/F: DWC-vs-SCS WIN RATE
# ═══════════════════════════════════════════════════════════════════════════════
def build_run_level_outcome_table(gsa_dir):
    out_path = os.path.join(gsa_dir, 'run_level_outcome_summary.csv')
    pooled_path = os.path.join(gsa_dir, 'segment_level_pooled.csv')
    if not os.path.exists(pooled_path):
        raise FileNotFoundError(
            f"{pooled_path} not found -- run "
            f"Main_sensitivity.build_segment_level_pooled_table first."
        )

    usecols = ['run_tag', 'y']
    df = pd.read_csv(pooled_path, usecols=usecols)
    grp = df.groupby(['run_tag', 'y']).size().unstack(fill_value=0)
    for col in ['DWC', 'SCS', 'None']:
        if col not in grp.columns:
            grp[col] = 0
    grp = grp.rename(columns={'DWC': 'n_dwc', 'SCS': 'n_scs', 'None': 'n_none'})
    grp['n_total'] = grp[['n_dwc', 'n_scs', 'n_none']].sum(axis=1)
    grp['dwc_share'] = grp['n_dwc'] / grp['n_total'].replace(0, np.nan)
    grp['scs_share'] = grp['n_scs'] / grp['n_total'].replace(0, np.nan)

    def _winner(row):
        if row['n_dwc'] > row['n_scs']:
            return 'DWC'
        if row['n_scs'] > row['n_dwc']:
            return 'SCS'
        return 'Tie'
    grp['winner'] = grp.apply(_winner, axis=1)
    grp = grp.reset_index()
    grp.to_csv(out_path, index=False)
    print(f"Run-level outcome summary saved -> {out_path}  ({len(grp)} runs)")
    return grp


def plot_dwc_vs_scs_win_rate(gsa_dir, df_runs=None):
    """[SUPPLEMENTARY] DWC vs SCS win-rate bar + margin histogram."""
    if df_runs is None:
        df_runs = build_run_level_outcome_table(gsa_dir)

    n_runs = len(df_runs)
    win_counts = df_runs['winner'].value_counts().reindex(['DWC', 'SCS', 'Tie']).fillna(0)
    color_map = {'DWC': COLOR_DWC, 'SCS': COLOR_SCS, 'Tie': COLOR_NONE}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.8))

    bars = ax1.bar(win_counts.index, win_counts.values,
                   color=[color_map[k] for k in win_counts.index],
                   edgecolor='white', linewidth=0.6)
    for b, v in zip(bars, win_counts.values):
        pct = 100.0 * v / max(n_runs, 1)
        ax1.text(b.get_x() + b.get_width() / 2, v + max(win_counts.values) * 0.02,
                 f"{int(v)}\n({pct:.0f}%)", ha='center', va='bottom', fontsize=8.5)
    ax1.set_ylabel('Number of GSA runs')
    ax1.set_title(f'[Supplementary] DWC vs SCS — run-level winner  (N={n_runs})', loc='left')
    ax1.set_ylim(0, max(win_counts.values) * 1.25 if win_counts.values.max() > 0 else 1)

    margin = (df_runs['n_dwc'] - df_runs['n_scs']).values
    bins = np.linspace(margin.min(), margin.max(), 30) if margin.max() > margin.min() else 10
    n, bins_edges, patches = ax2.hist(margin, bins=bins, edgecolor='white', linewidth=0.4)
    for patch, left_edge, right_edge in zip(patches, bins_edges[:-1], bins_edges[1:]):
        mid = 0.5 * (left_edge + right_edge)
        patch.set_facecolor(COLOR_DWC if mid >= 0 else COLOR_SCS)
    ax2.axvline(0, color='black', linewidth=0.9, linestyle='--')
    ax2.set_xlabel('Segments DWC − segments SCS  (per run)')
    ax2.set_ylabel('Number of GSA runs')
    ax2.set_title('Margin of DWC favorability per run', loc='left')

    legend_handles = [
        Patch(facecolor=COLOR_DWC, label='DWC favored'),
        Patch(facecolor=COLOR_SCS, label='SCS favored'),
    ]
    ax2.legend(handles=legend_handles, loc='upper right')

    fig.tight_layout()
    fig_dir = _ensure_fig_dir(gsa_dir)
    out = _save_fig(fig, fig_dir, 'supp_dwc_vs_scs_win_rate')
    plt.close(fig)
    return out


def plot_dwc_scs_share_vs_parameter(gsa_dir, param_name, df_runs=None, ctx=None):
    """[SUPPLEMENTARY] Scatter of realized DWC/SCS share vs one swept parameter."""
    if df_runs is None:
        df_runs = build_run_level_outcome_table(gsa_dir)
    lhs_path = os.path.join(gsa_dir, 'lhs_parameter_samples.csv')
    if not os.path.exists(lhs_path):
        raise FileNotFoundError(f"{lhs_path} not found.")
    df_lhs = pd.read_csv(lhs_path)
    if param_name not in df_lhs.columns:
        raise KeyError(f"'{param_name}' not found in lhs_parameter_samples.csv")

    merged = df_runs.merge(df_lhs[['run_tag', param_name]], on='run_tag', how='left')
    display_names = (ctx or {}).get('display_names', _DEFAULT_DISPLAY_NAMES)

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.scatter(merged[param_name], merged['dwc_share'], s=14, alpha=0.6,
               color=COLOR_DWC, linewidths=0, label='DWC share')
    ax.scatter(merged[param_name], merged['scs_share'], s=14, alpha=0.6,
               color=COLOR_SCS, linewidths=0, label='SCS share')
    ax.set_xlabel(display_names.get(param_name, param_name))
    ax.set_ylabel('Share of segments selected (per run)')
    ax.set_title(f'[Supplementary] Realized DWC/SCS share vs {param_name}', loc='left')
    ax.legend(loc='best')

    fig_dir = _ensure_fig_dir(gsa_dir)
    basename = f'supp_dwc_scs_share_vs_{param_name}'
    out = _save_fig(fig, fig_dir, basename)
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — TOP-LEVEL ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════
def generate_all_plots(gsa_dir, param_display_names=None, n_sample=15000,
                        top_k_marginal=3, share_vs_params=None):
    """
    Generate every figure described above — reading ONLY from already-saved
    CSVs in gsa_dir. Never re-fits, never re-solves.

    Main paper set (Figs 1–5):
        fig_parameter_importance
        fig_marginal_effect_curves
        fig_phase_diagram_C_DWC_base_vs_g_ev
        fig_breakeven_frontier
        fig_corridor_deployment_frequency

    Supplementary set:
        supp_bootstrap_rank_stability
        supp_input_correlation_structure
        supp_parameter_space_coverage
        supp_class_balance
        supp_dwc_vs_scs_win_rate
        supp_dwc_scs_share_vs_<param>  (for top swept params)
    """
    print(f"\nGenerating sensitivity plots from: {gsa_dir}")
    manifest = _load_manifest(gsa_dir)
    if manifest:
        print(f"  Study provenance: N_feasible={manifest.get('n_feasible')}, "
              f"seed={manifest.get('seed')}, "
              f"fixed_overrides={manifest.get('fixed_overrides')}")

    # build_readable_influence_table(gsa_dir, param_display_names=param_display_names)
    # ctx = _build_ctx(gsa_dir, param_display_names=param_display_names, n_sample=n_sample)

    # ── Main paper figures ────────────────────────────────────────────────────
    print("\n── Main paper figures ──")
    # plot_parameter_importance(gsa_dir, ctx=ctx)
    # plot_marginal_effect_curves(gsa_dir, ctx=ctx, top_k=top_k_marginal)
    # plot_phase_diagram(gsa_dir, ctx=ctx,
    #                     x_param='C_DWC_base', y_param='g_ev')
    # plot_breakeven_frontier(gsa_dir, ctx=ctx,
    #                          cost_param='C_DWC_base', ev_param='g_ev')
    # plot_corridor_deployment_frequency(gsa_dir, ctx=ctx)

    # ── Supplementary figures ─────────────────────────────────────────────────
    print("\n── Supplementary figures ──")
    # plot_bootstrap_rank_stability(gsa_dir, ctx=ctx)
    # plot_input_correlation_structure(gsa_dir, ctx=ctx)
    # plot_parameter_space_coverage(gsa_dir, ctx=ctx)
    # plot_class_balance(gsa_dir)

    # df_runs = build_run_level_outcome_table(gsa_dir)
    # plot_dwc_vs_scs_win_rate(gsa_dir, df_runs=df_runs)

    # sweep_params = share_vs_params or _top_theta_params(ctx, k=top_k_marginal)
    # for p in sweep_params:
    #     try:
    #         plot_dwc_scs_share_vs_parameter(gsa_dir, p, df_runs=df_runs, ctx=ctx)
    #     except (FileNotFoundError, KeyError) as e:
    #         print(f"  Skipping share-vs-{p} plot: {e}")

    print("  Sensitivity plots skipped for this run configuration.")

    # print(f"\nAll figures written to: {os.path.join(gsa_dir, 'Figures')}/")
    # print(f"  Main paper  : fig_parameter_importance, fig_marginal_effect_curves,")
    # print(f"                fig_phase_diagram_*, fig_breakeven_frontier,")
    # print(f"                fig_corridor_deployment_frequency")
    # print(f"  Supplementary: supp_bootstrap_rank_stability, supp_input_correlation_*,")
    # print(f"                 supp_parameter_space_coverage, supp_class_balance,")
    # print(f"                 supp_dwc_vs_scs_win_rate, supp_dwc_scs_share_vs_*")
    # print(f"Readable table written to: {os.path.join(gsa_dir, 'parameter_influence_table.csv')}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — STANDALONE CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════
def _autodiscover_gsa_dir():
    cands = sorted(glob.glob("GSA_*/"), key=os.path.getmtime, reverse=True)
    if not cands:
        cands = sorted(glob.glob("*/GSA_*/"), key=os.path.getmtime, reverse=True)
    return cands[0].rstrip("/") if cands else None


def main():
    """
        python Sensitivity_Plots.py
        python Sensitivity_Plots.py /path/to/GSA_20260618_120000
        python Sensitivity_Plots.py /path/to/GSA_20260618_120000 --top-k-marginal 3
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate sensitivity-analysis plots from a saved GSA_<timestamp> "
                    "directory (read-only; never re-solves or re-fits anything)."
    )
    parser.add_argument("gsa_dir", nargs="?", default=None,
                        help="Path to the GSA_<timestamp> directory. If omitted, "
                             "auto-discovers the most recently modified GSA_*/ "
                             "directory in the current working directory.")
    parser.add_argument("--n-sample", type=int, default=15000,
                        help="Subsample size for pooled-table scatter plots.")
    parser.add_argument("--top-k-marginal", type=int, default=3,
                        help="Number of top swept parameters used for the "
                             "marginal-effect-curve and share-vs-parameter figures.")
    args = parser.parse_args()

    gsa_dir = args.gsa_dir or _autodiscover_gsa_dir()
    if gsa_dir is None:
        print("ERROR: no GSA_<timestamp> directory given and none could be "
              "auto-discovered in the current directory.")
        raise SystemExit(1)
    print(f"Using GSA directory: {gsa_dir}")

    generate_all_plots(
        gsa_dir,
        n_sample=args.n_sample,
        top_k_marginal=args.top_k_marginal,
    )


if __name__ == '__main__':
    main()
