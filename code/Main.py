#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# Main.py  —  Sensitivity Analysis Runner
#
# Imports from:
#   Main_optimizer.py   — load_data, run_single_optimization, save_results,
#                          run_batch, DEFAULT_PARAMS
#   Main_plots.py       — generate_all_plots, generate_sensitivity_plots
#                          (single-run / OAT plots — unrelated to GSA)
#   Main_sensitivity.py — run_full_sensitivity_study, run_convergence_check
#                          (GSA sampling, MILP runs, and pooled-data assembly)
#
# ── Sensitivity Modes ─────────────────────────────────────────────────────────
#   'none'           Run once with the baseline parameters.
#   'dwc_cost'       Sweep C_DWC_base across a list of multipliers.
#   'ev_growth'      Sweep g_ev across low / moderate / high scenarios.
#   'traffic_growth' Sweep g_aadt across low / moderate / high scenarios.
#   'eta_split'      Sweep ETA_DWC and ETA_SCS jointly across paired lists.
#   'eta_cost_sweep' Sweep ETA_DWC from 0 to 1 while ETA_SCS = 1 - ETA_DWC;
#                    identify the minimum delivered cost and maximum NPV.
#   'custom'         Fully user-specified list of param dicts.
#   'breakeven_grid_dwc'   2D controlled grid: EV growth × DWC cost multiplier.
#   'breakeven_grid_scs'   2D controlled grid: EV growth × SCS cost multiplier.
#   'breakeven_grid_both'  2D controlled grid: EV growth × common cost multiplier.
#   'project_life_ev_growth_grid'
#                    2D controlled grid: project life × EV growth.
#   'gsa'            Draw constrained Latin-hypercube-based parameter samples,
#                    solve the MILP once per sample, and assemble the pooled
#                    run-by-segment dataset for later statistical analysis.
#   'gsa_convergence' Generate independent GSA datasets at several sample sizes
#                    (e.g. 300/600/1000). It does not fit or compare models.
#
# ── Per-run outputs (OAT modes) ───────────────────────────────────────────────
#   Results_<run_tag>/       — all CSVs (same schema as Main_optimizer.py)
#   Plots_<run_tag>/         — single-run plots (if PLOT_EACH_RUN=True)
#
# ── GSA outputs (SENSITIVITY_MODE='gsa') ─────────────────────────────────────
#   GSA_<timestamp>/
#     ├── lhs_parameter_samples.csv         — N×k sampled theta matrix
#     ├── gsa_run_manifest.json             — full provenance: bounds, fixed
#     │                                        overrides, mip_gap, discrete
#     │                                        params, segment feature cols
#     ├── Runs/Results_gsa_XXXX/            — full per-run CSV sets (always
#     │                                        saved for pooled-data assembly)
#     ├── segment_level_pooled.csv          — the flat (run × segment) table
#     │                                        for downstream analysis
#     ├── class_balance.csv                 — DWC/SCS/None frequency
#     └── (for gsa_convergence) convergence_data_generation_summary.csv
#
# ═══════════════════════════════════════════════════════════════════════════════

import os
import copy
import datetime
import gurobipy as gp
import pandas as pd

from Main_optimizer import (
    load_data,
    run_single_optimization,
    save_results,
    run_batch,
    DEFAULT_PARAMS,
)
from Main_plots import (
    generate_all_plots,
    generate_sensitivity_plots,
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — USER CONFIGURATION  (only section you need to edit)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Input data directory ──────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.dirname(SCRIPT_DIR)
INPUT_DIR      = os.environ.get(
    "DWC_INPUT_DIR", os.path.join(REPO_ROOT, "data", "processed")
)

USE_EXISTING_SCS = True
EXCLUDE_TOP_PCT  = 1          # diagnostic AADT filter quantile (1.0 = keep all)

# ── Sensitivity mode ──────────────────────────────────────────────────────────
# 'none' | 'dwc_cost' | 'ev_growth' | 'traffic_growth' | 'eta_split'
# 'eta_cost_sweep' | 'custom' | 'breakeven_grid_dwc' | 'breakeven_grid_scs'
# 'breakeven_grid_both' | 'project_life_ev_growth_grid' | 'gsa' | 'gsa_convergence'
SENSITIVITY_MODE = 'none'     # Safe default: one baseline run
PLOT_EACH_RUN   = True


# ── Baseline parameters ───────────────────────────────────────────────────────
# These are the "center" of the GSA design space and the values used for every
# non-swept OAT run. GSA samples override these per-draw.
BASELINE_OVERRIDES = {
    # ── EV adoption & traffic capture ────────────────────────────────────────
    'rho_ev'               : 0.005,
    'ETA_DWC'              : 0.5,
    'ETA_SCS'              : 0.5,    # overwritten per-draw by the GSA module
    'Utilization_rate_SCS' : 1.0,
    # ── Deployment budget ────────────────────────────────────────────────────
    'p_param'              : 200,     # baseline for non-GSA sensitivity modes
    # ── Retail charging prices ────────────────────────────────────────────────
    'P_CHARGE_DWC'         : 0.54,
    'P_CHARGE_SCS'         : 0.54,
    # ── Financial ─────────────────────────────────────────────────────────────
    'discount_rate'        : 0.07,
    'project_life'         : 30,
    'H_op'                 : 8760.0,
    'P_veh_DWC_kw'         : 300.0,
    'N_SCS_stalls'         : 6,
    # ── Regional cost multiplier bounds ─────────────────────────────────────
    'LAMBDA_MIN'           : 0.70,
    'LAMBDA_MAX'           : 1.30,
    # ── Growth rates (CAGR) ───────────────────────────────────────────────────
    'g_aadt'               : 0.011,
    'g_ev'                 : 0.15,
    'g_price_dwc'          : 0.02,
    'g_price_scs'          : 0.02,
    'g_lcoe'               : 0.015,
    'g_om'                 : 0.020,
}

# ── OAT sensitivity configs (used when SENSITIVITY_MODE is an OAT mode) ──────
DWC_COST_MULTIPLIERS    = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
EV_GROWTH_SCENARIOS     = {'low': 0.15, 'moderate': 0.20, 'high': 0.25}
TRAFFIC_GROWTH_SCENARIOS= {'low': 0.005, 'moderate': 0.010, 'high': 0.020}
ETA_DWC_VALUES          = [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
ETA_SCS_VALUES          = [round(0.9 - v, 2) for v in ETA_DWC_VALUES]
SA_CUSTOM_PARAMS        = ['project_life']
SA_CUSTOM_VALUES        = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
SA_CUSTOM_LABEL         = "Project Life"

# ── ETA allocation sweep ─────────────────────────────────────────────────────
# ETA_SCS is set automatically to 1 - ETA_DWC.
# A 0.05 step gives 21 runs, not thousands.
ETA_COST_SWEEP_DWC_VALUES = [i / 100 for i in range(101)]

# ── Breakeven grid configs (controlled 2D sweeps) ─────────────────────────────
BREAKEVEN_EV_GROWTH_VALUES = [
    0.05, 0.075, 0.10,
    0.125, 0.15, 0.175, 0.20, 0.225, 0.25
]

BREAKEVEN_DWC_COST_MULTIPLIERS = [
    0.50, 0.75, 1.00, 1.25, 1.50,
    1.75, 2.00, 2.25, 2.50,
]

BREAKEVEN_SCS_COST_MULTIPLIERS = [
    0.50, 0.75, 1.00, 1.25, 1.50,
    1.75, 2.00, 2.25, 2.50,
]

# ── Controlled 2D grid: project life × EV growth ─────────────────────────────
PROJECT_LIFE_EV_GRID_PROJECT_LIFE_VALUES = [
    10, 15, 20, 25, 30, 35, 40,
]

PROJECT_LIFE_EV_GRID_EV_GROWTH_VALUES = [
    0.05, 0.075, 0.10,
    0.125, 0.15, 0.175, 0.20, 0.225, 0.25,
]

# ── Output and plot flags ─────────────────────────────────────────────────────
BASE_OUTPUT_DIR = os.environ.get(
    "DWC_OUTPUT_DIR", os.path.join(REPO_ROOT, "results")
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0b — GSA CONFIGURATION
# (Only relevant when SENSITIVITY_MODE in {'gsa', 'gsa_convergence'})
#
# PARAMETER BOUNDS — anchored to BASELINE_OVERRIDES, matching the bounds you
# specified directly:
#   ETA_DWC               baseline 0.50 -> [0.10, 0.90]
#   ETA_SCS               baseline 0.50 -> [0.10, 0.90]
#   allocation constraint ETA_DWC + ETA_SCS <= 1.0
#   g_aadt                baseline 0.011 -> [0.008, 0.02]
#   g_ev                   baseline 0.15  -> [0.10, 0.18]
#   project_life            baseline 30    -> [25, 40]   (TRUE discrete integers)
#   C_DWC_base               baseline $10M  -> [$8M, $12M]
#   C_SCS_base_per_port       baseline $250k -> [$180k, $300k]
#   N_SCS_stalls             baseline 6     -> [4, 8]      (TRUE discrete integers)
#   g_lcoe                   baseline 0.015 -> [0.005, 0.025]
#
# p_param is INTENTIONALLY HELD FIXED at 200 deployed units (via
# GSA_FIXED_OVERRIDES below) across every GSA draw — we are measuring how a
# FIXED deployment budget gets reallocated between DWC / SCS / neither as the
# economic environment varies,
# not how the answer changes by simply allowing more infrastructure.
# ═══════════════════════════════════════════════════════════════════════════════

# ── Number of MILP solves for the GSA sweep ───────────────────────────────────
# k=9 parameters. Each draw produces one MILP run and contributes all segments
# to the pooled downstream dataset.
GSA_N_SAMPLES    = 1000       # <<< ADJUST based on single-run solve time

# ── Sample sizes used by SENSITIVITY_MODE='gsa_convergence' ──────────────────
GSA_CONVERGENCE_SAMPLE_SIZES = (300, 600, 1000)

# ── RNG seed for full reproducibility ────────────────────────────────────────
GSA_SEED         = 42

# ── GSA fixed (non-swept) parameters ─────────────────────────────────────────
# These override base_params after every sampled draw, so they are immutably
# fixed across all N GSA runs. p_param is always included.
GSA_FIXED_OVERRIDES = {
    'p_param' : 200,
    # 'mip_gap': 0.01,   # relax MIP gap for faster overnight solves if needed
}

# ── MIP gap for GSA runs ──────────────────────────────────────────────────────
GSA_MIP_GAP = 0.01    # <<< 1% optimality gap across all GSA draws


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PARAMETER BUILDERS  (OAT modes; unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

def _base_params():
    """Return a baseline params dict with BASELINE_OVERRIDES applied."""
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update(BASELINE_OVERRIDES)
    return p


def build_dwc_cost_params(base_params, multipliers):
    base_cost    = base_params['C_DWC_base']
    params_list, param_values = [], []
    for m in multipliers:
        p = copy.deepcopy(base_params)
        p['C_DWC_base'] = base_cost * m
        p['run_tag']    = f"dwc_cost_x{m:.2f}_p{p['p_param']}"
        params_list.append(p)
        param_values.append(round(base_cost * m / 1e6, 2))
    return params_list, param_values, "DWC Base Capital Cost ($M)"


def build_ev_growth_params(base_params, scenarios):
    params_list, param_values = [], []
    for label, g_ev_val in scenarios.items():
        p = copy.deepcopy(base_params)
        p['g_ev']    = float(g_ev_val)
        p['run_tag'] = f"ev_growth_{label}_gev{int(g_ev_val*100):02d}pct_p{p['p_param']}"
        params_list.append(p)
        param_values.append(round(float(g_ev_val), 4))
    return params_list, param_values, "EV Penetration Growth Rate  g_ev  (CAGR)"


def build_traffic_growth_params(base_params, scenarios):
    params_list, param_values = [], []
    for label, g_aadt_val in scenarios.items():
        p = copy.deepcopy(base_params)
        p['g_aadt']  = float(g_aadt_val)
        p['run_tag'] = f"traffic_growth_{label}_gaadt{int(g_aadt_val*1000):03d}bps_p{p['p_param']}"
        params_list.append(p)
        param_values.append(round(float(g_aadt_val), 4))
    return params_list, param_values, "AADT Growth Rate  g_aadt  (CAGR)"


def build_eta_split_params(base_params, eta_dwc_values, eta_scs_values):
    if len(eta_dwc_values) != len(eta_scs_values):
        raise ValueError("ETA_DWC_VALUES and ETA_SCS_VALUES must have the same length.")
    params_list, param_values = [], []
    for eta_d, eta_s in zip(eta_dwc_values, eta_scs_values):
        p = copy.deepcopy(base_params)
        p['ETA_DWC'] = float(eta_d)
        p['ETA_SCS'] = float(eta_s)
        p['run_tag'] = f"eta_d{eta_d:.2f}_s{eta_s:.2f}_p{p['p_param']}"
        params_list.append(p)
        param_values.append(round(float(eta_d), 4))
    return params_list, param_values, "ETA_DWC  (DWC traffic-capture efficiency)"


def build_eta_cost_sweep_params(base_params, eta_dwc_values):
    """
    Sweep ETA_DWC while holding total assumed capture fixed:

        ETA_DWC + ETA_SCS = 1.

    ETA remains an exogenous scenario parameter; it is not optimized inside
    the MILP.
    """
    params_list, param_values = [], []

    for eta_dwc in eta_dwc_values:
        eta_dwc = round(float(eta_dwc), 6)
        eta_scs = round(1.0 - eta_dwc, 6)

        p = copy.deepcopy(base_params)
        p['ETA_DWC'] = eta_dwc
        p['ETA_SCS'] = eta_scs
        p['run_tag'] = (
            f"eta_cost_sweep_dwc{eta_dwc:.2f}_"
            f"scs{eta_scs:.2f}_p{p['p_param']}"
        )

        params_list.append(p)
        param_values.append({
            'eta_dwc': eta_dwc,
            'eta_scs': eta_scs,
        })

    return params_list, param_values, "ETA allocation sweep"


def build_breakeven_grid_dwc_params(base_params, ev_growth_values, dwc_cost_multipliers):
    params_list, param_values = [], []
    base_dwc_cost = base_params['C_DWC_base']
    for gev in ev_growth_values:
        for m_dwc in dwc_cost_multipliers:
            p = copy.deepcopy(base_params)
            p['g_ev'] = float(gev)
            p['C_DWC_base'] = base_dwc_cost * float(m_dwc)
            p['run_tag'] = (
                f"breakeven_dwc_gev{int(round(gev * 1000)):03d}bps_"
                f"cdwcx{m_dwc:.2f}_p{p['p_param']}"
            )
            params_list.append(p)
            param_values.append({'g_ev': float(gev), 'dwc_cost_multiplier': float(m_dwc)})
    return params_list, param_values, "Breakeven grid: EV growth × DWC cost"


def build_breakeven_grid_scs_params(base_params, ev_growth_values, scs_cost_multipliers):
    params_list, param_values = [], []
    base_scs_cost = base_params['C_SCS_base_per_port']
    for gev in ev_growth_values:
        for m_scs in scs_cost_multipliers:
            p = copy.deepcopy(base_params)
            p['g_ev'] = float(gev)
            p['C_SCS_base_per_port'] = base_scs_cost * float(m_scs)
            p['run_tag'] = (
                f"breakeven_scs_gev{int(round(gev * 1000)):03d}bps_"
                f"cscsx{m_scs:.2f}_p{p['p_param']}"
            )
            params_list.append(p)
            param_values.append({'g_ev': float(gev), 'scs_cost_multiplier': float(m_scs)})
    return params_list, param_values, "Breakeven grid: EV growth × SCS cost"


def build_breakeven_grid_both_costs_params(base_params, ev_growth_values, cost_multipliers):
    params_list, param_values = [], []
    base_dwc_cost = base_params['C_DWC_base']
    base_scs_cost = base_params['C_SCS_base_per_port']
    for gev in ev_growth_values:
        for m in cost_multipliers:
            p = copy.deepcopy(base_params)
            p['g_ev'] = float(gev)
            p['C_DWC_base'] = base_dwc_cost * float(m)
            p['C_SCS_base_per_port'] = base_scs_cost * float(m)
            p['run_tag'] = (
                f"breakeven_both_gev{int(round(gev * 1000)):03d}bps_"
                f"costx{m:.2f}_p{p['p_param']}"
            )
            params_list.append(p)
            param_values.append({'g_ev': float(gev), 'cost_multiplier': float(m)})
    return params_list, param_values, "Breakeven grid: EV growth × common cost"


def build_project_life_ev_growth_grid_params(base_params, project_life_values, ev_growth_values):
    params_list, param_values = [], []
    for project_life in project_life_values:
        for gev in ev_growth_values:
            p = copy.deepcopy(base_params)
            p['project_life'] = int(project_life)
            p['g_ev'] = float(gev)
            p['run_tag'] = (
                f"projectlife_evgrid_life{int(project_life):02d}yr_"
                f"gev{int(round(gev * 1000)):03d}bps_p{p['p_param']}"
            )
            params_list.append(p)
            param_values.append({'project_life': int(project_life), 'g_ev': float(gev)})
    return params_list, param_values, "Project life × EV growth grid"


def build_custom_params(base_params, param_names, values, label):
    if isinstance(param_names, str):
        param_names = [param_names]
    params_list = []
    for val in values:
        p = copy.deepcopy(base_params)
        for key in param_names:
            p[key] = val
        safe_label = label.replace(' ', '_')
        p['run_tag'] = f"custom_{safe_label}_{val}_p{p['p_param']}"
        params_list.append(p)
    return params_list, list(values), label


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — INITIALIZE GUROBI ENV AND LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 65)
print("  Joint DWC + SCS Sensitivity Analysis Runner")
print(f"  Mode     : {SENSITIVITY_MODE}")
print(f"  Input dir: {INPUT_DIR}")
print("=" * 65)

try:
    env = gp.Env()
    print("Gurobi environment initialised.")
except gp.GurobiError as e:
    print(f"Gurobi error: {e}")
    env = None

base = _base_params()
data = load_data(
    input_dir        = INPUT_DIR,
    use_existing_scs = USE_EXISTING_SCS,
    exclude_top_pct  = EXCLUDE_TOP_PCT,
    max_sub_rank     = 10,
    lambda_min       = base.get('LAMBDA_MIN', 0.80),
    lambda_max       = base.get('LAMBDA_MAX', 1.20),
)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — GLOBAL SENSITIVITY DATA-GENERATION BRANCH
# ═══════════════════════════════════════════════════════════════════════════════

if SENSITIVITY_MODE in ('gsa', 'gsa_convergence'):
    from Main_sensitivity import (
        run_full_sensitivity_study,
        run_convergence_check,
        GSA_PARAM_BOUNDS,
    )

    gsa_fixed = copy.deepcopy(GSA_FIXED_OVERRIDES)
    gsa_fixed['mip_gap'] = GSA_MIP_GAP

    print("\n" + "=" * 65)
    print(f"  GSA parameter bounds (k={len(GSA_PARAM_BOUNDS)}):")
    for k, v in GSA_PARAM_BOUNDS.items():
        print(f"    {k:30s}: [{v[0]}, {v[1]}]")
    print(f"  Fixed overrides for every draw: {gsa_fixed}")
    print("=" * 65)

    if SENSITIVITY_MODE == 'gsa_convergence':
        print(f"\nGenerating GSA datasets across N = {GSA_CONVERGENCE_SAMPLE_SIZES}")
        print("This independently re-solves the full MILP sweep at each sample "
              "size — expensive, intended as a one-time check before committing "
              "to a full study.\n")

        convergence_dir, df_compare = run_convergence_check(
            data              = data,
            base_params       = _base_params(),
            output_root       = BASE_OUTPUT_DIR,
            sample_sizes      = GSA_CONVERGENCE_SAMPLE_SIZES,
            seed              = GSA_SEED,
            env               = env,
            fixed_overrides   = gsa_fixed,
        )

        print(f"\n{'='*65}")
        print(f"  MULTI-SIZE GSA DATA GENERATION COMPLETE.")
        print(f"  Dataset summary saved to:")
        print(f"  {os.path.join(convergence_dir, 'convergence_data_generation_summary.csv')}")
        print(f"{'='*65}")

    else:  # SENSITIVITY_MODE == 'gsa'
        print(f"\nStarting full GSA study: N={GSA_N_SAMPLES} draws, seed={GSA_SEED}")

        gsa_dir = run_full_sensitivity_study(
            data                = data,
            base_params         = _base_params(),
            output_root         = BASE_OUTPUT_DIR,
            n_samples           = GSA_N_SAMPLES,
            seed                = GSA_SEED,
            env                 = env,
            fixed_overrides     = gsa_fixed,
        )

        print(f"\n{'='*65}")
        print(f"  GSA DATA GENERATION COMPLETE. All outputs saved to:")
        print(f"  {gsa_dir}")
        print(f"\n  Use the saved tables later with:")
        print(f"  Paper_Figures/Figure 3/Logistic_Regression.py")
        print(f"{'='*65}")

    # ── Done — skip OAT sections below ───────────────────────────────────────
    import sys; sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — OAT SCENARIO LIST  (all modes except 'gsa' / 'gsa_convergence')
# ═══════════════════════════════════════════════════════════════════════════════

if SENSITIVITY_MODE == 'none':
    p = _base_params()
    p['run_tag'] = (
        f"baseline_p{p['p_param']}"
        f"_etaDWC{p['ETA_DWC']:.2f}"
        f"_etaSCS{p['ETA_SCS']:.2f}"
        f"_gev{int(p['g_ev']*100):02d}pct"
        f"_gaadt{int(p['g_aadt']*1000):03d}bps"
    )
    params_list, param_values, param_name = [p], [0], "Baseline"

elif SENSITIVITY_MODE == 'dwc_cost':
    params_list, param_values, param_name = build_dwc_cost_params(
        _base_params(), DWC_COST_MULTIPLIERS)

elif SENSITIVITY_MODE == 'ev_growth':
    params_list, param_values, param_name = build_ev_growth_params(
        _base_params(), EV_GROWTH_SCENARIOS)

elif SENSITIVITY_MODE == 'traffic_growth':
    params_list, param_values, param_name = build_traffic_growth_params(
        _base_params(), TRAFFIC_GROWTH_SCENARIOS)

elif SENSITIVITY_MODE == 'eta_split':
    params_list, param_values, param_name = build_eta_split_params(
        _base_params(), ETA_DWC_VALUES, ETA_SCS_VALUES)

elif SENSITIVITY_MODE == 'eta_cost_sweep':
    params_list, param_values, param_name = build_eta_cost_sweep_params(
        _base_params(), ETA_COST_SWEEP_DWC_VALUES)

elif SENSITIVITY_MODE == 'custom':
    if not SA_CUSTOM_PARAMS:
        raise ValueError(
            "SENSITIVITY_MODE='custom' but SA_CUSTOM_PARAMS is empty.")
    params_list, param_values, param_name = build_custom_params(
        _base_params(), SA_CUSTOM_PARAMS, SA_CUSTOM_VALUES, SA_CUSTOM_LABEL)

elif SENSITIVITY_MODE == 'breakeven_grid_dwc':
    params_list, param_values, param_name = build_breakeven_grid_dwc_params(
        _base_params(), BREAKEVEN_EV_GROWTH_VALUES, BREAKEVEN_DWC_COST_MULTIPLIERS)

elif SENSITIVITY_MODE == 'breakeven_grid_scs':
    params_list, param_values, param_name = build_breakeven_grid_scs_params(
        _base_params(), BREAKEVEN_EV_GROWTH_VALUES, BREAKEVEN_SCS_COST_MULTIPLIERS)

elif SENSITIVITY_MODE == 'breakeven_grid_both':
    params_list, param_values, param_name = build_breakeven_grid_both_costs_params(
        _base_params(), BREAKEVEN_EV_GROWTH_VALUES, BREAKEVEN_DWC_COST_MULTIPLIERS)

elif SENSITIVITY_MODE == 'project_life_ev_growth_grid':
    params_list, param_values, param_name = build_project_life_ev_growth_grid_params(
        _base_params(), PROJECT_LIFE_EV_GRID_PROJECT_LIFE_VALUES,
        PROJECT_LIFE_EV_GRID_EV_GROWTH_VALUES)

else:
    raise ValueError(
        f"Unknown SENSITIVITY_MODE '{SENSITIVITY_MODE}'. "
        "Choose: none | dwc_cost | ev_growth | traffic_growth | "
        "eta_split | eta_cost_sweep | custom | breakeven_grid_dwc | breakeven_grid_scs | breakeven_grid_both | project_life_ev_growth_grid | gsa | gsa_convergence")

print(f"\nScenarios to run: {len(params_list)}")
print(f"{'─'*65}")
for i, (p, v) in enumerate(zip(params_list, param_values)):
    print(f"  [{i+1:3d}]  run_tag = {p['run_tag']}")
    print(f"         {param_name} = {v}"
          f"   |  g_ev={p['g_ev']:.4f}  g_aadt={p['g_aadt']:.4f}"
          f"   |  rho_ev={p['rho_ev']:.4f}")
print(f"{'─'*65}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — RUN ALL OAT SCENARIOS
# ═══════════════════════════════════════════════════════════════════════════════

results_list = []
for i, (params, pval) in enumerate(zip(params_list, param_values)):
    run_tag = params['run_tag']
    print(f"\n{'╔'+'═'*62+'╗'}")
    print(f"  Scenario {i+1}/{len(params_list)}  —  {run_tag}")
    print(f"  {param_name} = {pval}")
    print(f"{'╚'+'═'*62+'╝'}")

    result = run_single_optimization(data, params, env=env)
    if result is None:
        print(f"  WARNING: no feasible solution — skipping {run_tag}")
        continue

    out_dir = os.path.join(BASE_OUTPUT_DIR, f"Results_{run_tag}")
    save_results(result, out_dir)

    if PLOT_EACH_RUN:
        plot_dir = os.path.join(BASE_OUTPUT_DIR, f"Plots_{run_tag}")
        generate_all_plots(result, plot_dir)

    results_list.append(result)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — OAT SENSITIVITY COMPARISON PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

solved_tags = {r['run_tag'] for r in results_list}
pv_filtered = [v for p, v in zip(params_list, param_values)
               if p['run_tag'] in solved_tags]

_GRID_MODES = (
    'breakeven_grid_dwc',
    'breakeven_grid_scs',
    'breakeven_grid_both',
    'project_life_ev_growth_grid',
)

if len(results_list) > 1 and SENSITIVITY_MODE == 'eta_cost_sweep':
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    sa_dir = os.path.join(BASE_OUTPUT_DIR, f"ETA_Cost_Sweep_{ts}")
    os.makedirs(sa_dir, exist_ok=True)
    print(f"\nETA cost sweep complete. Summary outputs will be written to: {sa_dir}/")

elif len(results_list) > 1 and SENSITIVITY_MODE not in _GRID_MODES:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    sa_dir = os.path.join(BASE_OUTPUT_DIR, f"SA_Plots_{SENSITIVITY_MODE}_{ts}")
    # generate_sensitivity_plots(results_list, param_name, pv_filtered, sa_dir)

    if SENSITIVITY_MODE == 'eta_split':
        # from Main_plots import plot_sensitivity_eta_split
        # plot_sensitivity_eta_split(results_list, sa_dir)
        pass

    os.makedirs(sa_dir, exist_ok=True)
    print(f"\nSensitivity comparison plots skipped. Plot directory would have been: {sa_dir}/")

elif len(results_list) > 1 and SENSITIVITY_MODE in _GRID_MODES:
    print("\nControlled 2D grid mode: no built-in OAT sensitivity comparison plots generated.")
    print("Use the saved Results_* folders and SA_summary_table.csv for the dedicated grid/heatmap plot.")
    sa_dir = BASE_OUTPUT_DIR

elif len(results_list) == 1 and SENSITIVITY_MODE == 'none':
    print("\nSingle-run mode: no sensitivity comparison plots generated.")
    sa_dir = os.path.join(BASE_OUTPUT_DIR, f"Results_{results_list[0]['run_tag']}")

else:
    print("\nNo results to compare (all scenarios infeasible).")
    sa_dir = BASE_OUTPUT_DIR


# ── ETA-cost metrics: used only by SENSITIVITY_MODE='eta_cost_sweep' ─────────
def _eta_sweep_npv_factor(growth_rate, discount_rate, project_life):
    ratio = (1.0 + growth_rate) / (1.0 + discount_rate)
    if abs(ratio - 1.0) < 1e-12:
        return float(project_life) / (1.0 + discount_rate)
    return (
        (1.0 / (1.0 + discount_rate))
        * (1.0 - ratio ** project_life)
        / (1.0 - ratio)
    )


def _eta_sweep_numeric_sum(df, column):
    if df is None or df.empty or column not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[column], errors='coerce').fillna(0.0).sum())


def _compute_eta_cost_metrics(result):
    """Compute portfolio-level lifetime delivered cost using demand growth only."""
    meta = result['meta']
    df_dwc = result['df_dwc']
    df_scs = result['df_scs']

    rho = float(meta['discount_rate'])
    life = int(meta['project_life'])
    gamma_energy = _eta_sweep_npv_factor(float(meta['g_d']), rho, life)
    gamma_var = _eta_sweep_npv_factor(float(meta['g_var']), rho, life)
    gamma_fix = _eta_sweep_npv_factor(float(meta['g_om']), rho, life)

    capex = (
        _eta_sweep_numeric_sum(df_dwc, 'Capex_Total')
        + _eta_sweep_numeric_sum(df_scs, 'Capex_Total')
    )

    base_energy = (
        _eta_sweep_numeric_sum(df_dwc, 'E_del_DWC')
        + _eta_sweep_numeric_sum(df_scs, 'E_del_SCS')
    )
    lifetime_energy = base_energy * gamma_energy

    variable_base = 0.0
    fixed_base = 0.0

    if not df_dwc.empty:
        variable_base += float((
            pd.to_numeric(df_dwc['LCOE'], errors='coerce').fillna(0.0)
            * pd.to_numeric(df_dwc['E_del_DWC'], errors='coerce').fillna(0.0)
        ).sum())
        fixed_base += float((
            pd.to_numeric(df_dwc['Lambda'], errors='coerce').fillna(0.0)
            * float(meta['c_om_base_dwc'])
        ).sum())

    if not df_scs.empty:
        variable_base += float((
            pd.to_numeric(df_scs['LCOE'], errors='coerce').fillna(0.0)
            * pd.to_numeric(df_scs['E_del_SCS'], errors='coerce').fillna(0.0)
        ).sum())
        fixed_base += float((
            pd.to_numeric(df_scs['Lambda'], errors='coerce').fillna(0.0)
            * float(meta['c_om_base_scs'])
        ).sum())

    fixed_pv = fixed_base * gamma_fix
    variable_pv = variable_base * gamma_var
    lifetime_cost = capex + fixed_pv + variable_pv
    lcde = lifetime_cost / lifetime_energy if lifetime_energy > 0 else float('nan')

    return {
        'eta_dwc': float(meta.get('eta_dwc', result['params'].get('ETA_DWC', 0.0))),
        'eta_scs': float(meta.get('eta_scs', result['params'].get('ETA_SCS', 0.0))),
        'n_total': int(meta['n_dwc_selected']) + int(meta['n_scs_selected']),
        'lifetime_energy_kwh': lifetime_energy,
        'lifetime_cost_usd': lifetime_cost,
        'fixed_om_pv_usd': fixed_pv,
        'variable_om_pv_usd': variable_pv,
        'lcde_usd_per_kwh': lcde,
        'lcde_cents_per_kwh': 100.0 * lcde,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — OAT SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════

if results_list:
    from Main_plots import _compute_run_summary
    rows = []
    for res, pval in zip(results_list, pv_filtered):
        s = _compute_run_summary(res)
        s['param_value'] = pval
        s['project_life']= res['params'].get('project_life', None)
        s['g_ev']        = res['params'].get('g_ev',        None)
        s['g_aadt']      = res['params'].get('g_aadt',       None)
        s['g_price_dwc'] = res['params'].get('g_price_dwc',  None)
        s['g_lcoe']      = res['params'].get('g_lcoe',       None)
        s['g_om']        = res['params'].get('g_om',         None)
        rows.append(s)

    df_summary = pd.DataFrame(rows)[[
        'param_value', 'run_tag',
        'project_life', 'g_ev', 'g_aadt', 'g_price_dwc', 'g_lcoe', 'g_om',
        'n_dwc', 'n_scs',
        'obj_val', 'total_capex', 'net_op',
        'pb_disc', 'pb_simple', 'pb_unit_med',
    ]]
    df_summary.columns = [
        'Param', 'Run Tag',
        'project_life', 'g_ev', 'g_aadt', 'g_price_DWC', 'g_LCOE', 'g_OM',
        'N_DWC', 'N_SCS',
        'Obj ($)', 'CAPEX ($)', 'Net Op ($/yr)',
        'PB_disc (yr)', 'PB_simple (yr)', 'PB_unit_med (yr)',
    ]

    print(f"\n{'═'*90}")
    print("  SENSITIVITY SUMMARY TABLE")
    print(f"{'═'*90}")
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    print(f"{'═'*90}")

    summary_path = os.path.join(sa_dir, "SA_summary_table.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSummary table saved → {summary_path}")

    if SENSITIVITY_MODE == 'eta_cost_sweep':
        eta_rows = []
        for res in results_list:
            m = _compute_eta_cost_metrics(res)
            eta_rows.append({
                'Run Tag': res['run_tag'],
                'ETA_DWC': m['eta_dwc'],
                'ETA_SCS': m['eta_scs'],
                'N_DWC': int(res['meta']['n_dwc_selected']),
                'N_SCS': int(res['meta']['n_scs_selected']),
                'N_Total': m['n_total'],
                'NPV_USD': float(res['meta']['obj_val']),
                'Lifetime_Energy_kWh': m['lifetime_energy_kwh'],
                'Lifetime_Cost_USD': m['lifetime_cost_usd'],
                'Fixed_OM_PV_USD': m['fixed_om_pv_usd'],
                'Variable_OM_PV_USD': m['variable_om_pv_usd'],
                'LCDE_USD_per_kWh': m['lcde_usd_per_kwh'],
                'LCDE_cents_per_kWh': m['lcde_cents_per_kwh'],
            })

        df_eta = pd.DataFrame(eta_rows).sort_values('ETA_DWC').reset_index(drop=True)
        eta_path = os.path.join(sa_dir, 'ETA_cost_sweep_summary.csv')
        df_eta.to_csv(eta_path, index=False)

        best_cost = df_eta.loc[df_eta['LCDE_cents_per_kWh'].idxmin()]
        best_npv = df_eta.loc[df_eta['NPV_USD'].idxmax()]

        print("\nMinimum-LCDE ETA allocation:")
        print(best_cost.to_string())
        print("\nMaximum-NPV ETA allocation:")
        print(best_npv.to_string())
        print(f"\nETA sweep summary saved → {eta_path}")

print("\nDone.")
