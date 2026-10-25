# ═══════════════════════════════════════════════════════════════════════════════
# Main_optimizer.py
# Joint DWC + SCS Optimization — All logic refactored into callable functions.
#
# Public API
# ──────────
#   load_data(input_dir, use_existing_scs, exclude_top_pct, ...)
#       → data dict  (static; load once, reuse across all sensitivity runs)
#
#   run_single_optimization(data, params, env)
#       → results dict  (all solution data + DataFrames + metadata)
#
#   save_results(results, output_dir)
#       → writes the exact same CSV set as the original File 1
#         so that the plotting files (Main_plots.py) work unchanged
#
#   run_batch(data, params_list, env)
#       → list of results dicts  (one per element of params_list)
#
#   DEFAULT_PARAMS  — baseline parameter dict; override keys for sensitivity
#
# NO logic has been changed.  The only differences from the original File 1:
#   • Everything is inside functions.
#   • Parameters come in through a dict rather than module-level globals.
#   • Data loading is separated from the solve so it can be cached.
# ═══════════════════════════════════════════════════════════════════════════════

import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import os

# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONSTANTS  (never change between runs)
# ══════════════════════════════════════════════════════════════════════════════
M_TO_MILES = 0.000621371
MPH_TO_MPS = 0.44704
MW_TO_KW   = 1000.0
KW_TO_MW   = 1.0 / 1000.0

# Hourly traffic profile — fixed shape, computed once
_HOURS       = np.arange(24)
_TRAFFIC_RAW = (
    0.4 * np.exp(-0.5 * ((_HOURS - 8)  / 2)**2) +
    0.6 * np.exp(-0.5 * ((_HOURS - 17) / 3)**2) +
    0.05
)
TRAFFIC_WEIGHTS          = _TRAFFIC_RAW / _TRAFFIC_RAW.sum()
TRAFFIC_PEAK_HOUR_WEIGHT = float(np.max(TRAFFIC_WEIGHTS))

# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT PARAMETERS
# Every parameter the sensitivity analysis might vary is exposed here.
# The runner overrides only the keys it wants to sweep.
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_PARAMS = {
    # ── EV adoption & traffic capture ────────────────────────────────────────
    'rho_ev'               : 0.005,
    # 'sigma'                : 0.7,
    'ETA_DWC'              : 0.7,       # typically set equal to sigma
    'ETA_SCS'              : 0.0,
    'Utilization_rate_SCS' : 0.16,
    # ── Deployment budget ────────────────────────────────────────────────────
    'p_param'              : 10,
    # ── Retail charging prices ────────────────────────────────────────────────
    'P_CHARGE_DWC'         : 0.54,      # $/kWh
    'P_CHARGE_SCS'         : 0.54,      # $/kWh
    # ── Financial ─────────────────────────────────────────────────────────────
    'discount_rate'        : 0.07,
    'project_life'         : 30,
    'H_op'                 : 8760.0,
    # ── Growth rates (CAGR basis) — γ^AADT, ρ^EV, γ^price, γ^LCOE, γ^O&M ───
    'g_aadt'               : 0.01,      # AADT annual growth rate
    'g_ev'                 : 0.05,      # EV penetration annual growth rate (applied to rho_ev)
    # ── Nominal price growth rates (monetary streams) ─────────────────────────
    'g_price_dwc'          : 0.025,     # nominal growth rate of DWC retail price ($/kWh)
    'g_price_scs'          : 0.025,     # nominal growth rate of SCS retail price ($/kWh)
    'g_lcoe'               : 0.020,     # nominal growth rate of LCOE / electricity cost ($/kWh)
    'g_om'                 : 0.025,     # nominal growth rate of fixed O&M costs ($/yr per asset)
    # ── DWC capital ───────────────────────────────────────────────────────────
    'C_DWC_base'           : 2_000_000 * 5,
    'C_line_int_base'      : 70_000,
    'C_line_adj'           : 400_000,
    'C_line_ext_base'      : 1_390_000,
    'C_connection'         : 600_000,
    'P_DWC_coil_per_mile'  : 2,
    'P_veh_DWC_kw'         : 300.0,     # kW; on-board wireless charging receiver power per EV
    # ── SCS hardware ─────────────────────────────────────────────────────────
    'C_SCS_base_per_port'  : 250_000,
    'N_SCS_stalls'         : 40,
    'P_SCS_per_stall_kw'   : 250.0,     # kW; converted to MW inside optimizer
    # ── Penalty ──────────────────────────────────────────────────────────────
    'E_req_kwh'            : 150,
    # ── Regional cost multiplier bounds ─────────────────────────────────────
    'LAMBDA_MIN'           : 0.80,
    'LAMBDA_MAX'           : 1.50,
    # ── Solver ───────────────────────────────────────────────────────────────
    'mip_gap'              : 0.005,
    # ── Run identity (always override per sensitivity scenario) ──────────────
    # 'USE_EXISTING_SCS'     : True,
    'run_tag'              : 'default',
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — per-vehicle power draw
# ══════════════════════════════════════════════════════════════════════════════
# def calculate_p_drive(velocity_mph):
#     """Per-vehicle power draw [kW] — P_traff,i,h in the formulation."""
#     if not velocity_mph or velocity_mph <= 0:
#         velocity_mph = 55.0
#     v_mps      = velocity_mph * MPH_TO_MPS
#     P_traction = (0.5 * 0.65 * 10.2 * 1.225 * v_mps**3 +
#                   0.0068 * 30.5 * 1000 * 9.8 * v_mps)
#     P_RX       = (1 / 0.95) * P_traction + 7500
#     return P_RX / 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — cumulative present-value weight for a growing annual cash-flow stream
# ══════════════════════════════════════════════════════════════════════════════
def _npv_factor(g, rho, T):
    """
    Γ(g) = Σ_{y=1}^{T} (1+g)^{y-1} / (1+ρ)^y
    Cumulative present-value weight for a cash-flow stream whose base-year
    (year-1) value grows at CAGR g and is discounted at annual rate ρ over T
    years.  Collapses the year dimension so the NPV of a growing annual stream
    equals its base-year value multiplied by Γ(g).
    """
    r = (1.0 + g) / (1.0 + rho)
    if abs(r - 1.0) < 1e-12:
        return float(T) / (1.0 + rho)
    return (1.0 / (1.0 + rho)) * (1.0 - r ** T) / (1.0 - r)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (static; call once, reuse across all sensitivity runs)
# ══════════════════════════════════════════════════════════════════════════════
def load_data(input_dir,
              use_existing_scs = True,
              exclude_top_pct  = 1, #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
              max_sub_rank     = 10,
              lambda_min       = 0.80,
              lambda_max       = 1.20):
    """
    Load and pre-process all CSV inputs.  Returns a `data` dict that is
    independent of the sensitivity parameters (rho_ev, prices, etc.) and
    can be reused across all runs in a sensitivity sweep.
    """

    # ── Raw CSV loads ─────────────────────────────────────────────────────────
    df_segments    = pd.read_csv(os.path.join(input_dir, 'NHS_US_5miles_with_SCS.csv'))
    df_adjacency   = pd.read_csv(os.path.join(input_dir, 'NHS_USA_5miles_Adjacent.csv'))
    df_sub_dist    = pd.read_csv(os.path.join(input_dir, 'NHS_US_SubstationDistance.csv'))
    df_substations = (pd.read_csv(os.path.join(input_dir, 'SubstationsProject.csv'))
                      .drop_duplicates(subset='Sub_ID_OG'))

    scs_file    = 'US_SCS_unique_existing.csv' if use_existing_scs else 'US_SCS.csv'
    df_scs_info = pd.read_csv(os.path.join(input_dir, scs_file))

    # ── Segment index & attribute lookups ─────────────────────────────────────
    I_all        = df_segments['SEGMENT_ID_OG'].unique().tolist()
    seg_index    = df_segments.set_index('SEGMENT_ID_OG')
    aadt_com     = seg_index['AADT_COM'].to_dict()
    speed_lim    = seg_index['SPEED_LIMI'].to_dict()
    lambda_raw   = seg_index['Norm_index'].to_dict()
    lcoe_rate    = (seg_index['Total_Electricity_Rate_Nom_2024'] / 100.0).to_dict()
    seg_coords   = seg_index[['MID_X', 'MID_Y']].T.to_dict('list')
    seg_length_m = seg_index['Shape_Length'].to_dict()
    seg_near_scs = seg_index['NEAR_SCS_ID'].to_dict()

    # Clamp lambda to [lambda_min, lambda_max]
    lambda_i = {k: max(lambda_min, min(lambda_max, v)) for k, v in lambda_raw.items()}

    # ── Diagnostic AADT filter ────────────────────────────────────────────────
    _aadt_series = pd.Series({i: (aadt_com.get(i, 0) or 0) for i in I_all})
    _threshold   = _aadt_series.quantile(exclude_top_pct)
    _top_segs    = set(_aadt_series[_aadt_series > _threshold].index.tolist())

    print(f"\nDIAGNOSTIC FILTER: excluding segments above "
          f"{exclude_top_pct*100:.0f}th percentile AADT")
    print(f"  AADT threshold            : {int(_threshold):,}")
    print(f"  Segments removed          : {len(_top_segs)}")
    _remaining = _aadt_series.drop(index=list(_top_segs))
    print(f"  AADT range after  filter  : "
          f"{int(_remaining.min()):,} – {int(_remaining.max()):,}")
    print(f"  AADT median after filter  : {int(_remaining.median()):,}")

    I     = [i for i in I_all if i not in _top_segs]
    I_set = set(I)
    seg_length_miles = {i: seg_length_m[i] * M_TO_MILES for i in I}

    # ── SCS sets: all sites referenced by surviving segments ──────────────────
    all_scs_ids_from_segments = set(
        int(v) for k, v in seg_near_scs.items()
        if k in I_set and pd.notna(v) and int(v) != -1
    )

    # Remove sites with no valid substation (NEAR_FID == -1)
    if 'SCS_ID' in df_scs_info.columns and 'NEAR_FID' in df_scs_info.columns:
        scs_ids_no_substation = set(
            int(row['SCS_ID'])
            for _, row in df_scs_info.iterrows()
            if pd.isna(row['NEAR_FID']) or int(row['NEAR_FID']) == -1
        )
    else:
        scs_ids_no_substation = set()

    all_scs_ids_from_segments -= scs_ids_no_substation
    print(f"\nSCS sites with no valid substation (NEAR_FID=-1), excluded: "
          f"{len(scs_ids_no_substation)}")

    # ── Existing-SCS partition (USE_EXISTING_SCS formulation) ─────────────────
    # ── Existing-SCS partition (USE_EXISTING_SCS formulation) ─────────────────
    if use_existing_scs:
        # Rule: any SCS_ID whose EXISTING_SCS_ID is not -1 (and not NaN) is
        # pre-served by an existing station and must be excluded from the
        # optimisation entirely.  No deduplication by existing-station ID is
        # applied; every such candidate is excluded independently.
        excluded_cand_ids = set()
        if 'EXISTING_SCS_ID' in df_scs_info.columns and 'SCS_ID' in df_scs_info.columns:
            for _, row in df_scs_info.iterrows():
                ex_id   = row['EXISTING_SCS_ID']
                cand_id = int(row['SCS_ID'])
                if cand_id in scs_ids_no_substation:
                    continue
                # Exclude if EXISTING_SCS_ID is a real value (not NaN, not -1)
                if not pd.isna(ex_id) and int(ex_id) != -1:
                    excluded_cand_ids.add(cand_id)

        existing_scs_ids = excluded_cand_ids  # kept for print/diagnostic parity
        R_cand  = sorted(all_scs_ids_from_segments - excluded_cand_ids)
        R_exist = sorted(all_scs_ids_from_segments & excluded_cand_ids)

        print(f"\nSCS partition (derived from segment file):")
        print(f"  Total SCS sites (after no-sub filter)    : {len(all_scs_ids_from_segments)}")
        print(f"  Candidates excluded (existing stn nearby): {len(excluded_cand_ids)}")
        print(f"  R_exist (excluded; existing stn present) : {len(R_exist)}")
        print(f"  R_cand  (candidates for optimisation)    : {len(R_cand)}")
    else:
        existing_map      = {}
        excluded_cand_ids = set()
        existing_scs_ids  = set()
        R_cand  = sorted(all_scs_ids_from_segments)
        R_exist = []
        print(f"\nSCS sites (all candidate, after no-sub filter): {len(R_cand)}")

    # ── SCS ↔ segment reverse mappings ───────────────────────────────────────
    scs_to_segs_cand  = {r: [] for r in R_cand}
    scs_to_segs_exist = {r: [] for r in R_exist}
    for seg_id, row in seg_index.iterrows():
        if seg_id not in I_set:
            continue
        r_id_raw = row.get('NEAR_SCS_ID')
        if pd.isna(r_id_raw):
            continue
        r_id = int(r_id_raw)
        if r_id in scs_to_segs_cand:
            scs_to_segs_cand[r_id].append(seg_id)
        if r_id in scs_to_segs_exist:
            scs_to_segs_exist[r_id].append(seg_id)

    # ── Substation sets and coordinate lookups ────────────────────────────────
    df_sub_dist_filtered = df_sub_dist[df_sub_dist['NEAR_RANK'] <= max_sub_rank]
    sub_coords = df_substations.set_index('Sub_ID_OG')[['X', 'Y']].T.to_dict('list')

    # ── SCS candidate substation assignments ─────────────────────────────────
    if use_existing_scs:
        df_scs_cand_info = df_scs_info[
            ~df_scs_info['SCS_ID'].isin(excluded_cand_ids)
        ].copy() if 'SCS_ID' in df_scs_info.columns else pd.DataFrame()
    else:
        df_scs_cand_info = df_scs_info.copy()

    if not df_scs_cand_info.empty and 'SCS_ID' in df_scs_cand_info.columns:
        scs_coords_cand = (df_scs_cand_info.set_index('SCS_ID')[['MID_X', 'MID_Y']]
                           .T.to_dict('list')
                           if 'MID_X' in df_scs_cand_info.columns else {})
        scs_nearest_sub_cand  = (df_scs_cand_info.set_index('SCS_ID')['NEAR_FID'].to_dict()
                                  if 'NEAR_FID'  in df_scs_cand_info.columns else {})
        scs_nearest_dist_cand = (df_scs_cand_info.set_index('SCS_ID')['NEAR_DIST'].to_dict()
                                  if 'NEAR_DIST' in df_scs_cand_info.columns else {})
    else:
        scs_coords_cand = scs_nearest_sub_cand = scs_nearest_dist_cand = {}

    # ── D_{r(i),s}: candidate SCS → nearest substation distance [miles] ───────
    scs_sub_dist_pairs = {}
    for r in R_cand:
        s_near = scs_nearest_sub_cand.get(r)
        d_near = scs_nearest_dist_cand.get(r)
        if s_near is None or pd.isna(s_near) or int(s_near) == -1:
            continue
        if d_near is None or pd.isna(d_near) or float(d_near) < 0:
            continue
        scs_sub_dist_pairs[(int(s_near), r)] = float(d_near) * M_TO_MILES

    # Remove R_cand sites with no valid substation entry
    _valid_scs_keys = set(r for _, r in scs_sub_dist_pairs.keys())
    r_cand_no_sub   = [r for r in R_cand if r not in _valid_scs_keys]
    if r_cand_no_sub:
        print(f"  WARNING: {len(r_cand_no_sub)} R_cand sites have no valid substation "
              f"after all filters — removing from R_cand.")
        R_cand = [r for r in R_cand if r not in r_cand_no_sub]

    S_SCS = list(set(s for s, _ in scs_sub_dist_pairs.keys()))

    # SCS coordinate fallback — derive from host segment if MID_X missing
    for r in R_cand:
        if r not in scs_coords_cand or scs_coords_cand[r][0] is None:
            hosts = scs_to_segs_cand.get(r, [])
            if hosts:
                rep = hosts[0]
                scs_coords_cand[r] = [seg_coords.get(rep, [None, None])[0],
                                      seg_coords.get(rep, [None, None])[1]]

    # Existing SCS coords (informational only)
    if use_existing_scs:
        df_scs_exist_info = df_scs_info[
            df_scs_info['SCS_ID'].isin(excluded_cand_ids)
        ].copy() if 'SCS_ID' in df_scs_info.columns else pd.DataFrame()
        scs_coords_exist = (
            df_scs_exist_info.set_index('SCS_ID')[['MID_X', 'MID_Y']].T.to_dict('list')
            if not df_scs_exist_info.empty and 'MID_X' in df_scs_exist_info.columns
            else {}
        )
    else:
        scs_coords_exist = {}

    # ── D_{i,s}: DWC segment → substation distances [miles] ──────────────────
    sub_dist_pairs = {}
    for _, row in df_sub_dist_filtered.iterrows():
        i = int(row['IN_FID'])
        s = int(row['NEAR_FID'])
        if i not in I_set:
            continue
        sub_dist_pairs[(s, i)] = float(row['NEAR_DIST']) * M_TO_MILES

    # Retain only substations connected to surviving segments. This is
    # equivalent for full-network runs and avoids thousands of unused
    # variables when exclude_top_pct is used for a reduced diagnostic run.
    S_DWC = sorted({s for s, _ in sub_dist_pairs})

    # ── Adjacency map ─────────────────────────────────────────────────────────
    adj_map   = {i: [] for i in I}
    adj_pairs = []
    for _, row in df_adjacency.iterrows():
        src, dst = int(row['TARGET_FID']), int(row['JOIN_FID'])
        if src not in I_set or dst not in I_set:
            continue
        if dst not in adj_map[src]:
            adj_map[src].append(dst)
            adj_pairs.append((src, dst))
        if src not in adj_map[dst]:
            adj_map[dst].append(src)
            adj_pairs.append((dst, src))
    undirected_edges = set(tuple(sorted(p)) for p in adj_pairs)

    # ── Remove R_cand sites with no host segment ──────────────────────────────
    r_cand_no_host = [r for r in R_cand if not scs_to_segs_cand.get(r)]
    if r_cand_no_host:
        print(f"  WARNING: {len(r_cand_no_host)} R_cand sites have no host segment "
              f"— removing from R_cand.")
        R_cand = [r for r in R_cand if scs_to_segs_cand.get(r)]
        for r in r_cand_no_host:
            scs_to_segs_cand.pop(r, None)
    print(f"  R_cand after no-host filter: {len(R_cand)}")

    # Sync scs_sub_dist_pairs with final R_cand
    _r_cand_set        = set(R_cand)
    scs_sub_dist_pairs = {(s, r): d for (s, r), d in scs_sub_dist_pairs.items()
                          if r in _r_cand_set}
    S_SCS = list(set(s for s, _ in scs_sub_dist_pairs.keys()))

    print(f"\nSets: |I|={len(I)}  |R_cand|={len(R_cand)}  |R_exist|={len(R_exist)}")
    print(f"      |S_DWC|={len(S_DWC)}  |S_SCS|={len(S_SCS)}")
    print(f"DWC sub–seg pairs  : {len(sub_dist_pairs)}")
    print(f"SCS sub–ramp pairs : {len(scs_sub_dist_pairs)}")
    print(f"Adjacency pairs    : {len(adj_pairs)}  ({len(undirected_edges)} undirected)")

    return {
        # Segment sets
        'I'                    : I,
        'I_set'                : I_set,
        # SCS sets
        'R_cand'               : R_cand,
        'R_exist'              : R_exist,
        # Substation sets
        'S_DWC'                : S_DWC,
        'S_SCS'                : S_SCS,
        # Distance pair dicts
        'sub_dist_pairs'       : sub_dist_pairs,
        'scs_sub_dist_pairs'   : scs_sub_dist_pairs,
        # Adjacency
        'adj_pairs'            : adj_pairs,
        'adj_map'              : adj_map,
        'undirected_edges'     : undirected_edges,
        # Segment attribute lookups
        'aadt_com'             : aadt_com,
        'speed_lim'            : speed_lim,
        'lambda_i'             : lambda_i,
        'lcoe_rate'            : lcoe_rate,
        'seg_coords'           : seg_coords,
        'seg_length_m'         : seg_length_m,
        'seg_length_miles'     : seg_length_miles,
        'seg_near_scs'         : seg_near_scs,
        # SCS mappings
        'scs_to_segs_cand'     : scs_to_segs_cand,
        'scs_to_segs_exist'    : scs_to_segs_exist,
        'scs_nearest_sub_cand' : scs_nearest_sub_cand,
        'scs_coords_cand'      : scs_coords_cand,
        'scs_coords_exist'     : scs_coords_exist,
        # Coordinate lookups
        'sub_coords'           : sub_coords,
        # Flags
        'use_existing_scs'     : use_existing_scs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER-DEPENDENT HELPERS  (called inside run_single_optimization)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_segment_params(I, aadt_com, speed_lim, lcoe_rate, lambda_i,
                             seg_length_miles,
                             rho_ev, ETA_DWC, P_DWC_coil_per_mile, P_CHARGE_DWC,
                             P_veh_DWC,
                             print_sample=5):
    """
    Compute per-segment DWC parameters that depend on sensitivity variables.
    Reproduces the original Section 5 loop exactly.
    """
    segment_params = {}
    print(f"\n{'─'*90}")
    print(f"SEGMENT DIAGNOSTIC SAMPLE (first {print_sample} segments)")
    print(f"{'─'*90}")
    print(f"  {'Seg_ID':>8}  {'AADT':>7}  {'Speed(mph)':>10}  {'Len(mi)':>7}  "
          f"{'P_WC(kW)':>8}  {'t_hr(avg)':>9}  {'E_traff(kWh)':>13}  "
          f"{'P_peak_MW':>10}  {'λ':>6}  {'LCOE($/kWh)':>12}")
    print(f"  {'':->8}  {'':->7}  {'':->10}  {'':->7}  "
          f"{'':->8}  {'':->9}  {'':->13}  {'':->10}  {'':->6}  {'':->12}")

    for idx, seg_id in enumerate(I):
        v_mph = speed_lim.get(seg_id, 60) or 60
        # v_mph = 45
        aadt  = aadt_com.get(seg_id, 0)   or 0
        lmp   = lcoe_rate.get(seg_id, 0.12)
        lam   = lambda_i.get(seg_id, 1.0)
        L_mi  = seg_length_miles[seg_id]
        # p_kw  = calculate_p_drive(v_mph)          # kept for diagnostics only
        t_h   = L_mi / v_mph if v_mph > 0 else 0.0

        # E^traff,DWC_i = 365·Σ_h AADT_{i,h}·η^DWC·P^veh,WC·(L_i/v_i)  [kWh/yr]
        # P_veh_DWC is the on-board receiver rated power (kW), not propulsion draw.
        e_traff_dwc = 365.0 * sum(
            aadt * TRAFFIC_WEIGHTS[h] * rho_ev * ETA_DWC * P_veh_DWC * t_h
            for h in range(24)
        )
        # bar{P}^DWC_i = P_DWC_coil_per_mile·L_i  [MW]
        p_peak_mw = P_DWC_coil_per_mile * L_mi

        segment_params[seg_id] = {
            'e_traff_dwc': e_traff_dwc,
            'p_peak_mw'  : p_peak_mw,
            'c_premium'  : P_CHARGE_DWC,
            'lcoe'       : lmp,
            'lambda'     : lam,
            'length_mi'  : L_mi,
            'speed_mph'  : v_mph,
            't_seg_hr'   : t_h,
            # 'p_kw'       : p_kw,
        }

        if idx < print_sample:
            print(f"  {seg_id:>8}  {int(aadt):>7,}  {v_mph:>10.1f}  {L_mi:>7.4f}  "
                  f"{P_veh_DWC:>8.1f}  {t_h:>9.5f}  {e_traff_dwc:>13,.1f}  "
                  f"{p_peak_mw:>10.4f}  {lam:>6.3f}  {lmp:>12.6f}")

    print(f"{'─'*90}")
    return segment_params


def _compute_scs_params_for_site(r, host_pool, aadt_com, lcoe_rate, lambda_i,
                                  rho_ev, ETA_SCS, E_req_kwh, P_CHARGE_SCS):
    """
    Compute SCS parameters for one candidate site.
    Reproduces the original compute_scs_params() inner function exactly.
    """
    if not host_pool:
        return {'e_traff_scs': 0.0, 'c_premium': P_CHARGE_SCS,
                'lcoe': 0.12, 'lambda': 1.0, 'host_seg': None}

    total_aadt = sum(aadt_com.get(seg, 0) or 0 for seg in host_pool)

    if total_aadt > 0:
        lmp_r = sum((aadt_com.get(seg, 0) or 0) * lcoe_rate.get(seg, 0.12)
                    for seg in host_pool) / total_aadt
        lam_r = sum((aadt_com.get(seg, 0) or 0) * lambda_i.get(seg, 1.0)
                    for seg in host_pool) / total_aadt
    else:
        rep   = max(host_pool, key=lambda x: aadt_com.get(x, 0))
        lmp_r = lcoe_rate.get(rep, 0.12)
        lam_r = lambda_i.get(rep, 1.0)

    rep = max(host_pool, key=lambda x: aadt_com.get(x, 0))

    # E^traff,SCS_i = 365·(Σ_h total_aadt·η^SCS)·E^req  [kWh/yr]
    e_traff_scs = 365.0 * (
        sum(total_aadt * TRAFFIC_WEIGHTS[h] * rho_ev * ETA_SCS for h in range(24))
        * E_req_kwh
    )
    return {
        'e_traff_scs': e_traff_scs,
        'c_premium'  : P_CHARGE_SCS,
        'lcoe'       : lmp_r,
        'lambda'     : lam_r,
        'host_seg'   : rep,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN OPTIMIZATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
def run_single_optimization(data, params, env=None):
    """
    Build and solve the joint DWC+SCS MIP for a single parameter set.

    Parameters
    ----------
    data   : dict   — output of load_data(); never mutated here
    params : dict   — sensitivity-variable parameters; see DEFAULT_PARAMS
    env    : gp.Env or None

    Returns
    -------
    results : dict  — all solution data, DataFrames, and metadata needed
                      by save_results() and Main_plots.py
    """

    # ── Unpack data ───────────────────────────────────────────────────────────
    I                  = data['I']
    I_set              = data['I_set']
    R_cand             = data['R_cand']
    R_exist            = data['R_exist']
    S_DWC              = data['S_DWC']
    S_SCS              = data['S_SCS']
    sub_dist_pairs     = data['sub_dist_pairs']
    scs_sub_dist_pairs = data['scs_sub_dist_pairs']
    adj_pairs          = data['adj_pairs']
    adj_map            = data['adj_map']
    undirected_edges   = data['undirected_edges']
    aadt_com           = data['aadt_com']
    speed_lim          = data['speed_lim']
    lambda_i           = data['lambda_i']
    lcoe_rate          = data['lcoe_rate']
    seg_coords         = data['seg_coords']
    seg_length_m       = data['seg_length_m']
    seg_length_miles   = data['seg_length_miles']
    seg_near_scs       = data['seg_near_scs']
    scs_to_segs_cand   = data['scs_to_segs_cand']
    scs_to_segs_exist  = data['scs_to_segs_exist']
    scs_nearest_sub_cand = data['scs_nearest_sub_cand']
    scs_coords_cand    = data['scs_coords_cand']
    scs_coords_exist   = data['scs_coords_exist']
    sub_coords         = data['sub_coords']
    use_existing_scs   = data['use_existing_scs']

    # ── Unpack params ─────────────────────────────────────────────────────────
    rho_ev               = params['rho_ev']
    # sigma                = params['sigma']
    ETA_DWC              = params['ETA_DWC']
    ETA_SCS              = params['ETA_SCS']
    Utilization_rate_SCS = params['Utilization_rate_SCS']
    p_param              = params['p_param']
    P_CHARGE_DWC         = params['P_CHARGE_DWC']
    P_CHARGE_SCS         = params['P_CHARGE_SCS']
    discount_rate        = params['discount_rate']
    project_life         = params['project_life']
    H_op                 = params['H_op']
    C_DWC_base           = params['C_DWC_base']
    C_line_int_base      = params['C_line_int_base']
    C_line_adj_cost      = params['C_line_adj']
    C_line_ext_base      = params['C_line_ext_base']
    C_connection         = params['C_connection']
    P_DWC_coil_per_mile  = params['P_DWC_coil_per_mile']
    P_veh_DWC            = params['P_veh_DWC_kw']          # on-board WC receiver power [kW]
    C_SCS_base_per_port  = params['C_SCS_base_per_port']
    N_SCS_stalls         = params['N_SCS_stalls']
    P_SCS_per_stall      = params['P_SCS_per_stall_kw'] * KW_TO_MW  # convert kW → MW
    E_req_kwh            = params['E_req_kwh']
    run_tag              = params.get('run_tag', 'run')
    mip_gap              = params.get('mip_gap', 0.005)
    # ── Growth rates (CAGR) — γ^AADT, ρ^EV_y, γ^price, γ^LCOE, γ^O&M ────────
    g_aadt               = params['g_aadt']
    g_ev                 = params['g_ev']
    g_price_dwc          = params['g_price_dwc']
    g_price_scs          = params['g_price_scs']
    g_lcoe               = params['g_lcoe']
    g_om                 = params['g_om']

    # ── Derived parameters ────────────────────────────────────────────────────
    if discount_rate == 0.0:
        alpha = 1.0 / project_life
    else:
        alpha = (discount_rate * (1 + discount_rate)**project_life) / \
                ((1 + discount_rate)**project_life - 1)
    C_SCS_base    = C_SCS_base_per_port * N_SCS_stalls
    C_OM_base_DWC = 0.015 * (C_DWC_base + C_connection)
    C_OM_base_SCS = 0.015 * (C_SCS_base + C_connection * N_SCS_stalls * P_SCS_per_stall)

    # g_d: combined demand growth — AADT × EV-penetration compound rate.
    g_d = (1.0 + g_aadt) * (1.0 + g_ev) - 1.0

    # g_rev_dwc: DWC revenue grows at demand rate × DWC retail price growth.
    g_rev_dwc = (1.0 + g_d) * (1.0 + g_price_dwc) - 1.0

    # g_rev_scs: SCS revenue grows at demand rate × SCS retail price growth.
    g_rev_scs = (1.0 + g_d) * (1.0 + g_price_scs) - 1.0

    # g_var: variable O&M (energy procurement) grows at demand rate × LCOE growth.
    g_var = (1.0 + g_d) * (1.0 + g_lcoe) - 1.0

    # Γ_rev_dwc: PV weight for DWC revenue stream.
    Gamma_rev_dwc = _npv_factor(g_rev_dwc, discount_rate, project_life)

    # Γ_rev_scs: PV weight for SCS revenue stream.
    Gamma_rev_scs = _npv_factor(g_rev_scs, discount_rate, project_life)

    # Γ_var: PV weight for variable O&M (energy cost) — same grid for both techs.
    Gamma_var = _npv_factor(g_var, discount_rate, project_life)

    # Γ_fix_om: PV weight for fixed O&M — grows at O&M cost inflation only.
    Gamma_fix_om = _npv_factor(g_om, discount_rate, project_life)

    # ── Parameters summary ────────────────────────────────────────────────────
    startup_lines = []
    startup_lines.append("=" * 60)
    startup_lines.append("PARAMETERS SUMMARY")
    startup_lines.append(f"  Run tag                 : {run_tag}")
    startup_lines.append(f"  Formulation             : "
                         f"{'With Existing SCS (R_exist)' if use_existing_scs else 'No Existing SCS'}")
    startup_lines.append(f"  DWC budget (p)          : {p_param} segments")
    startup_lines.append(f"  rho_ev (base year)      : {rho_ev:.4f}")
    startup_lines.append(f"  Planning horizon (T)    : {project_life} years")
    startup_lines.append(f"  CRF (α)                 : {alpha:.4f}")
    startup_lines.append(f"  η^DWC (ETA_DWC)         : {ETA_DWC:.2f}")
    startup_lines.append(f"  η^SCS (ETA_SCS)         : {ETA_SCS:.2f}")
    startup_lines.append(f"  P_CHARGE_DWC            : ${P_CHARGE_DWC:.2f}/kWh")
    startup_lines.append(f"  P_CHARGE_SCS            : ${P_CHARGE_SCS:.2f}/kWh")
    startup_lines.append(f"  DWC coil density        : {P_DWC_coil_per_mile} MW/mile")
    startup_lines.append(f"  P^veh,DWC (receiver kW)  : {P_veh_DWC:.1f} kW")
    startup_lines.append(f"  SCS per-port base cost  : ${C_SCS_base_per_port:,.0f}/port")
    startup_lines.append(f"  SCS station base cost   : ${C_SCS_base:,.0f}/station  "
                         f"({N_SCS_stalls} ports × ${C_SCS_base_per_port:,.0f})")
    startup_lines.append(f"  SCS station capacity    : {N_SCS_stalls} × "
                         f"{P_SCS_per_stall*MW_TO_KW:.1f} kW  "
                         f"({N_SCS_stalls * P_SCS_per_stall:.4f} MW total)")
    startup_lines.append(f"  SCS O&M base (per stn)  : ${C_OM_base_SCS:,.0f}/yr")
    startup_lines.append(f"  SCS E^req per vehicle   : {E_req_kwh} kWh")
    startup_lines.append(f"  g_AADT  (γ^AADT CAGR)          : {g_aadt:.4f}")
    startup_lines.append(f"  g_EV    (ρ^EV growth CAGR)     : {g_ev:.4f}")
    # ADD this replacement block:
    startup_lines.append(f"  g_d          (demand CAGR)            : {g_d:.4f}")
    startup_lines.append(f"  g_price_dwc  (DWC retail price CAGR)  : {g_price_dwc:.4f}")
    startup_lines.append(f"  g_price_scs  (SCS retail price CAGR)  : {g_price_scs:.4f}")
    startup_lines.append(f"  g_lcoe       (LCOE growth CAGR)        : {g_lcoe:.4f}")
    startup_lines.append(f"  g_om         (fixed O&M cost CAGR)     : {g_om:.4f}")
    startup_lines.append(f"  g_rev_dwc    (DWC revenue CAGR)        : {g_rev_dwc:.4f}")
    startup_lines.append(f"  g_rev_scs    (SCS revenue CAGR)        : {g_rev_scs:.4f}")
    startup_lines.append(f"  g_var        (variable O&M CAGR)       : {g_var:.4f}")
    startup_lines.append(f"  Γ_rev_dwc    (PV factor, DWC revenue)  : {Gamma_rev_dwc:.4f}")
    startup_lines.append(f"  Γ_rev_scs    (PV factor, SCS revenue)  : {Gamma_rev_scs:.4f}")
    startup_lines.append(f"  Γ_var        (PV factor, variable O&M) : {Gamma_var:.4f}")
    startup_lines.append(f"  Γ_fix_om     (PV factor, fixed O&M)    : {Gamma_fix_om:.4f}")
    startup_lines.append("=" * 60)
    startup_text = "\n".join(startup_lines)
    print(startup_text)

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 5 — PARAMETER-DEPENDENT SEGMENT & SCS COMPUTATIONS
    # ═════════════════════════════════════════════════════════════════════════

    segment_params = _compute_segment_params(
        I, aadt_com, speed_lim, lcoe_rate, lambda_i, seg_length_miles,
        rho_ev, ETA_DWC, P_DWC_coil_per_mile, P_CHARGE_DWC,
        P_veh_DWC
    )

    gamma_d_T = (1.0 + g_d) ** (project_life - 1)

    # SCS physical hardware energy ceiling [kWh/yr] — year-T quantity.
    # C6e: γ^d_T · E^del,SCS_i ≤ E_cap_SCS_physical · δ^SCS_i
    # The ceiling is NOT divided by γ^d_T here; instead γ^d_T is moved
    # to the LHS of the constraint so the base-year variable is bounded
    # only by C6d (traffic demand), not artificially deflated by growth.
    E_cap_SCS_physical = N_SCS_stalls * P_SCS_per_stall * H_op * MW_TO_KW * Utilization_rate_SCS
    print(f"\nSCS physical hardware ceiling (C6e RHS) : {E_cap_SCS_physical:,.1f} kWh/yr")
    print(f"  γ^d_T (year-{project_life} demand multiplier)  : {gamma_d_T:.4f}")
    print(f"  Implied base-year max if C6e binds      : {E_cap_SCS_physical / gamma_d_T:,.1f} kWh/yr")

    # SCS candidate params (parameter-dependent via rho_ev, ETA_SCS)
    scs_params_cand = {
        r: _compute_scs_params_for_site(
            r, scs_to_segs_cand.get(r, []),
            aadt_com, lcoe_rate, lambda_i,
            rho_ev, ETA_SCS, E_req_kwh, P_CHARGE_SCS
        )
        for r in R_cand
    }

    DIAG_PRINT_COUNT = 5
    print(f"\n{'─'*90}")
    print(f"SCS CANDIDATE DIAGNOSTIC SAMPLE (first {DIAG_PRINT_COUNT})")
    print(f"{'─'*90}")
    print(f"  {'SCS_ID':>8}  {'Host_Seg':>9}  {'AADT':>7}  "
          f"{'E_traff(kWh)':>13}  {'λ':>6}  {'LCOE':>7}  {'Sub_ID':>8}  {'Dist(mi)':>9}")
    print(f"  {'':->8}  {'':->9}  {'':->7}  {'':->13}  {'':->6}  {'':->7}  {'':->8}  {'':->9}")
    for idx, r in enumerate(R_cand[:DIAG_PRINT_COUNT]):
        p = scs_params_cand[r]
        host   = p['host_seg']
        aadt_h = aadt_com.get(host, 0) if host else 0
        s_near = scs_nearest_sub_cand.get(r)
        d_near = scs_sub_dist_pairs.get((int(s_near), r)) if s_near and int(s_near) != -1 else float('nan')
        print(f"  {r:>8}  {str(host):>9}  {int(aadt_h or 0):>7,}  "
              f"{p['e_traff_scs']:>13,.1f}  {p['lambda']:>6.3f}  {p['lcoe']:>7.5f}  "
              f"{str(s_near or 'N/A'):>8}  "
              f"{d_near if not np.isnan(float(d_near or 'nan')) else 'N/A':>9}")
    print(f"{'─'*90}")

    # Per-segment DWC hardware energy ceiling [kWh/yr]
    # γ^d_T: year-T demand growth multiplier.
    # C6b*: E^del,DWC_i ≤ (bar{P}^DWC_i · H^op · 1000) / γ^d_T · δ^DWC_i
    E_cap_DWC_physical = {
    i: segment_params[i]['p_peak_mw'] * H_op * MW_TO_KW
    for i in I
    }

    # Per-unit capital cost component breakdown (for plotting)
    dwc_capex_components = {
        i: {
            'base_hardware': segment_params[i]['lambda'] * C_DWC_base,
            'internal_line': segment_params[i]['lambda'] * C_line_int_base,
        }
        for i in I
    }
    scs_capex_components = {
        r: {'base_hardware': scs_params_cand[r]['lambda'] * C_SCS_base}
        for r in R_cand
    }

    # Big-M values
    M_virt  = p_param + 1
    M_power = sum(s['p_peak_mw'] for s in segment_params.values())
    print(f"\nBig-M virtual : {M_virt} | Big-M power: {M_power:.2f} MW")
    print(f"Candidate SCS with valid host : "
          f"{sum(1 for v in scs_params_cand.values() if v['host_seg'] is not None)}"
          f"/{len(R_cand)}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 6 — MODEL SETUP & DECISION VARIABLES
    # ═════════════════════════════════════════════════════════════════════════
    model = gp.Model("Joint_DWC_SCS_Optimization", env=env)

    delta_dwc      = model.addVars(I,                         vtype=GRB.BINARY,     name="delta_DWC")
    delta_scs      = model.addVars(R_cand,                    vtype=GRB.BINARY,     name="delta_SCS")
    delta_subs     = model.addVars(sub_dist_pairs.keys(),     vtype=GRB.BINARY,     name="delta_subs")
    delta_adj      = model.addVars(adj_pairs,                 vtype=GRB.BINARY,     name="delta_adj")
    delta_subs_scs = model.addVars(scs_sub_dist_pairs.keys(), vtype=GRB.BINARY,     name="delta_subs_SCS")

    u_si     = model.addVars(sub_dist_pairs.keys(),   lb=0.0, vtype=GRB.CONTINUOUS, name="u_si")
    u_ji     = model.addVars(adj_pairs,               lb=0.0, vtype=GRB.CONTINUOUS, name="u_ji")
    w_rs     = model.addVars(S_DWC,                   lb=0.0, vtype=GRB.CONTINUOUS, name="w_rs")
    f_ji     = model.addVars(adj_pairs,               lb=0.0, vtype=GRB.CONTINUOUS, name="f_ji")
    p_tap_si = model.addVars(sub_dist_pairs.keys(),   lb=0.0, vtype=GRB.CONTINUOUS, name="P_tap_si")
    p_tap_scs_rs = model.addVars(
        scs_sub_dist_pairs.keys(), lb=0.0, vtype=GRB.CONTINUOUS, name="P_tap_SCS_rs"
    )
    e_del_dwc    = model.addVars(I,      lb=0.0, vtype=GRB.CONTINUOUS, name="E_del_DWC")
    e_del_scs    = model.addVars(R_cand, lb=0.0, vtype=GRB.CONTINUOUS, name="E_del_SCS")

    model.update()
    print(f"\nVariables: {model.NumVars} total")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 7 — OBJECTIVE FUNCTION
    # max NPV Z = Γ_rev·R^total - C^cap - Γ_fix,om·C^{O&M,fix}
    #           - Γ_var,om·C^{O&M,var} - Γ_NC·C^NC  (Eq. objective)
    # All revenue, O&M, and penalty terms are base-year (year-1) quantities;
    # the Γ scalars convert each stream to its lifetime present value.
    # Capital C^cap is a one-time cost at t=0 and is not discounted further.
    # ═════════════════════════════════════════════════════════════════════════

    # R^DWC = Σ_i C^prem_i·E^del,DWC_i  (base-year revenue)
    R_dwc = gp.quicksum(
        segment_params[i]['c_premium'] * e_del_dwc[i] for i in I
    )
    # R^SCS = Σ_{i∈I_R,cand} C^SCS,prem_i·E^del,SCS_i  (base-year revenue)
    R_scs = gp.quicksum(
        scs_params_cand[r]['c_premium'] * e_del_scs[r] for r in R_cand
    )

    # C^cap (Eq. capex) — DWC base
    C_cap_dwc_base = gp.quicksum(
        segment_params[i]['lambda'] * delta_dwc[i] * (C_DWC_base + C_line_int_base)
        for i in I
    )
    # Adjacency link capital
    C_cap_adj = gp.quicksum(
        ((segment_params[a]['lambda'] + segment_params[b]['lambda']) / 2.0) *
        (delta_adj.get((a, b), 0) + delta_adj.get((b, a), 0)) * C_line_adj_cost
        for a, b in undirected_edges
        if a in segment_params and b in segment_params
    )
    # DWC substation feeder capital
    C_cap_subs_dwc = gp.quicksum(
        segment_params[i]['lambda'] * delta_subs[s, i] *
        sub_dist_pairs[s, i] * C_line_ext_base
        for s, i in sub_dist_pairs.keys()
    )
    # DWC grid connection capital
    C_cap_conn_dwc = gp.quicksum(
        segment_params[i]['lambda'] * p_tap_si[s, i] * C_connection
        for s, i in sub_dist_pairs.keys()
    )
    # SCS base capital
    C_cap_scs_base = gp.quicksum(
        scs_params_cand[r]['lambda'] * delta_scs[r] * C_SCS_base for r in R_cand
    )
    # SCS feeder capital
    C_cap_subs_scs = gp.quicksum(
        scs_params_cand[r]['lambda'] * delta_subs_scs[s, r] *
        scs_sub_dist_pairs[s, r] * C_line_ext_base
        for s, r in scs_sub_dist_pairs.keys()
    )
    # SCS grid connection capital
    C_cap_conn_scs = gp.quicksum(
        scs_params_cand[r]['lambda'] * p_tap_scs_rs[s, r] * C_connection
        for s, r in scs_sub_dist_pairs.keys()
    )

    # C^cap — one-time present-value cost at t=0; not annualised
    C_cap_total = (
        C_cap_dwc_base + C_cap_adj + C_cap_subs_dwc + C_cap_conn_dwc +
        C_cap_scs_base + C_cap_subs_scs + C_cap_conn_scs
    )

    # C^{O&M,fix,DWC} — base-year fixed per-deployment cost; grows at g_om
    # λ_i scales fixed O&M to reflect regional maintenance difficulty
    C_om_fix_dwc = gp.quicksum(
        segment_params[i]['lambda'] * delta_dwc[i] * C_OM_base_DWC
        for i in I
    )
    # C^{O&M,var,DWC} — base-year energy procurement cost.
    C_om_var_dwc = gp.quicksum(
        segment_params[i]['lcoe'] * e_del_dwc[i]
        for i in I
    )
    # C^{O&M,fix,SCS} — base-year fixed per-station cost; grows at g_om
    C_om_fix_scs = gp.quicksum(
        scs_params_cand[r]['lambda'] * delta_scs[r] * C_OM_base_SCS
        for r in R_cand
    )
    # C^{O&M,var,SCS} — base-year energy procurement cost; grows at g_var_om
    C_om_var_scs = gp.quicksum(
        scs_params_cand[r]['lcoe'] * e_del_scs[r]
        for r in R_cand
    )

    # ADD:
    # Objective — Eq. (objective):  max Z = -C^cap
    #   + Γ_rev · [R^DWC + R^SCS - C^{O&M,var,DWC} - C^{O&M,var,SCS}]
    #   - Γ_fix_om · [C^{O&M,fix,DWC} + C^{O&M,fix,SCS}]
    # Capital C^cap is a one-time cost at t=0; no discounting factor applied here.
    model.setObjective(
        Gamma_rev_dwc * R_dwc
        + Gamma_rev_scs * R_scs
        - Gamma_var     * (C_om_var_dwc + C_om_var_scs)
        - C_cap_total
        - Gamma_fix_om  * (C_om_fix_dwc + C_om_fix_scs),
        GRB.MAXIMIZE)
    
    model.update()
    print(f"\nObjective set. Total vars: {model.NumVars}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 8 — CONSTRAINTS
    # ═════════════════════════════════════════════════════════════════════════

    # F1: Segment virtual flow balance
    for i in I:
        model.addConstr(
            u_ji.sum('*', i) + u_si.sum('*', i) - u_ji.sum(i, '*') == delta_dwc[i],
            name=f"F1_{i}"
        )
    # F2: Substation virtual flow balance
    for s in S_DWC:
        model.addConstr(w_rs[s] - u_si.sum(s, '*') == 0, name=f"F2_{s}")
    # F3: Total virtual supply
    model.addConstr(w_rs.sum('*') == delta_dwc.sum('*'), name="F3")

    # C1: Global deployment count (≤ budget)
    # Σ_i δ^DWC_i + Σ_{i∈I_R} δ^SCS_i = p
    model.addConstr(delta_dwc.sum('*') + delta_scs.sum('*') <= p_param, name="C1_budget")
    # model.addConstr(delta_dwc.sum('*') <= p_param, name="C1_budget")

    # C2: Physical power flow balance (DWC)
    for i in I:
        model.addConstr(
            p_tap_si.sum('*', i) + f_ji.sum('*', i) ==
            delta_dwc[i] * segment_params[i]['p_peak_mw'] + f_ji.sum(i, '*'),
            name=f"C2_{i}"
        )

    # C3: Infrastructure–flow coupling — DWC (big-M)
    for s, i in sub_dist_pairs.keys():
        model.addConstr(u_si[s, i]       <= M_virt  * delta_subs[s, i], name=f"C3a_u_{s}_{i}")
        model.addConstr(p_tap_si[s, i]   <= M_power * delta_subs[s, i], name=f"C3a_p_{s}_{i}")
        model.addConstr(delta_subs[s, i] <= delta_dwc[i],               name=f"C3d_{s}_{i}")
    for j, i in adj_pairs:
        model.addConstr(u_ji[j, i]      <= M_virt  * delta_adj[j, i], name=f"C3b_u_{j}_{i}")
        model.addConstr(f_ji[j, i]      <= M_power * delta_adj[j, i], name=f"C3b_f_{j}_{i}")
        model.addConstr(delta_adj[j, i] <= delta_dwc[i],               name=f"C3c_to_{j}_{i}")
        model.addConstr(delta_adj[j, i] <= delta_dwc[j],               name=f"C3c_fr_{j}_{i}")

    # C3e: Adjacency link anti-symmetry
    c3e_count = 0
    for a, b in undirected_edges:
        if (a, b) in delta_adj and (b, a) in delta_adj:
            model.addConstr(
                delta_adj[a, b] + delta_adj[b, a] <= 1, name=f"C3e_{a}_{b}"
            )
            c3e_count += 1
    print(f"C3e anti-symmetry constraints added: {c3e_count}")

    # C4: Infrastructure–flow coupling — SCS (big-M)
    for s, r in scs_sub_dist_pairs.keys():
        model.addConstr(p_tap_scs_rs[s, r]   <= M_power * delta_subs_scs[s, r], name=f"C4_p_{s}_{r}")
        model.addConstr(delta_subs_scs[s, r] <= delta_scs[r],                   name=f"C4_logic_{s}_{r}")

    # C4a: SCS power balance
    for r in R_cand:
        model.addConstr(
            p_tap_scs_rs.sum('*', r) == delta_scs[r] * N_SCS_stalls * P_SCS_per_stall,
            name=f"C4a_{r}"
        )

    # C4b: Single substation per SCS
    for r in R_cand:
        relevant = [s for s in S_SCS if (s, r) in scs_sub_dist_pairs]
        if relevant:
            model.addConstr(
                gp.quicksum(delta_subs_scs[s, r] for s in relevant) <= 1,
                name=f"C4b_{r}"
            )

    # C6a–C6b: DWC delivered energy bounds
    for i in I:
        model.addConstr(
            e_del_dwc[i] <= segment_params[i]['e_traff_dwc'] * delta_dwc[i],
            name=f"C6a_{i}"
        )
        model.addConstr(
            gamma_d_T * e_del_dwc[i] <= E_cap_DWC_physical[i] * delta_dwc[i],
            name=f"C6b_{i}"
        )

    # C6d–C6e: SCS delivered energy bounds
    for r in R_cand:
        # C6d: base-year delivery ≤ base-year traffic demand
        model.addConstr(
            e_del_scs[r] <= scs_params_cand[r]['e_traff_scs'] * delta_scs[r],
            name=f"C6d_{r}"
        )
        # C6e: γ^d_T · E^del,SCS_i ≤ physical ceiling · δ^SCS_i
        # LHS is year-T delivered energy; RHS is the fixed hardware ceiling.
        # This avoids artificially deflating the base-year variable and
        # preserves selection neutrality across high- and low-AADT sites.
        model.addConstr(
            gamma_d_T * e_del_scs[r] <= E_cap_SCS_physical * delta_scs[r],
            name=f"C6e_{r}"
        )

    model.update()
    print(f"\nAll constraints added. Total constraints: {model.NumConstrs}")

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 9 — SOLVE
    # ═════════════════════════════════════════════════════════════════════════
    model.Params.MIPGap = mip_gap
    model.optimize()

    if model.Status not in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        print("No feasible solution found.")
        return None

    # ═════════════════════════════════════════════════════════════════════════
    # SECTION 10 — RESULTS EXTRACTION
    # ═════════════════════════════════════════════════════════════════════════
    electrified_segs         = [i for i in I      if delta_dwc[i].X > 0.5]
    selected_scs_cand        = [r for r in R_cand if delta_scs[r].X > 0.5]
    selected_substations_dwc = set()
    selected_substations_scs = set()

    print(f"\n{'='*60}")
    print(f"Objective (lifetime NPV Z)     : ${model.ObjVal:,.0f}")
    print(f"DWC segments selected          : {len(electrified_segs)}")
    print(f"New SCS stations selected      : {len(selected_scs_cand)}")
    if use_existing_scs:
        print(f"Existing SCS (pre-built, fixed): {len(R_exist)}")
    print(f"{'='*60}")

    # ── DWC results table with capital cost components ────────────────────────
    results_dwc = []
    for i in electrified_segs:
        conn_type, from_id = "Isolated", i
        feeder_cost = connection_cost = 0.0
        for s in S_DWC:
            if (s, i) in delta_subs and delta_subs[s, i].X > 0.5:
                conn_type, from_id = "Substation Tap", s
                selected_substations_dwc.add(s)
                lam  = segment_params[i]['lambda']
                dist = sub_dist_pairs.get((s, i), 0.0)
                feeder_cost = lam * dist * C_line_ext_base
                p_tap_val   = sum(
                    p_tap_si[s, i].X for s in S_DWC if (s, i) in p_tap_si
                )
                connection_cost = lam * p_tap_val * C_connection
                break
        if conn_type == "Isolated":
            for j in adj_map.get(i, []):
                if (j, i) in delta_adj and delta_adj[j, i].X > 0.5:
                    conn_type, from_id = "Adjacent Link", j
                    break
        capex_total = (dwc_capex_components[i]['base_hardware'] +
                       dwc_capex_components[i]['internal_line'] +
                       feeder_cost + connection_cost)
        results_dwc.append({
            "Segment_ID"       : i,
            "Conn_Type"        : conn_type,
            "From_ID"          : from_id,
            "AADT_COM"         : aadt_com.get(i),
            "MW_Load"          : segment_params[i]['p_peak_mw'],
            "E_traff_DWC"      : segment_params[i]['e_traff_dwc'],
            "E_del_DWC"        : e_del_dwc[i].X,
            "Lambda"           : segment_params[i]['lambda'],
            "LCOE"             : segment_params[i]['lcoe'],
            "MID_X"            : seg_coords.get(i, [None, None])[0],
            "MID_Y"            : seg_coords.get(i, [None, None])[1],
            "Capex_Base_Hardware": dwc_capex_components[i]['base_hardware'],
            "Capex_Internal_Line": dwc_capex_components[i]['internal_line'],
            "Capex_Feeder"     : feeder_cost,
            "Capex_Connection" : connection_cost,
            "Capex_Total"      : capex_total,
            "Ann_Revenue"      : segment_params[i]['c_premium'] * e_del_dwc[i].X,
            # Ann_Opex: base-year fixed O&M (λ-scaled) + variable energy cost (no λ)
            "Ann_Opex"         : (segment_params[i]['lambda'] * C_OM_base_DWC +
                                  segment_params[i]['lcoe'] * e_del_dwc[i].X),
        })
    df_dwc = pd.DataFrame(results_dwc, columns=[
        "Segment_ID", "Conn_Type", "From_ID", "AADT_COM", "MW_Load",
        "E_traff_DWC", "E_del_DWC", "Lambda", "LCOE", "MID_X", "MID_Y",
        "Capex_Base_Hardware", "Capex_Internal_Line", "Capex_Feeder",
        "Capex_Connection", "Capex_Total", "Ann_Revenue", "Ann_Opex",
    ])

    # ── SCS results table with capital cost components ────────────────────────
    results_scs = []
    for r in selected_scs_cand:
        sub_conn = None
        feeder_cost = connection_cost = 0.0
        for s in S_SCS:
            if (s, r) in delta_subs_scs and delta_subs_scs[s, r].X > 0.5:
                sub_conn = s
                selected_substations_scs.add(s)
                lam  = scs_params_cand[r]['lambda']
                dist = scs_sub_dist_pairs.get((s, r), 0.0)
                feeder_cost     = lam * dist * C_line_ext_base
                connection_cost = lam * N_SCS_stalls * P_SCS_per_stall * C_connection
                break
        dist_to_sub = scs_sub_dist_pairs.get((sub_conn, r)) if sub_conn else None
        capex_total = (scs_capex_components[r]['base_hardware'] +
                       feeder_cost + connection_cost)
        results_scs.append({
            "SCS_ID"           : r,
            "Type"             : "New Candidate",
            "Sub_ID"           : sub_conn,
            "Dist_to_Sub_mi"   : dist_to_sub,
            "Host_Seg"         : scs_params_cand[r]['host_seg'],
            "E_traff_SCS"      : scs_params_cand[r]['e_traff_scs'],
            "E_del_SCS"        : e_del_scs[r].X,
            "Lambda"           : scs_params_cand[r]['lambda'],
            "LCOE"             : scs_params_cand[r]['lcoe'],
            "MID_X"            : scs_coords_cand.get(r, [None, None])[0],
            "MID_Y"            : scs_coords_cand.get(r, [None, None])[1],
            "Capex_Base_Hardware": scs_capex_components[r]['base_hardware'],
            "Capex_Feeder"     : feeder_cost,
            "Capex_Connection" : connection_cost,
            "Capex_Total"      : capex_total,
            "Ann_Revenue"      : scs_params_cand[r]['c_premium'] * e_del_scs[r].X,
            # Ann_Opex: base-year fixed O&M (λ-scaled) + variable energy cost (no λ)
            "Ann_Opex"         : (scs_params_cand[r]['lambda'] * C_OM_base_SCS +
                                  scs_params_cand[r]['lcoe'] * e_del_scs[r].X),
        })
    df_scs = pd.DataFrame(results_scs, columns=[
        "SCS_ID", "Type", "Sub_ID", "Dist_to_Sub_mi", "Host_Seg",
        "E_traff_SCS", "E_del_SCS", "Lambda", "LCOE", "MID_X", "MID_Y",
        "Capex_Base_Hardware", "Capex_Feeder", "Capex_Connection",
        "Capex_Total", "Ann_Revenue", "Ann_Opex",
    ])

    # ── Existing SCS summary ──────────────────────────────────────────────────
    results_exist = []
    if use_existing_scs:
        for r in R_exist:
            hosts  = scs_to_segs_exist.get(r, [])
            host   = max(hosts, key=lambda x: aadt_com.get(x, 0)) if hosts else None
            lam_r  = lambda_i.get(host, 1.0) if host else 1.0
            cx, cy = scs_coords_exist.get(r, [None, None])
            results_exist.append({
                "SCS_ID"      : r,
                "Type"        : "Existing",
                "Host_Seg"    : host,
                "Capture_Frac": ETA_SCS,
                "Lambda"      : lam_r,
                "MID_X"       : cx,
                "MID_Y"       : cy,
            })
    df_exist = pd.DataFrame(results_exist)

    # ── Connection geometry tables ────────────────────────────────────────────
    connection_table = []
    for r in results_dwc:
        seg_id = r['Segment_ID']
        tc = seg_coords.get(seg_id)
        if not tc:
            continue
        if r['Conn_Type'] == "Substation Tap":
            fc = sub_coords.get(int(r['From_ID']))
            if fc:
                connection_table.append({
                    'Conn_Type': 'Substation Tap', 'From_Type': 'Substation',
                    'From_ID': int(r['From_ID']), 'To_ID': seg_id,
                    'FromX': fc[0], 'FromY': fc[1], 'ToX': tc[0], 'ToY': tc[1],
                    'Dist_Miles': sub_dist_pairs.get((int(r['From_ID']), seg_id), 0.0)
                })
        elif r['Conn_Type'] == "Adjacent Link":
            fc = seg_coords.get(int(r['From_ID']))
            if fc:
                connection_table.append({
                    'Conn_Type': 'Adjacent Link', 'From_Type': 'Segment',
                    'From_ID': int(r['From_ID']), 'To_ID': seg_id,
                    'FromX': fc[0], 'FromY': fc[1], 'ToX': tc[0], 'ToY': tc[1],
                    'Dist_Miles': 0.0
                })
    connection_columns = [
        "Conn_Type", "From_Type", "From_ID", "To_ID",
        "FromX", "FromY", "ToX", "ToY", "Dist_Miles",
    ]
    df_connections = pd.DataFrame(connection_table, columns=connection_columns)

    scs_connection_table = []
    for r in results_scs:
        scs_id = r['SCS_ID']
        tc = scs_coords_cand.get(scs_id)
        if not tc or r['Sub_ID'] is None:
            continue
        fc = sub_coords.get(int(r['Sub_ID']))
        if fc:
            scs_connection_table.append({
                'Conn_Type': 'SCS Tap', 'From_Type': 'Substation',
                'From_ID': int(r['Sub_ID']), 'To_ID': scs_id,
                'FromX': fc[0], 'FromY': fc[1], 'ToX': tc[0], 'ToY': tc[1],
                'Dist_Miles': r['Dist_to_Sub_mi'] or 0.0
            })
    df_scs_connections = pd.DataFrame(
        scs_connection_table, columns=connection_columns
    )

    # ── Master diagnostic dataframe ───────────────────────────────────────────
    master_data = []
    for i in I:
        aadt_v  = aadt_com.get(i, 0) or 0
        lam     = segment_params[i]['lambda']
        rev     = segment_params[i]['c_premium'] * segment_params[i]['e_traff_dwc']
        dists   = [d for (s, seg_id), d in sub_dist_pairs.items() if seg_id == i]
        min_d   = min(dists) if dists else 1.0
        x, y    = seg_coords.get(i, [np.nan, np.nan])
        master_data.append({
            'Segment_ID'      : i,
            'X': x, 'Y': y,
            'AADT_COM'        : aadt_v,
            'Min_Sub_Dist'    : min_d,
            'Total_Revenue'   : rev,
            'Lambda'          : lam,
            'Rev_Dist_Ratio'  : rev / (min_d + 0.1),
            'Traff_Dist_Ratio': aadt_v / (min_d + 0.1),
            'Is_DWC'          : 1 if i in electrified_segs else 0,
        })
    df_diag = pd.DataFrame(master_data)

    # Financial scalars — all base-year (year-1) quantities
    total_rev_dwc  = sum(segment_params[i]['c_premium'] * e_del_dwc[i].X
                         for i in electrified_segs)
    total_rev_scs  = sum(scs_params_cand[r]['c_premium'] * e_del_scs[r].X
                         for r in selected_scs_cand)
    # total_opex: base-year fixed O&M (λ-scaled) + variable energy cost (no λ)
    total_opex_dwc = sum(
        segment_params[i]['lambda'] * C_OM_base_DWC +
        segment_params[i]['lcoe'] * e_del_dwc[i].X
        for i in electrified_segs
    )
    total_opex_scs = sum(
        scs_params_cand[r]['lambda'] * C_OM_base_SCS +
        scs_params_cand[r]['lcoe'] * e_del_scs[r].X
        for r in selected_scs_cand
    )

    # ── Metadata dict (matches original run_metadata.csv schema) ─────────────
    meta = {
        'use_existing_scs'         : use_existing_scs,
        # 'sigma'                    : sigma,
        'rho_ev'                   : rho_ev,
        'eta_dwc'                  : ETA_DWC,
        'eta_scs'                  : ETA_SCS,
        'utilization_rate_scs'     : Utilization_rate_SCS,
        'discount_rate'            : discount_rate,
        'project_life'             : project_life,
        'alpha'                    : alpha,
        'h_op'                     : H_op,
        'p_param'                  : p_param,
        'p_charge_dwc'             : P_CHARGE_DWC,
        'p_charge_scs'             : P_CHARGE_SCS,
        'c_dwc_base'               : C_DWC_base,
        'c_line_int_base'          : C_line_int_base,
        'c_line_adj'               : C_line_adj_cost,
        'c_line_ext_base'          : C_line_ext_base,
        'c_connection'             : C_connection,
        'c_om_base_dwc'            : C_OM_base_DWC,
        'p_dwc_coil_per_mile'      : P_DWC_coil_per_mile,
        'p_veh_dwc_kw'             : P_veh_DWC,
        'n_scs_stalls'             : N_SCS_stalls,
        'p_scs_per_stall_kw'       : P_SCS_per_stall * MW_TO_KW,
        'c_scs_base'               : C_SCS_base,
        'c_scs_base_per_port'      : C_SCS_base_per_port,
        'c_om_base_scs'            : C_OM_base_SCS,
        'e_req_kwh'                : E_req_kwh,
        # Growth rates and derived NPV factors
        'g_aadt'                   : g_aadt,
        'g_ev'                     : g_ev,
        # ADD these replacements in their place:
        # 'g_inf'                    : g_inf,       # single inflation rate for all monetary streams
        'g_price_dwc'              : g_price_dwc,
        'g_price_scs'              : g_price_scs,
        'g_lcoe'                   : g_lcoe,
        'g_om'                     : g_om,
        'g_rev_dwc'                : g_rev_dwc,
        'g_rev_scs'                : g_rev_scs,
        'g_var'                    : g_var,
        'gamma_rev_dwc'            : Gamma_rev_dwc,
        'gamma_rev_scs'            : Gamma_rev_scs,
        'gamma_var'                : Gamma_var,
        'gamma_fix_om'             : Gamma_fix_om,
        'gamma_d_T'                : gamma_d_T,
        'g_d'                      : g_d,
        'n_dwc_selected'           : len(electrified_segs),
        'n_scs_selected'           : len(selected_scs_cand),
        'n_scs_existing'           : len(R_exist),
        'obj_val'                  : model.ObjVal,
        'total_rev_dwc'            : total_rev_dwc,
        'total_rev_scs'            : total_rev_scs,
        'total_opex_dwc'           : total_opex_dwc,
        'total_opex_scs'           : total_opex_scs,
        'run_tag'                  : run_tag,
        'output_dir'               : f"Results_{run_tag}",
    }

    return {
        # Identification
        'run_tag'                 : run_tag,
        'params'                  : params,
        'meta'                    : meta,
        # Core solution lists
        'electrified_segs'        : electrified_segs,
        'selected_scs_cand'       : selected_scs_cand,
        'selected_substations_dwc': selected_substations_dwc,
        'selected_substations_scs': selected_substations_scs,
        # DataFrames (exact column names expected by plotting files)
        'df_dwc'                  : df_dwc,
        'df_scs'                  : df_scs,
        'df_exist'                : df_exist,
        'df_connections'          : df_connections,
        'df_scs_connections'      : df_scs_connections,
        'df_diag'                 : df_diag,
        # Coordinate frames
        'I'                       : I,
        'seg_coords'              : seg_coords,
        'sub_coords'              : sub_coords,
        'scs_coords_cand'         : scs_coords_cand,
        'scs_coords_exist'        : scs_coords_exist,
        # Scalars for sensitivity comparison
        'total_rev_dwc'           : total_rev_dwc,
        'total_rev_scs'           : total_rev_scs,
        'total_opex_dwc'          : total_opex_dwc,
        'total_opex_scs'          : total_opex_scs,
        'obj_val'                 : model.ObjVal,
        # Report text
        'startup_text'            : startup_text,
        # Derived params (needed by some plots)
        'alpha'                   : alpha,
        'discount_rate'           : discount_rate,
        'project_life'            : project_life,
    }


# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS  — writes the exact CSV set expected by Main_plots.py
# ══════════════════════════════════════════════════════════════════════════════
def save_results(results, output_dir=None):
    """
    Write all CSV outputs for one optimization run.  Directory structure and
    file names are identical to the original File 1 so that Main_plots.py
    works without any changes.
    """
    if output_dir is None:
        output_dir = results['meta']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    def outpath(f):
        return os.path.join(output_dir, f)

    # Update output_dir in meta so plots can find files
    results['meta']['output_dir'] = output_dir

    df_dwc         = results['df_dwc']
    df_scs         = results['df_scs']
    df_exist       = results['df_exist']
    df_connections = results['df_connections']
    df_scs_conn    = results['df_scs_connections']
    df_diag        = results['df_diag']
    meta           = results['meta']
    I              = results['I']
    seg_coords     = results['seg_coords']
    sub_coords     = results['sub_coords']
    sel_sub_dwc    = results['selected_substations_dwc']
    sel_sub_scs    = results['selected_substations_scs']
    scs_coords_cand  = results['scs_coords_cand']
    scs_coords_exist = results['scs_coords_exist']
    use_existing_scs = meta['use_existing_scs']

    df_connections.to_csv(outpath('DWC_Network_FromTo_Table.csv'),   index=False)
    df_scs_conn.to_csv(outpath('SCS_Network_FromTo_Table.csv'),      index=False)
    df_diag.to_csv(outpath('DWC_Selection_Master_Diagnostic.csv'),   index=False)

    if not df_dwc.empty:
        df_dwc.to_csv(outpath('DWC_Selected_Segments.csv'), index=False)
        print(f"DWC_Selected_Segments.csv saved ({len(df_dwc)} rows)")

    if not df_scs.empty:
        df_scs.to_csv(outpath('SCS_Selected_Stations.csv'), index=False)
        print(f"SCS_Selected_Stations.csv saved ({len(df_scs)} rows)")

    if use_existing_scs and not df_exist.empty:
        df_exist.to_csv(outpath('SCS_Existing_Stations.csv'), index=False)
        print(f"SCS_Existing_Stations.csv saved ({len(df_exist)} rows)")

    pd.DataFrame({
        'Sub_ID': sorted(sel_sub_dwc),
        'Role'  : 'DWC',
        'X'     : [sub_coords.get(s, [None, None])[0] for s in sorted(sel_sub_dwc)],
        'Y'     : [sub_coords.get(s, [None, None])[1] for s in sorted(sel_sub_dwc)],
    }).to_csv(outpath('DWC_Selected_Substations.csv'), index=False)

    pd.DataFrame({
        'Sub_ID': sorted(sel_sub_scs),
        'Role'  : 'SCS',
        'X'     : [sub_coords.get(s, [None, None])[0] for s in sorted(sel_sub_scs)],
        'Y'     : [sub_coords.get(s, [None, None])[1] for s in sorted(sel_sub_scs)],
    }).to_csv(outpath('SCS_Selected_Substations.csv'), index=False)

    pd.DataFrame([meta]).to_csv(outpath('run_metadata.csv'), index=False)

    pd.DataFrame([
        {'Segment_ID': i, 'X': seg_coords[i][0], 'Y': seg_coords[i][1]}
        for i in I if i in seg_coords
    ]).to_csv(outpath('All_Segments_Coords.csv'), index=False)

    pd.DataFrame([
        {'Sub_ID': s, 'X': v[0], 'Y': v[1]}
        for s, v in sub_coords.items()
    ]).to_csv(outpath('All_Substations_Coords.csv'), index=False)

    if use_existing_scs and scs_coords_exist:
        pd.DataFrame([
            {'SCS_ID': r, 'MID_X': v[0], 'MID_Y': v[1]}
            for r, v in scs_coords_exist.items()
        ]).to_csv(outpath('SCS_Exist_Coords.csv'), index=False)

    # Plain-text report (same format as original)
    SEP  = "─" * 70
    alpha         = meta['alpha']
    discount_rate = meta['discount_rate']
    project_life  = meta['project_life']
    use_existing  = meta['use_existing_scs']
    # sigma         = meta['sigma']
    p_param       = meta['p_param']
    g_d           = meta.get('g_d',          0.0)
    gamma_rev_dwc = meta.get('gamma_rev_dwc', 1.0)
    gamma_rev_scs = meta.get('gamma_rev_scs', 1.0)
    gamma_var     = meta.get('gamma_var',     1.0)
    gamma_fix_om  = meta.get('gamma_fix_om',  1.0)

    report_lines = [results['startup_text'], ""]
    report_lines += [
        f"{'═'*70}",
        "  OPTIMISATION RESULTS SUMMARY",
        f"  Formulation : {'With Existing SCS' if use_existing else 'No Existing SCS'}",
        f"  p = {p_param}  |  ETA_DWC = {meta['eta_dwc']:.2f}  |  ETA_SCS = {meta['eta_scs']:.2f}  |  T = {project_life} yr  |  g_d = {g_d:.4f}",
        f"{'═'*70}",
        f"\n{SEP}",
        f"  SELECTED DWC SEGMENTS  ({len(df_dwc)} selected)",
        f"{SEP}",
    ]
    if not df_dwc.empty:
        report_lines.append(
            f"  {'Seg ID':>8}  {'Connection':>16}  {'From':>12}  "
            f"{'AADT':>7}  {'E_del (MWh)':>12}  {'λ':>6}  {'LCOE':>6}"
        )
        for _, row in df_dwc.iterrows():
            report_lines.append(
                f"  {row['Segment_ID']:>8}  {row['Conn_Type']:>16}  "
                f"{str(row['From_ID']):>12}  {int(row['AADT_COM'] or 0):>7,}  "
                f"{row['E_del_DWC']/1e3:>12,.1f}  {row['Lambda']:>6.3f}  {row['LCOE']:>6.4f}"
            )

    report_lines += [
        f"\n{SEP}",
        f"  SELECTED SCS STATIONS  ({len(df_scs)} selected)",
        f"{SEP}",
    ]
    if df_scs.empty:
        report_lines.append("  [No SCS stations selected]")
    else:
        report_lines.append(
            f"  {'SCS_ID':>8}  {'Host_Seg(s)':>20}  {'Sub_ID':>8}  {'Dist(mi)':>10}  "
            f"{'E_del(MWh)':>12}  {'λ':>6}  {'LCOE':>6}"
        )
        for _, row in df_scs.iterrows():
            d = f"{row['Dist_to_Sub_mi']:.2f}" if row['Dist_to_Sub_mi'] is not None else "N/A"
            # Collect all segment IDs that map to this SCS site from the full mapping
            report_lines.append(
                f"  {row['SCS_ID']:>8}  {str(row['Host_Seg'] or 'N/A'):>20}  "
                f"{str(row['Sub_ID'] or 'N/A'):>8}  {d:>10}  "
                f"{row['E_del_SCS']/1e3:>12,.1f}  {row['Lambda']:>6.3f}  {row['LCOE']:>6.4f}"
            )

    report_lines += [
            f"\n{SEP}",
            "  FINANCIAL SNAPSHOT (base-year quantities; Γ factors for lifetime NPV)",
            # ADD this replacement:
            f"  T={project_life} yr  |  ρ={discount_rate:.0%}  |  "
            f"Γ_rev_DWC={gamma_rev_dwc:.3f}  Γ_rev_SCS={gamma_rev_scs:.3f}  "
            f"Γ_var={gamma_var:.3f}  Γ_fix={gamma_fix_om:.3f}",
            f"{SEP}",
            f"  R^DWC  (base-yr revenue DWC)  : ${meta['total_rev_dwc']:>14,.0f} /yr",
            f"  R^SCS  (base-yr revenue SCS)  : ${meta['total_rev_scs']:>14,.0f} /yr",
            f"  C^{{O&M,DWC}}  (base-yr total)   : ${meta['total_opex_dwc']:>14,.0f} /yr",
            f"  C^{{O&M,SCS}}  (base-yr total)   : ${meta['total_opex_scs']:>14,.0f} /yr",
            f"  Lifetime NPV Z               : ${meta['obj_val']:>14,.0f}",
            f"{'═'*70}\n",
        ]
    report_text = "\n".join(report_lines)

    with open(outpath('model_results_summary.txt'), 'w') as f:
        f.write(report_text)

    print(f"\nAll outputs written to: {output_dir}/")
    return output_dir


# ══════════════════════════════════════════════════════════════════════════════
# BATCH RUNNER  — run a list of parameter dicts sequentially
# ══════════════════════════════════════════════════════════════════════════════
def run_batch(data, params_list, env=None, save=True, base_output_dir='.'):
    """
    Run the optimizer for every element of params_list.

    Parameters
    ----------
    data           : dict  — from load_data()
    params_list    : list of dicts  — each must have a unique 'run_tag'
    env            : gp.Env or None
    save           : bool  — if True, call save_results() after each solve
    base_output_dir: str   — parent directory; each run gets its own subdir

    Returns
    -------
    results_list : list of result dicts (same order as params_list)
    """
    results_list = []
    n = len(params_list)
    for idx, p in enumerate(params_list):
        tag = p.get('run_tag', f'run_{idx:04d}')
        print(f"\n{'╔'+'═'*58+'╗'}")
        print(f"  Scenario {idx+1}/{n}  —  {tag}")
        print(f"{'╚'+'═'*58+'╝'}")
        result = run_single_optimization(data, p, env=env)
        if result is None:
            print(f"  WARNING: scenario {tag} returned no feasible solution — skipping.")
            continue
        if save:
            out_dir = os.path.join(base_output_dir, f"Results_{tag}")
            save_results(result, out_dir)
        results_list.append(result)
    return results_list
