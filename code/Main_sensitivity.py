#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# Main_sensitivity.py
#
# Global Sensitivity Analysis (GSA) data-generation layer for the joint DWC+SCS
# optimizer. This module samples parameter combinations, runs the optimizer,
# and assembles reusable output tables. Statistical modeling and technology-
# preference analysis belong in Paper_Figures/Figure 3/Logistic_Regression.py.
#
# ── ETA DESIGN ───────────────────────────────────────────────────────────────
# ETA_DWC and ETA_SCS are sampled as separate traffic-allocation parameters
# subject to ETA_DWC + ETA_SCS <= 1. Both columns are saved for downstream
# regression and visualization work.
#
# ── WHAT THIS MODULE PRODUCES ────────────────────────────────────────────────
# A reproducible constrained parameter design, one saved optimizer result per
# feasible draw, and a pooled run-by-segment table with y in {DWC, SCS, None}.
# It does not fit a regression, calculate feature importance, rank parameters,
# or construct a technology-preference contrast.
#
# ── METHOD ────────────────────────────────────────────────────────────────────
# 1. Draw N constrained Latin-hypercube-based samples of theta (9 parameters).
# 2. Solve the MILP once per draw via run_single_optimization() (Main_optimizer's
#    existing API -- untouched). Persist every run's full CSV output via the
#    existing save_results(), so nothing about run storage is GSA-specific.
# 3. Build ONE flat table: one row per (run, segment), columns = raw theta_r
#    (repeated across all 9000 segments within a run) + fixed segment
#    features x_s (AADT, substation distance -- repeated across all runs for a
#    given segment) + the categorical outcome y_rs.
#
# ── EVERYTHING IS PERSISTED TO DISK ──────────────────────────────────────────
#   GSA_<timestamp>/
#     ├── lhs_parameter_samples.csv         <- N x k sampled theta vectors
#     ├── gsa_run_manifest.json             <- full provenance: bounds, fixed overrides, mip_gap, seed
#     ├── Runs/
#     │     └── Results_gsa_XXXX/           <- full per-run CSV set from save_results() (if enabled)
#     ├── segment_level_pooled.csv          <- THE flat (run x segment) table used for the regression
#     ├── class_balance.csv                 <- how imbalanced y_rs was (diagnostic, always saved)
#     └── convergence_check/                <- (optional) N=300/600/1000 data generation
#           ├── convergence_data_generation_summary.csv
#           └── convergence_manifest.json
# ═══════════════════════════════════════════════════════════════════════════════

import os
import json
import datetime
import numpy as np
import pandas as pd

from SALib.sample import latin

from Main_optimizer import run_single_optimization, save_results


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PARAMETER SPACE DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════
# Every bound is anchored to the values actually used in Main.py's
# BASELINE_OVERRIDES / DEFAULT_PARAMS, taken as the realistic reference point.
#
# p_param is intentionally NOT swept -- it is held FIXED (via GSA_FIXED_OVERRIDES)
# for every single GSA draw. This isolates how the optimizer reallocates a FIXED
# deployment budget across DWC vs SCS vs nothing as the other economic and
# behavioral parameters vary -- not how the answer changes if you simply allow
# yourself more infrastructure.
#
# ── ETA parameterization ─────────────────────────────────────────────────────
# ETA_DWC and ETA_SCS are sampled separately over [0.1, 0.9], subject to:
#
#   ETA_DWC + ETA_SCS <= 1
#
# The inequality leaves genuine two-dimensional variation, allowing the
# regression to estimate each allocation parameter's influence while
# controlling for the other. Constrained pairs are drawn from LHS proposal
# batches and infeasible pairs are rejected; the other continuous parameters
# retain their ordinary LHS draws.
#
# ── Discrete parameters ───────────────────────────────────────────────────────
# project_life and N_SCS_stalls are sampled as TRUE discrete integers directly
# (np.random.Generator.integers over the bound range), never as continuous LHS
# draws that get rounded after the fact. Rounding continuous draws creates
# uneven probability mass at the boundary integers (e.g. both 4.49 and 4.51
# round differently despite near-identical sampled values); sampling discretely
# avoids that entirely.
GSA_PARAM_BOUNDS = {
    'ETA_DWC'              : [0.10, 0.90],
    'ETA_SCS'              : [0.10, 0.90],
    'g_aadt'               : [0.008, 0.02],
    'g_ev'                 : [0.10, 0.18],
    'project_life'         : [25, 40],
    'C_DWC_base'           : [8_000_000, 12_000_000],
    'C_SCS_base_per_port'  : [180_000, 300_000],
    'N_SCS_stalls'         : [4, 8],
    'g_lcoe'               : [0.005, 0.025],
}

GSA_ALLOCATION_SUM_MAX = 1.0

# Parameters sampled as TRUE discrete integers (inclusive bounds), never
# continuous-then-rounded.
GSA_DISCRETE_PARAMS = {'project_life', 'N_SCS_stalls'}

# Fixed (non-swept) parameters applied to every GSA draw, matching Main.py's
# BASELINE_OVERRIDES so the GSA study describes the same fixed-budget regime.
GSA_FIXED_OVERRIDES = {
    'p_param': 200,
}

# Segment-level control features (x_s in the regression). Must be columns
# present in DWC_Selection_Master_Diagnostic.csv (df_diag), which is saved by
# Main_optimizer.save_results() for every run regardless of solution content.
SEGMENT_FEATURE_COLS = ['AADT_COM', 'Min_Sub_Dist']

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONSTRAINED LATIN-HYPERCUBE-BASED SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════
def build_lhs_param_list(base_params, bounds_dict, n_samples, seed=42,
                          fixed_overrides=None, discrete_params=None):
    """
    Draw N constrained Latin-hypercube-based samples and return full param dicts
    (base_params overridden by each sampled draw), ready to feed into
    run_single_optimization() one at a time.

    All continuous parameters vary simultaneously per run. The unconstrained
    dimensions use Latin Hypercube sampling. Discrete parameters
    (named in `discrete_params`) are sampled directly as integers via
    np.random.Generator.integers -- never as continuous draws that get rounded.

    ETA allocation handling:
        ETA_DWC and ETA_SCS are sampled separately within their bounds. LHS
        proposal pairs are accepted only when ETA_DWC + ETA_SCS <= 1. This
        produces genuine variation in both allocation fractions instead of
        making either one a deterministic function of the other.

    Parameters
    ----------
    base_params : dict -- baseline parameter dict; LHS draws override this
    bounds_dict : dict -- {param_name: [lo, hi]}; CONTINUOUS params only
                           (discrete params should NOT be passed here with
                           non-integer bounds -- they still need an entry so
                           LHS knows the design dimensionality, but their
                           sampled column is discarded and replaced below)
    n_samples   : int
    seed        : int
    fixed_overrides : dict or None -- applied last, always wins
    discrete_params : set or None -- names (subset of bounds_dict keys) to
                       resample as true integers instead of using the LHS draw

    Returns
    -------
    params_list : list of dicts (length n_samples), each with a unique run_tag.
    df_X        : pd.DataFrame (n_samples rows) -- the full sampled design
                  matrix, including both ETA allocation columns, saved
                  verbatim to lhs_parameter_samples.csv. This is also the
                  per-run theta_r table merged into the downstream dataset.
    """
    discrete_params = discrete_params or set()
    names  = list(bounds_dict.keys())
    bounds = [bounds_dict[n] for n in names]
    problem = {'num_vars': len(names), 'names': names, 'bounds': bounds}

    rng_master = np.random.default_rng(seed)

    # LHS draw for all named params (continuous bounds used as placeholders for
    # discrete ones; ETA columns are replaced by constrained draws below).
    X = latin.sample(problem, n_samples, seed=seed)

    if 'ETA_DWC' in names and 'ETA_SCS' in names:
        eta_pairs = _sample_constrained_eta_pairs(
            bounds_dict, n_samples, seed=seed,
        )
        X[:, names.index('ETA_DWC')] = eta_pairs[:, 0]
        X[:, names.index('ETA_SCS')] = eta_pairs[:, 1]

    # Overwrite discrete columns with TRUE discrete draws (no rounding of
    # continuous values).
    for name in discrete_params:
        if name not in names:
            continue
        col = names.index(name)
        lo, hi = bounds_dict[name]
        X[:, col] = rng_master.integers(int(lo), int(hi) + 1, size=n_samples)

    fixed_overrides = fixed_overrides or {}

    params_list = []
    for n_idx in range(n_samples):
        p   = _copy_dict_deep(base_params)
        row = dict(zip(names, X[n_idx, :]))

        for name in discrete_params:
            if name in row:
                row[name] = int(round(row[name]))

        for name, val in row.items():
            p[name] = val

        p['ETA_DWC'] = float(row.get('ETA_DWC', p.get('ETA_DWC', 0.0)))
        p['ETA_SCS'] = float(row.get('ETA_SCS', p.get('ETA_SCS', 0.0)))

        for fk, fv in fixed_overrides.items():
            p[fk] = fv

        p['run_tag'] = f"gsa_{n_idx:04d}"
        params_list.append(p)

    df_X = pd.DataFrame(X, columns=names)
    df_X.insert(0, 'run_tag', [p['run_tag'] for p in params_list])

    return params_list, df_X


def _sample_constrained_eta_pairs(bounds_dict, n_samples, seed=42):
    """Draw feasible ETA pairs from LHS proposal batches via rejection."""
    eta_names = ['ETA_DWC', 'ETA_SCS']
    min_sum = sum(float(bounds_dict[name][0]) for name in eta_names)
    if min_sum > GSA_ALLOCATION_SUM_MAX:
        raise ValueError(
            "ETA lower bounds are infeasible: ETA_DWC + ETA_SCS must be <= "
            f"{GSA_ALLOCATION_SUM_MAX}."
        )
    eta_problem = {
        'num_vars': 2,
        'names': eta_names,
        'bounds': [bounds_dict[name] for name in eta_names],
    }
    rng = np.random.default_rng(seed)
    accepted = []
    n_accepted = 0

    while n_accepted < n_samples:
        n_remaining = n_samples - n_accepted
        # The feasible region occupies about half of the requested square, so
        # 2.5x proposals normally fills the remainder in one iteration.
        batch_size = max(64, int(np.ceil(2.5 * n_remaining)))
        batch_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
        candidates = latin.sample(eta_problem, batch_size, seed=batch_seed)
        feasible = candidates[candidates[:, 0] + candidates[:, 1]
                              <= GSA_ALLOCATION_SUM_MAX + 1e-12]
        if len(feasible):
            accepted.append(feasible)
            n_accepted += len(feasible)

    eta_pairs = np.vstack(accepted)[:n_samples]
    rng.shuffle(eta_pairs)
    return eta_pairs


def _copy_dict_deep(d):
    import copy
    return copy.deepcopy(d)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — RUN THE GSA SWEEP  (solve N MILPs, persist everything)
# ═══════════════════════════════════════════════════════════════════════════════
def run_gsa_sweep(data, base_params, output_root, n_samples=1200, seed=42,
                   env=None, save_each_run=True, fixed_overrides=None,
                   gsa_dir=None):
    """
    Draw N constrained Latin-hypercube-based samples and solve the MILP once
    per sample, persisting every run's full CSV output via the existing
    save_results(). This is pure data generation -- no analysis happens here.

    Parameters
    ----------
    data         : dict   -- output of Main_optimizer.load_data()
    base_params  : dict   -- baseline parameter dict
    output_root  : str    -- parent directory; a new GSA_<timestamp> dir is
                              created here unless `gsa_dir` is given explicitly
    n_samples    : int
    seed         : int
    env          : gp.Env or None
    save_each_run: bool   -- if True, persist every run's full segment-level
                              CSV set (required -- the pooled dataset needs the
                              per-segment Is_DWC outcome from every run)
    fixed_overrides : dict or None
    gsa_dir      : str or None -- if given, reuse this directory instead of
                              creating a new GSA_<timestamp> one (used by the
                              convergence check to keep sub-studies organized)

    Returns
    -------
    gsa_dir      : str -- path to the directory containing Runs/ and the
                           saved sample matrix
    df_X         : pd.DataFrame -- the sampled theta matrix
    run_tags_feasible : list of str -- run_tags that returned a feasible
                           solution (others are skipped; downstream code must
                           not assume every gsa_XXXX run_tag has a results dir)
    """
    if save_each_run is False:
        raise ValueError(
            "save_each_run=False is not supported: the segment-level "
            "pooled dataset requires every run's per-segment Is_DWC/Is_SCS "
            "outcome, which only exists in the saved per-run CSVs."
        )

    if gsa_dir is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        gsa_dir = os.path.join(output_root, f"GSA_{ts}")
    runs_dir = os.path.join(gsa_dir, "Runs")
    os.makedirs(runs_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  GSA SWEEP — drawing {n_samples} constrained "
          f"Latin-hypercube-based samples")
    print(f"  Parameters (k)  : {len(GSA_PARAM_BOUNDS)}")
    print(f"  Fixed overrides : {fixed_overrides}")
    print(f"  Output directory: {gsa_dir}")
    print(f"{'='*70}\n")

    params_list, df_X = build_lhs_param_list(
        base_params, GSA_PARAM_BOUNDS, n_samples, seed=seed,
        fixed_overrides=fixed_overrides, discrete_params=GSA_DISCRETE_PARAMS,
    )

    df_X.to_csv(os.path.join(gsa_dir, 'lhs_parameter_samples.csv'), index=False)
    print(f"Saved parameter sample matrix -> lhs_parameter_samples.csv "
          f"({df_X.shape[0]} rows x {df_X.shape[1]} cols)")

    run_tags_feasible = []
    n_infeasible = 0
    for idx, p in enumerate(params_list):
        print(f"\n--- GSA run {idx+1}/{n_samples}  (run_tag={p['run_tag']}) ---")
        result = run_single_optimization(data, p, env=env)

        if result is None:
            print(f"  INFEASIBLE — skipping this draw entirely "
                  f"(no segment-level outcome to record).")
            n_infeasible += 1
            continue

        run_out_dir = os.path.join(runs_dir, f"Results_{p['run_tag']}")
        save_results(result, run_out_dir)
        run_tags_feasible.append(p['run_tag'])

    print(f"\nGSA sweep complete: {len(run_tags_feasible)}/{n_samples} feasible "
          f"({n_infeasible} infeasible draws skipped).")

    manifest = {
        'timestamp_dir'     : os.path.basename(gsa_dir),
        'n_samples_requested': n_samples,
        'n_feasible'        : len(run_tags_feasible),
        'n_infeasible'      : n_infeasible,
        'seed'              : seed,
        'sampling_method'   : 'constrained Latin-hypercube-based sampling',
        'param_bounds'      : GSA_PARAM_BOUNDS,
        'allocation_constraint': 'ETA_DWC + ETA_SCS <= 1.0',
        'discrete_params'   : sorted(GSA_DISCRETE_PARAMS),
        'fixed_overrides'   : fixed_overrides or {},
        'segment_feature_cols': SEGMENT_FEATURE_COLS,
        'base_params_used'  : {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                                for k, v in base_params.items()},
    }
    with open(os.path.join(gsa_dir, 'gsa_run_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Manifest saved -> gsa_run_manifest.json (includes fixed_overrides, "
          f"discrete_params, segment_feature_cols — full provenance).")

    return gsa_dir, df_X, run_tags_feasible


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BUILD THE POOLED (run x segment) DATASET
# ═══════════════════════════════════════════════════════════════════════════════
def build_segment_level_pooled_table(gsa_dir, df_X, run_tags_feasible,
                                      segment_feature_cols=None):
    """
    Build the single flat table consumed by downstream analysis:
        one row per (run_tag, Segment_ID), columns =
            [run_tag, Segment_ID, <theta columns repeated per run>,
             <segment feature columns repeated per segment>, y]
    where y in {'DWC', 'SCS', 'None'}.

    theta_r (the 9 swept parameters, including both ETA allocations) is constant
    across all ~9000 rows of one run. x_s (AADT, substation distance) is
    constant across all run-repetitions of one segment. All segments from all
    feasible runs are retained without subsampling for downstream analysis.

    SCS assignment at the segment level: a segment is "Is_SCS=1" if it is the
    representative host segment (Host_Seg) of an SCS site selected in that run
    (SCS_Selected_Stations.csv). This mirrors the Is_DWC flag already computed
    by Main_optimizer and saved directly in DWC_Selection_Master_Diagnostic.csv
    for the DWC case.

    Returns
    -------
    df_pooled : pd.DataFrame -- saved to segment_level_pooled.csv as a side
                effect; also returned in memory to the orchestrator.
    """
    segment_feature_cols = segment_feature_cols or SEGMENT_FEATURE_COLS
    runs_dir = os.path.join(gsa_dir, "Runs")
    theta_cols = [c for c in df_X.columns if c != 'run_tag']

    df_X_idx = df_X.set_index('run_tag')

    all_rows = []
    n_used = 0
    for run_tag in run_tags_feasible:
        run_path  = os.path.join(runs_dir, f"Results_{run_tag}")
        diag_path = os.path.join(run_path, "DWC_Selection_Master_Diagnostic.csv")
        scs_path  = os.path.join(run_path, "SCS_Selected_Stations.csv")

        if not os.path.exists(diag_path):
            print(f"  WARNING: missing diagnostic CSV for {run_tag} — skipping.")
            continue

        df_diag = pd.read_csv(diag_path)
        missing_feats = [c for c in segment_feature_cols if c not in df_diag.columns]
        if missing_feats:
            raise KeyError(
                f"Segment feature column(s) {missing_feats} not found in "
                f"{diag_path}. Update SEGMENT_FEATURE_COLS to match the "
                f"actual DWC_Selection_Master_Diagnostic.csv schema."
            )

        df_diag['Is_DWC'] = df_diag['Is_DWC'].fillna(0).astype(int)
        df_diag['Is_SCS'] = 0
        if os.path.exists(scs_path):
            df_scs = pd.read_csv(scs_path)
            if 'Host_Seg' in df_scs.columns:
                host_segs = set(df_scs['Host_Seg'].dropna().tolist())
                df_diag['Is_SCS'] = df_diag['Segment_ID'].isin(host_segs).astype(int)

        def _label(row):
            if row['Is_DWC'] == 1:
                return 'DWC'
            if row['Is_SCS'] == 1:
                return 'SCS'
            return 'None'
        df_diag['y'] = df_diag.apply(_label, axis=1)

        keep_cols = ['Segment_ID', 'y'] + segment_feature_cols
        df_sub = df_diag[keep_cols].copy()
        df_sub['run_tag'] = run_tag

        theta_row = df_X_idx.loc[run_tag, theta_cols]
        for c in theta_cols:
            df_sub[c] = theta_row[c]

        all_rows.append(df_sub)
        n_used += 1

    if not all_rows:
        raise ValueError("No usable runs found — cannot build pooled table.")

    df_pooled = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(gsa_dir, 'segment_level_pooled.csv')
    df_pooled.to_csv(out_path, index=False)
    print(f"\nPooled (run x segment) table saved -> segment_level_pooled.csv")
    print(f"  Runs used     : {n_used}/{len(run_tags_feasible)}")
    print(f"  Total rows    : {len(df_pooled):,}")
    print(f"  Columns       : {list(df_pooled.columns)}")

    class_counts = df_pooled['y'].value_counts()
    class_balance = (class_counts / len(df_pooled)).rename('fraction').reset_index()
    class_balance.columns = ['class', 'fraction']
    class_balance['count'] = class_counts.reindex(class_balance['class']).values
    class_balance.to_csv(os.path.join(gsa_dir, 'class_balance.csv'), index=False)
    print(f"\nClass balance (y outcome across all pooled rows):")
    print(class_balance.to_string(index=False))

    return df_pooled


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MULTI-SIZE DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════
def run_convergence_check(data, base_params, output_root, sample_sizes=(300, 600, 1000),
                           seed=42, env=None, fixed_overrides=None):
    """
    Generate independent sweep and pooled datasets at each requested sample
    size. This function performs no regression or parameter-ranking analysis;
    those comparisons belong in the downstream analysis script.

    Each sample size gets its OWN GSA_<timestamp> directory (full MILP re-solve
    each time — this is intentionally expensive and is a one-time check, not
    part of the regular study pipeline). Results are also written to a shared
    convergence_check/ directory containing a data-generation summary.

    Returns
    -------
    convergence_dir : str -- path to the convergence_check/ directory
    df_compare       : pd.DataFrame -- data-generation summary for each N
    """
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    convergence_dir = os.path.join(output_root, f"convergence_check_{ts}")
    os.makedirs(convergence_dir, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  CONVERGENCE CHECK across sample sizes: {sample_sizes}")
    print(f"  Each size requires a FULL independent GSA sweep (expensive).")
    print(f"{'#'*70}")

    rows = []
    for n in sample_sizes:
        print(f"\n>>> Generating GSA dataset at N={n} <<<")
        gsa_dir, df_X, run_tags = run_gsa_sweep(
            data, base_params, convergence_dir, n_samples=n, seed=seed,
            env=env, save_each_run=True, fixed_overrides=fixed_overrides,
        )
        df_pooled = build_segment_level_pooled_table(gsa_dir, df_X, run_tags)
        rows.append({
            'n_samples_requested': int(n),
            'n_feasible_runs': int(len(run_tags)),
            'n_pooled_rows': int(len(df_pooled)),
            'gsa_dir': gsa_dir,
        })
    df_compare = pd.DataFrame(rows)

    out_path = os.path.join(
        convergence_dir, 'convergence_data_generation_summary.csv'
    )
    df_compare.to_csv(out_path, index=False)

    manifest = {
        'sample_sizes': list(sample_sizes),
        'seed': seed,
        'fixed_overrides': fixed_overrides or {},
        'note': 'Each sample size generated optimizer outputs and a pooled '
                'segment table only. Regression and ranking comparisons are '
                'deferred to the downstream Figure 3 analysis.',
    }
    with open(os.path.join(convergence_dir, 'convergence_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nData-generation summary saved -> {out_path}")
    print(df_compare.to_string(index=False))

    return convergence_dir, df_compare


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — TOP-LEVEL ORCHESTRATOR  (what Main.py actually calls)
# ═══════════════════════════════════════════════════════════════════════════════
def run_full_sensitivity_study(data, base_params, output_root,
                                n_samples=1200, seed=42, env=None,
                                fixed_overrides=None):
    """
    Generate the complete dataset for the downstream Figure 3 analysis:
    constrained parameter sampling -> N MILP solves -> pooled run-by-segment
    table. No regression, coefficient ranking, bootstrap, or plotting is run.

    Returns
    -------
    gsa_dir : str -- directory containing all generated run and pooled data.
    """
    fixed_overrides = fixed_overrides if fixed_overrides is not None else GSA_FIXED_OVERRIDES

    gsa_dir, df_X, run_tags_feasible = run_gsa_sweep(
        data, base_params, output_root, n_samples=n_samples, seed=seed,
        env=env, save_each_run=True, fixed_overrides=fixed_overrides,
    )

    df_pooled = build_segment_level_pooled_table(gsa_dir, df_X, run_tags_feasible)

    print(f"\n{'='*70}")
    print(f"  GSA DATA GENERATION COMPLETE")
    print(f"  Feasible runs : {len(run_tags_feasible):,}")
    print(f"  Pooled rows   : {len(df_pooled):,}")
    print(f"  Outputs saved : {gsa_dir}/")
    print(f"  Regression and Figure 3 analysis should be run later with:")
    print(f"  Paper_Figures/Figure 3/Logistic_Regression.py")
    print(f"{'='*70}")
    return gsa_dir
