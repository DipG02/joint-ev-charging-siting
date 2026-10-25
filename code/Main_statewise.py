#!/usr/bin/env python3
# ═══════════════════════════════════════════════════════════════════════════════
# Main_statewiseruns.py
#
# Statewise DWC + SCS optimizer runner.
#
# Purpose
# -------
# 1. Read the national input files.
# 2. Split them into one optimizer-ready input folder per state, following the
#    same FILE_NAME parsing/filtering logic used in Sorting.py.
# 3. Run the existing optimizer once per state.
# 4. Save outputs using the exact same save_results() structure used in Main.py:
#       Results_<run_tag>/
#           DWC_Selected_Segments.csv
#           SCS_Selected_Stations.csv
#           run_metadata.csv
#           ...
#
# Notes
# -----
# - This script does NOT modify Main_optimizer.py.
# - It imports the same optimizer API used by Main.py:
#       load_data, run_single_optimization, save_results, DEFAULT_PARAMS
# - By default, it uses an "unlimited" statewise budget:
#       p_param = number of candidate highway segments in that state
#   This removes the artificial state-level deployment-count constraint and
#   reveals each state's latent DWC/SCS preference.
# ═══════════════════════════════════════════════════════════════════════════════

import os
import re
import json
import copy
import shutil
import datetime
import traceback

import pandas as pd

try:
    import gurobipy as gp
except Exception:
    gp = None

from Main_optimizer import (
    load_data,
    run_single_optimization,
    save_results,
    DEFAULT_PARAMS,
)

# Optional, only used for summary table if available.
try:
    from Main_plots import _compute_run_summary
    _HAS_SUMMARY = True
except Exception:
    _HAS_SUMMARY = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# National input directory containing:
#   NHS_US_5miles_with_SCS.csv
#   NHS_USA_5miles_Adjacent.csv
#   NHS_US_SubstationDistance.csv
#   US_SCS_unique_existing.csv
#   SubstationsProject.csv
#
# Override with DWC_INPUT_DIR when using a different input folder.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
MASTER_INPUT_DIR = os.environ.get(
    "DWC_INPUT_DIR", os.path.join(REPO_ROOT, "data", "processed")
)

# Where state-specific input folders will be created.
STATE_INPUT_ROOT = os.path.join(REPO_ROOT, "Inputs_Statewise_AllStates")

# Where statewise optimizer outputs will be saved.
BASE_OUTPUT_DIR = os.path.join(REPO_ROOT, "Statewise_UnlimitedBudget_Results")

# Existing SCS handling, same meaning as in Main.py / Main_optimizer.py.
USE_EXISTING_SCS = True

# Keep all AADT values by default.
EXCLUDE_TOP_PCT = 1.0

# Same as Main.py.
MAX_SUB_RANK = 10

# Whether to generate per-run plots using Main_plots.generate_all_plots.
# Main_plots.generate_all_plots is currently limited to the placement map and
# total DWC/SCS units plot.
PLOT_EACH_RUN = True

# If True, existing state input folders are deleted and rebuilt.
REBUILD_STATE_INPUTS = False

# If True, states with existing Results_<run_tag>/run_metadata.csv are skipped.
SKIP_COMPLETED_STATES = False

# If None, all states found in FILE_NAME are run.
# Otherwise, provide a list such as ["TX", "IN", "CA"].
ONLY_STATES = None

# States to exclude if present in the data.
# Leave empty if you truly want all unique state prefixes from FILE_NAME.
EXCLUDE_STATES = []

# Budget mode:
#   "unlimited_by_segments" : p_param = len(data["I"]) for that state.
#   "fixed"                 : p_param stays as BASELINE_OVERRIDES["p_param"].
BUDGET_MODE = "unlimited_by_segments"

# Optional cap when using unlimited_by_segments.
# Example: set to 2000 if a huge state becomes too slow.
# Leave None for no cap.
MAX_STATE_P_PARAM = None

# Solver gap for all state runs.
MIP_GAP = 0.01

# Baseline parameters.
# These match the style of your Main.py. For statewise unlimited runs, only
# p_param is changed after data loading if BUDGET_MODE="unlimited_by_segments".
BASELINE_OVERRIDES = {
    # EV adoption and traffic capture
    "rho_ev"               : 0.005,
    "ETA_DWC"              : 0.5,
    "ETA_SCS"              : 0.5,
    "Utilization_rate_SCS" : 1.0,

    # Deployment budget
    # Used only if BUDGET_MODE="fixed".
    "p_param"              : 5000,

    # Retail charging prices
    "P_CHARGE_DWC"         : 0.54,
    "P_CHARGE_SCS"         : 0.54,

    # Financial
    "discount_rate"        : 0.07,
    "project_life"         : 32,
    "H_op"                 : 8760.0,

    # Technology assumptions
    "P_veh_DWC_kw"         : 300.0,
    "N_SCS_stalls"         : 6,

    # Regional cost multiplier bounds
    "LAMBDA_MIN"           : 0.70,
    "LAMBDA_MAX"           : 1.30,

    # Growth rates
    "g_aadt"               : 0.011,
    "g_ev"                 : 0.15,
    "g_price_dwc"          : 0.02,
    "g_price_scs"          : 0.02,
    "g_lcoe"               : 0.015,
    "g_om"                 : 0.020,

    # Solver
    "mip_gap"              : MIP_GAP,
}


# State names only affect folder/run labels.
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


# Required filenames expected by Main_optimizer.load_data().
REQUIRED_INPUT_FILES = {
    "segments"     : "NHS_US_5miles_with_SCS.csv",
    "adjacency"    : "NHS_USA_5miles_Adjacent.csv",
    "subdist"      : "NHS_US_SubstationDistance.csv",
    "scs_existing" : "US_SCS_unique_existing.csv",
    "substations"  : "SubstationsProject.csv",
}

OPTIONAL_INPUT_FILES = {
    "scs_all"       : "US_SCS.csv",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_name(text):
    """Create filesystem-safe folder/run labels."""
    text = str(text).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_\-]+", "", text)
    return text


def _base_params():
    """Return a fresh baseline parameter dict."""
    p = copy.deepcopy(DEFAULT_PARAMS)
    p.update(BASELINE_OVERRIDES)
    return p


def _path(input_dir, filename):
    return os.path.join(input_dir, filename)


def validate_master_inputs(master_input_dir):
    missing = []
    for _, fname in REQUIRED_INPUT_FILES.items():
        p = _path(master_input_dir, fname)
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        raise FileNotFoundError(
            "Missing required input files:\n" + "\n".join(f"  {m}" for m in missing)
        )


def extract_state_abbr_from_filename(file_name_value):
    """
    Sorting.py logic:
        FILE_NAME example: 'TX_NHS_2021'
        state abbreviation = split('_')[0]
    """
    if pd.isna(file_name_value):
        return None
    val = str(file_name_value).strip()
    if not val:
        return None
    return val.split("_")[0].strip()


def discover_states(master_input_dir):
    """
    Find all unique state prefixes in NHS_US_5miles_with_SCS.csv.
    """
    seg_path = _path(master_input_dir, REQUIRED_INPUT_FILES["segments"])
    df_base = pd.read_csv(seg_path, low_memory=False)

    if "FILE_NAME" not in df_base.columns:
        raise KeyError("NHS_US_5miles_with_SCS.csv must contain FILE_NAME column.")

    states = (
        df_base["FILE_NAME"]
        .apply(extract_state_abbr_from_filename)
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )

    states = sorted(states)

    if ONLY_STATES is not None:
        only = {s.upper() for s in ONLY_STATES}
        states = [s for s in states if s in only]

    exclude = {s.upper() for s in EXCLUDE_STATES}
    states = [s for s in states if s not in exclude]

    if not states:
        raise ValueError("No states found after applying ONLY_STATES / EXCLUDE_STATES.")

    return states


def discover_states_from_existing_inputs(state_input_root):
    """
    Find state abbreviations from already-created Inputs_Statewise_<STATE>_*
    folders.
    """
    if not os.path.isdir(state_input_root):
        raise FileNotFoundError(
            f"State input root not found:\n  {state_input_root}"
        )

    states = []
    for name in os.listdir(state_input_root):
        p = os.path.join(state_input_root, name)
        if not os.path.isdir(p):
            continue
        m = re.match(r"^Inputs_Statewise_([A-Z]{2})_", name)
        if m:
            states.append(m.group(1).upper())

    states = sorted(set(states))

    if ONLY_STATES is not None:
        only = {s.upper() for s in ONLY_STATES}
        states = [s for s in states if s in only]

    exclude = {s.upper() for s in EXCLUDE_STATES}
    states = [s for s in states if s not in exclude]

    if not states:
        raise ValueError(
            "No states found in existing state input folders after applying "
            "ONLY_STATES / EXCLUDE_STATES."
        )

    return states


def create_state_input_dataset(master_input_dir, state_abbr, state_input_root):
    """
    Create one optimizer-ready input folder for a given state.

    This follows Sorting.py:
    - Filter base segment file by FILE_NAME prefix.
    - valid_segments = SEGMENT_ID_OG from filtered segment file.
    - valid_scs = NEAR_SCS_ID from filtered segment file.
    - Filter adjacency where both endpoints are in valid_segments.
    - Filter substation distance by IN_FID in valid_segments.
    - Filter SCS candidate file by SCS_ID in valid_scs.
    - Copy SubstationsProject.csv unchanged.
    """
    state_abbr = state_abbr.upper()
    state_full = STATE_NAMES.get(state_abbr, state_abbr)
    state_folder_name = f"Inputs_Statewise_{state_abbr}_{_safe_name(state_full)}"
    out_dir = os.path.join(state_input_root, state_folder_name)

    if os.path.exists(out_dir) and REBUILD_STATE_INPUTS:
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n--- Creating state input dataset: {state_abbr} ({state_full}) ---")
    print(f"    Output input folder: {out_dir}")

    # 1. Base segment file.
    df_base = pd.read_csv(
        _path(master_input_dir, REQUIRED_INPUT_FILES["segments"]),
        low_memory=False,
    )

    df_base["_State_Abbr"] = (
        df_base["FILE_NAME"]
        .apply(extract_state_abbr_from_filename)
        .astype(str)
        .str.upper()
    )

    df_base_sub = df_base[df_base["_State_Abbr"] == state_abbr].copy()
    df_base_sub.drop(columns=["_State_Abbr"], inplace=True, errors="ignore")

    if df_base_sub.empty:
        raise ValueError(f"No segments found for state {state_abbr}.")

    df_base_sub.to_csv(
        os.path.join(out_dir, REQUIRED_INPUT_FILES["segments"]),
        index=False,
    )

    valid_segments = set(
        pd.to_numeric(df_base_sub["SEGMENT_ID_OG"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    valid_scs = set(
        pd.to_numeric(df_base_sub["NEAR_SCS_ID"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    valid_scs = {x for x in valid_scs if x != -1}

    print(f"    Segments: {len(valid_segments):,}")
    print(f"    Referenced SCS IDs: {len(valid_scs):,}")

    # 2. Adjacency file.
    df_adj = pd.read_csv(
        _path(master_input_dir, REQUIRED_INPUT_FILES["adjacency"]),
        low_memory=False,
    )

    for c in ["TARGET_FID", "JOIN_FID"]:
        df_adj[c] = pd.to_numeric(df_adj[c], errors="coerce")

    df_adj_sub = df_adj[
        df_adj["TARGET_FID"].isin(valid_segments)
        & df_adj["JOIN_FID"].isin(valid_segments)
    ].copy()

    df_adj_sub.to_csv(
        os.path.join(out_dir, REQUIRED_INPUT_FILES["adjacency"]),
        index=False,
    )

    # 3. Substation distance file.
    df_subdist = pd.read_csv(
        _path(master_input_dir, REQUIRED_INPUT_FILES["subdist"]),
        low_memory=False,
    )

    df_subdist["IN_FID"] = pd.to_numeric(df_subdist["IN_FID"], errors="coerce")
    df_subdist_sub = df_subdist[df_subdist["IN_FID"].isin(valid_segments)].copy()

    df_subdist_sub.to_csv(
        os.path.join(out_dir, REQUIRED_INPUT_FILES["subdist"]),
        index=False,
    )

    # 4. Existing SCS file.
    df_scs = pd.read_csv(
        _path(master_input_dir, REQUIRED_INPUT_FILES["scs_existing"]),
        low_memory=False,
    )

    df_scs["SCS_ID"] = pd.to_numeric(df_scs["SCS_ID"], errors="coerce")
    df_scs_sub = df_scs[df_scs["SCS_ID"].isin(valid_scs)].copy()

    df_scs_sub.to_csv(
        os.path.join(out_dir, REQUIRED_INPUT_FILES["scs_existing"]),
        index=False,
    )

    # 4b. Optional US_SCS.csv if you ever run with USE_EXISTING_SCS=False.
    optional_scs_path = _path(master_input_dir, OPTIONAL_INPUT_FILES["scs_all"])
    if os.path.exists(optional_scs_path):
        df_scs_all = pd.read_csv(optional_scs_path, low_memory=False)
        if "SCS_ID" in df_scs_all.columns:
            df_scs_all["SCS_ID"] = pd.to_numeric(df_scs_all["SCS_ID"], errors="coerce")
            df_scs_all_sub = df_scs_all[df_scs_all["SCS_ID"].isin(valid_scs)].copy()
        else:
            df_scs_all_sub = df_scs_all.copy()

        df_scs_all_sub.to_csv(
            os.path.join(out_dir, OPTIONAL_INPUT_FILES["scs_all"]),
            index=False,
        )

    # 5. Copy SubstationsProject unchanged, same as Sorting.py.
    shutil.copy(
        _path(master_input_dir, REQUIRED_INPUT_FILES["substations"]),
        os.path.join(out_dir, REQUIRED_INPUT_FILES["substations"]),
    )

    # Save a small manifest for traceability.
    manifest = {
        "state_abbr": state_abbr,
        "state_name": state_full,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "master_input_dir": os.path.abspath(master_input_dir),
        "state_input_dir": os.path.abspath(out_dir),
        "n_segments": int(len(valid_segments)),
        "n_referenced_scs": int(len(valid_scs)),
        "n_adjacency_rows": int(len(df_adj_sub)),
        "n_substation_distance_rows": int(len(df_subdist_sub)),
        "n_scs_rows": int(len(df_scs_sub)),
    }

    with open(os.path.join(out_dir, "state_input_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"    Adjacency rows: {len(df_adj_sub):,}")
    print(f"    Substation-distance rows: {len(df_subdist_sub):,}")
    print(f"    SCS rows: {len(df_scs_sub):,}")

    return out_dir, manifest


def load_existing_state_input_dataset(state_abbr, state_input_root):
    """
    Use an already-created optimizer-ready state input folder.
    Does not rewrite or rebuild any input CSVs.
    """
    state_abbr = state_abbr.upper()
    state_full = STATE_NAMES.get(state_abbr, state_abbr)
    state_folder_name = f"Inputs_Statewise_{state_abbr}_{_safe_name(state_full)}"
    state_input_dir = os.path.join(state_input_root, state_folder_name)

    if not os.path.isdir(state_input_dir):
        raise FileNotFoundError(
            f"Existing state input folder not found for {state_abbr}:\n"
            f"  {state_input_dir}"
        )

    missing = []
    for _, fname in REQUIRED_INPUT_FILES.items():
        p = os.path.join(state_input_dir, fname)
        if not os.path.exists(p):
            missing.append(p)

    if missing:
        raise FileNotFoundError(
            f"Existing state input folder is missing required files for {state_abbr}:\n"
            + "\n".join(f"  {m}" for m in missing)
        )

    manifest_path = os.path.join(state_input_dir, "state_input_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
    else:
        manifest = {
            "state_abbr": state_abbr,
            "state_name": state_full,
            "state_input_dir": os.path.abspath(state_input_dir),
            "source": "existing_folder_no_manifest",
        }

    manifest["state_abbr"] = state_abbr
    manifest["state_name"] = state_full
    manifest["state_input_dir"] = os.path.abspath(state_input_dir)

    print(f"\n--- Using existing state input dataset: {state_abbr} ({state_full}) ---")
    print(f"    Input folder: {state_input_dir}")

    return state_input_dir, manifest


def make_state_run_tag(state_abbr, params, n_segments):
    state_abbr = state_abbr.upper()
    if BUDGET_MODE == "unlimited_by_segments":
        budget_label = f"punlim{int(params['p_param'])}"
    else:
        budget_label = f"p{int(params['p_param'])}"

    return (
        f"state_{state_abbr}_"
        f"{budget_label}_"
        f"T{int(params['project_life'])}_"
        f"etaDWC{params['ETA_DWC']:.2f}_"
        f"etaSCS{params['ETA_SCS']:.2f}_"
        f"gev{int(round(params['g_ev'] * 100)):02d}pct_"
        f"gaadt{int(round(params['g_aadt'] * 1000)):03d}bps"
    )


def initialize_gurobi_env():
    if gp is None:
        print("WARNING: gurobipy import failed. Passing env=None.")
        return None

    try:
        env = gp.Env()
        print("Gurobi environment initialised.")
        return env
    except Exception as e:
        print(f"WARNING: Gurobi environment initialisation failed: {e}")
        print("Proceeding with env=None.")
        return None


def write_run_failure(output_dir, state_abbr, error_text):
    os.makedirs(output_dir, exist_ok=True)
    fail_path = os.path.join(output_dir, f"FAILED_{state_abbr}.txt")
    with open(fail_path, "w") as f:
        f.write(error_text)
    print(f"    Failure log saved: {fail_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MAIN STATEWISE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(STATE_INPUT_ROOT, exist_ok=True)
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    if REBUILD_STATE_INPUTS:
        validate_master_inputs(MASTER_INPUT_DIR)
        states = discover_states(MASTER_INPUT_DIR)
    else:
        # validate_master_inputs(MASTER_INPUT_DIR)
        # states = discover_states(MASTER_INPUT_DIR)
        states = discover_states_from_existing_inputs(STATE_INPUT_ROOT)

    print("=" * 80)
    print("  Statewise DWC + SCS Optimization Runner")
    print(f"  Master input dir : {MASTER_INPUT_DIR}")
    print(f"  State input root : {STATE_INPUT_ROOT}")
    print(f"  Output root      : {BASE_OUTPUT_DIR}")
    print(f"  States found     : {len(states)}")
    print(f"  Budget mode      : {BUDGET_MODE}")
    print(f"  Plot each run    : {PLOT_EACH_RUN}")
    print("=" * 80)
    print("States:")
    print("  " + ", ".join(states))

    env = initialize_gurobi_env()

    state_input_rows = []
    run_rows = []
    results_list = []

    # Optional plotting import.
    if PLOT_EACH_RUN:
        from Main_plots import generate_all_plots

    for idx, state_abbr in enumerate(states, start=1):
        state_abbr = state_abbr.upper()
        state_full = STATE_NAMES.get(state_abbr, state_abbr)

        print("\n" + "╔" + "═" * 78 + "╗")
        print(f"  State {idx}/{len(states)}: {state_abbr} ({state_full})")
        print("╚" + "═" * 78 + "╝")

        try:
            # 1. Use the already-created state input folder.
            # state_input_dir, input_manifest = create_state_input_dataset(
            #     master_input_dir=MASTER_INPUT_DIR,
            #     state_abbr=state_abbr,
            #     state_input_root=STATE_INPUT_ROOT,
            # )
            state_input_dir, input_manifest = load_existing_state_input_dataset(
                state_abbr=state_abbr,
                state_input_root=STATE_INPUT_ROOT,
            )
            state_input_rows.append(input_manifest)

            # 2. Build base params.
            params = _base_params()

            # 3. Load state data.
            data = load_data(
                input_dir=state_input_dir,
                use_existing_scs=USE_EXISTING_SCS,
                exclude_top_pct=EXCLUDE_TOP_PCT,
                max_sub_rank=MAX_SUB_RANK,
                lambda_min=params.get("LAMBDA_MIN", 0.80),
                lambda_max=params.get("LAMBDA_MAX", 1.20),
            )

            n_state_segments = len(data["I"])

            if n_state_segments == 0:
                raise ValueError(f"No optimizer-valid segments left for {state_abbr} after filtering.")

            # 4. Statewise budget handling.
            if BUDGET_MODE == "unlimited_by_segments":
                p_unlim = int(n_state_segments)
                if MAX_STATE_P_PARAM is not None:
                    p_unlim = min(p_unlim, int(MAX_STATE_P_PARAM))
                params["p_param"] = p_unlim
            elif BUDGET_MODE == "fixed":
                pass
            else:
                raise ValueError(
                    f"Unknown BUDGET_MODE={BUDGET_MODE}. "
                    "Use 'unlimited_by_segments' or 'fixed'."
                )

            params["mip_gap"] = MIP_GAP
            params["run_tag"] = make_state_run_tag(state_abbr, params, n_state_segments)

            out_dir = os.path.join(BASE_OUTPUT_DIR, f"Results_{params['run_tag']}")

            if SKIP_COMPLETED_STATES and os.path.exists(os.path.join(out_dir, "run_metadata.csv")):
                print(f"    Skipping completed state: {state_abbr}")
                run_rows.append({
                    "state_abbr": state_abbr,
                    "state_name": state_full,
                    "status": "skipped_completed",
                    "run_tag": params["run_tag"],
                    "output_dir": out_dir,
                    "n_segments": n_state_segments,
                    "p_param": params["p_param"],
                })
                continue

            print("\nState run parameters:")
            print(f"    run_tag      : {params['run_tag']}")
            print(f"    n_segments   : {n_state_segments:,}")
            print(f"    p_param      : {params['p_param']:,}")
            print(f"    project_life : {params['project_life']}")
            print(f"    g_ev         : {params['g_ev']}")
            print(f"    g_aadt       : {params['g_aadt']}")
            print(f"    mip_gap      : {params['mip_gap']}")

            # 5. Solve.
            result = run_single_optimization(data, params, env=env)

            if result is None:
                print(f"    WARNING: infeasible or no result for {state_abbr}.")
                run_rows.append({
                    "state_abbr": state_abbr,
                    "state_name": state_full,
                    "status": "no_result_or_infeasible",
                    "run_tag": params["run_tag"],
                    "output_dir": out_dir,
                    "n_segments": n_state_segments,
                    "p_param": params["p_param"],
                })
                continue

            # 6. Save exactly like Main.py.
            save_results(result, out_dir)

            # 7. Optional plots, same structure as Main.py.
            if PLOT_EACH_RUN:
                plot_dir = os.path.join(BASE_OUTPUT_DIR, f"Plots_{params['run_tag']}")
                generate_all_plots(result, plot_dir)

            # 8. Summary row.
            summary = {}
            if _HAS_SUMMARY:
                try:
                    summary = _compute_run_summary(result)
                except Exception as e:
                    print(f"    WARNING: _compute_run_summary failed for {state_abbr}: {e}")
                    summary = {}

            meta = result.get("meta", {})

            row = {
                "state_abbr": state_abbr,
                "state_name": state_full,
                "status": "solved",
                "run_tag": params["run_tag"],
                "output_dir": out_dir,
                "input_dir": state_input_dir,
                "n_segments": n_state_segments,
                "p_param": params["p_param"],
                "project_life": params["project_life"],
                "g_ev": params["g_ev"],
                "g_aadt": params["g_aadt"],
                "ETA_DWC": params["ETA_DWC"],
                "ETA_SCS": params["ETA_SCS"],
                "rho_ev": params["rho_ev"],
                "mip_gap": params["mip_gap"],
                "n_dwc_selected": meta.get("n_dwc_selected", None),
                "n_scs_selected": meta.get("n_scs_selected", None),
                "obj_val": meta.get("obj_val", None),
                "total_rev_dwc": meta.get("total_rev_dwc", None),
                "total_rev_scs": meta.get("total_rev_scs", None),
                "total_opex_dwc": meta.get("total_opex_dwc", None),
                "total_opex_scs": meta.get("total_opex_scs", None),
            }

            for k, v in summary.items():
                row[f"summary_{k}"] = v

            run_rows.append(row)
            results_list.append(result)

            print(f"\n    Completed state {state_abbr}.")
            print(f"    Results saved to: {out_dir}")

        except Exception:
            err = traceback.format_exc()
            print(f"\n    ERROR while processing state {state_abbr}:")
            print(err)

            write_run_failure(BASE_OUTPUT_DIR, state_abbr, err)

            run_rows.append({
                "state_abbr": state_abbr,
                "state_name": state_full,
                "status": "failed",
                "run_tag": None,
                "output_dir": None,
                "n_segments": None,
                "p_param": None,
                "error": err,
            })

            # Continue with next state.
            continue

        # Write incremental summary after each state so overnight runs are safe.
        pd.DataFrame(run_rows).to_csv(
            os.path.join(BASE_OUTPUT_DIR, "statewise_run_summary_partial.csv"),
            index=False,
        )

        pd.DataFrame(state_input_rows).to_csv(
            os.path.join(BASE_OUTPUT_DIR, "statewise_input_manifest_partial.csv"),
            index=False,
        )

    # Final summaries.
    df_runs = pd.DataFrame(run_rows)
    df_inputs = pd.DataFrame(state_input_rows)

    final_summary_path = os.path.join(BASE_OUTPUT_DIR, "statewise_run_summary.csv")
    final_manifest_path = os.path.join(BASE_OUTPUT_DIR, "statewise_input_manifest.csv")

    df_runs.to_csv(final_summary_path, index=False)
    df_inputs.to_csv(final_manifest_path, index=False)

    # Save global manifest.
    global_manifest = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "master_input_dir": os.path.abspath(MASTER_INPUT_DIR),
        "state_input_root": os.path.abspath(STATE_INPUT_ROOT),
        "base_output_dir": os.path.abspath(BASE_OUTPUT_DIR),
        "states_requested": states,
        "n_states_requested": len(states),
        "budget_mode": BUDGET_MODE,
        "max_state_p_param": MAX_STATE_P_PARAM,
        "use_existing_scs": USE_EXISTING_SCS,
        "exclude_top_pct": EXCLUDE_TOP_PCT,
        "max_sub_rank": MAX_SUB_RANK,
        "plot_each_run": PLOT_EACH_RUN,
        "baseline_overrides": BASELINE_OVERRIDES,
        "mip_gap": MIP_GAP,
        "n_solved": int((df_runs["status"] == "solved").sum()) if not df_runs.empty else 0,
        "n_failed": int((df_runs["status"] == "failed").sum()) if not df_runs.empty else 0,
    }

    with open(os.path.join(BASE_OUTPUT_DIR, "statewise_global_manifest.json"), "w") as f:
        json.dump(global_manifest, f, indent=2, default=str)

    print("\n" + "═" * 90)
    print("  STATEWISE RUNS COMPLETE")
    print("═" * 90)
    print(f"  Final run summary     : {final_summary_path}")
    print(f"  Final input manifest  : {final_manifest_path}")
    print(f"  Output root           : {BASE_OUTPUT_DIR}")
    if not df_runs.empty:
        print("\nStatus counts:")
        print(df_runs["status"].value_counts(dropna=False).to_string())
    print("═" * 90)


if __name__ == "__main__":
    main()
