# Requirements and configuration

This is the pre-run checklist for a new machine. Python packages are listed in `requirements.txt` and `environment.yml`; filesystem and license settings cannot be expressed in a Python requirements file, so they are documented here.

## 1. Software requirements

| Requirement | Purpose | Study/reference version |
|---|---|---|
| Python | Optimization and postprocessing | Python 3.10 or newer |
| Gurobi Optimizer | Mixed-integer optimization | 13.0.1 |
| Valid Gurobi license | Permission and model-size capacity | Academic/unrestricted for national runs |
| ArcGIS Pro | Rebuild source GIS inputs | 3.6.0 |
| GeoPandas spatial stack | Reconstruct result shapefiles | Installed through the supplied environment |

The size-limited Gurobi license can test installation and small diagnostic models, but the complete 9,053-segment national MILP requires an unrestricted-capacity license.

## 2. Install the Python environment

### Micromamba or Conda

```bash
micromamba env create -f environment.yml
micromamba activate dwc-planning
```

If an existing environment is being used:

```bash
micromamba activate YOUR_ENVIRONMENT
python -m pip install -r requirements.txt
```

### Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Configure Gurobi

Confirm the Python binding and license before launching the national model:

```bash
python - <<'PY'
import gurobipy as gp
print("Gurobi:", gp.gurobi.version())
m = gp.Model("license_check")
x = m.addVar(lb=0)
m.setObjective(x, gp.GRB.MAXIMIZE)
m.addConstr(x <= 1)
m.optimize()
print("Status:", m.Status, "objective:", m.ObjVal)
PY
```

If more than one license file exists, point Gurobi to the intended one:

```bash
export GRB_LICENSE_FILE=/absolute/path/to/gurobi.lic
```

Do not commit a license file. A `HostID mismatch` means the license is bound to a different machine identity and must be renewed, rebound, or replaced through the appropriate Gurobi licensing process.

## 4. Required optimizer paths

`code/Main.py` has repository-relative defaults, so no edit is needed for the included inputs. These environment variables override them:

```bash
export DWC_INPUT_DIR="$PWD/data/processed"
export DWC_OUTPUT_DIR="$PWD/results"
```

| Setting | Default | When to change it |
|---|---|---|
| `DWC_INPUT_DIR` | `data/processed` | Inputs are stored elsewhere |
| `DWC_OUTPUT_DIR` | `results` | Results should be written elsewhere |
| `SENSITIVITY_MODE` in `Main.py` | `none` | Running a sweep or GSA |
| `PLOT_EACH_RUN` in `Main.py` | `True` | Disable during large batches |
| `GSA_N_SAMPLES` in `Main.py` | `1000` | Running a deliberate GSA sample size |

The default `SENSITIVITY_MODE = "none"` runs the `p = 200` baseline. Review `BASELINE_OVERRIDES` before every scientific run.

## 5. Required GIS-export paths

To convert a completed result back to ArcGIS layers:

```bash
export DWC_RESULTS_DIR=/absolute/path/to/Results_run_tag
export DWC_HIGHWAY_GDB=/absolute/path/to/project.gdb
python code/Main_Shapefiles_updated.py
```

Also confirm `ORIGINAL_HIGHWAY_LAYER` in `code/Main_Shapefiles_updated.py`. Its documented default is:

```text
NHS_Interstate_5_miles_FINAL_filteredSegments
```

The original highway geodatabase is not included in this repository. It is produced during the ArcGIS workflow described in `docs/GIS_WORKFLOW.md`.

## 6. Other script-specific paths

| Script | Configuration |
|---|---|
| `Main_statewise.py` | Uses `DWC_INPUT_DIR`; state inputs/results default to repository-relative directories |
| `Plots_Statewise.py` | Set `RESULTS_ROOT`, `OUT_DIR`, and optionally `STATE_SHAPEFILE` |
| `Main_Eta_plots.py` | Set `DWC_UNCONSTRAINED_RESULTS_DIR` or edit the three result-folder names |
| `Policy_Insight_Analysis.py` | Prefer CLI arguments `--baseline_dir`, `--gsa_dir`, `--input_segments`, and `--output_dir` |
| `Sensitivity_Plots.py` | Pass the completed GSA directory through its documented interface |
| `Sorting.py` | Run from an input directory or update it before use; it currently reads CSVs from the working directory |

These analysis utilities are not needed for the primary baseline reproduction.

## 7. Pre-run checks

From the repository root:

```bash
python tools/validate_inputs.py data/processed
python examples/baseline_p200/verify_results.py
python -m py_compile code/*.py tools/*.py examples/baseline_p200/verify_results.py
sha256sum -c SHA256SUMS.txt
```

Then run the baseline:

```bash
python code/Main.py
```

## 8. Files that must remain private or local

Do not commit:

- Gurobi license files or credentials;
- personal Box or home-directory paths;
- proprietary ArcGIS project/geodatabase files unless their release is approved;
- large GSA pooled tables and generated run directories; or
- data whose upstream redistribution terms have not been cleared.

See `RELEASE_CHECKLIST.md` for the tagging checklist and `DATA_LICENSE.md` for source attribution.

