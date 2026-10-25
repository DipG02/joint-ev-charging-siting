# Joint Dynamic Wireless and Stationary EV Charging Station Siting Toolkit

Research software and curated geospatial inputs for jointly planning dynamic wireless charging (DWC), stationary charging stations (SCS), and their distribution-grid connections on the contiguous U.S. Interstate Highway System.

This repository accompanies:

> Dipanjan Ghose, Junjie Qin, and S. Sivaranjani, “Rethinking electrified transportation: The case for dynamic wireless charging,” arXiv:2610.18708v1, 2026. [Paper](https://arxiv.org/abs/2610.18708)

The model asks where DWC roadway segments and SCS sites should be deployed, which substations should supply them, and—uniquely for DWC—whether adjacent electrified roadway segments can share a direct grid connection. It maximizes lifetime net present value (NPV) subject to deployment, grid-access, adjacency, and terminal-year capacity constraints.

The repository contains the code, processed inputs, GIS workflow, and a verified baseline case study needed to reproduce and extend the paper's planning analysis.

## End-to-end workflow

```mermaid
flowchart LR
    A[Public geospatial sources] --> B[ArcGIS Pro preprocessing]
    B --> C[Consolidated CSV inputs]
    C --> D[Python + Gurobi optimization]
    D --> E[Result CSVs]
    E --> F[GeoPandas shapefile reconstruction]
    F --> G[ArcGIS Pro maps and figures]
```

The GIS stage is an essential part of the method. The current project uses ArcGIS Pro to segment and enrich the highway network, build candidate sites and grid-access relationships, and export the CSVs consumed by Python. The repository therefore provides a reproducible GIS tutorial and the resulting consolidated inputs; it does not claim that the source-to-input pipeline is currently a one-command Python data builder.

Detailed guides:

- [Installation, licenses, and path configuration](SETUP_AND_CONFIGURATION.md)
- [Full pipeline](docs/PIPELINE.md)
- [ArcGIS data-construction workflow](docs/GIS_WORKFLOW.md)
- [Public data sources](docs/PUBLIC_DATA_SOURCES.md)
- [Input data dictionary](docs/INPUT_DATA_DICTIONARY.md)
- [Model and assumptions](docs/MODEL_AND_ASSUMPTIONS.md)
- [Returning optimizer results to ArcGIS](docs/ARCGIS_RESULTS_WORKFLOW.md)

## Repository layout

```text
joint-ev-charging-siting/
├── code/                       Optimization, analysis, and GIS-export scripts
├── data/processed/             Consolidated CSV inputs used by the Python model
├── docs/                       Reproduction and methodology guides
├── examples/baseline_p200/     Curated national case study, results, and plots
├── tools/validate_inputs.py    Schema and relationship checks
├── requirements.txt
├── environment.yml
├── CITATION.cff
└── RELEASE_CHECKLIST.md
```

## Case-study scale

The study is framed around the 48,890-mile contiguous U.S. Interstate freight network. After segmentation and removal of residual pieces shorter than four miles, the processed highway table contains 9,053 candidate segments whose `Actual_Length` values sum to 45,171.01 miles. The paper's baseline run limits deployment to 200 sites and selects 96 DWC segments and 104 SCS sites. Of the selected DWC segments, 64 connect directly to substations and 32 are powered through adjacent selected DWC segments.

The processed candidate-site file contains 21,430 raw SCS candidate records. With the checked-in loader, all 9,053 segments, and `USE_EXISTING_SCS = True`, the current inputs produce 5,073 optimizable SCS IDs plus 1,412 mapped candidate IDs excluded because an existing station is nearby. These are post-mapping model sets and are distinct from the paper's reported 17,400 GIS candidate sites. The raw, GIS-candidate, segment-referenced, existing-site-excluded, and final model-set counts should all be reported separately rather than treated as interchangeable.

## Requirements

- Python 3.10 or later
- Gurobi Optimizer and a valid license (the study used Gurobi 13.0.1)
- ArcGIS Pro for recreating the public-data preprocessing workflow (the study used ArcGIS Pro 3.6.0)
- GeoPandas and a working spatial I/O stack for reconstructing shapefiles

Install the Python environment with either:

```bash
conda env create -f environment.yml
conda activate dwc-planning
```

or:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Gurobi licensing is separate from package installation. See the [Gurobi documentation](https://www.gurobi.com/documentation/) for license setup.

For all required packages, license checks, environment variables, and script-specific paths, follow [SETUP_AND_CONFIGURATION.md](SETUP_AND_CONFIGURATION.md).

## Reproduced case study

The repository includes the constrained national baseline as a compact case study: 96 DWC segments, 104 new SCS sites, selected grid connections, exact run metadata, and three plots. See [examples/baseline_p200/README.md](examples/baseline_p200/README.md), or verify the archived outputs with:

```bash
python examples/baseline_p200/verify_results.py
```

## Validate the included inputs

From the repository root:

```bash
python tools/validate_inputs.py data/processed
```

The validator checks required files and columns, uniqueness of the principal identifiers, and cross-file references. It does not reproduce the original GIS operations.

## Run the optimizer

The orchestration entry point is `code/Main.py`; the optimization model is implemented in `code/Main_optimizer.py`, and parameter sampling is implemented in `code/Main_sensitivity.py`.

The checked-in defaults use `data/processed`, write to `results`, and run one baseline case. You can edit the configuration block near the top of `code/Main.py`, or override the two directories without changing code:

```bash
export DWC_INPUT_DIR=/absolute/path/to/processed-inputs
export DWC_OUTPUT_DIR=/absolute/path/to/results
```

Before a substantial run:

1. Confirm that `INPUT_DIR` and `BASE_OUTPUT_DIR` resolve to the intended locations.
2. Confirm the baseline parameter overrides and `SENSITIVITY_MODE`.
3. For a large sensitivity run, review `GSA_N_SAMPLES`, `PLOT_EACH_RUN`, solver settings, and available compute/storage first.

Then run:

```bash
python code/Main.py
```

The paper's global sensitivity analysis used 1,000 optimization runs and produced a very large pooled segment-level table. The repository therefore defaults to `SENSITIVITY_MODE = "none"`; change it to `"gsa"` only after reviewing compute, storage, and solver requirements.

Supported modes in the current script are:

- `none`: one baseline optimization
- `dwc_cost`, `ev_growth`, `traffic_growth`, and `eta_split`: one-at-a-time sweeps
- `eta_cost_sweep`: paired DWC/SCS allocation sweep
- `custom`: user-specified parameter combinations
- `breakeven_grid_dwc`, `breakeven_grid_scs`, and `breakeven_grid_both`: controlled two-dimensional grids
- `project_life_ev_growth_grid`: project-life and EV-growth grid
- `gsa`: global scenario sensitivity analysis
- `gsa_convergence`: independent GSA datasets at several sample sizes

See [examples/README.md](examples/README.md) for a safer first-run recipe and [docs/MODEL_AND_ASSUMPTIONS.md](docs/MODEL_AND_ASSUMPTIONS.md) for the formulation represented in code.

## Reconstruct GIS layers from a run

`code/Main_Shapefiles_updated.py` converts optimizer result tables back to GIS layers. Configure:

- the result directory for the run of interest;
- the original five-mile highway feature class created during preprocessing;
- the join field (`SEGMENT_ID_OG` in the highway layer, corresponding to `Segment_ID` in results).

The package is written next to the selected results directory.

Then run:

```bash
export DWC_RESULTS_DIR=/absolute/path/to/Results_run_tag
export DWC_HIGHWAY_GDB=/absolute/path/to/project.gdb
python code/Main_Shapefiles_updated.py
```

The export distinguishes DWC segments with `Direct_Substation` grid access from `Adjacent_Powered` segments, and emits separate substation-to-charger taps and DWC adjacency links. Keep every `.shp` file together with its `.shx`, `.dbf`, `.prj`, and `.cpg` sidecars. The complete layer list and ArcGIS styling instructions are in [docs/ARCGIS_RESULTS_WORKFLOW.md](docs/ARCGIS_RESULTS_WORKFLOW.md).

## Data provenance and redistribution

The included consolidated CSVs were derived from public or research-accessible sources including USDOT's National Highway System and HPMS data, EIA/Esri electricity-rate data, Esri demographic data, an open U.S. substation dataset, OpenStreetMap, and DOE's Alternative Fuels Data Center. “Publicly accessible” does not automatically mean unrestricted redistribution. Each upstream source retains its own attribution, use, and redistribution terms.

See [docs/PUBLIC_DATA_SOURCES.md](docs/PUBLIC_DATA_SOURCES.md) and [DATA_LICENSE.md](DATA_LICENSE.md). Confirm all current source terms before publishing the processed files.

## Authors and acknowledgment

- Dipanjan Ghose — Elmore Family School of Electrical and Computer Engineering, Purdue University
- Junjie Qin — Edwardson School of Industrial Engineering, Purdue University
- S. Sivaranjani — Elmore Family School of Electrical and Computer Engineering, Purdue University

Correspondence: S. Sivaranjani (`sseetha@purdue.edu`).

This material is based in part upon work supported by the U.S. National Science Foundation through the ASPIRE Engineering Research Center. The authors acknowledge the data providers listed in the paper and in this repository's source guide.

## Citation

If you use the model, workflow, or processed inputs, cite the paper above. Machine-readable citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

The code and documentation may be used for non-commercial academic research, teaching, and reproduction of the accompanying paper under the [Academic and Non-Commercial Use License](LICENSE.md). Commercial use requires prior written permission. Included processed data retain the attribution and source-specific conditions described in [DATA_LICENSE.md](DATA_LICENSE.md).
