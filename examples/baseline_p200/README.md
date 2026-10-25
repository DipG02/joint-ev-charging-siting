# Case study: constrained national baseline (`p = 200`)

This example is a curated copy of the completed run:

```text
baseline_p200_etaDWC0.50_etaSCS0.50_gev15pct_gaadt011bps
```

It uses all 9,053 processed Interstate segments, enables existing-station screening, and permits at most 200 combined DWC and new SCS deployments.

## Configuration

| Setting | Value |
|---|---:|
| Deployment limit | 200 |
| Base-year EV share | 0.5% |
| DWC traffic share | 50% |
| SCS traffic share | 50% |
| Project life | 30 years |
| Discount rate | 7% |
| EV-adoption growth | 15%/year |
| Traffic growth | 1.1%/year |
| Charging price | $0.54/kWh |
| DWC base cost | $10 million/segment |
| SCS equipment | 6 ports at $250,000/port |

The complete machine-readable configuration is in [results/run_metadata.csv](results/run_metadata.csv).

## Principal results

| Result | Value |
|---|---:|
| Lifetime portfolio NPV | $2.858 billion |
| Selected DWC segments | 96 |
| Selected new SCS sites | 104 |
| DWC segments with direct substation taps | 64 |
| DWC segments powered through adjacent DWC | 32 |
| Unique DWC-serving substations | 64 |
| Unique SCS-serving substations | 103 |
| DWC direct feeder length | 32.75 miles |
| SCS direct feeder length | 61.11 miles |
| Base-year DWC delivered energy | 39,551.82 MWh |
| Base-year SCS delivered energy | 17,282.55 MWh |

The direct feeder totals are sums of `Dist_Miles` for direct substation taps. DWC adjacency links have zero direct feeder length because they represent corridor power sharing rather than an additional substation connection.

## Included outputs

`results/` contains selected DWC and SCS assets, grid connections, selected substations, screened existing SCS sites, exact metadata, and the text model report.

`plots/` contains:

- `map_proposed.png`: national deployment map;
- `total_units.png`: DWC/SCS deployment composition;
- `traffic_split.png`: technology traffic-allocation assumptions.

The original run also produced large diagnostics and coordinate tables. Those can be regenerated but are not duplicated here.

## Verify the archived example

From the repository root:

```bash
python examples/baseline_p200/verify_results.py
```

The verifier checks row counts, metadata, connection classifications, feeder-mile totals, and the presence of every included plot.

File hashes are recorded in [SHA256SUMS.txt](SHA256SUMS.txt) and can be checked with:

```bash
(cd examples/baseline_p200 && sha256sum -c SHA256SUMS.txt)
```

## Reproduce the optimization

1. Install the environment and configure Gurobi as described in [../../SETUP_AND_CONFIGURATION.md](../../SETUP_AND_CONFIGURATION.md).
2. From the repository root, set the input and output paths:

   ```bash
   export DWC_INPUT_DIR="$PWD/data/processed"
   export DWC_OUTPUT_DIR="$PWD/results"
   ```

3. Confirm these settings near the top of `code/Main.py`:

   ```python
   SENSITIVITY_MODE = "none"
   PLOT_EACH_RUN = True

   BASELINE_OVERRIDES = {
       # ...
       "rho_ev": 0.005,
       "ETA_DWC": 0.5,
       "ETA_SCS": 0.5,
       "p_param": 200,
       "P_CHARGE_DWC": 0.54,
       "P_CHARGE_SCS": 0.54,
       "discount_rate": 0.07,
       "project_life": 30,
       "g_aadt": 0.011,
       "g_ev": 0.15,
       # ...
   }
   ```

4. Run:

   ```bash
   python code/Main.py
   ```

This full national MILP requires an academic or otherwise unrestricted Gurobi license. Gurobi's size-limited license can run small diagnostic tests but cannot solve this national case.

The current default plot wrapper generates `map_proposed.png` and `total_units.png`. The archived traffic-share panel was generated from the same result object using `plot_traffic_split` in `code/Main_plots.py`; that optional call is present but commented in `generate_all_plots()`.

## Compare a reproduction

Compare the new selected tables and metadata with `results/`. Exact reproduction requires the same input checksums, Gurobi version, model code, solver tolerances, and configuration. Multiple mathematically equivalent solutions can occasionally differ in asset identity even when their objective values agree, so compare the objective, technology counts, and connection totals as well as row-by-row selections.

## Provenance

These files were copied from the archived constrained national baseline used by the project. They are reference outputs—not newly generated with the restricted smoke-test license—and retain their original run tag and metadata.

