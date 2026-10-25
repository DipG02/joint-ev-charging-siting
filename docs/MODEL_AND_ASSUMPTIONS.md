# Model and assumptions

This document is a practical orientation to the implemented model. The paper and Supplementary Information are the authoritative mathematical description.

## Decision structure

For each highway segment, the model can select DWC, select its associated SCS candidate, or leave it unserved. Selected DWC segments may connect directly to a qualifying substation or receive power through a chain of adjacent selected DWC segments. Selected SCS sites require a direct substation connection.

The mixed-integer formulation includes:

- binary DWC and SCS deployment decisions;
- binary direct grid-connection assignments;
- directed adjacency-link decisions for DWC;
- virtual-flow constraints that eliminate disconnected DWC islands;
- physical power flows over DWC adjacency links;
- substation capacities; and
- a terminal-year capacity constraint under demand growth.

## Objective

The objective maximizes lifetime NPV:

```text
present value of charging revenue
− initial infrastructure and grid-connection cost
− present value of electricity purchases
− present value of fixed operation and maintenance
```

The current code treats the deployment limit as an upper bound (`<= p`) even where an older inline comment may describe exactly `p` deployments. Report the implemented condition when documenting new runs.

## Baseline study parameters

| Parameter | Baseline value |
|---|---:|
| Analysis horizon | 30 years |
| Discount rate | 7% |
| Initial EV share | 0.5% |
| DWC adoption share | 50% |
| SCS adoption share | 50% |
| Annual EV-share growth | 15% |
| Annual traffic growth | 1.1% |
| Charging price | $0.54/kWh |
| Charging-price growth | 2%/year |
| Electricity-rate growth | 1.5%/year |
| Fixed O&M growth | 2%/year |
| DWC roadway cost | $10 million per nominal 5-mile segment |
| DWC internal electrical line | $70,000 |
| DWC adjacency link | $400,000 |
| Direct distribution line | $1.39 million/mile |
| Grid interconnection | $0.60 million/MW |
| DWC power density | 2 MW/mile |
| Vehicle receiver power | 300 kW |
| SCS chargers | 6 ports/site |
| SCS charger rating | 250 kW/port |
| SCS equipment cost | $250,000/port |
| Energy per SCS charging event | 150 kWh |
| SCS utilization factor | 1.0 |
| Annual operating hours | 8,760 |
| Fixed O&M | 1.5% of installed infrastructure plus grid-connection cost/year |

Currency values are nominal study assumptions. They are not bids, forecasts, or site-engineering estimates.

## Baseline experiment

The paper's constrained baseline uses `p = 200` and selects 96 DWC highway segments and 104 SCS sites. DWC uses 64 direct substation taps and 32 adjacency-powered segments. The paper describes the full Interstate freight network as 48,890 miles; after its stated short-segment filtering, the included 9,053-row candidate table sums to 45,171.01 miles in `Actual_Length`.

## Global sensitivity analysis

The paper reports 1,000 feasible model runs across nine uncertain inputs and 198,736 total deployed observations: 79,006 DWC and 119,730 SCS. The current implementation jointly samples:

- DWC traffic share: 0.10–0.90;
- SCS traffic share: 0.10–0.90, subject to the two shares summing to at most 1;
- annual traffic growth: 0.8%–2.0%;
- annual EV-adoption growth: 10%–18%;
- project lifetime: 25–40 years, in integer years;
- DWC capital cost: $8–12 million per segment;
- SCS hardware cost: $180,000–300,000 per port;
- SCS port count: 4–8, in integer ports; and
- annual commercial electricity-rate growth: 0.5%–2.5%.

The deployment limit is held fixed at 200 and the GSA uses a 1% optimality gap.

Consult `code/Main_sensitivity.py` for the exact distributions and current bounds before running or reporting a new analysis. The pooled segment-level output can be several gigabytes and is intentionally not included in this repository.

## Interpretation limits

- Candidate locations are planning units, not construction-ready sites.
- Straight-line substation distances approximate direct feeder requirements, not routed distribution designs.
- DWC adjacency represents modeled corridor power sharing, not an electrical protection design.
- Median household income is a bounded construction-cost proxy and omits terrain, permitting, labor, and right-of-way detail.
- Demand, adoption, prices, costs, and growth assumptions are scenario inputs, not predictions.
- Existing grid capacity and upgrade needs are represented at the resolution supported by the input data and model assumptions.

