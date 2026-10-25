# Reproduction pipeline

This guide describes the complete sequence used in the paper. It separates the authoritative GIS transformations from the optimization and postprocessing stages so that each artifact has a traceable origin.

## 1. Acquire and project source data

Download the sources listed in [PUBLIC_DATA_SOURCES.md](PUBLIC_DATA_SOURCES.md). Import them into an ArcGIS Pro project and project analysis layers to **USA Contiguous Albers Equal Area Conic** (`ESRI:102003`). Record the download date, source URL or item ID, source version, and any source-specific license or attribution.

## 2. Construct the highway analysis units

Use the National Highway System Interstate layer as the base. Transfer commercial electricity rates and the construction-cost proxy to the original Interstate features before dissolving. Dissolve by signed route identifier (`SIGN1`) with unsplit lines enabled, preserving spatially disconnected route fragments as separate features. Repair geometry and order the resulting route fragments geographically.

Generate geodesic points every five miles, including route endpoints, and split the routes at those points using a 10-foot search radius. Calculate geodesic length in U.S. miles and remove segments shorter than four miles. In the study, this reduced 10,321 initial pieces to 9,053 analysis segments.

## 3. Enrich segments and build relationships

Transfer HPMS traffic variables by a one-to-one spatial join using `INTERSECT` and the maximum overlapping value for total AADT, commercial AADT, single-unit AADT, and speed limit. Fill missing traffic fields with their dataset medians. Retain all substations within five miles of each segment rather than only the closest one.

Construct candidate SCS sites from OpenStreetMap `motorway_link` and `trunk_link` ramps that intersect the Interstate network. Convert ramp geometries to points, cluster them with DBSCAN using a 400 m neighborhood and a minimum of two points, and retain one candidate per cluster. Remove candidates within two miles of an existing public DC fast-charging station; when several candidates are near the same existing station, associate only the nearest candidate for that filtering step.

Construct DWC adjacency from valid shared roadway endpoints and roadway continuation—not from simple Euclidean proximity. Store both directed arcs for each valid undirected adjacency because the optimizer's power-flow variables are directed.

## 4. Export the consolidated model inputs

Export the six CSVs described in [INPUT_DATA_DICTIONARY.md](INPUT_DATA_DICTIONARY.md). Preserve the stable identifiers that allow model results to be joined back to GIS:

- highway segment: `SEGMENT_ID_OG` in GIS and `Segment_ID` in results;
- candidate SCS: `SCS_ID`;
- substation: `Sub_ID_OG` / `Substation_ID`.

Run the schema validator:

```bash
python tools/validate_inputs.py data/processed
```

## 5. Run the optimization

Configure `code/Main.py` and run it from an environment with a valid Gurobi license. `Main.py` loads and cleans the CSV inputs, constructs indexed sets and parameters, selects the requested experiment, and delegates the mixed-integer model to `Main_optimizer.py`.

The principal indexed sets are:

- `I`: highway segments;
- `R`: candidate SCS sites;
- `A`: substations;
- `K_DWC`: valid segment–substation connections;
- `K_SCS`: valid SCS–substation connections; and
- directed DWC adjacency arcs.

For every run, archive the effective configuration, solver version, input checksums, solution status, gap, and result directory name. Do not rely only on a hand-constructed run tag.

## 6. Reconstruct spatial outputs

Configure and run `code/Main_Shapefiles_updated.py`. It joins optimizer segment IDs to the original five-mile highway feature class, exports selected SCS and substations, and builds straight-line visualization geometries for substation taps and segment-to-segment DWC power links.

The straight tap and adjacency lines communicate model relationships; they are not engineered utility routes. See [ARCGIS_RESULTS_WORKFLOW.md](ARCGIS_RESULTS_WORKFLOW.md).

## 7. Produce maps and figures

Add the complete exported shapefile groups to ArcGIS Pro. Symbolize DWC by `dwc_conn`, substations by `tech_role`, and use distinct line styles for direct taps and adjacency links. Preserve map projection, source attribution, and the run tag in the figure metadata or caption.

## Reproducibility boundary

The included CSVs reproduce the Python model inputs. Rebuilding those CSVs from raw public data still requires the documented ArcGIS operations and access to the relevant source versions. A future automated data builder could replace individual GIS steps only after its outputs are tested against these reference tables and spatial relationships.

