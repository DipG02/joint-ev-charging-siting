# ArcGIS Pro data-construction workflow

This tutorial mirrors Supplementary Note 1 of the accompanying paper. Tool names refer to ArcGIS Pro; the study used version 3.6.0. Exact menus can change between releases.

## Project setup

1. Create a file geodatabase and add the source datasets listed in [PUBLIC_DATA_SOURCES.md](PUBLIC_DATA_SOURCES.md).
2. Project working layers to **USA Contiguous Albers Equal Area Conic** (`ESRI:102003`).
3. Add fields that preserve original feature identifiers before any dissolve, split, or clustering step.
4. Save the coordinate system, source date, and geoprocessing history with the project.

## Interstate segmentation

1. Select Interstate facilities from the National Highway System layer.
2. Spatially attach commercial electricity rate (CER) and construction-cost index (CCI) attributes to the original highway features.
3. Run **Dissolve** using `SIGN1` as the route field. Enable unsplit lines, while preserving disconnected spatial fragments as independent features.
4. Run **Repair Geometry** and establish a consistent geographic order for route fragments.
5. Run **Generate Points Along Lines** with:
   - distance: 5 U.S. miles;
   - method: geodesic;
   - include endpoints: yes.
6. Run **Split Line at Point** with a 10-foot search radius.
7. Calculate geodesic segment length in U.S. miles.
8. Remove pieces shorter than 4 miles. The study retained 9,053 of 10,321 initial pieces.
9. Create the persistent segment identifier `SEGMENT_ID_OG`.

## Traffic attributes

Spatially join Highway Performance Monitoring System records to the segmented network with:

- join operation: `JOIN_ONE_TO_ONE`;
- spatial relation: `INTERSECT`;
- merge rule: `MAX` for `AADT`, `AADT_COM`, `AADT_SINGL`, and `SPEED_LIMI`.

The maximum is used to avoid understating traffic when more than one HPMS feature overlaps an analysis segment. Replace remaining missing traffic values with the median for that variable and record the imputed fields if extending the dataset.

## Electricity-rate and construction-cost attributes

The 2024 commercial electricity-rate layer used in the study was derived by Esri from EIA Form 861. Normalize the assigned rates as required by the model and fill missing segment values with the dataset average.

The study uses median household income as a spatial proxy for relative construction costs. Normalize it and bound the multiplier to 0.7–1.3. This proxy does not directly represent terrain, permitting, labor contracts, or right-of-way constraints; preserve that limitation in derivative analyses.

## Substation access

Use a distance-based analysis to identify every candidate substation within 5 miles of each highway segment. Retain all qualifying relationships, with distance and rank, rather than only the nearest substation. A segment without a direct substation candidate can remain a DWC candidate if it can be supplied through valid adjacent selected segments.

## Candidate SCS construction

1. Extract OpenStreetMap `motorway_link` and `trunk_link` features that intersect the Interstate network.
2. Convert the interchange ramps to representative points.
3. Cluster ramp points using DBSCAN:
   - neighborhood distance: 400 m;
   - minimum features per cluster: 2.
4. Create one candidate charging site per cluster and assign `SCS_ID`.
5. Identify existing public DC fast-charging stations from DOE AFDC.
6. Exclude candidate SCS sites within 2 miles of an existing public DC fast-charging station. If multiple candidates are close to the same station, associate only the nearest candidate for this filtering operation.
7. Remove SCS-bound demand assigned to excluded candidates while retaining those roadway segments' DWC eligibility.

## DWC adjacency

Create an adjacency relationship only when two analysis segments share a valid roadway endpoint and represent a legitimate roadway continuation. Do not infer adjacency solely from geometric closeness, because nearby carriageways, crossings, and interchanges can create false links. Export both directions for each valid connection if the downstream model expects directed arcs.

## CSV export

Export the fields and files in [INPUT_DATA_DICTIONARY.md](INPUT_DATA_DICTIONARY.md). ArcGIS distance fields in the reference files are stored in meters and converted to miles by the current Python code. The main segment table's `Actual_Length` is in miles; `Shape_Length` follows the projected geometry units.

## Quality-control checks

- Confirm the segment count and unique `SEGMENT_ID_OG` values.
- Inspect route endpoints for short slivers and invalid cross-route adjacencies.
- Verify that foreign keys reference existing segment, SCS, and substation IDs.
- Map unusually large substation distances and SCS distances.
- Count missing traffic, rate, and income fields before and after imputation.
- Store geoprocessing history or a ModelBuilder artifact with the release record.
- Compare checksums and row counts against the included processed reference inputs.

