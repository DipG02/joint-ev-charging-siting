# Returning optimizer results to ArcGIS

`code/Main_Shapefiles_updated.py` packages a selected optimizer run into an ArcGIS-ready folder and ZIP archive. It uses GeoPandas for export and requires access to the original five-mile highway feature class.

## Configure the script

Set environment variables and, if needed, edit the layer-name configuration:

1. `DWC_RESULTS_DIR`: the selected results directory, for example `Results_baseline_p200_etaDWC0.50_etaSCS0.50_gev15pct_gaadt011bps`;
2. `DWC_HIGHWAY_GDB`: the original file geodatabase containing the highway feature class with `SEGMENT_ID_OG`;
3. `ORIGINAL_HIGHWAY_LAYER` in the script, if the feature-class name differs from the documented default.

The export folder and ZIP are created next to the selected result directory.

Use portable project-relative or environment-specific paths. Do not publish a personal Box, home-directory, or geodatabase path.

## Run

```bash
export DWC_RESULTS_DIR=/absolute/path/to/Results_run_tag
export DWC_HIGHWAY_GDB=/absolute/path/to/project.gdb
python code/Main_Shapefiles_updated.py
```

The script joins optimizer `Segment_ID` values to GIS `SEGMENT_ID_OG` values and writes the following principal layers:

| Output | Purpose |
|---|---|
| `01_DWC_Selected_Segments_Lines.shp` | Selected DWC highway geometries |
| `02_DWC_Selected_Segments_Join_Table.csv` | Helper table for checking the segment join |
| `03_SCS_Selected_Stations.shp` | Selected SCS point locations |
| `04_Selected_Substations_All.shp` | Selected substations and technology role |
| `05_DWC_SubstationTaps.shp` | Direct DWC-to-substation tap lines |
| `06_SCS_SubstationTaps.shp` | Direct SCS-to-substation tap lines |
| `07_Charging_Substation_Taps_All.shp` | Combined direct tap lines |
| `08_DWC_AdjacencyLinks.shp` | Segment-to-segment DWC power-sharing links |

Optional existing-SCS and segment-midpoint layers may also be produced when their inputs are available.

## ArcGIS Pro import

1. Extract the generated ZIP.
2. Use **Add Data** and add shapefiles from the extracted directory.
3. Symbolize `01_DWC_Selected_Segments_Lines.shp` by `dwc_conn`:
   - `Direct_Substation`: DWC segment with its own direct substation tap;
   - `Adjacent_Powered`: DWC segment supplied through an adjacent selected DWC segment.
4. Symbolize `04_Selected_Substations_All.shp` by `tech_role`:
   - `DWC`;
   - `SCS`;
   - `DWC+SCS`.
5. Display `07_Charging_Substation_Taps_All.shp` for all direct charger-to-substation relationships.
6. Display `08_DWC_AdjacencyLinks.shp` with a distinct line style for corridor sharing.

An adjacency-powered DWC segment has zero **direct feeder length** in summaries because it does not receive its own substation tap. The adjacency link is separately represented and should not be confused with a zero-distance substation connection.

## Interpretation

The generated tap and adjacency geometries are straight schematic connections between modeled assets. They support visual interpretation and length accounting based on the model inputs. They do not identify an approved distribution-line route, easement, conductor, voltage class, or interconnection design.

## Shapefile integrity

A shapefile is a group of files. Keep the `.shp`, `.shx`, `.dbf`, `.prj`, and `.cpg` files together. Field names can be truncated by the shapefile format; use the generated README/manifest and inspect the attributes after import. If extending the export, consider GeoPackage for a modern single-file alternative while retaining shapefiles for the documented ArcGIS workflow.
