# Input data dictionary

The Python workflow reads six processed CSV files. The included copies are the reference inputs used by the current project. Field names are case-sensitive in the scripts.

## `NHS_US_5miles_with_SCS.csv`

One record per five-mile-scale Interstate segment; 9,053 rows in the included file.

| Field | Meaning |
|---|---|
| `SEGMENT_ID_OG` | Stable GIS segment identifier and primary join key |
| `SIGN1` | Signed Interstate route identifier |
| `Actual_Length` | Segment length in miles |
| `ROUTEID`, `SIGNT1`, `SIGNN1` | Source route descriptors |
| `SPEED_LIMI` | Assigned speed limit |
| `AADT` | Annual average daily traffic |
| `AADT_COM` | Commercial-vehicle AADT |
| `FILE_NAME`, `STATE`, `ZIP` | Source/state/location attributes |
| `Cmrcl_Electricity_Rate_Nom_2024` | Nominal 2024 commercial electricity rate |
| `Total_Electricity_Rate_Nom_2024` | Nominal 2024 total electricity rate |
| `Norm_electricity` | Normalized electricity-rate field |
| `median_hh_income` | Assigned median household income |
| `Norm_index` | Bounded construction-cost multiplier |
| `MID_X`, `MID_Y` | Segment midpoint coordinates in the working CRS |
| `Shape_Length` | Projected geometry length |
| `NEAR_SCS_ID` | Associated candidate SCS identifier |
| `NEAR_SCS_Dist` | Distance to the associated candidate SCS |

The file also contains `AADT_SINGL` in the conceptual GIS workflow when available, but that field is not present in the included exported table and is not required by the current validator.

## `NHS_USA_5miles_Adjacent.csv`

Directed segment-adjacency relationships; 16,758 rows.

| Field | Meaning |
|---|---|
| `TARGET_FID` | Source segment identifier |
| `JOIN_FID` | Adjacent segment identifier |
| `SIGN1`, `SIGN1_1` | Route descriptors for the two segments |
| `Shape_Length` | Relationship/source geometry attribute retained from GIS |

Adjacency must represent a valid roadway continuation. The table contains directed relationships used by the model's virtual and physical flow constraints.

## `NHS_US_SubstationDistance.csv`

All qualifying segment-to-substation relationships within the GIS search threshold; 39,340 rows.

| Field | Meaning |
|---|---|
| `IN_FID` | Highway segment identifier |
| `NEAR_FID` | Substation identifier |
| `NEAR_DIST` | Segment-to-substation distance in projected units (meters in the reference workflow) |
| `NEAR_RANK` | Distance rank for the segment |

The current Python loader converts `NEAR_DIST` from meters to miles using `0.000621371`.

## `SubstationsProject.csv`

Substation locations; 41,083 rows.

| Field | Meaning |
|---|---|
| `Sub_ID_OG` | Stable substation identifier |
| `X`, `Y` | Projected coordinates |
| `lat`, `lon` | Geographic coordinates |

## `US_SCS.csv`

Raw candidate SCS locations and proximity attributes; 21,430 rows.

| Field | Meaning |
|---|---|
| `SCS_ID` | Candidate SCS identifier |
| `MID_X`, `MID_Y` | Candidate projected coordinates |
| `NEAR_FID` | Nearby source feature identifier |
| `NEAR_DIST` | Proximity distance |

## `US_SCS_unique_existing.csv`

Candidate sites joined to existing-station screening information; 21,430 rows.

| Field | Meaning |
|---|---|
| `CLUSTER_ID`, `ORIG_FID` | Candidate-cluster provenance fields |
| `MID_X`, `MID_Y` | Candidate projected coordinates |
| `NEAR_FID`, `NEAR_DIST` | Candidate proximity fields |
| `SCS_ID` | Candidate SCS identifier |
| `EXISTING_SCS_ID` | Associated existing station identifier, when present |
| `EXISTING_SCS_DIST` | Distance to the associated existing station |

## Identifier relationships

```text
NHS_US_5miles_with_SCS.SEGMENT_ID_OG
  ├── NHS_USA_5miles_Adjacent.TARGET_FID / JOIN_FID
  ├── NHS_US_SubstationDistance.IN_FID
  └── optimizer output Segment_ID

NHS_US_5miles_with_SCS.NEAR_SCS_ID
  └── US_SCS.SCS_ID

NHS_US_SubstationDistance.NEAR_FID
  └── SubstationsProject.Sub_ID_OG
```

Null or sentinel SCS references may represent segments without an eligible SCS mapping and must be handled by the loader rather than interpreted as candidate IDs.

## Reference counts versus modeled sets

CSV row counts are not always the cardinalities of the final model sets. The loader filters invalid, missing, duplicate, or unreferenced records when constructing indexed sets. The paper reports 17,400 GIS candidate SCS sites, while the included raw candidate table contains 21,430 rows. With all 9,053 segments and existing-site screening enabled, the current loader produces 5,073 optimizable candidate IDs and 1,412 mapped IDs excluded because an existing station is nearby. Reproduction reports should record the raw, GIS-candidate, segment-referenced, existing-site-excluded, and final model-set counts separately.

