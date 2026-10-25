# Public data sources and provenance

The following sources are described in the paper and its Supplementary Information. URLs and access terms can change; verify them at release time and archive the exact source date/version used.

| Purpose | Source used in the study | Study access/version note | Key derived fields |
|---|---|---|---|
| Interstate geometry | USDOT/BTS National Highway System, National Transportation Atlas Database | Service updated 8 Aug 2025; accessed 11 Aug 2026 | route identifier, geometry, segment IDs |
| Traffic and speed | FHWA Highway Performance Monitoring System shapefiles | State-level public shapefiles | AADT, commercial AADT, single-unit AADT, speed limit |
| Electricity rate | Esri layer based on EIA Form 861 | 2024 nominal commercial and total rates | CER and normalized rate |
| Construction-cost proxy | Esri demographic layer | Median household income | normalized index bounded to 0.7–1.3 |
| Substations | Xu et al., *A data-driven nationwide transmission network model for the United States*, IEEE PESGM, 2020 | Research dataset cited by the paper | substation IDs and coordinates |
| Interchange ramps | OpenStreetMap | `motorway_link` and `trunk_link` geometries | candidate SCS clusters |
| Existing public DCFC | U.S. DOE Alternative Fuels Data Center | Public station data download; the live dataset is updated continuously | existing-site exclusion within 2 miles |

## Source links

- [National Highway System ArcGIS REST service](https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/NTAD_National_Highway_System/FeatureServer)
- [FHWA HPMS shapefiles](https://www.fhwa.dot.gov/policyinformation/hpms/shapefiles.cfm)
- [Esri electricity-rate item](https://www.arcgis.com/home/item.html?id=1ec5831c62d04eb081b64a3f864d6f84)
- [EIA electricity sales, revenue, and price data](https://www.eia.gov/electricity/sales_revenue_price/)
- [Esri demographic item](https://www.arcgis.com/home/item.html?id=b0b3b31e531e406185f2de4fff596060)
- [OpenStreetMap](https://www.openstreetmap.org/)
- [DOE Alternative Fuels Data Center station-data download](https://afdc.energy.gov/data_download)

## Known source notices

- The current NTAD NHS service states that the 2025-08-08 dataset is a U.S. government work available for unrestricted public use, but requires its metadata entry to accompany any distribution and prohibits changing or deleting that metadata entry. Preserve a copy of the source metadata with a public data release.
- FHWA warns that spatial processing can leave gaps or section-level discrepancies in HPMS geospatial releases and that its Highway Statistics tables, rather than the spatial extracts, are the official national calculations.
- EIA has discontinued the standalone *Electric Sales, Revenue, and Average Price* product after the 2024 edition; the underlying information remains available through EIA-861 data files.
- OpenStreetMap data are licensed under the ODbL. A release using derived OSM ramp data must credit OpenStreetMap contributors, identify the ODbL, and assess the share-alike requirements for the distributed database.
- AFDC requires users of its interactive download form to accept its then-current terms and conditions. Archive the download date, filters, and accepted terms for the study snapshot.

## Derived-data notes

- The highway input is a processed research dataset, not a verbatim copy of one source layer.
- Electricity-rate and demographic attributes were spatially assigned before the route dissolve.
- Candidate SCS sites are derived from clustered ramp geometries rather than existing charging sites.
- Existing DC fast chargers are used to remove redundant candidates; they are not the candidate-site source.
- Substation tap and adjacency lines reconstructed after optimization are relationship visualizations, not utility routing plans.

## Attribution practice

When updating or redistributing the processed inputs, retain each source URL, access date, required attribution, and source-specific notice. Esri-hosted items can have item-specific terms, so record the item version used for a new dataset build.
