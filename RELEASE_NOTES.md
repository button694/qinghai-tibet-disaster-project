# Interactive hazard and rescue explorer — 28 October 2026

Open the [interactive map](https://button694.github.io/qinghai-tibet-disaster-project/#/map).

- Search provincial, municipal and county names or administrative codes. Select a region to filter the map and paginated location list together.
- Select a hazard point in the list or on the map to display its linked emergency facilities and saved routes. Select a facility card to isolate a route; select it again to show all linked routes.
- Switch between team-computed susceptibility results and externally sourced 2050/2100 glacial-lake data. The future lake data and the team's routing calculations are labeled separately.
- Switch Chinese/English for the new explorer, navigation and raster controls. Original place names, facility names and data records are preserved. Legacy catalog and analytical pages retain their original detailed text.
- Raster entries now link to the corresponding raster preview. Other catalog-only layers link to their dataset records; these are no longer presented as missing data.
- Existing emergency-route detail responses now resolve facilities from exported database identifiers.

## External lake data

Zhang, G. (2025). *Remote sensing monitoring data of hazardous glacial lakes in the Himalayan region*. National Tibetan Plateau / Third Pole Environment Data Center. DOI: [10.11888/Cryos.tpdc.302610](https://doi.org/10.11888/Cryos.tpdc.302610). Provider metadata describes 2050 and 2100 under RCP2.6. License: CC BY 4.0.

Required associated reference: Zheng et al. (2021), *Increasing risk of glacial lake outburst floods from future Third Pole deglaciation*, Nature Climate Change. DOI: [10.1038/s41558-021-01028-3](https://doi.org/10.1038/s41558-021-01028-3).

Raw imported HRGI, EXPOSURE and RRGI values are displayed without inventing definitions, units or equivalence to team susceptibility scores.

## Verification and limits

Type checking and production build passed. Targeted tests cover original-source route matching and polygon filtering with holes. All 8,169 route records resolve to preserved facility records. There are 11 routes in 2050 and 10 in 2100 without an original lake identifier; these remain in the route directory and are not assigned to lakes by guessing row order. Locations without saved associations show an explicit empty state.

Region filtering uses published administrative polygons simplified at 200 m tolerance; assignments near boundaries are for browsing. This viewer does not recompute the model or plan new routes from live road conditions. Distances remain the saved network distances. The CatBoost training code remains unavailable following the previously documented hard-drive failure.

`frontend-source.zip` preserves the implementation and tests. See its BUILD.md for reconstruction instructions using the data included in site.zip. No database backup or credentials are included.
