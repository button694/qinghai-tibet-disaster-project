# Susceptibility Screening and Emergency Rescue Network Analysis

See the [project overview](README.md) for the disaster-response motivation, study area, workflow, and contributions. This document describes the two preserved analysis scripts.

## Scripts and preserved training evidence

| Script | Purpose | Outputs |
| --- | --- | --- |
| `build_geohazard_settlement_analysis_v4.py` | Eleven-factor susceptibility scoring, candidate extraction, rule-based hazard classification, D8 downslope tracing, and settlement-impact screening | Susceptibility GeoTIFF; hazard points, paths and corridors in GeoPackage; settlement associations in CSV; weights and metadata |
| `build_hazard_rescue_network_analysis_v1_2.py` | Combined road/rail network, multi-source Dijkstra analysis, matching to three emergency facility types, and nearest-airport association | Routes and facilities in GeoPackage; route, exception and summary CSVs; airport associations and metadata |

**CatBoost training and SHAP interpretation preceded these scripts.** The hazard script's `Config.weights` stores the 11 previously derived, normalized SHAP weights. It does not train CatBoost or recompute SHAP values.

The original training and SHAP-computation code was lost following a hard-drive failure. The original output screenshot and downstream scripts survive. This repository does not include the fitted model, training samples, tuning logs, or evaluation files. The website's 1,582 points and 4,566 routes are saved outputs; different inputs need not reproduce these counts.

![Original SHAP weight console output](shap_weights_run.png)

`SHAP_weights.csv` transcribes the screenshot's mean absolute SHAP values, project weights, and literature reference percentages. Project percentages match `SHAP_ORIGINAL_PERCENT` in the hazard script and are normalized for scoring. The screenshot contains no sample count, train/test split, hyperparameters, accuracy, or AUC. Literature percentages are reference values, not project evaluation metrics. The original screenshot, source-data filenames, and script strings retain their original language to preserve evidence and data compatibility.

## Environment and inputs

Use a separate Python 3.10+ environment. Dependencies are not pinned to the original experiment versions. GeoPandas, Rasterio and Pyogrio require compatible GDAL support for reading FileGDB and writing GeoPackage. Fiona is an optional layer-listing fallback.

```bash
python -m pip install -r requirements.txt
```

Run from the repository root. Update `Config` paths, layer names, and field names for your inputs. Default layout:

```text
data/
  NewDefault.gdb/
  rasters/
    TNR_DEM_RS.tif
    TNR_PERMAFROST_RS.tif
    TNR_SLOPE_RS.tif             # optional; derivable from DEM
    TNR_ASPECT_RS.tif            # optional; derivable from DEM
    TNR_RELIEF_RS.tif            # optional; derivable from DEM
    BASE_POPULATION_RS.tif      # optional
    LCC_投影后/
      TNR_LANDCOVER_RS_LCC.tif
      TNR_PRECIP_ANNUAL_RS_*_LCC.tif
outputs/
```

Required layers are defined in each script's `required_layers`, `Config`, and configuration-check output. Hazard analysis uses glaciers, glacial lakes, historical hazards, historical earthquakes, rivers, roads, settlements, and environmental rasters. Network analysis uses hazard points, roads, railways, hospitals, rescue stations, fire stations, and airports. Sources are documented in the website catalog. Observe original data licenses. Complete FileGDB inputs, original TIFFs, and database backups are not distributed here.

## Execution order

Configuration checks also require the GIS dependencies imported by the scripts.

```bash
python build_geohazard_settlement_analysis_v4.py --check-config
python build_hazard_rescue_network_analysis_v1_2.py --check-config
python build_geohazard_settlement_analysis_v4.py
```

Hazard outputs go to `outputs/hazard_analysis`. The point layer in `QTP_RES_HAZARD_ANALYSIS_V4.gpkg` is named `RESULT_SUSCEPTIBILITY_PT`. **The network script does not automatically read that GeoPackage.** Import the layer into `data/NewDefault.gdb` as `RES_SUSCEPTIBILITY_PT` using GIS software, preserving hazard identifiers and required fields. Then run:

```bash
python build_hazard_rescue_network_analysis_v1_2.py
```

Network outputs go to `outputs/rescue_network`. Neither script writes directly to SQL Server; importing results into the database is a separate step. Repeated runs overwrite matching CSV/raster files, and routing replaces its matching output GeoPackage. Use separate output directories for separate runs.

## Method definitions and limitations

- **Spatial framework:** Lambert projection, central meridian 89.5 degrees, latitude of origin 32.5 degrees, standard parallels 27.5/37.5 degrees, and a 1 km grid. Eleven factor scores are combined with SHAP-derived weights. Scoring directions and categorical enrichment are constructed separately. Scores are not calibrated hazard probabilities.
- **Candidate extraction:** threshold 0.55; the 8 km parameter becomes a **17 by 17 cell** local-maximum window; at most 5,000 candidates. Equal-valued plateaus can yield multiple points. This is not a strict minimum pairwise distance.
- **Classification version difference:** the script's `score_level` starts the medium class at **50 points**, while the report/website describes **55 points**. Candidates are first filtered at 55, so the 50–55 interval does not enter candidate outputs. The difference matters for full-raster classification. Original script thresholds are preserved.
- **Settlement impacts:** D8 follows the steepest of eight neighboring DEM directions. Hazard-specific corridors and downslope filters provide regional exposure screening, not landslide dynamics or flood inundation simulation. Population estimates depend on raster units. Population on each settlement relationship refers to the whole hazard corridor; adding these rows would double-count exposure.
- **Network construction:** an undirected `MultiGraph` subdivides segments to approximately 2 km maximum length and quantizes nodes on a 20 m grid. It does not split/connect every geometric line intersection. Grade separation, bridges, tunnels, and road/rail transfer constraints are not modeled.
- **Routing objective:** hazards and facilities connect to their nearest network nodes within 20 km and 10 km respectively. Multi-source Dijkstra minimizes **distance inside the modeled network**. Endpoint connection distances are added after facility selection, so full distance including these connections is not guaranteed minimal. Facilities sharing a source node are resolved by shorter connection distance.
- **Airports and operational scope:** nearest-airport associations use planar straight-line distance. Airports do not participate in network routing. Travel speeds, live restrictions, disaster damage, and facility availability are absent. These are research and preliminary planning outputs, not real-time dispatch instructions.

## Publication changes and verification

The messaging-app suffix `(1)` was removed from public script filenames. Only developer-machine absolute paths in `Config` were replaced with relative `data/` and `outputs/` paths. Weights, thresholds, and function implementations were preserved. `provenance.json` records checksums of the supplied originals and public copies.

Both scripts passed syntax compilation. AST comparisons confirmed that 46 hazard-analysis functions and 26 network-analysis functions matched the supplied originals; the 11 normalized weights passed a sum check. The publication process did not install the full GIS environment or rerun the full original-data analysis. These checks validate preservation and syntax, not model accuracy.

## Method reference

Yang et al. (2026). *Landslide susceptibility on the Qinghai-Tibet Plateau: Key driving factors identified through machine learning*. Journal of Geographical Sciences, 36(1), 199–218. [doi:10.1007/s11442-026-2444-6](https://doi.org/10.1007/s11442-026-2444-6).

This paper informed the CatBoost/SHAP methodology. The project adapted the factors and added hazard-location screening, settlement-impact screening, and rescue-network analysis. The paper's reported accuracy is not a measured accuracy for this project.
