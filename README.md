# Qinghai–Tibet Plateau Geohazard Susceptibility and Emergency Response

A spatial database and GIS analysis project connecting potential geohazard locations, exposed settlements, emergency facilities, and transport networks across the part of the Qinghai–Tibet Plateau within China.

**[Interactive website](https://button694.github.io/qinghai-tibet-disaster-project/)** | **[Analysis methods and execution guide](ANALYSIS.md)**

Visitors can explore the published results without installing SQL Server or Python.

## Background and motivation

On 26 August 2026, a debris-flow disaster originating upstream in Nepal struck the Gyirong border port in Tibet, China, causing major casualties and leaving people missing. The China Geological Survey's preliminary assessment traced the event to a high-altitude ice collapse that entrained moraine material and evolved into a debris flow. This disaster motivated our team to explore how spatial data could support potential geohazard identification and emergency response planning. See the [China Geological Survey report](https://www.cgs.gov.cn/ywdt/ddyw/202608/t20260828_867531.html).

We built the database specifically to support these connected functions: organizing environmental and historical hazard data, screening locations potentially susceptible to future hazards, assessing possible settlement exposure, and linking these locations to emergency facilities through the regional transport network.

## Study area and data

The study area covers the portion of the Qinghai–Tibet Plateau within China. Our team assembled and processed environmental data describing climate, terrain, geomorphology, and related geological conditions, together with historical landslide, debris-flow, and other hazard records. Original dataset sources are documented in the website's data catalog.

The database integrates environmental rasters, historical hazards, settlements, hospitals, fire stations, rescue facilities, roads, railways, and analytical outputs. The implemented susceptibility model uses 11 factors: precipitation, elevation, distance to historical earthquakes, distance to glacial lakes, slope, permafrost type, distance to glaciers, land cover, distance to roads, distance to rivers, and aspect.

## Analytical workflow

1. **Data preparation and database construction.** Acquire source datasets, standardize spatial references, process vector and raster layers, and organize the results in a SQL Server spatial database.
2. **CatBoost and SHAP analysis.** Combine historical hazard records with environmental predictors to construct training data, train a CatBoost model, and derive factor importance from mean absolute SHAP values. Normalize these importance values into weights for subsequent spatial scoring.
3. **Potential hazard location screening.** Apply the saved SHAP-derived weights to the latest environmental factor layers available to the project, calculate susceptibility scores on a 1 km grid, and extract high-scoring local maxima. Rule-based classification and DEM-based downslope tracing support potential settlement-impact screening.
4. **Emergency route planning.** Combine roads and railways into a network and use multi-source Dijkstra analysis to associate candidate locations with hospitals, fire stations, and rescue facilities. The routing objective is shortest modeled network distance; airport associations use straight-line distance.
5. **Integration and visualization.** Store analytical results alongside source datasets and provide maps, a data catalog, result inspection, and route visualization.

These candidate locations represent regional susceptibility screening for possible future hazards. They do not predict when an event will occur or provide calibrated event probabilities. Routes do not account for real-time closures, disaster damage, travel speeds, or facility availability. Detailed assumptions are documented in [ANALYSIS.md](ANALYSIS.md).

## Preserved results

| Published content | Count |
| --- | ---: |
| Saved susceptibility result points | 1,582 |
| Integrated rescue routes | 4,566 |
| Hospital facilities | 1,529 |
| Dataset catalog records | 58 |
| Lightweight raster previews | 31 |

The website also includes saved 2050 and 2100 scenario products, with provenance distinguished from the team's own analytical outputs. The snapshot was exported on **27 October 2026**.

## Team contributions

**Yiyan Wang** was responsible for source-data acquisition, all data preprocessing, the design and construction of the SQL Server spatial database, CatBoost/SHAP factor analysis, and emergency rescue-route computation. Chenming Yang contributed to data acquisition, while Jiayi Chen and Xiaotong Deng developed the initial web interface in collaboration with Wang. Wang subsequently extended and refined the website.

## Available code and evidence

- `build_geohazard_settlement_analysis_v4.py`: weighted susceptibility scoring, candidate extraction, hazard-type rules, and settlement-impact screening.
- `build_hazard_rescue_network_analysis_v1_2.py`: emergency facility association and transport-network route computation.
- `shap_weights_run.png` and `SHAP_weights.csv`: original console screenshot and transcribed factor-weight results.
- `requirements.txt` and `provenance.json`: dependencies and checksums documenting the preserved script copies.

The CatBoost training and SHAP-computation code was lost following a hard-drive failure. The surviving hazard script consumes previously calculated weights; it does not train CatBoost. This repository preserves downstream analysis and result evidence rather than a complete reproducible training experiment. Training samples, the fitted model, tuning logs, and performance records are not included. Model accuracy cannot be inferred from the weight screenshot.

## Website publication and scope

`site.zip` contains the built static website. GitHub Actions extracts it and deploys it to GitHub Pages. It includes no database backup, credentials, connection settings, or complete source rasters.

The website displays saved results without a live SQL Server connection. Line and polygon geometries are simplified using a 200 m tolerance for browsing; exported point attributes retain their saved values. Some large layers are represented by catalog entries only. The static edition does not provide model recomputation, precise spatial queries, live warnings, real-time dispatch, or every resource-association detail. Saved network distances are not actual travel times.

Source attribution is retained in the data catalog. Publication does not change the rights or licenses of third-party datasets or software.

## Method reference

Yang et al. (2026). *Landslide susceptibility on the Qinghai-Tibet Plateau: Key driving factors identified through machine learning*. Journal of Geographical Sciences, 36(1), 199–218. [doi:10.1007/s11442-026-2444-6](https://doi.org/10.1007/s11442-026-2444-6).

This paper informed the CatBoost/SHAP methodology. The project adapted the factor set and added hazard screening, settlement-impact screening, and rescue-network analysis. The paper's reported performance is not a performance measurement for this project.
