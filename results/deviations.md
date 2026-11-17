# Methodological Decisions and Deviations from the Base Paper

This document records all methodological decisions, assumptions, and necessary deviations from the base paper:
*Morapedi TD and Obagbuwa IC (2023). Air pollution particulate matter (PM2.5) prediction in South African cities using machine learning techniques. Frontiers in Artificial Intelligence 6:1230087.*

---

### 1. Date Range Filtering
- **Decision**: Keep only dates between `2019-01-01` and `2022-12-31`.
- **Reason**: The raw WAQI CSV files include UTC boundary transition rows on `2018-12-31` and `2023-01-01`. Dropping these edge rows ensures a clean four-year period (2019 to 2022) without partial day artifacts.

### 2. Missing-Value Imputation and Temporal Reindexing
- **Decision**: Reindex each city to a complete daily frequency strictly between its own first and last available recorded dates. Missing `Median_PM25` values within each city's range are imputed with that city's mean `Median_PM25`.
- **Reason**: The paper states missing values were filled with the "mean of the PM2.5 median." Extending cities beyond their actual station monitoring period would generate synthetic observations. Stations with early cutoffs (Cape Town ending 2022-10-21, Richards Bay ending 2022-08-22, East London ending 2022-07-10) are bounded by their real monitoring windows. Klerksdorp has fewer observations (629 rows), which is preserved and documented.

### 3. Lag Feature Construction
- **Decision**: Define `lag1`, `lag2`, and `lag3` as `Median_PM25` shifted by 364, 728, and 1092 days (52, 104, and 156 weeks) per city.
- **Reason**: Figure 12 of the paper displays `lag1`, `lag2`, and `lag3` alongside calendar features, but the text never defines the lag intervals. Seasonal 52-week shifts preserve the day-of-week alignment across historical years. Rows with missing lags are preserved because CatBoost and XGBoost natively handle missing feature values.

### 4. Group Cross-Validation Alignment
- **Decision**: Primary cross-validation uses `GroupKFold(n_splits=5)` grouped by `City`.
- **Reason**: Figures 6 and 8 in the paper show fold train/test station splits of (9, 3), (9, 3), (10, 2), (10, 2), and (10, 2) cities, indicating that spatial leave-cities-out evaluation was conducted. A secondary `TimeSeriesSplit(n_splits=5, test_size=150, gap=24)` is also evaluated per city as described in Section 3.3.

### 5. XGBoost Estimators Discrepancy
- **Decision**: Set `n_estimators=1000` with `early_stopping_rounds=50`.
- **Reason**: Section 3.3 of the paper states 1,000 estimators, whereas the printed code output snippet in Figure 8 lists `n_estimators=500`. We follow the written text (1,000 estimators) and specify `early_stopping_rounds=50` (unspecified in the paper).

### 6. Date Split Evaluation
- **Decision**: Train on records prior to `2022-01-01` and evaluate on records from `2022-01-01` onward. Cities with no 2022 records are safely handled without runtime errors.
- **Reason**: Matches Section 3.3 (iii) of the paper.

### 7. Cities Without Recording Stations (Approximation)
- **Decision**: Train the top-performing regression model on all 12 stations using calendar features plus `Lat` and `Long` (no lags), and predict 2022 levels for 6 unmonitored cities (Kimberley, Kuruman, Lebowakgomo, Mahikeng, Upington, Vryburg).
- **Reason**: The base paper mentions augmenting with Sentinel-5P and Google Earth Engine layers, but provides neither code, satellite feature definitions, nor processed tables. Spatial coordinate interpolation serves as a transparent approximation. Leave-one-city-out cross-validation is reported to assess spatial generalization error.

### 8. Future Forecasting Protocol
- **Decision**: Models are trained on data up to `2022-11-25` only. Predictions are generated recursively day-by-day from `2022-11-26` to `2023-12-31`. Lags are dynamically updated using predicted values when inside the forecast window. Real data after `2022-11-25` is used strictly as a hold-out sanity check to compute test RMSE.
- **Reason**: In Figure 12, the authors forecasted from 2022-11-26 to 2023-12-31, but used hourly rows with frozen lag values (a known methodological flaw). Our recursive daily forecast prevents lookahead bias while providing a fair comparison against actual late-2022 observations.

### 9. AQI Status Classification Protocol
- **Decision**: Train on a 70/30 train/test split (`random_state=42`) using only `Median_PM25` as the input feature to predict the 6 EPA AQI categories.
- **Reason**: Matches the paper's reported test support (4,402 samples, exactly 30% of the dataset) and single-variable input. Note that this task exhibits direct label leakage because the target class is deterministically computed from the input feature via threshold inequalities, explaining the near-100% accuracy.

