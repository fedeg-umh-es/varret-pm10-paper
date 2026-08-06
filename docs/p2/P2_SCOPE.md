# P2 Scope Definition

## Scientific Question

How much of the observed skill relative to persistence can be explained by the
finite-memory linear autocorrelation structure of PM10?

## Series

- Elche (station 03014002_10_M, Elx-Agroalimentari, Alicante)
- Valencia Vivers (station 46250043_10_M, Valencia-Vivers, Valencia)
- Zarra EMEP (station 46263999_10_M, Zarra-EMEP, Valencia)

## Resolution

- Daily

## Forecast Horizons

- h = 1, 2, 3, 4, 5, 6, 7 (days ahead)

## Main Empirical Models (for best-model selection)

- `hgb_direct` — HistGradientBoosting direct multi-horizon
- `ridge_direct` — Ridge regression direct multi-horizon
- `sarima` — Seasonal ARIMA

## Reference-Only Models (not eligible for best-model selection envelope)

- `seasonal_naive`
- `stl_ridge_direct`

These two models appear in comparison tables but do not determine the best empirical model.

## Main Linear Reference

- AR(p)/Yule–Walker finite-memory optimal linear reference
- Order: p = 14
- Estimated from daily PM10 autocorrelation structure using calendar-aligned valid pairs
- Separate estimation per station

## Skill Definition

```
Skill_RMSE = 1 - RMSE_model / RMSE_persistence
```

where persistence repeats the last observed value (at origin date) for all horizons.

## Persistence Baseline

Simple lag-1 persistence: for each origin date t, the forecast for all horizons
h = 1,...,7 is y_t (the PM10 value observed at the origin date).

## Terminology

Permitted:
- "finite-memory optimal linear reference"
- "AR(p)/Yule–Walker linear predictability reference"

Prohibited:
- "universal linear predictability ceiling" (the reference is conditional on
  stationarity, quadratic loss, order p, information used, autocovariance estimation,
  and exact skill definition)

## Data Paths

| Station           | Raw Data Path                          | Predictions Path                                  |
|-------------------|----------------------------------------|---------------------------------------------------|
| Elche             | data/raw/pm10_daily.csv                | outputs/metrics/predictions.csv                   |
| Valencia Vivers   | data/raw/pm10_valencia_vivers.csv      | outputs/metrics/predictions_valencia_vivers.csv   |
| Zarra EMEP        | data/raw/pm10_zarra_emep.csv           | outputs/metrics/predictions_zarra_emep.csv        |

Multi-station skill reference: evidence/paper_a/aggregates/master_diagnostic_table.csv
