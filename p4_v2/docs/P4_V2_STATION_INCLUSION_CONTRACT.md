# P4-v2 Station Inclusion Contract

**Status: FROZEN_BEFORE_FORECASTING**

This contract is frozen for the new P4-v2 PM10 benchmark before any forecasting model is trained or evaluated. It is a data-selection contract only.

## Canonical identity

- Pollutant: PM10, represented by `MAGNITUD=10` in the official source.
- Canonical `station_id`: the original `PUNTO_MUESTREO` field.
- A point is stable only if it maps to exactly one `(PROVINCIA, MUNICIPIO, ESTACION, MAGNITUD)` tuple across 2020–2023 and has no duplicate station/date source records. Distinct `PUNTO_MUESTREO` values are never merged merely because they share a station code.

## Daily aggregation

For each source station-day, `pm10_daily` is the arithmetic mean of valid hourly PM10 observations belonging to that UTC calendar day, represented by the MITECO `ANNO/MES/DIA` label and H01–H24 fields. A day is usable only when `n_valid_hours >= 18`. Missing hourly cells are not imputed, interpolated, or forward-filled. A day below the threshold remains present in the canonical daily table with `pm10_daily` missing and is not counted as usable.

## Inclusion criteria

A PM10 `station_id` is included if and only if all conditions hold:

1. **Stable identity:** the canonical `PUNTO_MUESTREO` identifier passes the identity audit.
2. **Daily support:** daily values are usable only at the `18/24` rule above.
3. **Year support:** at least 3 calendar years in 2020–2023 contain at least 300 usable station-days each.
4. **Total usable-day coverage:** `coverage_total = usable_days / support_days >= 0.75`. `support_days` is the inclusive number of UTC calendar days from the station's first to last source-supported date in 2020–2023. This denominator counts days without a usable observation as missing and does not count dates outside the station's observed temporal support.
5. **Forecast-independence:** no model error, forecast skill, variance retention, event performance, ranking, or other predictive result is used.

## Rationale

- The original `PUNTO_MUESTREO` identifier avoids merging distinct measurement points and preserves the source's experimental series identity.
- The 18-hour rule prevents a daily mean from being driven by a small fraction of a day while retaining a transparent, fixed completeness rule.
- Three usable calendar years and 300 usable days per year require multi-year support for later rolling-origin design without selecting stations from predictive outcomes.
- The 0.75 usable-day coverage floor limits prolonged missingness over each station's actual 2020–2023 support.
- All thresholds are frozen data-contract choices, not regulatory limits, universal air-quality standards, or forecasting-performance thresholds.

## Frozen outcome

The deterministic application of this contract selects **324** of **415** PM10 sampling-point series. The count is an output of the frozen definitions and was not targeted.
