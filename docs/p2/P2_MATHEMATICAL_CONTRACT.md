# P2 Mathematical Contract

## 1. Predictor Vector

For a stationary daily PM10 series {y_t}, the predictor vector of order p is:

```
x_t = [y_t, y_{t-1}, ..., y_{t-p+1}]^T   (p × 1)
```

## 2. Forecast Target

The target for horizon h ≥ 1 is:

```
y_{t+h}
```

## 3. Autocovariance Matrix

The p × p autocovariance (Toeplitz) matrix is:

```
Γ_p(i, j) = γ(|i - j|),   i, j = 1, ..., p
```

where γ(k) = Cov(y_t, y_{t-k}) = E[(y_t - μ)(y_{t-k} - μ)].

Estimated using valid pairs aligned on the calendar grid (no compact compression of NaN).
Valid pair count for lag k: n_k = number of dates t where both y_t and y_{t-k} are observed.

## 4. Cross-Covariance Vector

The cross-covariance vector between x_t and y_{t+h} (in lag order matching x_t):

```
c_h = Cov(x_t, y_{t+h}) = [γ(h), γ(h+1), ..., γ(h+p-1)]^T   (p × 1)
```

Explicitly: the i-th element (i = 1, ..., p) is γ(h + i - 1).

## 5. Yule–Walker Coefficients

Optimal linear coefficients β_h are the solution to:

```
Γ_p · β_h = c_h
```

i.e.,

```
β_h = Γ_p^{-1} c_h
```

Solved via scipy.linalg.solve (or numpy.linalg.solve as fallback), NOT via explicit
matrix inversion. This avoids numerical amplification from explicit inversion.

## 6. Minimum MSE of the Optimal Linear Predictor

The minimum achievable MSE for the optimal linear predictor of order p at horizon h is:

```
σ²_{e,h} = γ(0) - c_h^T Γ_p^{-1} c_h
         = γ(0) - c_h^T β_h
```

Note: σ²_{e,h} ≥ 0 by the positive semi-definiteness of Γ_p.
If numerical error yields a small negative value, it is clamped to 0 and flagged.

## 7. Yule–Walker RMSE

```
yw_rmse_h = sqrt(σ²_{e,h})
```

## 8. Skill Definition

**The empirical skill definition used in this repository is RMSE-based:**

```
Skill_RMSE = 1 - RMSE_model / RMSE_persistence
```

Therefore the Yule–Walker linear reference skill is:

```
yw_skill_h = 1 - yw_rmse_h / persistence_rmse_h
```

where:
```
persistence_rmse_h = sqrt(MSE_persistence_h)
```
and MSE_persistence_h is computed from the matched (origin_date, date, horizon) pairs
in the predictions CSV.

**Caution:** Do NOT mix YW-MSE-based reference with RMSE-based empirical skill
without applying the square root correction. The Yule–Walker reference and empirical
metrics must use the same functional form.

## 9. AR(1) Special Case (Validation Reference)

For p = 1 at horizon h:
```
β_{h,1} = ρ(h) / ρ(0) = ρ(h) = ρ^h   (for AR(1) process with autocorrelation ρ)
σ²_{e,h} = γ(0)(1 - ρ^{2h})
yw_rmse_h = σ_y · sqrt(1 - ρ^{2h})
yw_skill_h = 1 - sqrt(1 - ρ^{2h}) · σ_y / persistence_rmse_h
```

## 10. Hybrid Reference (DIAGNOSTIC ONLY)

The repository contains a "hybrid" reference derived in earlier analysis (PM10-skill
persistence bound). Classification:

```
STATUS: DIAGNOSTIC_ONLY
```

It is NOT presented as a rigorous linear reference. Its formula must be recovered from
the repository source if cited. If the formula cannot be determined unambiguously:
`BLOCKED_BY_HYBRID_FORMULA_AMBIGUITY`.

## 11. AR(1) Approximation Reference (DIAGNOSTIC ONLY)

The AR(1) approximation reference at horizon h:

```
ar1_skill_h = 1 - sqrt(1 - ρ(1)^{2h}) · γ(0)^{1/2} / persistence_rmse_h
```

using only ρ(1) (the first autocorrelation). This is not the full Yule–Walker
reference but an approximation for diagnostic comparison.

Classification: `DIAGNOSTIC_ONLY` — not the main P2 result.

## 12. Missingness Protocol

1. Parse date column.
2. Sort chronologically.
3. Reindex to complete daily calendar.
4. Missing PM10 values remain as NaN — NOT interpolated, NOT dropped.
5. Autocovariance γ(k) estimated from valid pairs: pairs (y_t, y_{t-k}) where
   both values are non-NaN on the complete calendar grid.
6. Record n_k (valid pair count) for each lag k.

**Prohibited:**
```python
y = y[~np.isnan(y)]   # before lag computation or ACF estimation
```
Compact removal changes the temporal meaning of lags.

## 13. Stationarity Assumption

The Yule–Walker reference assumes weak stationarity of the PM10 series.
Deviations from stationarity (trends, seasonality, non-constant variance) mean
the estimated γ(k) is an average over all available time, not a local property.
This is a known limitation and is flagged in the results.

## 14. Conditional Nature of the Reference

The AR(p)/Yule–Walker reference is conditional on:
- Stationarity of the series
- Quadratic (MSE) loss function
- Order p (here p = 14)
- Information available (past p observations, no exogenous variables)
- Estimation of autocovariances from the observed sample
- Exact definition of skill

It is NOT a universal linear predictability ceiling for all possible linear models
or all possible information sets.
