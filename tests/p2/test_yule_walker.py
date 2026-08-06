"""Tests for P2 Yule–Walker finite-memory linear reference.

Synthetic tests validate implementation correctness.
They are NOT scientific evidence for the paper's thesis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.p2.yule_walker import (
    _build_toeplitz,
    _check_psd,
    _cross_covariance_vector,
    _estimate_autocovariance,
    _solve_yw,
    compute_yw_reference,
    load_calendar_series,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_calendar_series(values: np.ndarray, start: str = "2020-01-01") -> pd.Series:
    """Wrap array as DatetimeIndex Series on complete daily calendar."""
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def _ar1_series(n: int, rho: float, sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    eps = rng.normal(0, sigma, n)
    for t in range(1, n):
        y[t] = rho * y[t - 1] + eps[t]
    return y


# ── Test 1: Complete daily calendar is preserved ────────────────────────────────

def test_complete_daily_calendar(tmp_path):
    """load_calendar_series returns a complete daily DatetimeIndex."""
    import csv
    csv_file = tmp_path / "test.csv"
    dates = ["2020-01-01", "2020-01-03", "2020-01-05"]  # skip Jan 2, 4
    with open(csv_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "pm10"])
        for d, v in zip(dates, [10.0, 20.0, 30.0]):
            w.writerow([d, v])

    s = load_calendar_series(str(csv_file))
    assert len(s) == 5  # Jan 1–5 complete
    assert s.index.freq == "D" or pd.infer_freq(s.index) in ("D", "day", "d", None)
    assert s.isna().sum() == 2  # Jan 2 and Jan 4 missing


# ── Test 2: Missing dates are preserved as NaN ──────────────────────────────────

def test_missing_dates_preserved(tmp_path):
    """Missing days in source CSV become NaN, not dropped."""
    import csv
    csv_file = tmp_path / "test.csv"
    with open(csv_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "pm10"])
        w.writerow(["2020-01-01", 15.0])
        w.writerow(["2020-01-10", 25.0])  # 8 missing days between

    s = load_calendar_series(str(csv_file))
    assert len(s) == 10
    assert s.isna().sum() == 8
    assert float(s.iloc[0]) == pytest.approx(15.0)
    assert float(s.iloc[-1]) == pytest.approx(25.0)


# ── Test 3: Temporal compression is prohibited ──────────────────────────────────

def test_no_temporal_compression():
    """Dropping NaN before lag computation changes lag meaning — must NOT happen."""
    # Create series with gap: values at t=0,1, gap at t=2, value at t=3
    values = np.array([1.0, 2.0, np.nan, 4.0])
    s = _make_calendar_series(values)

    # lag-1 autocovariance using valid pairs on calendar (correct)
    gamma, n_pairs = _estimate_autocovariance(s, max_lag=2, min_pairs=1)

    # lag-1: pairs (t=0,t=1), (t=1,t=2=NaN), (t=2=NaN,t=3) → only (0,1) valid
    assert n_pairs[1] == 1  # only one valid pair for lag 1

    # lag-2: pairs (t=0,t=2=NaN), (t=1,t=3) → only (1,3) valid
    assert n_pairs[2] == 1


# ── Test 4: Valid pair counts per lag ──────────────────────────────────────────

def test_valid_pair_counts():
    """Verify valid pair counting against manual calculation."""
    values = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    s = _make_calendar_series(values)
    gamma, n_pairs = _estimate_autocovariance(s, max_lag=2, min_pairs=1)

    # lag 0: all 4 non-NaN values → 5 total but only non-NaN pairs: (0,0),(1,1),(3,3),(4,4) = 4
    assert n_pairs[0] == 4
    # lag 1: (0,1),(1,2=NaN),(2=NaN,3),(3,4) → valid: (0,1),(3,4) = 2
    assert n_pairs[1] == 2
    # lag 2: (0,2=NaN),(1,3),(2=NaN,4) → valid: (1,3) = 1
    assert n_pairs[2] == 1


# ── Test 5: Toeplitz matrix structure ──────────────────────────────────────────

def test_toeplitz_structure():
    """Γ_p must be symmetric Toeplitz."""
    gamma = np.array([10.0, 6.0, 3.0, 1.0])
    p = 3
    mat = _build_toeplitz(gamma, p)

    assert mat.shape == (p, p)
    # Symmetric
    np.testing.assert_allclose(mat, mat.T)
    # Toeplitz: all diagonals constant
    for k in range(p):
        diag_vals = np.diag(mat, k)
        assert np.allclose(diag_vals, diag_vals[0]), f"Diagonal {k} not constant"


# ── Test 6: Matrix symmetry ────────────────────────────────────────────────────

def test_matrix_symmetry():
    """Autocovariance matrix must be symmetric."""
    gamma = np.array([5.0, 3.0, 1.5, 0.5])
    mat = _build_toeplitz(gamma, 4)
    np.testing.assert_allclose(mat, mat.T, atol=1e-12)


# ── Test 7: Cross-covariance vector order ──────────────────────────────────────

def test_cross_covariance_vector():
    """c_h[i] = γ(h + i - 1) for i = 1,...,p."""
    gamma = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 1.0])
    p = 3
    h = 2
    c_h = _cross_covariance_vector(gamma, h, p)

    # c_h = [γ(h), γ(h+1), γ(h+2)] = [γ(2), γ(3), γ(4)]
    assert len(c_h) == p
    np.testing.assert_allclose(c_h, [gamma[2], gamma[3], gamma[4]])


# ── Test 8: AR(1) synthetic case ───────────────────────────────────────────────

def test_ar1_synthetic():
    """For AR(1) with known ρ, YW p=1 should recover optimal MSE ≈ γ(0)(1 - ρ^{2h})."""
    rho = 0.7
    n = 5000
    y = _ar1_series(n, rho, sigma=1.0, seed=0)
    s = _make_calendar_series(y)
    sigma2 = float(np.var(y, ddof=1))

    for h in [1, 2, 3]:
        res = compute_yw_reference(
            y_calendar=s,
            station="synthetic",
            horizon=h,
            p=1,
            persistence_mse=sigma2 * (1 + rho**2 - 2 * rho),  # approx for lag-1 persistence
            persistence_rmse=float(np.sqrt(sigma2 * (1 + rho**2 - 2 * rho))),
        )
        assert res.numerical_status in ("OK", "REGULARIZED"), f"h={h}: {res.numerical_status}"
        # Theoretical MSE for AR(1) p=1: γ(0)(1 - ρ^{2h})
        expected_mse = sigma2 * (1 - rho ** (2 * h))
        assert abs(res.yw_mse - expected_mse) / max(expected_mse, 1e-6) < 0.05, (
            f"h={h}: yw_mse={res.yw_mse:.4f}, expected≈{expected_mse:.4f}"
        )


# ── Test 9: White noise case ───────────────────────────────────────────────────

def test_white_noise():
    """For white noise (ρ=0), YW reference MSE ≈ γ(0) for all h."""
    rng = np.random.default_rng(7)
    y = rng.normal(0, 1, 2000)
    s = _make_calendar_series(y)
    sigma2 = float(np.var(y, ddof=1))

    res = compute_yw_reference(
        y_calendar=s,
        station="whitenoise",
        horizon=1,
        p=14,
        persistence_mse=sigma2,
        persistence_rmse=float(np.sqrt(sigma2)),
    )
    assert res.numerical_status in ("OK", "REGULARIZED")
    # YW should not improve beyond persistence for white noise
    assert res.yw_mse <= sigma2 * 1.1  # within 10%
    # yw_skill should be near 0 (or slightly negative due to estimation noise)
    assert res.yw_skill < 0.1


# ── Test 10: Missingness case ──────────────────────────────────────────────────

def test_missingness_case():
    """YW with 20% missing data should still produce a valid result."""
    rng = np.random.default_rng(42)
    n = 2000
    y = _ar1_series(n, 0.6, seed=1)
    miss_idx = rng.choice(n, size=n // 5, replace=False)
    y[miss_idx] = np.nan
    s = _make_calendar_series(y)

    # Persistence metrics approximation
    valid = y[~np.isnan(y)]
    sigma2 = float(np.var(valid, ddof=1))
    res = compute_yw_reference(
        y_calendar=s,
        station="missing20pct",
        horizon=1,
        p=5,
        persistence_mse=sigma2 * 2.0,
        persistence_rmse=float(np.sqrt(sigma2 * 2.0)),
    )
    assert res.numerical_status in ("OK", "REGULARIZED", "MISSING_AUTOCOVARIANCES")
    if res.numerical_status == "OK":
        assert res.yw_mse >= 0


# ── Test 11: MSE/RMSE consistency ─────────────────────────────────────────────

def test_mse_rmse_consistency():
    """yw_rmse == sqrt(yw_mse) and persistence_rmse == sqrt(persistence_mse)."""
    y = _ar1_series(1000, 0.5, seed=9)
    s = _make_calendar_series(y)
    sigma2 = float(np.var(y, ddof=1))

    pers_mse = sigma2 * 1.5
    pers_rmse = float(np.sqrt(pers_mse))
    res = compute_yw_reference(
        y_calendar=s,
        station="consistency",
        horizon=2,
        p=7,
        persistence_mse=pers_mse,
        persistence_rmse=pers_rmse,
    )

    if res.numerical_status == "OK" and not np.isnan(res.yw_mse):
        assert abs(res.yw_rmse - np.sqrt(res.yw_mse)) < 1e-10
        assert abs(res.persistence_rmse - np.sqrt(res.persistence_mse)) < 1e-10

    # Skill = 1 - yw_rmse / persistence_rmse
    if not np.isnan(res.yw_skill) and pers_rmse > 0:
        expected_skill = 1.0 - res.yw_rmse / pers_rmse
        assert abs(res.yw_skill - expected_skill) < 1e-10


# ── Test 12: Ill-conditioned matrix detection ─────────────────────────────────

def test_ill_conditioned_matrix():
    """Near-singular Γ_p yields high condition number and is flagged."""
    # Build a nearly degenerate matrix
    p = 4
    gamma = np.array([1.0, 0.9999, 0.9998, 0.9997, 0.9996, 0.9995])
    mat = _build_toeplitz(gamma, p)

    min_ev, max_ev, cond, psd_status = _check_psd(mat)
    assert cond > 1e3 or psd_status in ("PSD_MARGINAL", "NOT_PSD")


# ── Test 13: Best model selection (main models only) ─────────────────────────

def test_best_model_selection_main_only():
    """Best model is selected from hgb_direct, ridge_direct, sarima only."""
    from scripts.p2.run_p2_reference import build_model_selection_audit, MAIN_MODELS, REFERENCE_MODELS

    ref_rows = [
        {
            "station": "Test",
            "horizon": 1,
            "hgb_direct_skill": 0.10,
            "ridge_direct_skill": 0.15,
            "sarima_skill": 0.12,
            "seasonal_naive_skill": 0.20,  # highest but reference-only
            "stl_ridge_direct_skill": 0.25,  # highest but reference-only
            "best_model_skill": 0.15,
            "best_model_name": "ridge_direct",
            "skill_definition": "Skill_RMSE",
        }
    ]
    ref_df = pd.DataFrame(ref_rows)
    audit = build_model_selection_audit(ref_df)

    # stl_ridge_direct and seasonal_naive must not be eligible
    for _, row in audit.iterrows():
        if row["model"] in REFERENCE_MODELS:
            assert row["eligible_for_best"] is False
            assert row["is_best"] is False
        if row["model"] in MAIN_MODELS:
            assert row["eligible_for_best"] is True


# ── Test 14: stl_ridge_direct excluded from best-model ────────────────────────

def test_stl_ridge_direct_excluded():
    """stl_ridge_direct must never appear as best_model_name."""
    from scripts.p2.run_p2_reference import build_reference_table

    # Construct fake empirical_skills where stl_ridge_direct has highest skill
    emp_rows = []
    for station_key in ["elche", "valencia_vivers", "zarra_emep"]:
        label = {"elche": "Elche", "valencia_vivers": "Valencia Vivers", "zarra_emep": "Zarra EMEP"}[station_key]
        for h in range(1, 8):
            for model, skill in [
                ("hgb_direct", 0.10),
                ("ridge_direct", 0.12),
                ("sarima", 0.08),
                ("seasonal_naive", 0.30),  # high
                ("stl_ridge_direct", 0.50),  # highest — must be excluded
            ]:
                emp_rows.append({"station": station_key, "model": model, "horizon": h, "skill": skill})
    emp_df = pd.DataFrame(emp_rows)

    # Build dummy all_rows
    all_rows = []
    for station_key in ["elche", "valencia_vivers", "zarra_emep"]:
        label = {"elche": "Elche", "valencia_vivers": "Valencia Vivers", "zarra_emep": "Zarra EMEP"}[station_key]
        for h in range(1, 8):
            all_rows.append({
                "station": label,
                "station_key": station_key,
                "horizon": h,
                "p": 14,
                "rho1": 0.5,
                "rho_h": 0.3,
                "ar1_reference_skill": 0.09,
                "ar1_status": "DIAGNOSTIC_ONLY",
                "yw_linear_reference_skill": 0.11,
                "yw_mse": 90.0,
                "yw_rmse": 9.5,
                "gamma0": 100.0,
                "persistence_mse": 100.0,
                "persistence_rmse": 10.0,
                "valid_pair_count_rho_h": 1500,
                "min_valid_pair_count_required_lags": 1200,
                "covariance_min_eigenvalue": 0.5,
                "covariance_max_eigenvalue": 50.0,
                "covariance_condition_number": 100.0,
                "covariance_psd_status": "PSD",
                "regularization_applied": 0.0,
                "numerical_status": "OK",
                "rho1_control_value": 0.5,
                "rho1_diff_from_control": 0.0,
                "notes": "",
            })

    ref_df = build_reference_table(all_rows, emp_df, "Skill_RMSE")

    # stl_ridge_direct must never be best_model_name
    assert "stl_ridge_direct" not in ref_df["best_model_name"].values
    assert "seasonal_naive" not in ref_df["best_model_name"].values
    # Best should be ridge_direct (0.12, highest among main models)
    assert (ref_df["best_model_name"] == "ridge_direct").all()


# ── Test 15: Deterministic reproducibility ────────────────────────────────────

def test_deterministic_reproducibility():
    """Same input yields identical YW results on two runs."""
    y = _ar1_series(800, 0.55, seed=3)
    s = _make_calendar_series(y)
    sigma2 = float(np.var(y, ddof=1))

    res1 = compute_yw_reference(s, "repro", 3, 5, sigma2 * 2, float(np.sqrt(sigma2 * 2)))
    res2 = compute_yw_reference(s, "repro", 3, 5, sigma2 * 2, float(np.sqrt(sigma2 * 2)))

    assert res1.yw_mse == res2.yw_mse
    assert res1.yw_rmse == res2.yw_rmse
    assert res1.yw_skill == res2.yw_skill
    np.testing.assert_array_equal(res1.autocovariances, res2.autocovariances)
    np.testing.assert_array_equal(res1.valid_pair_counts, res2.valid_pair_counts)
