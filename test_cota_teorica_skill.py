#!/usr/bin/env python3
"""Test empirico de la cota teorica de skill vs persistencia.

Para una serie estacionaria, el skill maximo alcanzable frente a
persistencia a horizonte h depende solo de la ACF, no del modelo.
Bajo AR(1) con phi=rho(1):  Skill_max(h) = (1 - phi^h) / 2, que
crece con h y tiende a 0.5. Cota general con rho(h) empirico:
Skill_max(h) = 1 - (1 - phi^(2h)) / (2 (1 - rho(h))).
Compara skill_emp(h) de las predicciones con ambas cotas.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c in low:
            return low[c]
    return None

def acf_empirica(y, hmax):
    y = y[~np.isnan(y)]
    y = y - y.mean()
    denom = np.dot(y, y)
    out = np.empty(hmax + 1)
    out[0] = 1.0
    for h in range(1, hmax + 1):
        out[h] = np.dot(y[:-h], y[h:]) / denom
    return out

def cota_ar1(phi, hmax):
    h = np.arange(1, hmax + 1)
    return (1.0 - phi ** h) / 2.0

def cota_acf(rho, phi, hmax):
    h = np.arange(1, hmax + 1)
    num = 1.0 - phi ** (2 * h)
    den = 2.0 * (1.0 - rho[1:hmax + 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - num / den

def skill_empirico(pred):
    c_h = pick(pred.columns, "h", "horizon", "lead", "step")
    c_yt = pick(pred.columns, "y_true", "true", "target", "observed", "y", "actual")
    c_yp = pick(pred.columns, "y_pred", "pred", "prediction", "forecast", "yhat")
    c_bl = pick(pred.columns, "persistence", "baseline_pred", "baseline",
                "naive", "y_pred_persistence")
    c_m = pick(pred.columns, "model", "model_name", "algorithm")
    miss = [n for n, v in [("h", c_h), ("y_true", c_yt),
                           ("y_pred", c_yp), ("baseline", c_bl)] if v is None]
    if miss:
        raise SystemExit(f"Faltan columnas {miss}. Hay: {list(pred.columns)}")
    pred = pred.copy()
    pred["_sm"] = (pred[c_yt] - pred[c_yp]) ** 2
    pred["_sb"] = (pred[c_yt] - pred[c_bl]) ** 2
    keys = [c_h] + ([c_m] if c_m else [])
    g = pred.groupby(keys).agg(mse_model=("_sm", "mean"),
                               mse_base=("_sb", "mean")).reset_index()
    g["skill_emp"] = 1.0 - g["mse_model"] / g["mse_base"]
    return g.rename(columns={c_h: "h", c_m: "model"} if c_m else {c_h: "h"})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serie", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--nombre", default="serie")
    a = ap.parse_args()
    raw = pd.read_csv(a.serie)
    c_val = pick(raw.columns, "y", "target", "pm10", "value", "obs")
    if c_val is None:
        num = raw.select_dtypes("number").columns
        c_val = num[-1] if len(num) else raw.columns[-1]
    y = raw[c_val].to_numpy(dtype=float)
    emp = skill_empirico(pd.read_csv(a.pred))
    hmax = int(emp["h"].max())
    rho = acf_empirica(y, hmax)
    phi = float(rho[1])
    b_ar1 = cota_ar1(phi, hmax)
    b_acf = cota_acf(rho, phi, hmax)
    print(f"\n=== {a.nombre} ===  n={int(np.sum(~np.isnan(y)))}  "
          f"phi=rho(1)={phi:.3f}  Hmax={hmax}")
    print(f"{'h':>2} {'skill_AR1':>10} {'skill_ACF':>10} {'skill_emp':>10} {'gap':>8}")
    per_h = emp.groupby("h")["skill_emp"].mean()
    for h in range(1, hmax + 1):
        se = per_h.get(h, np.nan)
        gap = se - b_ar1[h - 1]
        print(f"{h:>2} {b_ar1[h-1]:>10.3f} {b_acf[h-1]:>10.3f} "
              f"{se:>10.3f} {gap:>+8.3f}")
    crece = b_ar1[-1] > b_ar1[0]
    cerca = np.nanmean(np.abs(per_h.reindex(range(1, hmax + 1)).to_numpy()
                              - b_ar1)) < 0.10
    print(f"\nCota crece con h: {crece}   |   "
          f"skill emp pegado a la cota (<0.10 medio): {cerca}")
    if crece and cerca:
        print("VEREDICTO: compatible — el skill sigue la ACF, no el modelo.")
    else:
        print("VEREDICTO: revisar — serie no AR(1) pura (estacionalidad / "
              "no estacionariedad). Mirar skill_ACF.")

if __name__ == "__main__":
    main()
