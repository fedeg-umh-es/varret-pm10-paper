"""Build a reproducible E1-RR variance-retention markdown report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PREDICTIONS_PATH = Path("outputs/metrics/predictions.csv")
SUMMARY_PATH = Path("outputs/tables/variance_retention_summary.csv")
REPORT_PATH = Path("outputs/reports/e1_rr_variance_retention_report.md")


def _format_model_list(models: list[str]) -> str:
    return ", ".join(models)


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _build_model_table(summary_df: pd.DataFrame) -> str:
    rows: list[str] = []
    header = (
        "| model | skill mean | skill min | skill max | alpha mean | alpha min | alpha max | "
        "skill_vp mean | skill_vp min | skill_vp max | collapse_flag count | inflation_flag count | "
        "near_ideal_flag count | low_sample_flag count |"
    )
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    rows.extend([header, sep])

    grouped = summary_df.groupby("model", sort=True)
    for model, group in grouped:
        total = len(group)
        rows.append(
            "| "
            + " | ".join(
                [
                    str(model),
                    _fmt(group["skill"].mean()),
                    _fmt(group["skill"].min()),
                    _fmt(group["skill"].max()),
                    _fmt(group["alpha"].mean()),
                    _fmt(group["alpha"].min()),
                    _fmt(group["alpha"].max()),
                    _fmt(group["skill_vp"].mean()),
                    _fmt(group["skill_vp"].min()),
                    _fmt(group["skill_vp"].max()),
                    f"{int(group['collapse_flag'].sum())}/{total}",
                    f"{int(group['inflation_flag'].sum())}/{total}",
                    f"{int(group['near_ideal_flag'].sum())}/{total}",
                    f"{int(group['low_sample_flag'].sum())}/{total}",
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _build_diagnostics(summary_df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    eval_models = [m for m in sorted(summary_df["model"].unique()) if m != "persistence"]
    if eval_models:
        positive_models = []
        for model in eval_models:
            model_skill_mean = float(summary_df.loc[summary_df["model"] == model, "skill"].mean())
            if model_skill_mean > 0:
                positive_models.append(model)
        if len(positive_models) == len(eval_models):
            lines.append(
                f"- Ambos modelos ({', '.join(eval_models)}) muestran **skill positivo** frente a persistencia."
            )

    low_alpha_models = []
    for model, group in summary_df.groupby("model"):
        if model == "persistence":
            continue
        if float(group["alpha"].mean()) < 0.5:
            low_alpha_models.append(model)
    if low_alpha_models:
        lines.append(
            f"- Los modelos {', '.join(low_alpha_models)} muestran **baja retención de varianza** (alpha medio < 0.5)."
        )

    for model, group in summary_df.groupby("model", sort=True):
        if model == "persistence":
            continue
        collapse = int(group["collapse_flag"].sum())
        total = len(group)
        lines.append(f"- `{model}` presenta **collapse** en **{collapse}/{total}** horizontes.")

    inflation_total = int(summary_df["inflation_flag"].sum())
    near_ideal_total = int(summary_df["near_ideal_flag"].sum())
    if inflation_total > 0:
        lines.append(f"- Se observaron casos de **inflation**: {inflation_total} filas con `inflation_flag=True`.")
    else:
        lines.append("- No hay evidencia de **inflation** (`inflation_flag=0`).")

    if near_ideal_total > 0:
        lines.append(f"- Se observaron casos **near-ideal**: {near_ideal_total} filas con `near_ideal_flag=True`.")
    else:
        lines.append("- No hay evidencia de comportamiento **near-ideal** (`near_ideal_flag=0`).")

    vp_gap = (summary_df["skill"] - summary_df["skill_vp"]).mean()
    if vp_gap > 0:
        lines.append(
            "- `Skill_VP` queda reducido respecto al skill bruto cuando `alpha` es bajo, consistente con pronósticos suavizados."
        )

    return lines


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions table: {PREDICTIONS_PATH}")
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing variance retention table: {SUMMARY_PATH}")

    predictions_df = pd.read_csv(PREDICTIONS_PATH)
    summary_df = pd.read_csv(SUMMARY_PATH)

    n_rows = len(predictions_df)
    models = sorted(predictions_df["model"].astype(str).unique().tolist())
    min_h = int(predictions_df["horizon"].min())
    max_h = int(predictions_df["horizon"].max())
    n_min = int(summary_df.groupby(["model", "horizon"], dropna=False)["n"].min().min())
    low_sample_count = int(summary_df["low_sample_flag"].sum())

    diagnostics = "\n".join(_build_diagnostics(summary_df))
    table = _build_model_table(summary_df)

    content = f"""# E1-RR Variance Retention Report (Daily Lags-Only)

## 1) Alcance

Este informe se limita estrictamente al paquete de trabajo de post-evaluación E1-RR:

- E1-RR daily lags-only.
- PM10 diario.
- Horizontes `h = 1..7`.
- Evaluación rolling-origin.
- Baseline obligatorio de persistencia.
- Sin variables meteorológicas.
- Sin E2-MET.
- Sin E3-PROB.

## 2) Validación de muestra

- Filas en `outputs/metrics/predictions.csv`: **{n_rows}**.
- Modelos presentes: **{_format_model_list(models)}**.
- Horizontes presentes: **{min_h}..{max_h}**.
- `n` mínimo por combinación `model/horizon`: **{n_min}**.
- Conteo de `low_sample_flag=True` en `variance_retention_summary.csv`: **{low_sample_count}**.

## 3) Tabla resumen por modelo

{table}

## 4) Diagnóstico textual

{diagnostics}

## 5) Guardrails de redacción

- No afirmar ausencia total de predictibilidad solo por bajo `alpha`.
- No afirmar invalidez global del modelo únicamente por `collapse_flag`.
- No presentar `Skill_VP` como métrica universal de desempeño.
- No mezclar esta lectura con E2-MET ni E3-PROB.
- Mantener el encuadre como **diagnóstico auxiliar de post-evaluación**.

## 6) Frase paper-ready

“Positive persistence-relative skill was observed across horizons, but the accompanying variance-retention diagnostics indicate that this skill was largely associated with smoothed forecasts rather than near-observed dynamic variability.”
"""

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(content, encoding="utf-8")
    print(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
