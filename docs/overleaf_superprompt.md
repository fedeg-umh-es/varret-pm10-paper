# Overleaf Superprompt — Real Data Update
## varret-pm10-paper · E1-RR Daily PM10 · Generated 2026-05-23

Paste the block below directly into the Overleaf AI assistant (or use as
a structured editing guide). Each CAMBIO is an independent, self-contained
replacement. Apply them in order; each one references the exact stale text
to replace and the exact new text to insert.

---

```
Eres un asistente de edición de manuscritos académicos en LaTeX (Overleaf).
Tu tarea es aplicar EXACTAMENTE los cambios descritos a continuación al
fichero fuente del artículo. No modifiques nada que no esté explícitamente
indicado. No reformules frases que no contengan valores desactualizados.

Los cambios se basan en la ejecución real del pipeline E1-RR con los
datos auténticos de la estación Elx-Agroalimentari (RVVCCA), 2017-2024.
```

---

## CAMBIO 1 — Abstract: rango cuantitativo de resultados

**Localiza** en el abstract cualquier frase que mencione valores de skill,
alpha o número de filas de predicciones. Reemplaza esos valores con:

```latex
Both evaluated models achieve positive persistence-relative skill across all
seven forecast horizons (skill range: 0.055--0.280), while simultaneously
collapsing forecast variance relative to observed variability
($\alpha$ range: 0.023--0.298). Variance-collapse diagnostic flags are
triggered at all seven horizons for both models, with no inflation,
near-ideal, or low-sample flags detected.
```

---

## CAMBIO 2 — §2 (Literature context): párrafo PRISMA completo

**Elimina** cualquier párrafo del §2 que contenga texto roto del tipo
`records were retrieved from using ,` o que tenga campos vacíos.
**Sustituye** por este párrafo único y coherente:

```latex
To quantify how visible these evaluation conditions are in the particulate
matter forecasting literature, we conducted a PRISMA~2020 audit of
PM$_{10}$/PM$_{2.5}$ forecasting studies. Records were retrieved from
Scopus and the Web of Science Core Collection (peer-reviewed journal
articles in English, 2015--2025) using the Boolean query:

\begin{verbatim}
%%BOOLEAN_QUERY_PLACEHOLDER%%
\end{verbatim}

After deduplication ($n = 807$), title/abstract screening, and full-text
eligibility assessment, 503 studies were included in the final corpus.
Abstract-level evidence was successfully retrieved for 486 of the
503 included studies (96.6\%); the remaining 17 were retained in the
corpus but excluded from abstract-level frequency estimation. Each
eligible study was coded for the explicit, abstract-level presence of
five evaluation-reporting dimensions. A practice was counted as present
only when explicitly stated in the abstract; absence of explicit reporting
was treated as a substantive audit outcome, not as missing data.
Reproducible materials are archived at Zenodo
(concept DOI: \href{https://doi.org/10.5281/zenodo.18675211}{10.5281/zenodo.18675211};
release: \href{https://doi.org/10.5281/zenodo.19125915}{10.5281/zenodo.19125915}).
```

> **NOTA**: Reemplaza `%%BOOLEAN_QUERY_PLACEHOLDER%%` con la cadena
> booleana real de Scopus v6 (ver `search_export/` en el repositorio SLR).
> Ejemplo de estructura esperada:
> `TITLE-ABS-KEY ( ( "PM10" OR "PM 10" OR "particulate matter" ) AND ( "forecast*" OR "predict*" ) ) AND PUBYEAR > 2014 AND PUBYEAR < 2026 AND LANGUAGE ( english ) AND DOCTYPE ( ar )`

---

## CAMBIO 3 — §4 (Dataset): estación y gestión de huecos

**Localiza** el párrafo del §4 que describe la fuente de datos
(probablemente menciona una estación placeholder o una ruta de archivo).
**Reemplaza** ese bloque con:

```latex
The case study uses daily PM$_{10}$ observations from the
Elx-Agroalimentari monitoring station in Spain, obtained from the RVVCCA
(\textit{Red Valenciana de Vigilancia i Control de la Contaminació
Atmosfèrica}) public air-quality monitoring network. The cleaned record
covers 2017-01-01 to 2024-12-31 and contains 2,350 observed daily
PM$_{10}$ values, together with meteorological covariates including
temperature, relative humidity, wind speed, and wind direction. The
meteorological covariates are retained in the raw data for provenance
but deliberately excluded from the E1-RR experiment: the design is
intentionally lags-only to isolate autoregressive information, without
mixing the present diagnostic with meteorological forcing or probabilistic
uncertainty.

The raw daily calendar spans 2,922 dates; 572 dates (19.6\%) have no
observed PM$_{10}$ value. The daily series was reindexed to a complete
daily calendar before lag construction. Dates with missing PM$_{10}$
values, missing lagged predictors, or unavailable verification targets
after horizon alignment were excluded by listwise deletion. No temporal
interpolation, forward filling, backward filling, or other imputation
was applied.
```

---

## CAMBIO 4 — §6 (Reproducible workflow): conteo de filas

**Localiza** la frase del §6 que menciona `predictions.csv` con un número
de filas (probablemente 23,016 o 5,850). **Reemplaza** ese segmento con:

```latex
The post-evaluation consumes a table of forecasts and verification values
(\texttt{predictions.csv}; 26,001 rows) spanning horizons $h = 1, \ldots, 7$
for \texttt{ridge\_direct}, \texttt{hgb\_direct}, and the mandatory
persistence baseline. The minimum number of verification samples per
model/horizon diagnostic cell is 1,215, and no low-sample flags are
triggered.
```

---

## CAMBIO 5 — §7 (Results, párrafo inicial): actualizar cifras

**Localiza** el primer párrafo del §7 que mencione el número total de filas
o el mínimo de muestras por celda diagnóstica. **Reemplaza** las cifras
desactualizadas con los valores reales:

- `23,016` → `26,001` (o `5,850` → `26,001`)
- `1,096` → `1,215`
- rango de fechas de verificación: `2020-01-23` a `2024-12-31`

Texto de reemplazo para el párrafo completo:

```latex
The rolling-origin evaluation yields 26,001 forecast–verification pairs
spanning $h = 1, \ldots, 7$ across three models (\texttt{hgb\_direct},
\texttt{ridge\_direct}, and persistence). Verification dates run from
2020-01-23 to 2024-12-31. The minimum number of verification samples
in any model/horizon cell is 1,215, and no cell triggers the low-sample
diagnostic flag ($n < 30$).
```

---

## CAMBIO 6 — Tabla 2 (Summary table): valores reales por modelo

**Localiza** la tabla de resumen (probablemente \texttt{tab:e1\_rr} o
\texttt{tab:skill\_summary}) con valores de skill/alpha por modelo.
**Reemplaza** las filas de datos con:

```latex
\begin{table}[htbp]
\centering
\caption{Summary of persistence-relative skill and variance-retention
diagnostics for the E1-RR daily lags-only post-evaluation
(horizons $h = 1, \ldots, 7$; real data, 2017--2024).}
\label{tab:e1_rr_variance_retention_summary}
\begin{tabular}{lrrrrrrr}
\toprule
Model & $\overline{\text{Skill}}$ & $\overline{\alpha}$ &
$\overline{\text{Skill}_{VP}}$ &
Collapse & Inflation & Near-ideal & Low-sample \\
\midrule
\texttt{hgb\_direct}   & 0.212 & 0.145 & 0.027 & 7/7 & 0/7 & 0/7 & 0/7 \\
\texttt{ridge\_direct} & 0.242 & 0.072 & 0.013 & 7/7 & 0/7 & 0/7 & 0/7 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## CAMBIO 7 — Tabla 3 (nueva): diagnóstico por horizonte

**Inserta** esta tabla completa inmediatamente después de la Tabla~2.
Si ya existe una tabla horizonte-a-horizonte con valores sintéticos,
reemplázala completamente.

```latex
\begin{table}[htbp]
\centering
\caption{Horizon-by-horizon variance-retention diagnostics.
Flag codes: C = collapse ($\alpha < 0.5$); -- = none triggered.
$n$ = number of forecast--verification pairs per cell.}
\label{tab:e1_rr_by_horizon}
\begin{tabular}{llrrrrr}
\toprule
Model & $h$ & $n$ & Skill & $\alpha$ & Skill$_{VP}$ & Flag \\
\midrule
\texttt{hgb\_direct}   & 1 & 1274 & 0.055 & 0.298 & 0.016 & C \\
\texttt{hgb\_direct}   & 2 & 1251 & 0.207 & 0.164 & 0.034 & C \\
\texttt{hgb\_direct}   & 3 & 1241 & 0.236 & 0.130 & 0.031 & C \\
\texttt{hgb\_direct}   & 4 & 1234 & 0.242 & 0.119 & 0.029 & C \\
\texttt{hgb\_direct}   & 5 & 1228 & 0.254 & 0.104 & 0.026 & C \\
\texttt{hgb\_direct}   & 6 & 1224 & 0.251 & 0.101 & 0.025 & C \\
\texttt{hgb\_direct}   & 7 & 1215 & 0.238 & 0.102 & 0.024 & C \\
\midrule
\texttt{ridge\_direct} & 1 & 1274 & 0.107 & 0.265 & 0.028 & C \\
\texttt{ridge\_direct} & 2 & 1251 & 0.226 & 0.080 & 0.018 & C \\
\texttt{ridge\_direct} & 3 & 1241 & 0.258 & 0.037 & 0.010 & C \\
\texttt{ridge\_direct} & 4 & 1234 & 0.272 & 0.023 & 0.006 & C \\
\texttt{ridge\_direct} & 5 & 1228 & 0.278 & 0.029 & 0.008 & C \\
\texttt{ridge\_direct} & 6 & 1224 & 0.280 & 0.037 & 0.010 & C \\
\texttt{ridge\_direct} & 7 & 1215 & 0.272 & 0.033 & 0.009 & C \\
\bottomrule
\end{tabular}
\end{table}
```

---

## CAMBIO 8 — §7 (Results, narrativa): actualizar texto interpretativo

**Localiza** el párrafo del §7 que interpreta los resultados por modelo.
**Reemplaza** con la narrativa real completa:

```latex
Table~\ref{tab:e1_rr_variance_retention_summary} summarises the diagnostic
outcomes. \texttt{hgb\_direct} achieves a mean persistence-relative skill
of 0.212 (range: 0.055--0.254 across horizons) with a mean
$\alpha = 0.145$ (range: 0.101--0.298) and a mean
Skill$_{VP} = 0.027$. \texttt{ridge\_direct} achieves a higher mean
skill of 0.242 (range: 0.107--0.280) but a substantially lower mean
$\alpha = 0.072$ (range: 0.023--0.265) and mean
Skill$_{VP} = 0.013$. Both models trigger collapse flags at all seven
evaluated horizons, with no inflation, near-ideal, or low-sample flags
detected.

The horizon-level detail (Table~\ref{tab:e1_rr_by_horizon}) reveals a
consistent pattern: skill improves with forecast horizon for both models,
while $\alpha$ decreases. For \texttt{hgb\_direct}, skill rises from
0.055 at $h=1$ to a peak of 0.254 at $h=5$, then declines slightly
for $h = 6, 7$. For \texttt{ridge\_direct}, skill rises from 0.107 at
$h=1$ to 0.280 at $h=6$ before a minor retreat. Both profiles are
consistent with the well-known difficulty of very short-horizon PM$_{10}$
forecasting under high day-to-day variability.

The variance-retention ratio $\alpha$ shows a sharper contrast between
models. \texttt{hgb\_direct} retains roughly 10--30\% of observed
variance at all horizons. \texttt{ridge\_direct} retains 26.5\% at
$h=1$ but collapses to near-zero retention (2.3\%) at $h=4$ before
recovering slightly. This pattern is consistent with the regularisation
bias of Ridge regression: the $\ell_2$ penalty shrinks coefficients
aggressively when the autoregressive signal weakens with horizon, driving
forecasts toward the training mean and suppressing variance almost
entirely.

The central diagnostic finding is that \texttt{ridge\_direct} ranks
higher than \texttt{hgb\_direct} on mean RMSE-based skill, but lower
on variance retention. Both orderings are operationally meaningful and
neither alone is sufficient: RMSE skill without $\alpha$ conceals the
degree to which the model has sacrificed discriminative variability for
average accuracy. The joint reporting of Skill and $\alpha$ — and their
product Skill$_{VP}$ — is precisely the diagnostic contribution this
framework offers.
```

---

## CAMBIO 9 — §8.1 (Discussion): añadir párrafo sobre colapso en h=1

**Localiza** la subsección §8.1 (o la primera subsección de Discussion).
**Inserta** este párrafo antes del primer párrafo existente, o al inicio
de la discusión si no hay subsección explícita:

```latex
A noteworthy feature of the diagnostic profile is that variance collapse
is present even at $h = 1$, the shortest forecast horizon. At $h = 1$,
\texttt{hgb\_direct} retains $\alpha = 0.298$ and
\texttt{ridge\_direct} retains $\alpha = 0.265$ — both well below the
collapse threshold of 0.5. This result contrasts with the intuitive
expectation that very short-horizon forecasts should closely track
observations and therefore inherit near-full variance. The finding
suggests that collapse is a structural property of lags-only linear and
tree-based models on this series, not an artefact of multi-step
accumulation. It reinforces the motivation for reporting $\alpha$
alongside Skill at every evaluated horizon rather than relying on a
single aggregated summary.
```

---

## CAMBIO 10 — Caption de Figura 2: actualizar rangos

**Localiza** el \caption de la figura que muestra el diagnóstico
skill–alpha (probablemente `figure2_skill_variance_retention`).
**Actualiza** cualquier valor numérico en el caption con los rangos reales:

```latex
\caption{Skill--variance-retention diagnostic for the E1-RR daily
lags-only post-evaluation. Each point represents one model/horizon
combination ($h = 1, \ldots, 7$; $n = 1{,}215$--$1{,}274$ pairs per
cell). Both models achieve positive persistence-relative skill
(range: 0.055--0.280) while collapsing forecast variance relative to
observed variability ($\alpha$ range: 0.023--0.298). The collapse
region ($\alpha < 0.5$) is shaded. No points fall in the inflation or
near-ideal zones. \texttt{ridge\_direct} dominates on skill but shows
more severe variance collapse at mid-to-long horizons.}
```

---

## CAMBIO 11 — Code Availability / Data Availability: actualizar filas

**Localiza** la sección de disponibilidad de código o datos que menciona
`predictions.csv` con un número de filas. **Actualiza** las cifras:

```latex
\texttt{predictions.csv} (26,001 rows; models: \texttt{hgb\_direct},
\texttt{ridge\_direct}, persistence; horizons $h = 1, \ldots, 7$;
verification dates 2020-01-23 to 2024-12-31).
```

Si hay una fila en una tabla de artefactos, reemplaza el valor de la
columna de filas de `23,016` (o `5,850`) a `26,001`.

---

## Resumen de valores clave para referencia rápida

| Artefacto | Valor desactualizado | Valor real |
|-----------|---------------------|------------|
| Filas en dataset | 2,323 | 2,350 |
| Rango de fechas | 2017-01-01 a 2023-05-12 | 2017-01-01 a 2024-12-31 |
| Filas en predictions.csv | 23,016 | 26,001 |
| Mínimo n por celda | 1,096 | 1,215 |
| hgb mean Skill | (sintético) | 0.212 |
| hgb mean α | 0.287 | 0.145 |
| hgb mean Skill_VP | (sintético) | 0.027 |
| ridge mean Skill | (sintético) | 0.242 |
| ridge mean α | 0.207 | 0.072 |
| ridge mean Skill_VP | (sintético) | 0.013 |
| Colapso hgb | (sintético) | 7/7 |
| Colapso ridge | (sintético) | 7/7 |
| Estación | placeholder | Elx-Agroalimentari, RVVCCA |
| PRISMA estudios incluidos | (vacío) | 503 |
| PRISMA abstract-coded | (vacío) | 486 / 503 (96.6%) |
