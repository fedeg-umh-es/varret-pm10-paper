# Manuscript Patch - Real E1-RR Daily PM10 Run

Use this patch to replace stale placeholder or synthetic-run text in the
manuscript PDF/Overleaf source.

## Section 2 - SLR Audit Text

Use one consolidated paragraph. Delete any duplicated sentence that reads like
`records were retrieved from using , ...`.

```latex
To quantify how visible these evaluation conditions are in the particulate matter forecasting literature, we conducted a PRISMA 2020 audit of PM$_{10}$/PM$_{2.5}$ forecasting studies. Records were retrieved from Scopus and the Web of Science Core Collection (peer-reviewed journal articles in English, 2015--2025). After deduplication ($n=807$), title/abstract screening, and full-text eligibility assessment, 503 studies were included in the final corpus. Abstract-level evidence was successfully retrieved for 486 of the 503 included studies (96.6\%); the remaining 17 were retained in the corpus but excluded from abstract-level frequency estimation. Each eligible study was coded for the explicit, abstract-level presence of five evaluation-reporting dimensions. A practice was counted as present only when explicitly stated in the abstract; absence of explicit reporting was treated as a substantive audit outcome, not as missing data. Reproducible materials are archived at Zenodo (concept DOI: 10.5281/zenodo.18675211; release: 10.5281/zenodo.19125915).
```

## Section 4 - Dataset And Gap Handling

Replace the station/source and gap-handling placeholders with:

```latex
The case study uses daily PM$_{10}$ observations from the Elx-Agroalimentari monitoring station in Spain, obtained from the RVVCCA public air-quality monitoring network. The cleaned record covers 2017-01-01 to 2024-12-31 and contains 2,350 observed daily PM$_{10}$ values, together with meteorological covariates including temperature, relative humidity, wind speed, and wind direction. The meteorological covariates are retained in the raw data for provenance but deliberately excluded from the E1-RR experiment: the design is intentionally lags-only to isolate autoregressive information, without mixing the present diagnostic with meteorological forcing or probabilistic uncertainty.

The daily series was reindexed to a complete daily calendar before lag construction. Dates with missing PM$_{10}$ values, missing lagged predictors, or unavailable verification targets after horizon alignment were excluded by listwise deletion. No temporal interpolation, forward filling, backward filling, or other imputation was applied.
```

## Section 6 - Reproducible Workflow

Replace stale row counts with:

```latex
The post-evaluation consumes a table of forecasts and verification values (`predictions.csv`; 26,001 rows) spanning horizons $h=1,\ldots,7$ for `ridge_direct`, `hgb_direct`, and the mandatory persistence baseline. The minimum number of verification samples per model/horizon diagnostic cell is 1,215, and no low-sample flags are triggered.
```

## Results - Summary Table

Use the regenerated table:

```latex
\begin{table}[htbp]
\centering
\caption{Summary of persistence-relative skill and variance-retention diagnostics for the E1-RR daily lags-only post-evaluation.}
\label{tab:e1_rr_variance_retention_summary}
\begin{tabular}{lrrrrrrr}
\toprule
Model & Mean skill & Mean alpha & Mean Skill\_VP & Collapse horizons & Inflation horizons & Near-ideal horizons & Low-sample horizons \\
\midrule
hgb\_direct & 0.212 & 0.145 & 0.027 & 7/7 & 0/7 & 0/7 & 0/7 \\
ridge\_direct & 0.242 & 0.072 & 0.013 & 7/7 & 0/7 & 0/7 & 0/7 \\
\bottomrule
\end{tabular}
\end{table}
```

## Results - Replacement Narrative

```latex
Aggregating across horizons, `hgb_direct` shows a mean persistence-relative skill of 0.212 with a mean $\alpha=0.145$ and a mean Skill$_{VP}=0.027$. The diagnostic categorisation indicates collapse in 7/7 horizons and no inflation, near-ideal, or low-sample horizons.

Similarly, `ridge_direct` shows a mean persistence-relative skill of 0.242, with mean $\alpha=0.072$ and mean Skill$_{VP}=0.013$. All horizons are categorised as collapse (7/7), with no inflation, near-ideal, or low-sample horizons.

The central diagnostic result is that `ridge_direct` has higher mean persistence-relative skill, while `hgb_direct` retains more forecast variance. Thus, the ranking implied by average RMSE-based improvement differs from the ranking implied by retained forecast variability. In both models, positive skill co-occurs with substantial variance shrinkage across all evaluated horizons.
```

## Do Not Use

These values came from the synthetic placeholder run and should not appear in
the final manuscript:

```text
2323 rows
2017-01-01 to 2023-05-12
predictions.csv = 23016 rows
minimum n = 1096
hgb_direct mean alpha = 0.287
ridge_direct mean alpha = 0.207
```
