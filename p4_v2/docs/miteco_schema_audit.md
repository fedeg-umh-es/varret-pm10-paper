# MITECO schema audit — P4-v2 Phase 0

Generated: `2026-08-16T08:19:35Z`

## Source

Official dataset: [Datos oficiales de calidad del aire](https://catalogo.datosabiertos.miteco.gob.es/catalogo/dataset/19458583-9953-4fe7-a494-e2cc26e89e58)
Format specification: [https://www.miteco.gob.es/content/dam/miteco/es/calidad-y-evaluacion-ambiental/temas/atmosfera-y-calidad-del-aire/formatodatoscsvmiteco_web_tcm30-501420.pdf](https://www.miteco.gob.es/content/dam/miteco/es/calidad-y-evaluacion-ambiental/temas/atmosfera-y-calidad-del-aire/formatodatoscsvmiteco_web_tcm30-501420.pdf)

The four annual PM10 files were extracted from official MITECO ZIP packages linked by the 2020, 2021, 2022 and 2023 official-data pages. No alternative dataset was used.

## Observed source schema

All four PM10 files had the same 32-column header:

```text
PROVINCIA;MUNICIPIO;ESTACION;MAGNITUD;PUNTO_MUESTREO;ANNO;MES;DIA;H01;H02;H03;H04;H05;H06;H07;H08;H09;H10;H11;H12;H13;H14;H15;H16;H17;H18;H19;H20;H21;H22;H23;H24
```

| Field group | Observed fields | Interpretation supported by source |
|---|---|---|
| Station identity | `PROVINCIA`, `MUNICIPIO`, `ESTACION`, `PUNTO_MUESTREO` | Preserved exactly; Phase 0 uses the original `PUNTO_MUESTREO` as `station_id` and retains the other identifiers. The union contains 415 distinct PM10 sampling-point IDs and 411 distinct province–municipality–`ESTACION` codes; four station codes expose two distinct PM10 sampling points, so they are not merged. |
| Pollutant | `MAGNITUD` | All extracted rows have `MAGNITUD=10`, the PM10 file supplied by MITECO. |
| Calendar fields | `ANNO`, `MES`, `DIA` | Source day label; no undocumented date repair was applied. |
| Hourly observations | `H01` … `H24` | Numeric hourly values; the official format document states that hourly reference is UTC and hour 1 is the period ending 01:00, through hour 24 ending 24:00. |
| Quality flags | No separate quality/validation columns observed in the extracted PM10 files | Empty hourly cells are retained as missing; no per-observation quality flag is fabricated. |
| Units | No unit column in the PM10 files | Unit metadata must be taken from MITECO pollutant metadata; this phase does not infer a unit from values. |
| Network/geography | No network, latitude or longitude columns in these files | Province, municipality and station codes are retained; network/geographic enrichment is not inferred. |

## Encoding and missingness

- Delimiter observed: semicolon (`;`).
- Encoding observed: UTF-8 (read with BOM tolerance).
- Empty hourly cells were observed and treated as unavailable observations; no imputation, interpolation or forward filling was performed.
- All rows had width 32, numeric non-empty hourly values, valid calendar dates, and `MAGNITUD=10` in this audit.
- The official specification is authoritative for semantic interpretation; this audit does not infer undocumented validation semantics from the CSV alone.

## Yearly observations

| Year | Source rows | Source points | Date range | Valid hourly cells | Empty hourly cells |
|---:|---:|---:|---|---:|---:|
| 2020 | 123865 | 355 | 2020-01-01 to 2020-12-31 | 2909665 | 63095 |
| 2021 | 128792 | 375 | 2021-01-01 to 2021-12-31 | 3028445 | 62563 |
| 2022 | 130335 | 371 | 2022-01-01 to 2022-12-31 | 3062958 | 65082 |
| 2023 | 132791 | 379 | 2023-01-01 to 2023-12-31 | 3120872 | 66112 |

## Derived canonical fields

The processed hourly Parquet table contains `station_id`, UTC timestamp, PM10 value, a null `quality_flag` because no source flag exists, `source_year`, and the original source identifiers. The daily candidate table is computed as the arithmetic mean of observed hourly values for each source day; it does not impute incomplete days.

This is a Phase-0 data contract audit only. No forecast model, error metric, event metric, variance-retention metric, ranking, or eligibility rule was computed.
