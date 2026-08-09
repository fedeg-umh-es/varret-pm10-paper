# Visual QA V5

The V5.1 PDF was rendered page by page with Poppler at 120 dpi and each
page was inspected. Scientific figures were also rendered independently
at higher resolution before the PDF review.

| Page | Element | Issue | Severity | Action | Final status |
|---:|---|---|---|---|---|
| 1 | Title, abstract, Introduction | No clipping, overlap, or unreadable text. | — | None | PASS |
| 2 | Introduction and Methods | Section transition and model description are readable. | — | None | PASS |
| 3 | Methods and Results opening | Results begins at the end of the page; no orphan heading or clipped content. | MINOR | Accepted as a natural section continuation; no scientific float crosses the Discussion boundary. | PASS |
| 4 | Table 1 and Figure 1 | Table and two-panel horizon plot are legible; shared legend and horizons 1--7 are visible. | — | None | PASS |
| 5 | Figure 2 and Results | Five facets and the sixth key cell are readable; formal discordance contours do not overlap labels or the caption. | — | None | PASS |
| 6 | Table 2 and Figure 3 | Decision table and sensitivity heatmap are readable and remain in Results. | — | None | PASS |
| 7 | Discussion | No Results floats appear after the Discussion heading. | — | None | PASS |
| 8 | Conclusions and declarations | No clipping, isolated heading, or broken hyperlink. | — | None | PASS |
| 9 | References | The bibliography is contiguous and complete. | — | None | PASS |
| 10 | Supplementary Table S1 | Station table is complete and readable. | — | None | PASS |
| 11 | Supplementary Tables S2--S3 | Supplementary tables are complete and legible at their intended scale. | — | None | PASS |

## Figure gates

- Figure 1: one shared legend, explicit horizons 1--7, balanced panels,
  visible reference lines, and readable labels: PASS.
- Figure 2: five enlarged common-axis facets, sixth-cell key, visible
  formal-discordance contours, distinct shaded geometric region, no text
  overlap, and no clipped labels: PASS.
- Figure 3: all nine percentages and the primary threshold point are
  visible without clipping: PASS.
- All scientific figures are deterministic data plots with zero LightGBM
  rows; no AI-assisted artwork is included.

`RESULT_FLOATS_AFTER_DISCUSSION_HEADING = 0`

`SUPPLEMENT_REFERENCE_INTERLEAVING = 0`

`FIGURE_2_VISUAL_QA = PASS`

`VISUAL_QA_V5_1 = PASS`
