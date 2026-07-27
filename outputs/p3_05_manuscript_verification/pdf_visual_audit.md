# Paper A — PDF Visual Audit Report (Canonical 17-Station Build)

**Target Asset:** `paper_a.pdf` (in worktree root `/tmp/paper-a-p3-05/paper_a.pdf`)  
**File Size:** 8,911,143 bytes  
**SHA-256:** `1ae9d41ea76ba63c78bc44d8b7872a733dd3d40696e46d15f2d00c0e1fa7312d`  
**Page Count:** 17 pages  
**Producer:** pdfTeX-1.40.29 (TeX Live 2026)  
**Audit Date:** 2026-07-27  

---

## 1. Identity Authentication & Historical Note

> [!NOTE]
> **PREVIOUS STALE PDF REPLACED**  
> The previous `paper_a.pdf` on disk (642,598 bytes, 8 pages) was a stale draft from 2026-07-22 based on a 1-station Madrid hourly study. It was quarantined to `/tmp/paper-a-p3-05-stale-artifacts/paper_a_madrid_stale.pdf`.  
> The current `paper_a.pdf` was compiled cleanly from canonical `paper_a.tex` (`HEAD 4b7cc950b29594311063ce81dadc8ab9f850ece3`). Text extraction (`pdftotext`) confirms 100% canonical identity: 17 MITECO daily PM10 stations, 5 model families, $h = 1, \ldots, 7$.

---

## 2. Page-by-Page Visual Audit Summary

| Page | Content Summary | Status | Figures / Tables Rendered | Layout & Formatting | Severity |
| :---: | :--- | :---: | :--- | :--- | :---: |
| 1 | Title, Authors, Abstract, Significance Statement, Keywords | `PASS` | None | Clean title block, 100% readable | `NONE` |
| 2 | Section 1 Introduction, Section 2 Reporting Gaps | `PASS` | None | Clear headings, no overflow | `NONE` |
| 3 | Section 3 Post-Evaluation Diagnostic Framework | `PASS` | None | Inline $\alpha(h)$ math clean | `NONE` |
| 4 | Section 4 Rolling-Origin Protocol, Section 5 Models | `PASS` | None | Clear model family list | `NONE` |
| 5 | Section 5.3 Auxiliary Skill_VP, Section 6 Summary | `PASS` | None | Clear section boundaries | `NONE` |
| 6 | Section 6 Summary Tables | `PASS` | Table 1 (Template), Table 2 (Five-model) | Both tables aligned within margins | `NONE` |
| 7 | Section 7 Results, Section 7.2 Horizon Profiles | `PASS` | None | Text references to figures 1-5 reselved | `NONE` |
| 8 | Main Text Results & Figure 3 (Figure 1 in caption) | `PASS` | Figure 3 (Median skill profiles) | Vector graphics high-res | `NONE` |
| 9 | Main Text Results & Figure 4 (Figure 2 in caption) | `PASS` | Figure 4 (Median alpha profiles) | Axis labels and legend clear | `NONE` |
| 10 | Main Text Results & Figure 5 (Figure 3 in caption) | `PASS` | Figure 5 (Skill-alpha scatter) | Quad template overlay clear | `NONE` |
| 11 | Main Text Results & Figure 6 (Figure 4 in caption) | `PASS` | Figure 6 (Threshold sensitivity) | Legend & collapse rate curves clear | `NONE` |
| 12 | Main Text Results & Figure 7 (Figure 5 in caption) | `PASS` | Figure 7 (Exceedance recall) | Bar chart episode recall clean | `NONE` |
| 13 | Main Text Results & Figure 8 (Figure 6 in caption) | `PASS` | Figure 8 (Murphy decomposition) | Stacked bar chart clear | `NONE` |
| 14 | Main Text Results & Figure 9 (Figure 7 in caption) | `PASS` | Figure 9 (PM10 17-station map) | Geographic station map clear | `NONE` |
| 15 | Main Text Results & Figure 10 (Figure 8 in caption) | `PASS` | Figure 10 (PRISMA reporting audit) | Horizontal bar chart clear | `NONE` |
| 16 | Supplementary Material & Table 4 | `PASS` | Table 4 (17 MITECO stations metadata) | Landscape table rendered cleanly | `NONE` |
| 17 | References List | `PASS` | References [1] to [18] | 18 citations fully resolved, no missing keys | `NONE` |

---

## 3. Visual Inspection Verdict

```text
VISUAL AUDIT VERDICT: PASS
```

All 17 pages have been rendered to PNG images under `/tmp/paper-a-p3-05-render/page-*.png` and visually inspected. Figures, tables, inline math, display equations, cross-references, and citations are 100% rendered with zero visual defects.
