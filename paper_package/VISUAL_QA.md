# Visual QA

PDF inspected: `paper_package/main.pdf`

Result: PASS. All 10 pages were rendered at 120 dpi and inspected. No
MAJOR or BLOCKING issues remain.

`FIGURE_2_VISUAL_QA = PASS`

| Page | Element | Issue | Severity | Action | Final status |
| ---: | --- | --- | --- | --- | --- |
| 1 | Title, abstract, introduction | No clipping or margin issue | MINOR | None | PASS |
| 2 | Introduction, gap, rules | Equations and references legible | MINOR | None | PASS |
| 3 | Methods opening | No clipping or excessive whitespace | MINOR | None | PASS |
| 4 | Model specification table, metrics | Compact table remains readable; no overfull content | MINOR | None | PASS |
| 5 | Table 2, Results, Table 1 | Tables and captions legible | MINOR | None | PASS |
| 6 | Figure 1, Results, Discussion | Figure readable and placed before later discussion | MINOR | None | PASS |
| 7 | Figure 2, Discussion | No text overlap; shaded region and formal contours are distinct | MINOR | None | PASS |
| 8 | Table 3, sensitivity table, conclusions | Tables and conclusion readable; no major float displacement | MINOR | None | PASS |
| 9 | Figure 3, availability, supplement entry | Sensitivity figure legible; no figure after Conclusions | MINOR | None | PASS |
| 10 | Supplementary station table, references | Table and references fit within margins | MINOR | None | PASS |

Figure 2 checks:

- shaded region denotes only positive skill with alpha below 0.50;
- black open contours identify the 101 formal discordant cells;
- legend does not obscure the dense lower-right cloud;
- Skill = 0 and alpha = 0.50 reference lines are visible;
- no labels are superimposed on the critical point cloud.
