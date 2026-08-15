#!/usr/bin/env python3
"""Build the self-contained SERRA Online Resource 1 source from final CSVs."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
Q0_PATH = ROOT / "serra_dm_q0_cell_level.csv"
AUTO_NW_PATH = ROOT / "serra_dm_auto_nw_cell_level.csv"
OUTPUT_PATH = ROOT / "ESM_1.tex"
SITES = [
    "Madrid Casa de Campo",
    "Birr (Co. Offaly)",
    "Dublin Airport",
    "Dundalk (Co. Louth)",
    "Pearse St. Dublin",
    "Ringsend Dublin",
    "Edenderry (Co. Offaly)",
    "Henry St. Limerick",
    "Portlaoise (Co. Laois)",
]
HORIZONS = [1, 6, 12, 24]


def load(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["site"], int(row["horizon"])): row for row in rows}


def latex_site(value: str) -> str:
    return value.replace("&", r"\&")


def main() -> None:
    q0 = load(Q0_PATH)
    auto = load(AUTO_NW_PATH)
    expected = {(site, horizon) for site in SITES for horizon in HORIZONS}
    if set(q0) != expected or set(auto) != expected:
        raise ValueError("final ledgers do not contain exactly 36 aligned cells")

    lines = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[landscape,margin=0.8cm]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{booktabs}",
        r"\usepackage{longtable}",
        r"\usepackage{array}",
        r"\usepackage{amsmath}",
        r"\usepackage{hyperref}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\begin{document}",
        r"\begin{center}",
        r"{\Large\bfseries When descriptive forecast-horizon gains do not imply global evidence\\[2pt]",
        r"of incremental meteorological predictability: a multi-site PM10 study\par}",
        r"\vspace{4pt}",
        r"{\normalsize Stochastic Environmental Research and Risk Assessment\par}",
        r"{\normalsize Federico García Crespí\par}",
        r"{\small Departamento de Ingeniería de Computadores, Universidad Miguel Hernández de Elche, Elche, Spain\par}",
        r"{\small ORCID: \href{https://orcid.org/0000-0002-7129-7791}{0000-0002-7129-7791}; corresponding email: \texttt{fedeg@umh.es}\par}",
        r"\vspace{8pt}",
        r"{\Large\bfseries Online Resource 1\par}",
        r"\vspace{3pt}",
        r"Complete site--horizon DM--HLN results for the fixed paired loss-differential ledger.",
        r"\end{center}",
        r"\noindent\textbf{Definitions and multiplicity.} The paired loss differential is",
        r"$d_t=L_{\mathrm{lags}}-L_{\mathrm{lags+meteorology}}$, so positive statistics favour the lags + meteorology information set.",
        r"The primary specification uses $q_{\mathrm{overlap}}=0$; the sensitivity uses the fixed Bartlett/Newey--West bandwidth",
        r"$q=\lfloor4(n/100)^{2/9}\rfloor$. The global Bonferroni family contains 36 planned comparisons, and",
        r"Benjamini--Hochberg adjustment is applied across the four horizons within each station. No cell survives global",
        r"Bonferroni correction. Non-significance is not evidence of equivalence.",
        r"\vspace{6pt}",
        r"\tiny",
        r"\renewcommand{\tablename}{Online Resource}",
        r"\begin{longtable}{@{}p{3.45cm}rrr rrrrr rrrrrr@{}}",
        r"\caption{Complete cell-level inferential results.}\label{tab:online_resource_1}\\",
        r"\toprule",
        r"Site & $h$ & $n$ & \multicolumn{5}{c}{Primary $q_{\mathrm{overlap}}=0$} & \multicolumn{6}{c}{Automatic Newey--West}\\",
        r" & & & DM--HLN & $p$ & $p_{\mathrm{Bonf}}$ & $p_{\mathrm{BH}}$ & BH & $q$ & DM--HLN & $p$ & $p_{\mathrm{Bonf}}$ & $p_{\mathrm{BH}}$ & BH\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Site & $h$ & $n$ & \multicolumn{5}{c}{Primary $q_{\mathrm{overlap}}=0$} & \multicolumn{6}{c}{Automatic Newey--West}\\",
        r" & & & DM--HLN & $p$ & $p_{\mathrm{Bonf}}$ & $p_{\mathrm{BH}}$ & BH & $q$ & DM--HLN & $p$ & $p_{\mathrm{Bonf}}$ & $p_{\mathrm{BH}}$ & BH\\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{14}{r@{}}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for site in SITES:
        for horizon in HORIZONS:
            a = q0[(site, horizon)]
            b = auto[(site, horizon)]
            lines.append(
                "{} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} & {} \\\\\\".format(
                    latex_site(site),
                    horizon,
                    a["n"],
                    a["dm_hln_stat"],
                    a["p_raw"],
                    a["p_bonferroni"],
                    a["p_bh_station"],
                    "Yes" if a["bh_station_reject"].lower() == "true" else "No",
                    b["q"],
                    b["dm_hln_stat"],
                    b["p_raw"],
                    b["p_bonferroni"],
                    b["p_bh_station"],
                    "Yes" if b["bh_station_reject"].lower() == "true" else "No",
                )
            )
    lines.extend([r"\end{longtable}", r"\end{document}", ""])
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
