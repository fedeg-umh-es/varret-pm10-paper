"""Build a short textual summary of the latest P33 run."""

from pathlib import Path


def main() -> None:
    """Write a concise run summary that references the main P33 table."""
    report_path = Path("outputs/reports/run_summary.txt")
    table_path = Path("outputs/tables/variance_retention_summary.csv")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "P33 run summary\n"
        "==============\n"
        f"Primary diagnostic table: {table_path}\n"
        "Review skill, alpha, and skill_vp jointly to assess variance retention.\n",
        encoding="utf-8",
    )
    print(f"Wrote run summary to {report_path}")


if __name__ == "__main__":
    main()
