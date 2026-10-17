"""Calculate combined metrics for every sheet in an Excel workbook."""

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = (
    "Correctness",
    "Faithfulness",
    "Precision",
    "Recall",
)


def format_average(value: Decimal) -> str:
    return str(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


def normalize_column_name(name: object) -> str:
    """Normalize a header so spacing and capitalization do not matter."""
    return "".join(character.casefold() for character in str(name) if character.isalnum())


def build_report(workbook_path: Path) -> str:
    sheets = pd.read_excel(workbook_path, sheet_name=None)
    if not sheets:
        raise ValueError(f"No worksheets found in {workbook_path}")

    report = [f"Metrics for {workbook_path.name}"]
    averages: list[tuple[str, Decimal, Decimal, Decimal]] = []
    for sheet_name, frame in sheets.items():
        if frame.empty:
            raise ValueError(f"Worksheet {sheet_name!r} has no data rows")

        columns_by_name: dict[str, object] = {}
        for column in frame.columns:
            normalized_name = normalize_column_name(column)
            columns_by_name[normalized_name] = column

        sums: dict[str, float] = {}
        decimal_sums: dict[str, Decimal] = {}
        for metric_column in METRIC_COLUMNS:
            normalized_name = normalize_column_name(metric_column)
            actual_column = columns_by_name[normalized_name]
            values = frame[actual_column]
            numeric_values = pd.to_numeric(values, errors="coerce")
            invalid_values = (
                values.notna()
                & numeric_values.isna()
                & values.astype(str).str.strip().ne("")
            )
            if invalid_values.any():
                examples = values[invalid_values].head(3).tolist()
                raise ValueError(
                    f"Worksheet {sheet_name!r}, column {metric_column!r}, "
                    f"contains non-numeric values: {examples}"
                )
            sums[metric_column] = float(numeric_values.sum())
            decimal_sums[metric_column] = sum(
                (Decimal(str(value)) for value in numeric_values.dropna()),
                Decimal(0),
            )

        gd = (sums["Correctness"] + sums["Faithfulness"]) / 2
        rd = (sums["Precision"] + sums["Recall"]) / 2
        cd = (0.5 * gd) + (0.5 * rd)
        row_count = Decimal(len(frame))
        average_gd = (
            decimal_sums["Correctness"] + decimal_sums["Faithfulness"]
        ) / (2 * row_count)
        average_rd = (
            decimal_sums["Precision"] + decimal_sums["Recall"]
        ) / (2 * row_count)
        averages.append(
            (sheet_name, average_gd, average_rd, (average_gd + average_rd) / 2)
        )

        report.extend(
            (
                "",
                f"Sheet: {sheet_name}",
                f"Sum of Correctness: {sums['Correctness']:.12g}",
                f"Sum of Faithfulness: {sums['Faithfulness']:.12g}",
                f"Gd: {gd:.12g}",
                f"Sum of Precision: {sums['Precision']:.12g}",
                f"Sum of Recall: {sums['Recall']:.12g}",
                f"Rd: {rd:.12g}",
                f"Cd: {cd:.12g}",
            )
        )

    headers = ("Sheet", "Average Gd", "Average Rd", "Average Cd")
    rows = [
        (
            sheet_name,
            format_average(average_gd),
            format_average(average_rd),
            format_average(average_cd),
        )
        for sheet_name, average_gd, average_rd, average_cd in averages
    ]
    column_widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    border = "+-" + "-+-".join("-" * width for width in column_widths) + "-+"

    def format_row(row: tuple[str, ...]) -> str:
        cells = (
            f" {value:<{column_widths[index]}} "
            if index == 0
            else f" {value:>{column_widths[index]}} "
            for index, value in enumerate(row)
        )
        return "|" + "|".join(cells) + "|"

    report.extend(("", "Average metrics by sheet:", border))
    report.append(format_row(headers))
    report.append(border)
    report.extend(format_row(row) for row in rows)
    report.append(border)

    return "\n".join(report) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("excel_file", type=Path)
    args = parser.parse_args()

    report = build_report(args.excel_file)
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{args.excel_file.stem}_metrics1.txt"
    output_path.write_text(report, encoding="utf-8")
    print(f"Metrics written to {output_path}")


if __name__ == "__main__":
    main()