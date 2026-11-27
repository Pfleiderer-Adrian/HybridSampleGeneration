"""Write the endpoint file for the coverage badge from a coverage JSON report."""

import json
import sys


def color(percent: float) -> str:
    for minimum, name in ((90, "brightgreen"), (80, "green"), (70, "yellowgreen"), (60, "yellow")):
        if percent >= minimum:
            return name
    return "orange" if percent >= 50 else "red"


def main(report_path: str, badge_path: str) -> None:
    with open(report_path, encoding="utf-8") as file:
        percent = float(json.load(file)["totals"]["percent_covered"])
    badge = {
        "schemaVersion": 1,
        "label": "coverage",
        "message": f"{percent:.0f}%",
        "color": color(percent),
    }
    with open(badge_path, "w", encoding="utf-8") as file:
        json.dump(badge, file)
        file.write("\n")
    print(f"Coverage without the study browser: {percent:.1f} %")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
