import json
import os
from collections import defaultdict
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET


AREAS = {
    "configuration": "Configuration",
    "datasets": "Datasets",
    "imaging": "Imaging",
    "generation": "Generation",
    "matching": "Matching",
    "fusion": "Fusion",
    "pipeline": "Pipeline",
    "persistence": "Persistence",
    "evaluation": "Evaluation",
    "example_cases": "Examples / MVTec AD2",
}


def get_area(classname):
    parts = classname.split(".")
    if parts and parts[0] == "tests":
        parts = parts[1:]
    return AREAS.get(parts[0] if parts else "", "Other")


def table_cell(value):
    return (
        value.replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("`", "'")
    )


repository = os.environ["GITHUB_REPOSITORY"]
token = os.environ["GITHUB_TOKEN"]
sha = os.environ["TEST_COMMIT_SHA"]
run_url = (
    f"{os.environ['GITHUB_SERVER_URL']}/{repository}"
    f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
)

reports = sorted(Path("test-results").glob("**/*.xml"))
if not reports:
    raise SystemExit("No XML test report found.")

areas = defaultdict(list)

for report in reports:
    root = ET.parse(report).getroot()

    for case in root.iter("testcase"):
        classname = case.get("classname", "")
        testname = case.get("name", "Unknown Test")
        name = f"{classname}.{testname}" if classname else testname

        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")
        problem = failure if failure is not None else error

        if problem is not None:
            status = "failure"
            details = (
                problem.text
                or problem.get("message")
                or "Test skipped."
            )
        elif skipped is not None:
            status = "skipped"
            details = (
                skipped.get("message")
                or skipped.text
                or "Test skipped."
            )
        else:
            status = "success"
            details = ""

        areas[get_area(classname)].append({
            "name": name,
            "status": status,
            "duration": case.get("time", "?"),
            "details": details,
        })

symbols = {
    "success": "✅ passed",
    "failure": "❌ failed",
    "skipped": "⏭️ skipped",
}

for area, tests in sorted(areas.items()):
    tests.sort(key=lambda test: test["name"])

    passed = sum(test["status"] == "success" for test in tests)
    failed = sum(test["status"] == "failure" for test in tests)
    skipped = sum(test["status"] == "skipped" for test in tests)

    if failed:
        conclusion = "failure"
    elif passed:
        conclusion = "success"
    else:
        conclusion = "skipped"

    summary = (
        f"**{len(tests)} Test Cases:** "
        f"{passed} passed, {failed} failed, "
        f"{skipped} skipped.\n\n"
        f"[Open workflow and complete logs]({run_url})"
    )

    sections = []

    # Show failure details before the complete test list.
    for test in tests:
        if test["status"] == "failure":
            sections.append(
                f"### ❌ {table_cell(test['name'])}\n\n"
                f"```text\n{test['details'][:12000]}\n```\n\n"
            )

    sections.append(
        "### Einzeltests\n\n"
        "| Status | Test | Time (s) |\n"
        "|---|---|---|\n"
    )

    for test in tests:
        sections.append(
            f"| {symbols[test['status']]} "
            f"| {table_cell(test['name'])} "
            f"| {table_cell(test['duration'])} |\n"
        )

    details = "".join(sections)

    # Limit the output size to stay below GitHub's check output limit.
    if len(details.encode("utf-8")) > 60000:
        details = (
            details.encode("utf-8")[:58000].decode("utf-8", errors="ignore")
            + "\n\nMore details see Workflow-Log."
        )

    payload = {
        "name": f"Tests / {area}",
        "head_sha": sha,
        "status": "completed",
        "conclusion": conclusion,
        "output": {
            "title": f"{area}: {passed} passed, {failed} failed",
            "summary": summary,
            "text": details,
        },
    }

    request = urllib.request.Request(
        f"https://api.github.com/repos/{repository}/check-runs",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )

    with urllib.request.urlopen(request) as response:
        result = json.load(response)

    print(f"{conclusion}: Tests / {area} — Check {result['id']}")
