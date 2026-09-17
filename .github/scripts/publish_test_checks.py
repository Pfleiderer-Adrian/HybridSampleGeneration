import json
import os
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET


repository = os.environ["GITHUB_REPOSITORY"]
token = os.environ["GITHUB_TOKEN"]
sha = os.environ["TEST_COMMIT_SHA"]
run_url = (
    f"{os.environ['GITHUB_SERVER_URL']}/{repository}"
    f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
)

reports = sorted(Path("test-results").glob("**/*.xml"))
if not reports:
    raise SystemExit("No XML-Test reports are found.")

for report in reports:
    root = ET.parse(report).getroot()

    for case in root.iter("testcase"):
        classname = case.get("classname", "")
        testname = case.get("name", "Unbekannter Test")
        name = f"{classname}.{testname}" if classname else testname

        failure = case.find("failure")
        error = case.find("error")
        skipped = case.find("skipped")

        problem = failure if failure is not None else error

        if problem is not None:
            conclusion = "failure"
            summary = (
                problem.text
                or problem.get("message")
                or "Test failed."
            )
        elif skipped is not None:
            conclusion = "skipped"
            summary = skipped.get("message") or "skip test."
        else:
            conclusion = "success"
            summary = "Test passed."

        duration = case.get("time", "?")
        summary = (
            f"Time: {duration} seconds\n\n"
            f"```text\n{summary[:15000]}\n```"
        )

        payload = {
            "name": name[:255],
            "head_sha": sha,
            "status": "completed",
            "conclusion": conclusion,
            "details_url": run_url,
            "output": {
                "title": name[:255],
                "summary": summary,
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

        print(f"{conclusion}: {name} — Check {result['id']}")
