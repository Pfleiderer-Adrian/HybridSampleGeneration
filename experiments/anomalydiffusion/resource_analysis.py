"""Summarize measured GPU energy coverage and full stage wall-clock runtimes."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def analyse(launch):
    now = datetime.now(timezone.utc)
    queue = read(launch / "status.json")
    queue_start = datetime.fromisoformat(queue["started_utc"])
    end = datetime.fromisoformat(queue["finished_utc"]) if queue.get("finished_utc") else now
    duration = (end - queue_start).total_seconds()
    devices = defaultdict(lambda: {"joules": 0., "measured_seconds": 0., "allocations_joules": {}, "methods": {}})
    sessions = []
    for metadata in sorted((launch / "telemetry").glob("*/metadata.json")):
        folder = metadata.parent
        sessions.append(read(metadata))
        intervals = folder / "intervals.jsonl"
        if not intervals.exists():
            continue
        with intervals.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A reader may catch the final line during append.
                value = devices[row["gpu_uuid"]]
                value["joules"] += row["joules"]
                value["measured_seconds"] += row["seconds"]
                key = row["allocation"]
                value["allocations_joules"][key] = value["allocations_joules"].get(key, 0.) + row["joules"]
                value["methods"][row["method"]] = value["methods"].get(row["method"], 0) + 1
    for value in devices.values():
        value.update(kwh=value["joules"] / 3.6e6,
                     average_power_w=value["joules"] / value["measured_seconds"] if value["measured_seconds"] else None,
                     observed_queue_time_fraction=value["measured_seconds"] / duration if duration else 0.,
                     unmeasured_queue_seconds=max(0., duration - value["measured_seconds"]))
    runtimes = []
    for path in sorted((launch / "jobs").glob("*/status.json")):
        state = read(path)
        for entry in state["stages"]:
            start = datetime.fromisoformat(entry["started_utc"])
            finish = datetime.fromisoformat(entry["finished_utc"]) if entry.get("finished_utc") else now
            runtimes.append({"category": state["category"], "gpu": state["gpu"], "stage": entry["stage"],
                             "status": entry["status"], "wall_seconds": (finish - start).total_seconds()})
    result = {"updated_utc": now.isoformat(), "queue_status": queue["status"],
        "queue_wall_seconds": duration, "measured_gpu_kwh": sum(v["kwh"] for v in devices.values()),
        "devices": dict(devices), "stage_wall_times": runtimes,
        "measurement_sessions": sessions,
        "scope": "GPU energy from measured intervals only. No backfill, CPU/RAM energy, cooling, PUE or CO2 estimate.",
        "cost_accounting": "Generator and generation energy are shared once per category. The two downstream variants have separate costs. Historical hybrid-generation energy is unavailable."}
    target = launch / "telemetry/resource_analysis.json"
    temporary = target.with_suffix(".json.part")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(target)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyse(args.launch.resolve()), indent=2))
