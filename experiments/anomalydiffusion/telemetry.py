"""Read-only GPU energy sidecar for an already running comparison queue."""
import argparse
from collections import defaultdict
import ctypes
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import time


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text())


def interval_energy(previous, current, seconds):
    before, after = previous.get("energy_total_mj"), current.get("energy_total_mj")
    if before is not None and after is not None and after >= before:
        return (after - before) / 1000., "nvml_energy_counter"
    return (previous["power_w"] + current["power_w"]) * .5 * seconds, "power_trapezoid_fallback"


class GPU:
    def __init__(self, library, index):
        self.library, self.index = library, index
        self.handle = ctypes.c_void_p()
        self.check(library.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)))
        buffer = ctypes.create_string_buffer(96)
        self.check(library.nvmlDeviceGetUUID(self.handle, buffer, len(buffer)))
        self.uuid = buffer.value.decode()

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"NVML error code {code}")

    def sample(self):
        power, energy = ctypes.c_uint(), ctypes.c_ulonglong()
        self.check(self.library.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(power)))
        code = self.library.nvmlDeviceGetTotalEnergyConsumption(self.handle, ctypes.byref(energy))
        return {"index": self.index, "uuid": self.uuid, "power_w": power.value / 1000.,
                "energy_total_mj": energy.value if code == 0 else None}


def alive(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].strip()[0] != "Z"
    except OSError:
        return False


def descendant(pid, parent):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == parent:
            return True
        seen.add(pid)
        try:
            status = Path(f"/proc/{pid}/status").read_text().splitlines()
            pid = int(next(line.split()[1] for line in status if line.startswith("PPid:")))
        except (OSError, StopIteration):
            return False
    return False


def processes():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                     "--format=csv,noheader,nounits"], text=True, timeout=10)
    result = defaultdict(list)
    for line in output.splitlines():
        uuid, pid = (part.strip() for part in line.split(","))
        result[uuid].append(int(pid))
    return result


def running_stages(launch):
    active = {}
    for path in (launch / "jobs").glob("*/status.json"):
        state = load(path)
        if state["status"] != "running" or not state["stages"]:
            continue
        stage = state["stages"][-1]
        pid = stage.get("pid")
        if stage["status"] == "running" and pid and alive(pid):
            active[state["gpu"]] = {"category": state["category"], "stage": stage["stage"], "pid": pid}
    return active


def allocation(previous, current):
    if previous.get("unexpected_gpu_pids") or current.get("unexpected_gpu_pids"):
        return "interference_unattributed"
    before, after = previous.get("job"), current.get("job")
    if before == after and before:
        if previous.get("unmapped_gpu_pids") or current.get("unmapped_gpu_pids"):
            return f"pid_unverified/{before['category']}/{before['stage']}"
        return f"{before['category']}/{before['stage']}"
    if before is None and after is None:
        return "idle_unattributed"
    return "transition_unattributed"


def run(launch, interval):
    folder = launch / "telemetry"
    folder.mkdir(exist_ok=True)
    with (folder / "monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A new session never pretends to bridge an unmeasured gap after a restart.
        session = folder / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        session.mkdir()
        save(folder / "latest_session.json", {"session": str(session), "pid": os.getpid()})
        plan, queue = load(launch / "plan.json"), load(launch / "status.json")
        library = ctypes.CDLL("libnvidia-ml.so.1")
        GPU.check(library.nvmlInit_v2())
        gpus = [GPU(library, index) for index in plan["gpus"]]
        metadata = {"pid": os.getpid(), "started_utc": utc(), "queue_started_utc": queue["started_utc"],
            "monitor_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "interval_seconds": interval, "gpus": [{"index": gpu.index, "uuid": gpu.uuid} for gpu in gpus],
            "boundary": "GPU device energy only; excludes CPU, RAM, storage, network, cooling and PUE",
            "coverage": "Partial: measurement begins after queue start; earlier energy is unknown",
            "attribution": "Transitions and observed foreign processes remain unattributed. pid_unverified groups use queue GPU assignment; device PIDs are not visible in this PID namespace.",
            "shared_costs": "Generator and generation costs are recorded once per category, separate from the two downstream variants"}
        save(session / "metadata.json", metadata)
        previous = None
        stopped = False
        def stop(signum, frame):
            nonlocal stopped
            stopped = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        totals = defaultdict(lambda: {"joules": 0., "seconds": 0., "intervals": 0, "methods": {}})
        errors = 0
        with (session / "samples.jsonl").open("a", buffering=1) as samples, (session / "intervals.jsonl").open("a", buffering=1) as intervals:
            try:
                while True:
                    started = time.monotonic()
                    try:
                        jobs, apps = running_stages(launch), processes()
                        readings = []
                        for gpu in gpus:
                            reading = gpu.sample()
                            reading["job"] = jobs.get(gpu.index)
                            reading["gpu_process_pids"] = apps[gpu.uuid]
                            pid = reading["job"]["pid"] if reading["job"] else None
                            reading["unmapped_gpu_pids"] = [p for p in apps[gpu.uuid] if not Path(f"/proc/{p}/status").exists()]
                            reading["unexpected_gpu_pids"] = [p for p in apps[gpu.uuid]
                                if p not in reading["unmapped_gpu_pids"] and (pid is None or not descendant(p, pid))]
                            readings.append(reading)
                        current = {"timestamp_utc": utc(), "monotonic_seconds": time.monotonic(), "gpus": readings}
                        samples.write(json.dumps(current) + "\n")
                        if previous:
                            seconds = current["monotonic_seconds"] - previous["monotonic_seconds"]
                            for before, after in zip(previous["gpus"], current["gpus"]):
                                joules, method = interval_energy(before, after, seconds)
                                group = allocation(before, after)
                                key = f"gpu{after['index']}/{group}"
                                row = {"start_utc": previous["timestamp_utc"], "end_utc": current["timestamp_utc"],
                                    "gpu_uuid": after["uuid"], "allocation": group, "seconds": seconds,
                                    "joules": joules, "method": method}
                                intervals.write(json.dumps(row) + "\n")
                                total = totals[key]
                                total["joules"] += joules
                                total["seconds"] += seconds
                                total["intervals"] += 1
                                total["methods"][method] = total["methods"].get(method, 0) + 1
                        previous = current
                        groups = {key: {**value, "kwh": value["joules"] / 3.6e6,
                                        "average_power_w": value["joules"] / value["seconds"] if value["seconds"] else None}
                                  for key, value in totals.items()}
                        queue = load(launch / "status.json")
                        status = "running" if not stopped and queue["status"] == "running" and alive(queue["pid"]) else "stopped"
                        save(session / "summary.json", {"status": status, "updated_utc": utc(),
                            "measured_gpu_kwh": sum(v["joules"] for v in totals.values()) / 3.6e6,
                            "groups": groups, "sample_errors": errors, "coverage": metadata["coverage"]})
                        if status != "running":
                            break
                    except Exception as error:
                        errors += 1
                        with (session / "errors.jsonl").open("a") as stream:
                            stream.write(json.dumps({"timestamp_utc": utc(), "error": repr(error)}) + "\n")
                    time.sleep(max(.1, interval - (time.monotonic() - started)))
            finally:
                library.nvmlShutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("Interval must be at least one second.")
    run(args.launch.resolve(), args.interval)


if __name__ == "__main__":
    main()
