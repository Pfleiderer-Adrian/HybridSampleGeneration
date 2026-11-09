"""Persistent two-GPU execution of both frozen V3 AnomalyDiffusion mixtures."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import traceback

from .prepare import sha256


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def check_inputs(job):
    for path, expected in job["input_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")


def stage_commands(plan, job):
    inputs = Path(job["inputs"])
    source = Path(plan["source_snapshot"])
    common = ["--inputs", str(inputs), "--checkpoint", plan["base_checkpoint"],
              "--latent-mask-policy", "maxpool"]
    generator = ["bash", str(source / "experiments/anomalydiffusion/run.sh")]
    stages = []
    for stage in ("mask", "appearance"):
        output = inputs / f"{stage}_training"
        command = generator + ["train", *common, "--output", str(output), "--stage", stage,
                               "--batch-size", "4"]
        stages.append((f"{stage}_training", command, output / "completed.json", "generator"))
    command = generator + ["generate", *common, "--output", str(inputs / "generated"),
        "--mask-weights", str(inputs / "mask_training/selection.json"),
        "--appearance-weights", str(inputs / "appearance_training/selection.json"),
        "--count", "1000", "--ddim-steps", "200", "--mask-ddim-steps", "100"]
    stages.append(("generation", command, inputs / "generated/manifest.json", "generator"))
    for variant in ("real_anomalydiffusion", "real_anomalydiffusion_hybrid"):
        output = Path(plan["downstream_root"]) / variant / job["category"]
        command = [plan["downstream_python"], "-m", "experiments.anomalydiffusion.downstream",
                   "--inputs", str(inputs), "--variant", variant, "--output", str(output)]
        stages.append((variant, command, None, "downstream"))
    return stages


def run_job(plan, job, gpu, launch):
    folder = launch / "jobs" / job["category"]
    folder.mkdir(parents=True, exist_ok=True)
    state = {"category": job["category"], "gpu": gpu, "status": "running", "started_utc": now(),
             "stages": [], "pid": os.getpid()}
    atomic_json(folder / "status.json", state)
    try:
        for name, command, artifact, environment in stage_commands(plan, job):
            check_inputs(job)
            entry = {"stage": name, "status": "running", "started_utc": now(), "command": command,
                     "log": str(folder / f"{name}.log")}
            state["stage"] = name
            state["stages"].append(entry)
            state["updated_utc"] = now()
            atomic_json(folder / "status.json", state)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8",
                       MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false")
            paths = [plan["source_snapshot"]]
            if environment == "downstream":
                paths.insert(0, plan["downstream_runtime_libs"])
                env.update(HF_HUB_CACHE=plan["downstream_hf_cache"], HF_HUB_OFFLINE="1")
            env["PYTHONPATH"] = os.pathsep.join(paths)
            with Path(entry["log"]).open("w") as log:
                process = subprocess.Popen(command, cwd=plan["source_snapshot"], env=env,
                                           stdout=log, stderr=subprocess.STDOUT)
                entry["pid"] = process.pid
                atomic_json(folder / "status.json", state)
                print(f"{now()} gpu={gpu} category={job['category']} stage={name} pid={process.pid}", flush=True)
                code = process.wait()
            entry.update(returncode=code, finished_utc=now())
            if code:
                entry["status"] = "failed"
                raise RuntimeError(f"{name} exited with {code}; see {entry['log']}")
            if artifact is not None and not artifact.is_file():
                raise RuntimeError(f"{name} did not produce its completion artifact: {artifact}")
            if environment == "downstream":
                runs = sorted((Path(plan["downstream_root"]) / name / job["category"] / "downstream").glob("dinov3_paired/*"))
                if (len(runs) != 1 or not (runs[0] / "checkpoints/best.pt").is_file()
                        or not (runs[0] / "metrics.json").is_file()):
                    raise RuntimeError(f"{name} did not produce exactly one completed downstream run.")
                entry["downstream_run"] = str(runs[0])
            entry["status"] = "complete"
            state["updated_utc"] = now()
            atomic_json(folder / "status.json", state)
        state.update(status="complete", finished_utc=now())
    except Exception as error:
        state.update(status="failed", error=str(error), traceback=traceback.format_exc(), finished_utc=now())
    atomic_json(folder / "status.json", state)
    print(f"{now()} gpu={gpu} category={job['category']} status={state['status']}", flush=True)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    launch = args.plan.resolve().parent
    plan = json.loads(args.plan.read_text())
    # A live lock prevents accidental duplicate launches of this same queue.
    with (launch / "queue.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {"pid": os.getpid(), "status": "running", "started_utc": now(),
                 "gpus": plan["gpus"], "categories": [j["category"] for j in plan["jobs"]]}
        atomic_json(launch / "status.json", state)
        jobs = Queue()
        for job in plan["jobs"]:
            jobs.put(job)
        def worker(gpu):
            results = []
            while True:
                try:
                    job = jobs.get_nowait()
                except Empty:
                    return results
                results.append(run_job(plan, job, gpu, launch))
        with ThreadPoolExecutor(max_workers=len(plan["gpus"])) as pool:
            results = [r for batch in pool.map(worker, plan["gpus"]) for r in batch]
        state.update(status="complete" if all(r["status"] == "complete" for r in results) else "failed",
                     finished_utc=now(), results=results)
        atomic_json(launch / "status.json", state)


if __name__ == "__main__":
    main()
