"""Resume E6 across four GPUs, checkpointing every lambda and seed.

Run only inside the journal:E6 tmux window. The three seed-0 native fits already
started manually occupy GPUs 0, 1, 3; this coordinator picks up after them.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from common import OUTPUT, ROOT, make_audit, read_json, view_uids, write_json

OUT = OUTPUT / "E6"
PYTHON = Path("/home/sm5/ys/ICASSP/archive/non_paper_research_20261011/.venvs/table_a2_factory/bin/python")
TASK = ROOT / "scripts/journal_20261024/e6_task.py"
PREP = ROOT / "scripts/journal_20261024/e6_prepare.py"
INITIAL = {0: ("heldout11", "routerdc", 0),
           1: ("full14", "routerdc", 0),
           3: ("heldout11", "vlc", 0)}
METHODS = ("routerdc", "vlc", "mlp", "cosinecls", "zooter")
PROTOCOLS = ("full14", "heldout11")
GPU_IDS = (0, 1, 2, 3)
PREFERRED = {0: ("routerdc", "vlc", "cosinecls", "zooter", "mlp"),
             1: ("cosinecls", "zooter", "routerdc", "vlc", "mlp"),
             2: ("mlp", "zooter", "cosinecls", "vlc", "routerdc"),
             3: ("vlc", "routerdc", "zooter", "cosinecls", "mlp")}


def log(**fields):
    print(json.dumps(fields, sort_keys=True, default=str), flush=True)


def ready(task):
    protocol, method, seed = task
    return (OUT / "tasks" / protocol / method / f"seed-{seed}" / "COMPLETE.json").is_file()


def running_initial(task):
    protocol, method, seed = task
    result = subprocess.run(["pgrep", "-af", "e6_task.py"], capture_output=True, text=True, check=False)
    needles = (f"--protocol {protocol}", f"--method {method}", f"--seed {seed}")
    return any(all(needle in line for needle in needles) for line in result.stdout.splitlines())


def external_tasks(all_tasks):
    """Adopt native fits left running by an earlier coordinator instance."""
    devices = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid",
                              "--format=csv,noheader"], capture_output=True,
                             text=True, check=True)
    uuid_to_gpu = {uuid.strip(): int(index.strip())
                   for index, uuid in (line.split(",", 1)
                                       for line in devices.stdout.splitlines())}
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid",
                           "--format=csv,noheader"], capture_output=True,
                          text=True, check=True)
    pid_to_gpu = {int(pid.strip()): uuid_to_gpu[uuid.strip()]
                  for pid, uuid in (line.split(",", 1)
                                    for line in apps.stdout.splitlines())
                  if uuid.strip() in uuid_to_gpu}
    result = subprocess.run(["pgrep", "-af", "e6_task.py"], capture_output=True,
                            text=True, check=False)
    found = {}
    for line in result.stdout.splitlines():
        try:
            pid_text, command = line.split(None, 1)
            pid = int(pid_text)
            args = shlex.split(command)
            if not any(Path(arg).name == "e6_task.py" for arg in args):
                continue
            task = (args[args.index("--protocol") + 1],
                    args[args.index("--method") + 1],
                    int(args[args.index("--seed") + 1]))
            if task not in all_tasks:
                continue
            gpu = pid_to_gpu.get(pid)
            if gpu not in GPU_IDS:
                continue
            if gpu in found:
                raise RuntimeError(f"more than one external E6 task on GPU {gpu}")
            found[gpu] = (pid, task)
        except (ValueError, IndexError, OSError):
            continue
    return found


def process_alive(pid):
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)],
                            capture_output=True, text=True, check=False)
    return result.returncode == 0 and not result.stdout.strip().startswith("Z")


def env(gpu):
    values = os.environ.copy()
    values.update(PYTHONDONTWRITEBYTECODE="1", HF_HOME=str(OUT / "hf_home"),
                  HF_HUB_CACHE=str(OUT / "hf_cache"), HF_HUB_OFFLINE="1",
                  TRANSFORMERS_OFFLINE="1", XDG_CACHE_HOME=str(OUT / "xdg"),
                  CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS="4",
                  OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    return values


def start_task(task, gpu):
    protocol, method, seed = task
    path = OUT / "tasks" / protocol / method / f"seed-{seed}"
    path.mkdir(parents=True, exist_ok=True)
    handle = (path / "run.log").open("a")
    command = [str(PYTHON), "-B", str(TASK), "--protocol", protocol,
               "--method", method, "--seed", str(seed)]
    process = subprocess.Popen(command, cwd=ROOT, env=env(gpu), stdout=handle,
                               stderr=subprocess.STDOUT)
    log(stage="task_started", gpu=gpu, protocol=protocol, method=method,
        seed=seed, pid=process.pid)
    return process, handle


def siglip_download_complete():
    model = OUT / "hf_cache/models--google--siglip-large-patch16-384"
    return any(path.is_file() and path.stat().st_size > 1_000_000_000
               for path in model.glob("snapshots/*/model.safetensors"))


def external_siglip_running():
    result = subprocess.run(["pgrep", "-af", "e6_prepare.py"], capture_output=True,
                            text=True, check=False)
    return any("e6_prepare.py siglip" in line and "pgrep -af" not in line
               for line in result.stdout.splitlines())


def discard_committed_monitor_models():
    """Remove only native epoch checkpoints after actions and metrics are frozen."""
    for committed in (OUT / "tasks").glob("*/*/seed-*/lambda-*/CANDIDATE.json"):
        monitor = committed.parent / "training_monitor"
        for name in ("best_model.pkl", "best_model_model.pt"):
            path = monitor / name
            if path.is_file():
                path.unlink()
                log(stage="ephemeral_checkpoint_removed", path=str(path.relative_to(ROOT)))


def main():
    sizes = {view: len(ids) for view, ids in view_uids().items()}
    audit = make_audit("E6", "journal:E6; e6_prepare.py; e6_task.py; e6_orchestrate.py",
        "All six training lambdas fit on source train; source dev selects comparable lambda. "
        "A separately labeled best_test diagnostic selects on each test view as required by E6.", sizes)
    existing_path = OUT / "AUDIT.json"
    if existing_path.is_file():
        existing = read_json(existing_path)
        if existing.get("status") == "running":
            for key in ("start_time", "start_time_note", "git_commit", "git_commit_note",
                        "released_code_url"):
                if key in existing:
                    audit[key] = existing[key]
    audit["status"] = "running"
    audit["gpu"] = "NVIDIA RTX A6000; CUDA_VISIBLE_DEVICES 0,1,2,3; one GPU per task"
    write_json(OUT / "AUDIT.json", audit)
    all_tasks = [(protocol, method, seed) for method in METHODS
                 for protocol in PROTOCOLS for seed in range(5)]
    external = external_tasks(all_tasks)
    for gpu, (pid, task) in external.items():
        log(stage="external_task_adopted", gpu=gpu, pid=pid, task=task)
    initial = dict(INITIAL)
    pending = [task for task in all_tasks if task not in initial.values()
               and task not in (item[1] for item in external.values()) and not ready(task)]
    active = {}
    failed = []
    siglip_process = None
    siglip_handle = None
    external_siglip_seen = False
    last_report = 0.0
    while pending or active or external or initial or siglip_process:
        discard_committed_monitor_models()
        # Other coordinators or adopted tasks may finish after this queue was built.
        pending[:] = [task for task in pending if not ready(task)]
        for gpu, (pid, task) in list(external.items()):
            if process_alive(pid):
                continue
            if ready(task):
                log(stage="external_task_complete", gpu=gpu, task=task)
            else:
                failed.append({"task": task, "reason": "adopted native fit exited without COMPLETE.json"})
                log(stage="external_task_failed", gpu=gpu, task=task)
            del external[gpu]
        for gpu, task in list(initial.items()):
            if running_initial(task):
                continue
            if ready(task):
                log(stage="initial_complete", gpu=gpu, task=task)
            else:
                failed.append({"task": task, "reason": "initial native fit exited without COMPLETE.json"})
                log(stage="initial_failed", gpu=gpu, task=task)
            del initial[gpu]
        for gpu, (process, task, handle) in list(active.items()):
            if process.poll() is not None:
                handle.close()
                if process.returncode != 0 or not ready(task):
                    failed.append({"task": task, "exit_code": process.returncode})
                    log(stage="task_failed", gpu=gpu, task=task, exit_code=process.returncode)
                else:
                    log(stage="task_complete", gpu=gpu, task=task)
                del active[gpu]
        if siglip_process and siglip_process.poll() is not None:
            siglip_handle.close()
            if siglip_process.returncode != 0 or not (OUT / "siglip-large-patch16-384.parquet").is_file():
                failed.append({"task": "siglip_feature_extraction", "exit_code": siglip_process.returncode})
                log(stage="siglip_failed", exit_code=siglip_process.returncode)
            else:
                log(stage="siglip_complete")
            siglip_process = None
        external_siglip = external_siglip_running()
        external_siglip_seen |= external_siglip
        if (2 not in active and siglip_process is None and not external_siglip and
                not (OUT / "siglip-large-patch16-384.parquet").is_file() and
                not any(item.get("task") == "siglip_feature_extraction" for item in failed) and
                siglip_download_complete()):
            if external_siglip_seen:
                failed.append({"task": "siglip_feature_extraction",
                               "reason": "external extractor exited without feature parquet"})
                log(stage="siglip_external_failed")
            else:
                siglip_handle = (OUT / "siglip_run.log").open("a")
                siglip_process = subprocess.Popen([str(PYTHON), "-B", str(PREP), "siglip"],
                    cwd=ROOT, env=env(2), stdout=siglip_handle, stderr=subprocess.STDOUT)
                log(stage="siglip_started", gpu=2, pid=siglip_process.pid)
        # A failed feature extractor makes MLP impossible. Record each skipped
        # seed and finish the other methods instead of waiting indefinitely.
        if any(item.get("task") == "siglip_feature_extraction" for item in failed):
            blocked = [task for task in pending if task[1] == "mlp"]
            for task in blocked:
                pending.remove(task)
                failed.append({"task": task, "reason": "SigLIP features unavailable"})
                log(stage="task_blocked", task=task, reason="SigLIP features unavailable")
        for gpu in GPU_IDS:
            if gpu in active or gpu in external or gpu in initial or (gpu == 2 and
                    (external_siglip or siglip_process is not None or
                     (not (OUT / "siglip-large-patch16-384.parquet").is_file() and
                      not any(item.get("task") == "siglip_feature_extraction" for item in failed)))):
                continue
            selected = None
            for method in PREFERRED[gpu]:
                selected = next((task for task in pending if task[1] == method and
                    (method != "mlp" or (OUT / "siglip-large-patch16-384.parquet").is_file())), None)
                if selected is not None:
                    break
            if selected is None:
                continue
            pending.remove(selected)
            process, handle = start_task(selected, gpu)
            active[gpu] = (process, selected, handle)
        now = time.monotonic()
        if now - last_report > 300:
            log(stage="progress", complete=sum(ready(task) for task in all_tasks),
                total=len(all_tasks), pending=len(pending), running=len(active), failures=len(failed))
            last_report = now
        time.sleep(15)
    if failed:
        write_json(OUT / "ORCHESTRATOR_FAILURES.json", failed)
    command = [str(PYTHON), "-B", str(ROOT / "scripts/journal_20261024/e6_aggregate.py")]
    completed = subprocess.run(command, cwd=ROOT, env=env(0), capture_output=True, text=True)
    log(stage="aggregate", exit_code=completed.returncode, stdout=completed.stdout.strip(),
        stderr=completed.stderr[-3000:])
    if completed.returncode != 0 or failed:
        raise RuntimeError("E6 has recorded task or aggregation failures")
    audit = read_json(OUT / "AUDIT.json")
    if audit["status"] != "complete":
        raise RuntimeError(f"E6 stopped at acceptance: {audit['status']}")
    progress = OUTPUT / "PROGRESS.md"
    lines = ["", "## E6 — complete", "",
             "Released RouterDC, VLC, CosineCls and ZOOTER native fits, plus the released MLP "
             "network under an Appendix-H AdamW adapter, were run for six lambdas and five seeds on both "
             "official 14-source and held-out 11-source protocols. Source-dev and diagnostic "
             "best-test selections are reported separately in E6/SUMMARY.csv; exact chosen "
             "lambdas are in E6/lambda_selections.csv. See E6/AUDIT.json for source code, "
             "image correction, and acceptance details.", ""]
    with progress.open("a") as handle:
        handle.write("\n".join(lines))
    log(stage="E6_complete")


if __name__ == "__main__":
    main()
