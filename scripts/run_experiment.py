from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.training.telemetry import collect_resource_snapshot


class SafetyStop(RuntimeError):
    pass


def _report_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _configured_path(paths: dict, name: str, environment: str, default: str) -> Path:
    value = os.environ.get(environment) or paths.get(name) or default
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _result_path(output_dir: Path, condition: str, representation: str, seed: int, suffix: str | None) -> Path:
    suffix_part = f"__{suffix}" if suffix else ""
    return output_dir / "per_seed" / f"seed_{seed}" / f"{condition}__{representation}{suffix_part}.json"


def _phase3_has_row(results_root: Path, job: dict, seed: int, phase: dict) -> bool:
    if seed != 42:
        return False
    reusable = {
        tuple(pair)
        for pair in phase.get("seed42_reuse", {}).get("representations", [])
    }
    pair = (job["condition"], job["representation"])
    if pair not in reusable:
        return False
    path = results_root / phase.get("seed42_reuse", {}).get("results_subdir", "Neural Accessibility") / "model_results.json"
    if not path.is_file():
        return False
    rows = json.loads(path.read_text(encoding="utf-8"))
    return any(
        row.get("condition") == pair[0]
        and row.get("representation") == pair[1]
        and int(row.get("seed", 42)) == 42
        for row in rows
    )


def _jobs(args, config: dict, results_root: Path) -> list[dict]:
    if args.phase:
        if args.phase not in config.get("phases", {}):
            raise ValueError(f"unknown phase in config: {args.phase}")
        phase = config["phases"][args.phase]
        jobs = []
        for item in phase.get("jobs", []):
            if args.condition and item["condition"] != args.condition:
                continue
            if args.representation and item["representation"] != args.representation:
                continue
            configured_seeds = [int(item["seed"])] if "seed" in item else [int(seed) for seed in item.get("seeds", phase.get("seeds", []))]
            if args.seed is not None:
                if "seed" in item and int(item["seed"]) != args.seed:
                    continue
                seeds = [args.seed]
            else:
                seeds = configured_seeds
            for seed in seeds:
                job = dict(item)
                job["seed"] = int(seed)
                jobs.append(job)
        phase_name = args.phase
        return [{**job, "phase_name": phase_name, "phase": phase} for job in jobs]
    if not args.condition or not args.representation or args.seed is None:
        raise ValueError("provide --phase or all of --condition, --representation, and --seed")
    return [{
        "condition": args.condition,
        "representation": args.representation,
        "seed": args.seed,
        "phase_name": "single_condition",
        "phase": {"results_subdir": args.output_subdir or "Phase 4 Confirmation", "cache_subdir": "neural_accessibility/imagenette_320", "dataset_subdir": "raw/imagenette"},
    }]


def _update_summary(summary: dict, snapshot: dict) -> None:
    process = snapshot.get("process", {})
    system = snapshot.get("system", {})
    cgroup = snapshot.get("cgroup", {})
    gpu = snapshot.get("gpu", {})
    for key, value in (
        ("minimum_disk_free_bytes", snapshot.get("disk_free_bytes")),
        ("minimum_memory_available_bytes", system.get("memory_available_bytes")),
    ):
        if value is not None:
            summary[key] = min(summary.get(key, value), value)
    for key, value in (
        ("maximum_process_rss_bytes", process.get("rss_bytes")),
        ("maximum_process_virtual_bytes", process.get("virtual_bytes")),
        ("maximum_cgroup_memory_bytes", cgroup.get("memory_current_bytes")),
        ("maximum_cgroup_swap_bytes", cgroup.get("swap_current_bytes")),
        ("maximum_commit_used_bytes", system.get("commit_used_bytes")),
        ("maximum_vram_used_mb", gpu.get("vram_used_mb")),
        ("maximum_gpu_utilization_percent", gpu.get("utilization_percent")),
    ):
        if value is not None:
            summary[key] = max(summary.get(key, value), value)
    commit_used = system.get("commit_used_bytes")
    commit_limit = system.get("commit_limit_bytes")
    if commit_used is not None and commit_limit:
        fraction = commit_used / commit_limit
        summary["maximum_commit_fraction"] = max(summary.get("maximum_commit_fraction", fraction), fraction)
    cgroup_limit = cgroup.get("memory_limit_bytes")
    cgroup_used = cgroup.get("memory_current_bytes")
    if cgroup_limit and cgroup_used is not None:
        headroom = cgroup_limit - cgroup_used
        summary["minimum_cgroup_headroom_bytes"] = min(summary.get("minimum_cgroup_headroom_bytes", headroom), headroom)


def _safety_reason(snapshot: dict, runtime: dict, commit_high_since: float | None) -> tuple[str | None, float | None]:
    disk_free = snapshot.get("disk_free_bytes")
    if disk_free is not None and disk_free < int(runtime["stop_free_gb"] * 1_000_000_000):
        return "free disk space fell below the configured stop threshold", commit_high_since
    system = snapshot.get("system", {})
    if snapshot.get("platform", "").startswith("win"):
        used = system.get("commit_used_bytes")
        limit = system.get("commit_limit_bytes")
        fraction = used / limit if used is not None and limit else None
        if fraction is None:
            return "Windows system commit telemetry is unavailable", commit_high_since
        if fraction >= runtime["windows_commit_stop"]:
            return "Windows system commit reached the stop threshold", commit_high_since
        if fraction >= runtime["windows_commit_high"]:
            commit_high_since = time.monotonic() if commit_high_since is None else commit_high_since
            if time.monotonic() - commit_high_since >= runtime["windows_commit_high_seconds"]:
                return "Windows system commit stayed high for the configured duration", commit_high_since
        else:
            commit_high_since = None
    else:
        available = system.get("memory_available_bytes")
        if available is not None and available < int(runtime["linux_stop_available_gib"] * (1024**3)):
            return "available system RAM fell below the configured stop threshold", commit_high_since
        cgroup = snapshot.get("cgroup", {})
        limit = cgroup.get("memory_limit_bytes")
        used = cgroup.get("memory_current_bytes")
        if limit and used is not None and limit - used < int(runtime["linux_stop_cgroup_headroom_gib"] * (1024**3)):
            return "cgroup memory headroom fell below the configured stop threshold", commit_high_since
    return None, commit_high_since


def _stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        process.wait(timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _check_start(snapshot: dict, runtime: dict) -> None:
    required_free = int(runtime["minimum_start_free_gb"] * 1_000_000_000)
    free = snapshot.get("disk_free_bytes")
    if free is None or free < required_free:
        raise SafetyStop(f"need {runtime['minimum_start_free_gb']} GB free before starting; found {free}")
    if snapshot.get("platform", "").startswith("win"):
        system = snapshot.get("system", {})
        used, limit = system.get("commit_used_bytes"), system.get("commit_limit_bytes")
        if used is None or not limit or used / limit >= runtime["windows_commit_start_max"]:
            raise SafetyStop("Windows system commit is unavailable or too high before child start")
    else:
        available = snapshot.get("system", {}).get("memory_available_bytes")
        if available is None or available < int(runtime["linux_minimum_available_gib"] * (1024**3)):
            raise SafetyStop("Linux available RAM is unavailable or below the configured start threshold")


def _run_child(
    command: list[str],
    output_path: Path,
    log_path: Path,
    environment: dict[str, str],
    runtime: dict,
    disk_path: Path,
) -> dict:
    disk_path = _existing_parent(disk_path)
    start_snapshot = collect_resource_snapshot(disk_path=disk_path)
    _check_start(start_snapshot, runtime)
    summary: dict = {}
    _update_summary(summary, start_snapshot)
    started = time.time()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            start_new_session=os.name != "nt",
        )
        commit_high_since = None
        status = "running"
        reason = None
        try:
            while process.poll() is None:
                snapshot = collect_resource_snapshot(pid=process.pid, disk_path=disk_path)
                _update_summary(summary, snapshot)
                reason, commit_high_since = _safety_reason(snapshot, runtime, commit_high_since)
                if reason:
                    _stop_child(process)
                    status = "stopped_by_safety_guard"
                    break
                time.sleep(max(0.1, float(runtime["poll_seconds"])))
        except KeyboardInterrupt:
            _stop_child(process)
            status = "interrupted"
            reason = "runner received KeyboardInterrupt"
        return_code = process.wait()
    if status == "running":
        status = "completed" if return_code == 0 else "failed"
    time.sleep(max(0, int(runtime["recovery_seconds"])))
    end_snapshot = collect_resource_snapshot(disk_path=disk_path)
    _update_summary(summary, end_snapshot)
    child_result = None
    if output_path.is_file():
        try:
            child_result = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            child_result = None
    if status == "completed" and not isinstance(child_result, dict):
        status = "invalid_result"
        reason = "child exited without writing a valid JSON result object"
    record = {
        "schema": "tamtd.runner-diagnostics.v1",
        "status": status,
        "reason": reason,
        "started_unix_seconds": started,
        "finished_unix_seconds": time.time(),
        "return_code": return_code,
        "result_path": _report_path(output_path),
        "log_path": _report_path(log_path),
        "disk_free_before_bytes": start_snapshot.get("disk_free_bytes"),
        "disk_free_minimum_bytes": summary.get("minimum_disk_free_bytes"),
        "disk_free_after_recovery_bytes": end_snapshot.get("disk_free_bytes"),
        "runner_resource_peaks_and_minima": summary,
        "child_resource_diagnostics": child_result.get("resource_diagnostics") if child_result else None,
    }
    _atomic_json(output_path.with_name(output_path.stem + ".runner-diagnostics.json"), record)
    if status != "completed":
        raise SafetyStop(reason or f"child exited with status {status} (code {return_code})")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("EXPERIMENT_CONFIG") or PROJECT_ROOT / "configs/experiment_runtime.yaml"),
    )
    parser.add_argument("--phase")
    parser.add_argument("--condition")
    parser.add_argument("--representation")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-subdir")
    parser.add_argument("--result-suffix")
    parser.add_argument("--device")
    parser.add_argument("--python-exe", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--byte-batch-mode", choices=("fixed", "token_budget"))
    parser.add_argument("--max-padded-tokens", type=int)
    parser.add_argument("--max-samples-per-batch", type=int)
    parser.add_argument("--effective-batch-size", type=int)
    parser.add_argument("--diagnostic", choices=("pbc_quality_store",))
    parser.add_argument("--diagnostic-output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths_config = config["paths"]
    runtime = config["runtime"]
    data = _configured_path(paths_config, "data_root", "DATA_ROOT", "data")
    cache = _configured_path(paths_config, "cache_root", "CACHE_ROOT", str(data / "cache"))
    results = _configured_path(paths_config, "results_root", "RESULTS_ROOT", "results")
    temp = _configured_path(paths_config, "temp_root", "TEMP_ROOT", "tmp/tamtd")
    pbc = os.environ.get("PBC_ROOT") or paths_config.get("pbc_root") or None
    if pbc:
        pbc_path = Path(pbc).expanduser()
        pbc = str((pbc_path if pbc_path.is_absolute() else PROJECT_ROOT / pbc_path).resolve())
    device = args.device or os.environ.get("CUDA_DEVICE") or os.environ.get("DEVICE") or runtime["device"]
    python = args.python_exe or (Path(runtime["python"]) if runtime.get("python") else Path(sys.executable))
    if not python.is_absolute():
        python = (PROJECT_ROOT / python).resolve()
    if not python.is_file():
        raise FileNotFoundError(f"configured Python executable not found: {python}")
    environment = os.environ.copy()
    environment.update({
        "DATA_ROOT": str(data),
        "CACHE_ROOT": str(cache),
        "RESULTS_ROOT": str(results),
        "TEMP_ROOT": str(temp),
        "TMPDIR": str(temp),
        "TEMP": str(temp),
        "TMP": str(temp),
        "CUDA_DEVICE": device,
        "EXPERIMENT_CONFIG": str(config_path.resolve()),
    })
    if pbc:
        environment["PBC_ROOT"] = pbc

    if args.diagnostic:
        output_path = args.diagnostic_output
        if output_path is None:
            output_path = results / config["diagnostics"][args.diagnostic]["output"]
        elif not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        output_path = output_path.resolve()
        if output_path.is_file():
            raise FileExistsError(f"refusing to overwrite diagnostic output: {output_path}")
        command = [
            str(python), str(PROJECT_ROOT / "scripts/diagnose_byte_batching.py"),
            "--cache-root", str(cache / "neural_accessibility" / "imagenette_320"),
            "--output", str(output_path), "--device", device,
            "--temp-root", str(temp),
            "--max-padded-tokens", str(args.max_padded_tokens or runtime["max_padded_tokens"]),
            "--max-samples-per-batch", str(args.max_samples_per_batch or runtime["max_samples_per_batch"]),
            "--effective-batch-size", str(args.effective_batch_size or runtime["effective_batch_size"]),
        ]
        jobs = [{"name": args.diagnostic, "command": command, "output": output_path}]
    else:
        jobs = []
        for job in _jobs(args, config, results):
            phase = job["phase"]
            output_dir = results / args.output_subdir if args.output_subdir else results / phase["results_subdir"]
            suffix = args.result_suffix or job.get("result_suffix")
            output = _result_path(output_dir, job["condition"], job["representation"], job["seed"], suffix)
            if output.is_file():
                try:
                    existing_result = json.loads(output.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as error:
                    raise RuntimeError(f"existing result is not valid JSON; refusing to overwrite: {output}") from error
                if not isinstance(existing_result, dict):
                    raise RuntimeError(f"existing result is not a JSON object; refusing to overwrite: {output}")
                continue
            if _phase3_has_row(results, job, job["seed"], phase):
                continue
            command = [
                str(python), str(PROJECT_ROOT / "scripts/run_phase4_condition.py"),
                "--condition", job["condition"],
                "--representation", job["representation"],
                "--seed", str(job["seed"]),
                "--device", str(args.device or job.get("device") or device),
                "--cache-root", str(cache / phase["cache_subdir"]),
                "--phase4-cache-root", str(cache / phase["cache_subdir"] / "phase4_streams"),
                "--dataset-root", str(data / phase["dataset_subdir"]),
                "--output-dir", str(output_dir),
                "--checkpoint-dir", str(cache / phase["cache_subdir"] / "checkpoints" / "phase4"),
                "--training-settings", str(results / paths_config["training_settings"]),
                "--temp-root", str(temp),
            ]
            batch_size = args.batch_size or job.get("batch_size")
            if batch_size is not None:
                command.extend(["--batch-size", str(batch_size)])
            mode = args.byte_batch_mode or job.get("byte_batch_mode") or runtime.get("byte_batch_mode", "fixed")
            command.extend(["--byte-batch-mode", mode])
            if mode == "token_budget":
                command.extend([
                    "--max-padded-tokens", str(args.max_padded_tokens or runtime["max_padded_tokens"]),
                    "--max-samples-per-batch", str(args.max_samples_per_batch or runtime["max_samples_per_batch"]),
                    "--effective-batch-size", str(args.effective_batch_size or runtime["effective_batch_size"]),
                    "--bucket-multiplier", str(runtime["bucket_multiplier"]),
                ])
            if suffix:
                command.extend(["--result-suffix", suffix])
            if pbc:
                command.extend(["--pbc-root", pbc])
            jobs.append({"name": f"{job['condition']}/{job['representation']} seed={job['seed']}", "command": command, "output": output, "log_dir": output_dir / "logs" / "portable_runner"})

    if args.dry_run:
        print(json.dumps({"python": str(python), "device": device, "data_root": str(data), "cache_root": str(cache), "results_root": str(results), "temp_root": str(temp), "pending_jobs": [{"name": job["name"], "command": job["command"], "output": str(job["output"])} for job in jobs], "count": len(jobs)}, indent=2))
        return

    temp.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        output = job["output"]
        log_dir = job.get("log_dir", output.parent / "logs")
        log_path = log_dir / f"{output.stem}.log"
        try:
            _run_child(job["command"], output, log_path, environment, runtime, cache)
        except (SafetyStop, RuntimeError) as error:
            print(f"STOPPED {job['name']}: {error}", file=sys.stderr, flush=True)
            raise SystemExit(2) from error
        print(f"DONE {job['name']}: {output}", flush=True)
    print(json.dumps({"completed_or_already_present": len(jobs), "pending_jobs": 0, "results_root": str(results)}, indent=2))


if __name__ == "__main__":
    main()
