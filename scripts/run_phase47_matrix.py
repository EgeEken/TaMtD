from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tamtd.training.telemetry import collect_resource_snapshot


CONDITIONS = ("webp_q30", "webp_q75", "avif_q40", "avif_q60")
SEEDS = (42, 123, 777)
METRIC_FIELDS = (
    "condition", "stage", "representation", "seed", "n_samples", "accuracy", "cross_entropy_nats",
    "class_prior_cross_entropy_nats", "u_probe_bits", "decoded_rgb_accuracy", "decoded_rgb_u_probe_bits",
    "relative_u_probe_utility", "relative_accuracy", "accuracy_ratio", "best_epoch", "epochs_completed",
    "training_seconds", "model_parameters", "checkpoint",
)


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...] | None = None) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_fields = fields or tuple(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=selected_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _disk_snapshot() -> dict:
    snapshot = collect_resource_snapshot(disk_path=Path("C:\\"), include_gpu=True)
    system = snapshot.get("system", {})
    gpu = snapshot.get("gpu", {})
    values = {
        "disk_free_bytes": snapshot.get("disk_free_bytes"),
        "memory_available_bytes": system.get("memory_available_bytes"),
        "commit_free_bytes": (
            system.get("commit_limit_bytes") - system.get("commit_used_bytes")
            if system.get("commit_limit_bytes") is not None and system.get("commit_used_bytes") is not None
            else None
        ),
        "vram_free_mb": gpu.get("vram_free_mb"),
        "vram_used_mb": gpu.get("vram_used_mb"),
        "system_commit_used_bytes": system.get("commit_used_bytes"),
        "system_commit_limit_bytes": system.get("commit_limit_bytes"),
    }
    if any(values[name] is None for name in ("disk_free_bytes", "memory_available_bytes", "commit_free_bytes", "vram_free_mb")):
        raise RuntimeError(f"Phase 4.7 preflight telemetry is incomplete: {values}")
    return values


def _preflight() -> dict:
    values = _disk_snapshot()
    minimums = {
        "disk_free_bytes": 20 * 1024**3,
        "memory_available_bytes": 4 * 1024**3,
        "commit_free_bytes": 5 * 1024**3,
        "vram_free_mb": 1024,
    }
    for name, floor in minimums.items():
        if values[name] < floor:
            raise RuntimeError(f"preflight guard failed: {name}={values[name]} < {floor}")
    return values


def _record_path(records_dir: Path, condition: str, seed: int, gate: bool = False, force_new: bool = False) -> Path:
    stem = "overfit_gate" if gate else f"{condition}__seed{seed}"
    primary = records_dir / f"{stem}.json"
    if force_new:
        retry = 1
        while True:
            candidate = records_dir / f"{stem}__retry{retry}.json"
            if not candidate.exists():
                return candidate
            retry += 1
    if not primary.exists():
        return primary
    try:
        previous = json.loads(primary.read_text(encoding="utf-8"))
        if previous.get("status") == "complete" and (not gate or previous.get("passed")):
            return primary
    except (OSError, json.JSONDecodeError):
        pass
    retry = 1
    while True:
        candidate = records_dir / f"{stem}__retry{retry}.json"
        if not candidate.exists():
            return candidate
        try:
            previous = json.loads(candidate.read_text(encoding="utf-8"))
            if previous.get("status") == "complete" and (not gate or previous.get("passed")):
                return candidate
        except (OSError, json.JSONDecodeError):
            pass
        retry += 1


def _run_child(condition: str, seed: int, record_path: Path, args, gate: bool = False) -> tuple[dict, dict, dict, int, dict]:
    before = _preflight()
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_phase47_probe.py"),
        "--condition", condition,
        "--seed", str(seed),
        "--cache-root", str(args.cache_root),
        "--results-dir", str(args.results_dir),
        "--record-path", str(record_path),
        "--disk-path", "C:\\",
    ]
    if gate:
        command.append("--overfit-gate")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    environment["PYTHONUNBUFFERED"] = "1"
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
    elapsed = time.perf_counter() - start
    after = _disk_snapshot()
    if not record_path.is_file():
        raise RuntimeError(f"training child exited {completed.returncode} without writing its result record: {record_path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if completed.returncode != 0 or record.get("status") != "complete":
        raise RuntimeError(f"training child failed for {condition}/seed{seed}: exit={completed.returncode}, status={record.get('status')}, error={record.get('error')}")
    if gate and record.get("passed") is not True:
        raise RuntimeError("tiny overfit gate did not pass; stopping before the model matrix")
    if not gate and (record.get("condition") != condition or int(record.get("seed", -1)) != seed or not record.get("checkpoint")):
        raise RuntimeError("training child wrote a mismatched condition, seed, or missing checkpoint")
    if not gate and not (PROJECT_ROOT / record["checkpoint"]).is_file():
        raise RuntimeError("training child reported a checkpoint that is not present")
    recovery = {
        "condition": condition,
        "seed": seed,
        "child_type": "tiny_overfit_gate" if gate else "structured_probe",
        "child_elapsed_seconds": elapsed,
        "exit_code": completed.returncode,
        "disk_free_before_bytes": before["disk_free_bytes"],
        "disk_free_after_exit_bytes": after["disk_free_bytes"],
        "disk_recovered_bytes": after["disk_free_bytes"] - before["disk_free_bytes"],
        "host_memory_available_before_bytes": before["memory_available_bytes"],
        "host_memory_available_after_exit_bytes": after["memory_available_bytes"],
        "commit_free_before_bytes": before["commit_free_bytes"],
        "commit_free_after_exit_bytes": after["commit_free_bytes"],
        "vram_free_before_mb": before["vram_free_mb"],
        "vram_free_after_exit_mb": after["vram_free_mb"],
    }
    return record, before, after, completed.returncode, recovery


def _metric_row(record: dict) -> dict:
    return {field: record.get(field) for field in METRIC_FIELDS}


def run(args) -> dict:
    output_csv = args.results_dir / "structured_probe_per_seed.csv"
    resource_csv = args.results_dir / "training_resource_recovery.csv"
    records_dir = args.cache_root / "phase47" / "training" / "results"
    records_dir.mkdir(parents=True, exist_ok=True)
    completed_rows = _read_csv(output_csv)
    completed_keys = {(row["condition"], int(row["seed"])) for row in completed_rows}
    recovery_rows = _read_csv(resource_csv)
    rerun_conditions = set(args.rerun_conditions)

    rerun_gate = "webp_q30" in rerun_conditions
    gate_path = _record_path(records_dir, "webp_q30", 42, gate=True, force_new=rerun_gate)
    if rerun_gate or not gate_path.is_file() or json.loads(gate_path.read_text(encoding="utf-8")).get("passed") is not True:
        record, _, _, _, recovery = _run_child("webp_q30", 42, gate_path, args, gate=True)
        recovery_rows.append(recovery)
        _write_csv(resource_csv, recovery_rows)
        if record.get("passed") is not True:
            raise RuntimeError("tiny overfit gate did not pass")

    new_rows = []
    for condition in args.conditions:
        for seed in args.seeds:
            key = (condition, seed)
            if key in completed_keys and condition not in rerun_conditions:
                print(f"already recorded; skipping {condition}/seed{seed}", flush=True)
                continue
            if condition in rerun_conditions:
                record_path = _record_path(records_dir, condition, seed, force_new=True)
                record, _, _, _, recovery = _run_child(condition, seed, record_path, args)
                row = _metric_row(record)
                completed_rows = [item for item in completed_rows if (item["condition"], int(item["seed"])) != key]
                completed_rows.append(row)
                _write_csv(output_csv, completed_rows, METRIC_FIELDS)
                recovery_rows.append(recovery)
                _write_csv(resource_csv, recovery_rows)
                completed_keys.add(key)
                print(f"verified corrected result row {condition}/seed{seed}", flush=True)
                continue
            record_path = _record_path(records_dir, condition, seed)
            if record_path.is_file():
                existing = json.loads(record_path.read_text(encoding="utf-8"))
                if existing.get("status") == "complete":
                    record = existing
                    if not record.get("checkpoint") or not (PROJECT_ROOT / record["checkpoint"]).is_file():
                        raise RuntimeError(f"completed child result has no readable checkpoint: {condition}/seed{seed}")
                    before = after = None
                    recovery = None
                else:
                    record, before, after, _, recovery = _run_child(condition, seed, _record_path(records_dir, condition, seed), args)
            else:
                record, before, after, _, recovery = _run_child(condition, seed, record_path, args)
            row = _metric_row(record)
            new_rows.append(row)
            completed_rows = [item for item in completed_rows if (item["condition"], int(item["seed"])) != key]
            completed_rows.append(row)
            _write_csv(output_csv, completed_rows, METRIC_FIELDS)
            if recovery is not None:
                recovery_rows.append(recovery)
                _write_csv(resource_csv, recovery_rows)
            completed_keys.add(key)
            print(f"verified result row {condition}/seed{seed}", flush=True)

    expected = {(condition, seed) for condition in args.conditions for seed in args.seeds}
    matrix_rows = [row for row in completed_rows if (row["condition"], int(row["seed"])) in expected]
    if len({(row["condition"], int(row["seed"])) for row in matrix_rows}) != len(expected):
        raise RuntimeError("structured probe matrix is incomplete after child execution")
    return {
        "conditions": list(args.conditions),
        "seeds": list(args.seeds),
        "completed_rows": len(matrix_rows),
        "expected_rows": len(expected),
        "rerun_conditions": sorted(rerun_conditions),
        "overfit_gate": str(gate_path.relative_to(PROJECT_ROOT)),
        "results": str(output_csv.relative_to(PROJECT_ROOT)),
        "resource_recovery": str(resource_csv.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "data/cache/neural_accessibility/imagenette_320")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results/Phase 4.7 Standard Codec Stages")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--rerun-conditions", nargs="+", choices=CONDITIONS, default=[])
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
