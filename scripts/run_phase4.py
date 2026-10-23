import argparse
import ctypes
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "Phase 4 Confirmation"
PHASE3_RESULTS = PROJECT_ROOT / "results" / "Neural Accessibility" / "model_results.json"
PHASE3_RGB_REFERENCES = PROJECT_ROOT / "results" / "Neural Accessibility" / "decoded_rgb_references.json"
CACHE_ROOT = PROJECT_ROOT / "data" / "cache" / "neural_accessibility" / "imagenette_320"
PHASE4_CACHE = CACHE_ROOT / "phase4_streams"
BUILDER = PROJECT_ROOT / "scripts" / "build_phase4_raw_byte_cache.py"
TRAINER = PROJECT_ROOT / "scripts" / "run_phase4_condition.py"
TIMING_RUNNER = PROJECT_ROOT / "scripts" / "measure_phase4_jpeg_timings.py"
MINIMUM_START_FREE = 20_000_000_000
STOP_FREE = 12_000_000_000
SEEDS = (42, 123, 777)
PBC_CONDITIONS = tuple(
    (f"pbc_{preset}", representation)
    for preset in ("compression", "balanced", "quality")
    for representation in ("pbc_lzma_bytes", "pbc_store_bytes", "pbc_canonical_ir")
)
JPEG_CONDITIONS = tuple(
    (f"jpeg_q{quality}", representation)
    for quality in (10, 60)
    for representation in ("jpeg_file_bytes", "jpeg_dct_ir")
)
BASE_TRAIN_CONDITIONS = PBC_CONDITIONS + JPEG_CONDITIONS + (("original_rgb", "decoded_rgb_reference"),)
CROSS_CODEC_CONDITIONS = ("webp_q30", "webp_q75", "avif_q40", "avif_q60", "jxl_d6", "jxl_d3")
BATCH_SIZE_OVERRIDES = {
    ("pbc_balanced", "pbc_lzma_bytes"): 8,
    ("pbc_balanced", "pbc_store_bytes"): 8,
    ("pbc_quality", "pbc_lzma_bytes"): 8,
    ("pbc_quality", "pbc_store_bytes"): 8,
}
DEVICE_OVERRIDES = {
    ("pbc_balanced", "pbc_lzma_bytes"): "cpu",
    ("pbc_balanced", "pbc_store_bytes"): "cpu",
    ("pbc_quality", "pbc_lzma_bytes"): "cpu",
    ("pbc_quality", "pbc_store_bytes"): "cpu",
}


class MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", wintypes.DWORD),
        ("dwMemoryLoad", wintypes.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class ProcessMemoryCountersEx(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


class SafetyStop(RuntimeError):
    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


_ACTIVE_CHILD: subprocess.Popen | None = None


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _free_bytes() -> int:
    return shutil.disk_usage("C:/").free


def _commit() -> dict:
    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    query = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
    query.argtypes = [ctypes.POINTER(MemoryStatusEx)]
    query.restype = wintypes.BOOL
    if not query(ctypes.byref(status)):
        return {}
    return {
        "commit_used_bytes": int(status.ullTotalPageFile - status.ullAvailPageFile),
        "commit_limit_bytes": int(status.ullTotalPageFile),
        "physical_available_bytes": int(status.ullAvailPhys),
    }


def _process_memory(pid: int) -> int | None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    handle = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
    if not handle:
        return None
    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCountersEx), wintypes.DWORD]
    query.restype = wintypes.BOOL
    success = query(handle, ctypes.byref(counters), counters.cb)
    value = int(counters.PeakWorkingSetSize) if success else 0
    kernel32.CloseHandle(handle)
    return value if value > 0 else None


def _gpu_memory() -> int | None:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        return max(values) if values else None
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _python_executable(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else [PROJECT_ROOT / ".venv313/Scripts/python.exe", PROJECT_ROOT / ".venv/Scripts/python.exe"]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("No project Python found; pass --python-exe")


def _recorded_pbc_root() -> Path | None:
    configured = os.environ.get("PBC_ROOT")
    if configured:
        candidate = Path(configured).expanduser().resolve()
        return candidate if (candidate / "PBC3.py").is_file() else None
    config = json.loads((PROJECT_ROOT / "results" / "Neural Accessibility" / "config.json").read_text(encoding="utf-8"))
    model_path = config["pbc_effective_configurations"]["compression"].get("learned_filler_model_path")
    if model_path:
        candidate = Path(model_path).parent.resolve()
        return candidate if (candidate / "PBC3.py").is_file() else None
    return None


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


def _assert_no_active_child() -> None:
    global _ACTIVE_CHILD
    if _ACTIVE_CHILD is None:
        return
    if _ACTIVE_CHILD.poll() is None:
        raise RuntimeError("a previous Phase 4 child is still running")
    _ACTIVE_CHILD.wait()
    _ACTIVE_CHILD = None


def _run_job(name: str, command: list[str], log_path: Path, expected_paths: list[Path], poll_seconds: float = 5.0) -> dict:
    global _ACTIVE_CHILD
    _assert_no_active_child()
    free_before = _free_bytes()
    if free_before < MINIMUM_START_FREE:
        raise SafetyStop(f"only {free_before / 1e9:.2f} GB is free on C: before {name}; need 20 GB")
    memory_before = _commit()
    commit_limit_before = memory_before.get("commit_limit_bytes")
    commit_fraction_before = None if not commit_limit_before else memory_before["commit_used_bytes"] / commit_limit_before
    if commit_fraction_before is None or commit_fraction_before >= 0.95:
        raise SafetyStop(
            f"system commit is unavailable or at least 95% before {name}; refusing to start a child",
            {"system_commit_before": memory_before, "C_free_bytes": free_before},
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = _now()
    min_free = free_before
    peak_commit = memory_before.get("commit_used_bytes")
    peak_working_set = None
    peak_gpu = _gpu_memory()
    print(f"START {name}; C: free={free_before / 1e9:.2f} GB; system commit={commit_fraction_before:.1%}", flush=True)
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=creationflags)
        _ACTIVE_CHILD = process
        last_report = 0.0
        commit_high_since = None
        try:
            while process.poll() is None:
                free = _free_bytes()
                min_free = min(min_free, free)
                memory = _commit()
                commit = memory.get("commit_used_bytes")
                if commit is not None:
                    peak_commit = max(peak_commit or commit, commit)
                commit_fraction = None if commit is None or not memory.get("commit_limit_bytes") else commit / memory["commit_limit_bytes"]
                if commit_fraction is not None and commit_fraction >= 0.95:
                    if commit_high_since is None:
                        commit_high_since = time.monotonic()
                else:
                    commit_high_since = None
                working = _process_memory(process.pid)
                if working is not None:
                    peak_working_set = max(peak_working_set or working, working)
                gpu = _gpu_memory()
                if gpu is not None:
                    peak_gpu = max(peak_gpu or gpu, gpu)
                if free < STOP_FREE:
                    _stop_child(process)
                    _ACTIVE_CHILD = None
                    raise SafetyStop(
                        f"C: fell below 12 GB during {name}; child was stopped",
                        {
                            "minimum_free_bytes": min_free,
                            "system_commit_peak_bytes": peak_commit,
                            "system_commit_limit_bytes": memory.get("commit_limit_bytes"),
                            "system_commit_peak_fraction": None if not peak_commit else peak_commit / memory.get("commit_limit_bytes", 1),
                            "peak_process_working_set_bytes": peak_working_set,
                            "gpu_memory_peak_mb": peak_gpu,
                        },
                    )
                commit_stop = commit_fraction is not None and (
                    commit_fraction >= 0.985
                    or commit_high_since is not None and time.monotonic() - commit_high_since >= 300
                )
                if commit_stop:
                    _stop_child(process)
                    _ACTIVE_CHILD = None
                    raise SafetyStop(
                        f"system commit stayed at least 95% for 5 minutes or reached 98.5% during {name}; child was stopped",
                        {
                            "minimum_free_bytes": min_free,
                            "system_commit_peak_bytes": peak_commit,
                            "system_commit_limit_bytes": memory.get("commit_limit_bytes"),
                            "system_commit_peak_fraction": None if not peak_commit else peak_commit / memory.get("commit_limit_bytes", 1),
                            "physical_available_bytes_at_stop": memory.get("physical_available_bytes"),
                            "peak_process_working_set_bytes": peak_working_set,
                            "gpu_memory_peak_mb": peak_gpu,
                        },
                    )
                if time.monotonic() - last_report >= 60:
                    try:
                        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        latest = lines[-1].strip()[-180:] if lines else ""
                    except OSError:
                        latest = ""
                    print(f"RUNNING {name}; C: free={free / 1e9:.2f} GB; {latest}", flush=True)
                    last_report = time.monotonic()
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            _stop_child(process)
            _ACTIVE_CHILD = None
            raise
    code = process.wait()
    if process.poll() is None:
        raise RuntimeError(f"child process for {name} did not exit after wait")
    _ACTIVE_CHILD = None
    free_after = _free_bytes()
    min_free = min(min_free, free_after)
    if code != 0:
        raise RuntimeError(f"{name} exited with code {code}; see {log_path}")
    missing = [str(path) for path in expected_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"{name} exited without writing expected output(s): {missing}")
    time.sleep(10)
    free_after_recovery = _free_bytes()
    memory_after_recovery = _commit()
    commit_limit_after_recovery = memory_after_recovery.get("commit_limit_bytes")
    record = {
        "job": name,
        "started_at": started,
        "finished_at": _now(),
        "free_before_bytes": free_before,
        "minimum_free_bytes": min_free,
        "free_after_exit_bytes": free_after,
        "free_after_recovery_bytes": free_after_recovery,
        "free_recovered_vs_minimum_bytes": free_after_recovery - min_free,
        "system_commit_before_bytes": memory_before.get("commit_used_bytes"),
        "system_commit_before_fraction": commit_fraction_before,
        "peak_process_working_set_bytes": peak_working_set,
        "system_commit_peak_bytes": peak_commit,
        "system_commit_limit_bytes": commit_limit_after_recovery,
        "system_commit_peak_fraction": None if not peak_commit else peak_commit / (commit_limit_after_recovery or 1),
        "system_commit_after_recovery_bytes": memory_after_recovery.get("commit_used_bytes"),
        "system_commit_after_recovery_fraction": None if not commit_limit_after_recovery else memory_after_recovery["commit_used_bytes"] / commit_limit_after_recovery,
        "gpu_memory_peak_mb": peak_gpu,
        "log": str(log_path),
    }
    print(f"DONE {name}; C: min={min_free / 1e9:.2f} GB after={free_after_recovery / 1e9:.2f} GB", flush=True)
    return record


def _phase3_rows() -> set[tuple[str, str, int]]:
    rows = json.loads(PHASE3_RESULTS.read_text(encoding="utf-8"))
    return {(row["condition"], row["representation"], int(row["seed"])) for row in rows}


def _result_path(seed: int, condition: str, representation: str) -> Path:
    variant = "__cpu_batch8" if (condition, representation) == ("pbc_balanced", "pbc_lzma_bytes") else ""
    return RESULTS_DIR / "per_seed" / f"seed_{seed}" / f"{condition}__{representation}{variant}.json"


def _job_priority(name: str) -> int:
    if name.startswith("evaluate seed 42 RGB validation references"):
        return -1
    if any(name.startswith(f"{condition}/codec_file_bytes seed=42") for condition in CROSS_CODEC_CONDITIONS):
        return 0
    if name in {
        "pbc_quality/pbc_canonical_ir seed=123",
        "jpeg_q10/jpeg_file_bytes seed=123",
        "jpeg_q10/jpeg_dct_ir seed=123",
        "jpeg_q60/jpeg_file_bytes seed=123",
        "jpeg_q60/jpeg_dct_ir seed=123",
        "original_rgb/decoded_rgb_reference seed=123",
    }:
        return 1
    if name.startswith("pbc_quality/pbc_lzma_bytes seed=123") or name.startswith("pbc_quality/pbc_store_bytes seed=123"):
        return 2
    if "seed=777" in name:
        representation = name.split("/", 1)[1].split(" seed=", 1)[0].split(" batch=", 1)[0]
        if representation in {"pbc_canonical_ir", "jpeg_file_bytes", "jpeg_dct_ir", "decoded_rgb_reference"}:
            return 3
        if name.startswith("pbc_compression/"):
            return 4
        return 5
    return 6


def _cache_is_complete(condition: str, expected_rows: int) -> bool:
    stream_path = PHASE4_CACHE / f"{condition}.bin"
    manifest_path = PHASE4_CACHE / f"{condition}.jsonl"
    if not stream_path.is_file() or not manifest_path.is_file():
        return False
    count = 0
    last = None
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            count += 1
            last = json.loads(line)
    if count != expected_rows or last is None or last.get("cache_index") != expected_rows - 1:
        return False
    entry = last.get("stream", {})
    return entry.get("offset", -1) + entry.get("length", 0) == stream_path.stat().st_size


def _validate_seed42_sources() -> None:
    existing = _phase3_rows()
    missing = [(condition, representation) for condition, representation in BASE_TRAIN_CONDITIONS if (condition, representation, 42) not in existing]
    if missing:
        raise RuntimeError(f"Phase 3 seed-42 results are incomplete for Phase 4 reuse: {missing}")
    references = json.loads(PHASE3_RGB_REFERENCES.read_text(encoding="utf-8"))
    required = {f"pbc_{preset}" for preset in ("compression", "balanced", "quality")} | {"jpeg_q10", "jpeg_q60"}
    if not required.issubset(references):
        raise RuntimeError(f"Phase 3 decoded-RGB test references are missing: {sorted(required - set(references))}")


def _run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-exe", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pbc-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    python_exe = _python_executable(args.python_exe)
    args.pbc_root = args.pbc_root.resolve() if args.pbc_root else _recorded_pbc_root()
    if args.pbc_root is None:
        raise RuntimeError("PBC checkout is unavailable; set PBC_ROOT or pass --pbc-root")
    _validate_seed42_sources()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []

    validation_refs = RESULTS_DIR / "per_seed" / "seed_42" / "decoded_rgb_validation.json"
    if not validation_refs.exists():
        jobs.append(("evaluate seed 42 RGB validation references", [str(python_exe), str(TRAINER), "--condition", "original_rgb", "--representation", "decoded_rgb_reference", "--seed", "42", "--device", args.device, "--eval-existing-rgb"], [validation_refs]))

    for seed in SEEDS:
        for condition, representation in BASE_TRAIN_CONDITIONS:
            output = _result_path(seed, condition, representation)
            batch_size = BATCH_SIZE_OVERRIDES.get((condition, representation))
            device = DEVICE_OVERRIDES.get((condition, representation), args.device)
            if seed == 42 and (condition, representation, seed) in _phase3_rows() and batch_size is None and device == args.device:
                continue
            if output.exists():
                continue
            name = f"{condition}/{representation} seed={seed}"
            command = [str(python_exe), str(TRAINER), "--condition", condition, "--representation", representation, "--seed", str(seed), "--device", device]
            if batch_size is not None:
                name += f" batch={batch_size}"
                command.extend(["--batch-size", str(batch_size)])
            if device != args.device:
                name += f" device={device}"
            if (condition, representation) == ("pbc_balanced", "pbc_lzma_bytes"):
                command.extend(["--result-suffix", "cpu_batch8"])
            jobs.append((name, command, [output]))

    expected_cache_rows = sum(1 for _ in (CACHE_ROOT / "manifest.jsonl").open(encoding="utf-8"))
    incomplete_caches = [condition for condition in CROSS_CODEC_CONDITIONS if not _cache_is_complete(condition, expected_cache_rows)]
    if incomplete_caches:
        raise RuntimeError(f"Phase 4 caches are incomplete; refusing to rebuild them: {incomplete_caches}")
    timing_output = RESULTS_DIR / "jpeg_timing_refresh.json"
    if not timing_output.exists():
        raise RuntimeError("Phase 4 JPEG timing refresh is missing; refusing to rerun it")
    for condition in CROSS_CODEC_CONDITIONS:
        output = _result_path(42, condition, "codec_file_bytes")
        if not output.exists():
            jobs.append((f"{condition}/codec_file_bytes seed=42", [str(python_exe), str(TRAINER), "--condition", condition, "--representation", "codec_file_bytes", "--seed", "42", "--device", args.device], [output]))

    for _, command, _ in jobs:
        if len(command) > 1 and Path(command[1]) == TRAINER:
            command.extend(["--pbc-root", str(args.pbc_root)])

    jobs.sort(key=lambda job: _job_priority(job[0]))

    if args.dry_run:
        print(json.dumps({"python": str(python_exe), "pending_jobs": [name for name, _, _ in jobs], "count": len(jobs), "C_free_bytes": _free_bytes()}, indent=2))
        return

    diagnostics = json.loads((RESULTS_DIR / "resource_diagnostics.json").read_text(encoding="utf-8")) if (RESULTS_DIR / "resource_diagnostics.json").exists() else []
    diagnostics_path = RESULTS_DIR / "resource_diagnostics.json"
    for name, command, expected in jobs:
        try:
            record = _run_job(name, command, RESULTS_DIR / "logs" / (name.replace("/", "__").replace(" ", "_") + ".log"), expected, args.poll_seconds)
        except SafetyStop as error:
            diagnostics.append({"job": name, "stopped": True, "reason": str(error), "C_free_bytes": _free_bytes(), **error.details})
            diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
            print(f"STOPPED: {error}", flush=True)
            return
        diagnostics.append(record)
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"completed_jobs": len(diagnostics), "C_free_bytes": _free_bytes(), "diagnostics": str(diagnostics_path)}, indent=2), flush=True)


if __name__ == "__main__":
    _run()
