import argparse
import ctypes
import datetime
import json
import os
import signal
import shutil
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "Neural Accessibility"
RESULTS_PATH = RESULTS_DIR / "model_results.json"
DIAGNOSTICS_PATH = RESULTS_DIR / "resource_diagnostics.json"
RUNNER_PATH = PROJECT_ROOT / "scripts" / "run_neural_accessibility.py"
PENDING_ORDER = [
    ("jpeg_q80", "jpeg_dct_ir"),
    ("original_rgb", "decoded_rgb_reference"),
]
MINIMUM_START_FREE = 20_000_000_000
STOP_FREE = 12_000_000_000


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


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def _free_bytes() -> int:
    return shutil.disk_usage("C:/").free


def _pagefile_bytes() -> int | None:
    try:
        return os.stat(r"C:\pagefile.sys").st_size
    except OSError:
        return None


def _memory_status() -> dict:
    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel32.GlobalMemoryStatusEx
    query.argtypes = [ctypes.POINTER(MemoryStatusEx)]
    query.restype = wintypes.BOOL
    if not query(ctypes.byref(status)):
        return {}
    return {
        "physical_total_bytes": status.ullTotalPhys,
        "physical_available_bytes": status.ullAvailPhys,
        "commit_limit_bytes": status.ullTotalPageFile,
        "commit_used_bytes": status.ullTotalPageFile - status.ullAvailPageFile,
        "commit_available_bytes": status.ullAvailPageFile,
    }


def _open_process_memory(pid: int):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    return kernel32.OpenProcess(0x0400 | 0x0010, False, pid)


def _peak_working_set(handle) -> int | None:
    if not handle:
        return None
    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    query = psapi.GetProcessMemoryInfo
    query.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCountersEx), wintypes.DWORD]
    query.restype = wintypes.BOOL
    if not query(handle, ctypes.byref(counters), counters.cb):
        return None
    peak = int(counters.PeakWorkingSetSize)
    return peak if peak >= 64 * 1024 * 1024 else None


def _close_process_handle(handle) -> None:
    if handle:
        close = ctypes.windll.kernel32.CloseHandle
        close.argtypes = [wintypes.HANDLE]
        close.restype = wintypes.BOOL
        close(handle)


def _read_results() -> tuple[list[dict], dict[tuple[str, str], dict]]:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"model results not found: {RESULTS_PATH}")
    rows = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    lookup = {(row["condition"], row["representation"]): row for row in rows}
    if len(lookup) != len(rows):
        raise RuntimeError("model_results.json contains duplicate condition/representation rows")
    return rows, lookup


def _write_diagnostics(rows: list[dict]) -> None:
    DIAGNOSTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DIAGNOSTICS_PATH.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _stop_child(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def _run_condition(condition: str, representation: str, python_exe: Path, device: str, recovery_seconds: int, poll_seconds: float, diagnostics: list[dict]) -> bool:
    key = (condition, representation)
    rows_before, lookup_before = _read_results()
    if key in lookup_before:
        print(f"Already complete; skipping {condition} / {representation}", flush=True)
        return True

    free_before = _free_bytes()
    if free_before < MINIMUM_START_FREE:
        raise RuntimeError(f"pre-start C: free space {free_before} bytes is below the 20 GB requirement")

    command = [
        str(python_exe), str(RUNNER_PATH), "--device", device,
        "--only-condition", condition, representation,
    ]
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    started = time.monotonic()
    started_at = _now()
    system_start = _memory_status()
    process = subprocess.Popen(command, cwd=PROJECT_ROOT, creationflags=flags)
    process_handle = _open_process_memory(process.pid) if os.name == "nt" else None
    minimum_free = free_before
    peak_commit = dict(system_start)
    peak_rss = _peak_working_set(process_handle)
    pagefile_start = _pagefile_bytes()
    pagefile_peak = pagefile_start
    stop_for_disk = False
    last_log = started

    print(f"Starting {condition} / {representation}; C: free={free_before / 1e9:.2f} GB; child PID={process.pid}", flush=True)
    while process.poll() is None:
        current_free = _free_bytes()
        minimum_free = min(minimum_free, current_free)
        current_memory = _memory_status()
        if current_memory.get("commit_used_bytes", 0) > peak_commit.get("commit_used_bytes", 0):
            peak_commit = dict(current_memory)
        current_rss = _peak_working_set(process_handle)
        if current_rss is not None:
            peak_rss = max(peak_rss or 0, current_rss)
        current_pagefile = _pagefile_bytes()
        if current_pagefile is not None:
            pagefile_peak = max(pagefile_peak or 0, current_pagefile)
        if current_free < STOP_FREE:
            stop_for_disk = True
            print(f"C: free fell below 12 GB ({current_free / 1e9:.2f} GB); stopping child", flush=True)
            _stop_child(process)
            break
        if time.monotonic() - last_log >= 10:
            commit_used = current_memory.get("commit_used_bytes")
            commit_limit = current_memory.get("commit_limit_bytes")
            commit_text = f"commit={commit_used / 1e9:.2f}/{commit_limit / 1e9:.2f} GB" if commit_used is not None and commit_limit else "commit=n/a"
            print(f"Monitor {condition}: C: free={current_free / 1e9:.2f} GB, {commit_text}, peak RSS={(peak_rss or 0) / 1e9:.2f} GB", flush=True)
            last_log = time.monotonic()
        time.sleep(poll_seconds)

    exit_code = process.wait()
    final_rss = _peak_working_set(process_handle)
    if final_rss is not None:
        peak_rss = max(peak_rss or 0, final_rss)
    _close_process_handle(process_handle)

    free_after_exit = _free_bytes()
    post_exit_max_free = free_after_exit
    post_exit_memory = _memory_status()
    end_recovery = time.monotonic() + recovery_seconds
    while time.monotonic() < end_recovery:
        time.sleep(min(5.0, max(0.1, end_recovery - time.monotonic())))
        post_exit_max_free = max(post_exit_max_free, _free_bytes())
        post_exit_memory = _memory_status()
        current_pagefile = _pagefile_bytes()
        if current_pagefile is not None:
            pagefile_peak = max(pagefile_peak or 0, current_pagefile)

    rows_after, lookup_after = _read_results()
    row = lookup_after.get(key)
    old_rows_preserved = all(lookup_after.get(old_key) == old_row for old_key, old_row in lookup_before.items())
    result_written = row is not None
    exactly_one_added = len(rows_after) == len(rows_before) + int(result_written)
    success = exit_code == 0 and not stop_for_disk and result_written and exactly_one_added and old_rows_preserved
    finished_at = _now()
    diagnostic = {
        "condition": condition,
        "representation": representation,
        "started_at": started_at,
        "finished_at": finished_at,
        "wall_seconds": time.monotonic() - started,
        "child_pid": process.pid,
        "exit_code": exit_code,
        "result_written": result_written,
        "completed_successfully": success,
        "stopped_for_low_disk": stop_for_disk,
        "workers": 0,
        "pin_memory": False,
        "feature_mmap_mode": "read-only" if representation == "jpeg_dct_ir" else None,
        "free_before_bytes": free_before,
        "minimum_free_during_child_bytes": minimum_free,
        "free_immediately_after_exit_bytes": free_after_exit,
        "maximum_free_during_recovery_window_bytes": post_exit_max_free,
        "free_recovery_after_exit_bytes": max(0, post_exit_max_free - minimum_free),
        "free_after_recovery_window_bytes": _free_bytes(),
        "process_peak_working_set_bytes": peak_rss,
        "process_peak_working_set_status": "captured" if peak_rss is not None else "unavailable; Windows returned no plausible process working-set value",
        "system_memory_start": system_start,
        "system_commit_peak": peak_commit,
        "system_memory_after_exit": post_exit_memory,
        "pagefile_size_start_bytes": pagefile_start,
        "pagefile_size_peak_bytes": pagefile_peak,
        "pagefile_size_after_exit_bytes": _pagefile_bytes(),
        "gpu_peak_vram_mb": row.get("peak_vram_mb") if row else None,
    }
    diagnostics.append(diagnostic)
    _write_diagnostics(diagnostics)

    print(
        f"Finished {condition}: exit={exit_code}, C: min={minimum_free / 1e9:.2f} GB, "
        f"after exit={free_after_exit / 1e9:.2f} GB, recovered={diagnostic['free_recovery_after_exit_bytes'] / 1e9:.2f} GB",
        flush=True,
    )
    if not success:
        raise RuntimeError(f"condition did not complete cleanly: {condition} / {representation}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=PROJECT_ROOT / ".venv313" / "Scripts" / "python.exe")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--recovery-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows, lookup = _read_results()
    expected = {
        *((f"pbc_{preset}", representation) for preset in ("compression", "balanced", "quality") for representation in ("pbc_lzma_bytes", "pbc_store_bytes", "pbc_parsed_fields", "pbc_canonical_ir")),
        *((f"jpeg_q{quality}", representation) for quality in (10, 20, 60, 80) for representation in ("jpeg_file_bytes", "jpeg_dct_ir")),
        ("original_rgb", "decoded_rgb_reference"),
    }
    unknown = set(lookup) - expected
    missing = [key for key in PENDING_ORDER if key not in lookup]
    unexpected_missing = expected - set(lookup) - set(PENDING_ORDER)
    if unknown or unexpected_missing:
        raise RuntimeError(f"result matrix mismatch; unknown={sorted(unknown)}, other_missing={sorted(unexpected_missing)}")

    print(f"Verified {len(rows)} recorded model results; pending={missing}", flush=True)
    if args.dry_run:
        print("Dry run: no child process launched", flush=True)
        return 0
    if not args.python.exists():
        raise FileNotFoundError(f"training interpreter not found: {args.python}")

    diagnostics = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8")) if DIAGNOSTICS_PATH.exists() else []
    for condition, representation in PENDING_ORDER:
        _, current = _read_results()
        if (condition, representation) in current:
            continue
        _run_condition(condition, representation, args.python.resolve(), args.device, args.recovery_seconds, args.poll_seconds, diagnostics)
    final_rows, final_lookup = _read_results()
    if len(final_rows) != len(expected) or set(final_lookup) != expected:
        raise RuntimeError(f"model matrix remains incomplete: {sorted(expected - set(final_lookup))}")
    print(f"Phase 3 model matrix complete: {len(final_rows)} results", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
