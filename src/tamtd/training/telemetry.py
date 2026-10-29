from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


def _quantity_map(text: str) -> dict[str, int]:
    values = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2:
            try:
                value = int(fields[1])
            except ValueError:
                continue
            values[fields[0].rstrip(":")] = value * (1024 if len(fields) > 2 and fields[2] == "kB" else 1)
    return values


def parse_proc_status(text: str) -> dict[str, int | None]:
    values = _quantity_map(text)
    return {
        "rss_bytes": values.get("VmRSS"),
        "virtual_bytes": values.get("VmSize"),
    }


def parse_meminfo(text: str) -> dict[str, int | None]:
    values = _quantity_map(text)
    available = values.get("MemAvailable")
    if available is None:
        available = sum(values.get(key, 0) for key in ("MemFree", "Buffers", "Cached")) or None
    swap_total = values.get("SwapTotal")
    swap_free = values.get("SwapFree")
    return {
        "memory_total_bytes": values.get("MemTotal"),
        "memory_available_bytes": available,
        "swap_total_bytes": swap_total,
        "swap_used_bytes": max(swap_total - swap_free, 0) if swap_total is not None and swap_free is not None else None,
    }


def _read_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
        return int(value) if value != "max" else None
    except (OSError, ValueError):
        return None


def _cgroup_memory(pid: int, proc_root: Path, cgroup_root: Path) -> dict[str, int | None]:
    result = {
        "memory_current_bytes": None,
        "memory_peak_bytes": None,
        "memory_limit_bytes": None,
        "swap_current_bytes": None,
        "swap_limit_bytes": None,
    }
    try:
        entries = (proc_root / str(pid) / "cgroup").read_text(encoding="utf-8").splitlines()
    except OSError:
        return result

    for entry in entries:
        fields = entry.split(":", 2)
        if len(fields) != 3:
            continue
        _, controllers, relative = fields
        if not controllers:
            base = cgroup_root / relative.lstrip("/")
            if (base / "memory.current").is_file():
                result.update(
                    memory_current_bytes=_read_integer(base / "memory.current"),
                    memory_peak_bytes=_read_integer(base / "memory.peak"),
                    memory_limit_bytes=_read_integer(base / "memory.max"),
                    swap_current_bytes=_read_integer(base / "memory.swap.current"),
                    swap_limit_bytes=_read_integer(base / "memory.swap.max"),
                )
                return result
        elif "memory" in controllers.split(","):
            base = cgroup_root / "memory" / relative.lstrip("/")
            current = _read_integer(base / "memory.usage_in_bytes")
            if current is not None:
                memory_limit = _read_integer(base / "memory.limit_in_bytes")
                memsw_current = _read_integer(base / "memory.memsw.usage_in_bytes")
                memsw_limit = _read_integer(base / "memory.memsw.limit_in_bytes")
                result.update(
                    memory_current_bytes=current,
                    memory_peak_bytes=_read_integer(base / "memory.max_usage_in_bytes"),
                    memory_limit_bytes=memory_limit,
                    swap_current_bytes=max(memsw_current - current, 0) if memsw_current is not None else None,
                    swap_limit_bytes=max(memsw_limit - memory_limit, 0) if memsw_limit is not None and memory_limit is not None else None,
                )
                return result
    return result


def _windows_memory() -> dict[str, int | None]:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    try:
        query = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
        query.argtypes = [ctypes.POINTER(MemoryStatus)]
        query.restype = ctypes.c_int
        if not query(ctypes.byref(status)):
            raise OSError(ctypes.get_last_error(), "GlobalMemoryStatusEx failed")
    except (AttributeError, OSError) as error:
        return {"status": f"unavailable: {error}"}
    return {
        "memory_total_bytes": int(status.ullTotalPhys),
        "memory_available_bytes": int(status.ullAvailPhys),
        "commit_used_bytes": int(status.ullTotalPageFile - status.ullAvailPageFile),
        "commit_limit_bytes": int(status.ullTotalPageFile),
    }


def _gpu_snapshot(torch_module: Any = None, device: Any = None) -> dict[str, int | float | None]:
    result: dict[str, int | float | None] = {
        "cuda_allocated_bytes": None,
        "cuda_allocated_peak_bytes": None,
        "cuda_reserved_bytes": None,
        "cuda_reserved_peak_bytes": None,
        "vram_used_mb": None,
        "vram_free_mb": None,
        "vram_total_mb": None,
        "utilization_percent": None,
    }
    if torch_module is not None:
        try:
            if torch_module.cuda.is_available():
                selected = device if device is not None and getattr(device, "type", None) == "cuda" else torch_module.device("cuda")
                result.update(
                    cuda_allocated_bytes=int(torch_module.cuda.memory_allocated(selected)),
                    cuda_allocated_peak_bytes=int(torch_module.cuda.max_memory_allocated(selected)),
                    cuda_reserved_bytes=int(torch_module.cuda.memory_reserved(selected)),
                    cuda_reserved_peak_bytes=int(torch_module.cuda.max_memory_reserved(selected)),
                )
        except (AttributeError, RuntimeError, TypeError):
            pass
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            completed = subprocess.run(
                [
                    executable,
                    "--query-gpu=memory.free,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            rows = [tuple(int(part.strip()) for part in line.split(",")) for line in completed.stdout.splitlines() if line.strip()]
            if rows:
                result["vram_free_mb"] = min(free for free, _, _, _ in rows)
                result["vram_used_mb"] = max(used for _, used, _, _ in rows)
                result["vram_total_mb"] = max(total for _, _, total, _ in rows)
                result["utilization_percent"] = max(utilization for _, _, _, utilization in rows)
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    return result


def collect_resource_snapshot(
    torch_module: Any = None,
    device: Any = None,
    disk_path: str | Path | None = None,
    pid: int | None = None,
    platform_name: str | None = None,
    proc_root: str | Path = "/proc",
    cgroup_root: str | Path = "/sys/fs/cgroup",
    include_gpu: bool = True,
) -> dict:
    platform_name = platform_name or sys.platform
    pid = os.getpid() if pid is None else int(pid)
    proc_root = Path(proc_root)
    if platform_name.startswith("linux"):
        try:
            process = parse_proc_status((proc_root / str(pid) / "status").read_text(encoding="utf-8"))
            process["status"] = "available_from_procfs"
        except OSError as error:
            process = {"rss_bytes": None, "virtual_bytes": None, "status": f"unavailable: {error}"}
        try:
            system = parse_meminfo((proc_root / "meminfo").read_text(encoding="utf-8"))
        except OSError as error:
            system = {"status": f"unavailable: {error}"}
        cgroup = _cgroup_memory(pid, proc_root, Path(cgroup_root))
        process["rss_reliability"] = "procfs resident set"
    elif platform_name.startswith("win"):
        process = {
            "rss_bytes": None,
            "virtual_bytes": None,
            "status": "unavailable: process working-set counters are not trusted on Windows",
            "rss_reliability": "unavailable",
        }
        system = _windows_memory()
        cgroup = {key: None for key in ("memory_current_bytes", "memory_peak_bytes", "memory_limit_bytes", "swap_current_bytes", "swap_limit_bytes")}
    else:
        process = {"rss_bytes": None, "virtual_bytes": None, "status": "unavailable: unsupported platform"}
        system = {"status": "unavailable: unsupported platform"}
        cgroup = {key: None for key in ("memory_current_bytes", "memory_peak_bytes", "memory_limit_bytes", "swap_current_bytes", "swap_limit_bytes")}
    try:
        disk_free = shutil.disk_usage(Path(disk_path) if disk_path else Path.cwd()).free
    except OSError:
        disk_free = None
    return {
        "timestamp": time.time(),
        "platform": platform_name,
        "process": process,
        "system": system,
        "cgroup": cgroup,
        "disk_free_bytes": disk_free,
        "gpu": _gpu_snapshot(torch_module, device) if include_gpu else {},
    }


class ResourceMonitor:
    def __init__(
        self,
        torch_module: Any = None,
        device: Any = None,
        disk_path: str | Path | None = None,
        process_id: int | None = None,
        interval_seconds: float = 1.0,
        gpu_interval_seconds: float = 5.0,
    ) -> None:
        self.torch_module = torch_module
        self.device = device
        self.disk_path = disk_path
        self.process_id = process_id
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.gpu_interval_seconds = max(self.interval_seconds, float(gpu_interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._sample_count = 0
        self._first: dict | None = None
        self._last: dict | None = None
        self._peaks: dict[str, int | float] = {}
        self._minima: dict[str, int | float] = {}
        self._gpu_utilization_sum = 0
        self._gpu_utilization_samples = 0
        self._gpu_due = 0.0

    def _capture(self) -> None:
        now = time.monotonic()
        include_gpu = now >= self._gpu_due
        snapshot = collect_resource_snapshot(
            torch_module=self.torch_module,
            device=self.device,
            disk_path=self.disk_path,
            pid=self.process_id,
        )
        if not include_gpu:
            snapshot["gpu"] = {}
        else:
            self._gpu_due = now + self.gpu_interval_seconds
        with self._lock:
            if self._first is None:
                self._first = snapshot
            self._last = snapshot
            self._sample_count += 1
            for category, fields in (
                ("process", ("rss_bytes", "virtual_bytes")),
                ("system", ("commit_used_bytes", "swap_used_bytes")),
                ("cgroup", ("memory_current_bytes", "memory_peak_bytes", "swap_current_bytes")),
                ("gpu", ("cuda_allocated_peak_bytes", "cuda_reserved_peak_bytes", "vram_used_mb", "utilization_percent")),
            ):
                source = snapshot.get(category, {})
                for field in fields:
                    value = source.get(field)
                    if value is not None:
                        self._peaks[field] = max(self._peaks.get(field, value), value)
            disk_free = snapshot.get("disk_free_bytes")
            if disk_free is not None:
                self._minima["disk_free_bytes"] = min(self._minima.get("disk_free_bytes", disk_free), disk_free)
            utilization = snapshot.get("gpu", {}).get("utilization_percent")
            if utilization is not None:
                self._gpu_utilization_sum += int(utilization)
                self._gpu_utilization_samples += 1
            vram_free = snapshot.get("gpu", {}).get("vram_free_mb")
            if vram_free is not None:
                self._minima["vram_free_mb"] = min(self._minima.get("vram_free_mb", vram_free), vram_free)
            available = snapshot.get("system", {}).get("memory_available_bytes")
            if available is not None:
                self._minima["memory_available_bytes"] = min(
                    self._minima.get("memory_available_bytes", available), available
                )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._capture()

    def start(self) -> "ResourceMonitor":
        if self.torch_module is not None and self.device is not None and getattr(self.device, "type", None) == "cuda":
            try:
                self.torch_module.cuda.reset_peak_memory_stats(self.device)
            except (AttributeError, RuntimeError):
                pass
        self._capture()
        self._thread = threading.Thread(target=self._run, name="tamtd-resource-monitor", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 2))
        self._capture()
        with self._lock:
            return {
                "schema": "tamtd.resource-diagnostics.v1",
                "sample_count": self._sample_count,
                "sampling_interval_seconds": self.interval_seconds,
                "start": self._first,
                "end": self._last,
                "peaks": dict(self._peaks),
                "minima": dict(self._minima),
                "gpu_utilization_mean_percent": (
                    self._gpu_utilization_sum / self._gpu_utilization_samples
                    if self._gpu_utilization_samples
                    else None
                ),
            }
