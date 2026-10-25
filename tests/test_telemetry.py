import tempfile
import unittest
from pathlib import Path

from tamtd.training.telemetry import collect_resource_snapshot, parse_meminfo, parse_proc_status


class TelemetryTests(unittest.TestCase):
    def test_linux_process_memory_and_meminfo_parsers(self):
        process = parse_proc_status("Name:\tpython\nVmSize:\t2048 kB\nVmRSS:\t512 kB\n")
        self.assertEqual(process, {"rss_bytes": 512 * 1024, "virtual_bytes": 2048 * 1024})
        system = parse_meminfo("MemTotal: 8192 kB\nMemAvailable: 4096 kB\nSwapTotal: 2048 kB\nSwapFree: 1536 kB\n")
        self.assertEqual(system["memory_total_bytes"], 8192 * 1024)
        self.assertEqual(system["memory_available_bytes"], 4096 * 1024)
        self.assertEqual(system["swap_used_bytes"], 512 * 1024)

    def test_linux_snapshot_reads_cgroup_v2_and_process_stats(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            proc = root / "proc"
            cgroup = root / "sys/fs/cgroup"
            pid = 42
            (proc / str(pid)).mkdir(parents=True)
            (cgroup / "job").mkdir(parents=True)
            (proc / str(pid) / "status").write_text("VmRSS:\t100 kB\nVmSize:\t200 kB\n")
            (proc / str(pid) / "cgroup").write_text("0::/job\n")
            (proc / "meminfo").write_text("MemTotal: 100000 kB\nMemAvailable: 50000 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n")
            (cgroup / "job/memory.current").write_text("12345\n")
            (cgroup / "job/memory.peak").write_text("23456\n")
            (cgroup / "job/memory.max").write_text("50000\n")
            (cgroup / "job/memory.swap.current").write_text("0\n")
            (cgroup / "job/memory.swap.max").write_text("max\n")
            snapshot = collect_resource_snapshot(
                pid=pid,
                platform_name="linux",
                proc_root=proc,
                cgroup_root=cgroup,
                disk_path=root,
                include_gpu=False,
            )
            self.assertEqual(snapshot["process"]["rss_bytes"], 100 * 1024)
            self.assertEqual(snapshot["process"]["virtual_bytes"], 200 * 1024)
            self.assertEqual(snapshot["system"]["memory_available_bytes"], 50000 * 1024)
            self.assertEqual(snapshot["cgroup"]["memory_current_bytes"], 12345)
            self.assertEqual(snapshot["cgroup"]["memory_peak_bytes"], 23456)
            self.assertIsNone(snapshot["cgroup"]["swap_limit_bytes"])
            self.assertEqual(snapshot["gpu"], {})

    def test_linux_cgroup_v1_reports_swap_without_double_counting_memory(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            proc = root / "proc"
            cgroup = root / "sys/fs/cgroup"
            pid = 43
            (proc / str(pid)).mkdir(parents=True)
            (cgroup / "memory/job").mkdir(parents=True)
            (proc / str(pid) / "cgroup").write_text("5:memory:/job\n")
            (proc / str(pid) / "status").write_text("VmRSS:\t10 kB\nVmSize:\t20 kB\n")
            (proc / "meminfo").write_text("MemTotal: 100000 kB\nMemAvailable: 50000 kB\nSwapTotal: 1000 kB\nSwapFree: 500 kB\n")
            (cgroup / "memory/job/memory.usage_in_bytes").write_text("1000\n")
            (cgroup / "memory/job/memory.max_usage_in_bytes").write_text("1500\n")
            (cgroup / "memory/job/memory.limit_in_bytes").write_text("5000\n")
            (cgroup / "memory/job/memory.memsw.usage_in_bytes").write_text("1300\n")
            (cgroup / "memory/job/memory.memsw.limit_in_bytes").write_text("7000\n")
            snapshot = collect_resource_snapshot(
                pid=pid,
                platform_name="linux",
                proc_root=proc,
                cgroup_root=cgroup,
                disk_path=root,
                include_gpu=False,
            )
            self.assertEqual(snapshot["cgroup"]["memory_current_bytes"], 1000)
            self.assertEqual(snapshot["cgroup"]["swap_current_bytes"], 300)
            self.assertEqual(snapshot["cgroup"]["swap_limit_bytes"], 2000)

    def test_windows_snapshot_marks_process_rss_unavailable(self):
        snapshot = collect_resource_snapshot(platform_name="win32", include_gpu=False)
        self.assertIsNone(snapshot["process"]["rss_bytes"])
        self.assertIsNone(snapshot["process"]["virtual_bytes"])
        self.assertIn("not trusted on Windows", snapshot["process"]["status"])
        if "commit_used_bytes" in snapshot["system"]:
            self.assertGreaterEqual(snapshot["system"]["commit_limit_bytes"], snapshot["system"]["commit_used_bytes"])
        else:
            self.assertIn("unavailable", snapshot["system"]["status"])


if __name__ == "__main__":
    unittest.main()
