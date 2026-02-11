"""Hardware metadata for reproducible benchmarking.

Records the exact system configuration so results can be compared
across machines (e.g. M1 Pro vs M4 Max).
"""
from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass(frozen=True)
class HardwareInfo:
    chip: str
    cpu_cores_total: int
    cpu_cores_performance: int
    cpu_cores_efficiency: int
    gpu_cores: int
    ram_gb: int
    os_version: str
    python_version: str
    architecture: str

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.chip} | {self.cpu_cores_total} CPU cores "
            f"({self.cpu_cores_performance}P+{self.cpu_cores_efficiency}E) | "
            f"{self.gpu_cores}-core GPU | {self.ram_gb} GB RAM | "
            f"macOS {self.os_version} | Python {self.python_version}"
        )


def detect_hardware() -> HardwareInfo:
    """Detect hardware on macOS Apple Silicon. Falls back to manual values."""
    os_version = platform.mac_ver()[0] or "unknown"
    python_version = platform.python_version()
    architecture = platform.machine()

    # Try sysctl for chip info
    chip = _sysctl("machdep.cpu.brand_string") or "Apple M1 Pro"
    ram_bytes = _sysctl("hw.memsize")
    ram_gb = int(int(ram_bytes) / (1024 ** 3)) if ram_bytes else 32

    # CPU core counts
    total_cores = int(_sysctl("hw.ncpu") or "8")
    perf_cores = int(_sysctl("hw.perflevel0.logicalcpu") or "6")
    eff_cores = int(_sysctl("hw.perflevel1.logicalcpu") or "2")

    # GPU cores (not directly available via sysctl, use known values)
    gpu_cores = _detect_gpu_cores(chip)

    return HardwareInfo(
        chip=chip,
        cpu_cores_total=total_cores,
        cpu_cores_performance=perf_cores,
        cpu_cores_efficiency=eff_cores,
        gpu_cores=gpu_cores,
        ram_gb=ram_gb,
        os_version=os_version,
        python_version=python_version,
        architecture=architecture,
    )


def _sysctl(key: str) -> Optional[str]:
    try:
        result = subprocess.check_output(
            ["sysctl", "-n", key], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return result if result else None
    except Exception:
        return None


def _detect_gpu_cores(chip: str) -> int:
    """Return GPU core count based on chip name."""
    chip_lower = chip.lower()
    known = {
        "m1 pro": 14, "m1 max": 24, "m1 ultra": 48, "m1": 7,
        "m2 pro": 16, "m2 max": 30, "m2 ultra": 60, "m2": 8,
        "m3 pro": 14, "m3 max": 30, "m3 ultra": 60, "m3": 8,
        "m4 pro": 16, "m4 max": 32, "m4 ultra": 64, "m4": 10,
    }
    for name, cores in sorted(known.items(), key=lambda x: -len(x[0])):
        if name in chip_lower:
            return cores
    return 14  # conservative default


# Singleton for use across test suite
HARDWARE = detect_hardware()
