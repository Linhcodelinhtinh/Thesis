#!/usr/bin/env python3
"""Hardware and System Inventory Script (Phase 1).

Captures CPU, GPU, VRAM, Driver, CUDA, RAM, and OS specifications
and writes the structured inventory to resources/manifests/hardware_inventory.json.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "resources" / "manifests" / "hardware_inventory.json"


def get_cpu_info():
    info = {
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "logical_cores": os.cpu_count(),
    }
    if sys.platform == "win32":
        try:
            cmd = "Get-CimInstance Win32_Processor | Select-Object -Property Name, NumberOfCores, NumberOfLogicalProcessors | ConvertTo-Json"
            out = subprocess.check_output(["powershell", "-Command", cmd], text=True)
            data = json.loads(out)
            if isinstance(data, list):
                data = data[0]
            info["name"] = data.get("Name", "").strip()
            info["physical_cores"] = data.get("NumberOfCores")
            info["logical_cores"] = data.get("NumberOfLogicalProcessors")
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r") as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("model name"):
                    info["name"] = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
    return info


def get_ram_info():
    info = {"total_gb": None}
    if sys.platform == "win32":
        try:
            cmd = "(Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize"
            out = subprocess.check_output(["powershell", "-Command", cmd], text=True).strip()
            info["total_gb"] = round(float(out) / (1024 * 1024), 2)
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = float(line.split()[1])
                        info["total_gb"] = round(kb / (1024 * 1024), 2)
                        break
        except Exception:
            pass
    return info


def get_gpu_info():
    info = {
        "available": False,
        "name": None,
        "driver_version": None,
        "cuda_version": None,
        "total_vram_mib": None,
    }
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            query = "name,driver_version,memory.total"
            cmd = [
                nvidia_smi,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ]
            out = subprocess.check_output(cmd, text=True).strip()
            if out:
                parts = [p.strip() for p in out.splitlines()[0].split(",")]
                info["available"] = True
                info["name"] = parts[0]
                info["driver_version"] = parts[1]
                info["total_vram_mib"] = float(parts[2])
        except Exception as e:
            info["error"] = str(e)

        try:
            cmd = [nvidia_smi]
            out = subprocess.check_output(cmd, text=True)
            for line in out.splitlines():
                if "CUDA Version:" in line:
                    parts = line.split("CUDA Version:")
                    info["cuda_version"] = parts[1].split()[0].replace("|", "").strip()
                    break
        except Exception:
            pass

    return info


def collect_inventory():
    cpu = get_cpu_info()
    ram = get_ram_info()
    gpu = get_gpu_info()

    inventory = {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "compiler": platform.python_compiler(),
        },
        "cpu": cpu,
        "ram": ram,
        "gpu": gpu,
    }
    return inventory


def main():
    print("=" * 60)
    print("VLA Policy Evaluation Sandbox V1 — System Inventory (Phase 1)")
    print("=" * 60)

    inv = collect_inventory()

    print(f"OS:     {inv['platform']['system']} {inv['platform']['release']} ({inv['platform']['machine']})")
    print(f"Python: {inv['python']['version']} ({inv['python']['executable']})")
    print(f"CPU:    {inv['cpu'].get('name', inv['cpu']['processor'])} | Cores: {inv['cpu'].get('physical_cores', 'N/A')} physical, {inv['cpu'].get('logical_cores')} logical")
    print(f"RAM:    {inv['ram'].get('total_gb')} GB")
    if inv['gpu']['available']:
        print(f"GPU:    {inv['gpu']['name']} | VRAM: {inv['gpu']['total_vram_mib']} MiB | Driver: {inv['gpu']['driver_version']} | CUDA: {inv['gpu']['cuda_version']}")
    else:
        print("GPU:    Not detected or nvidia-smi unavailable")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2)

    print("-" * 60)
    print(f"Hardware inventory saved to: {OUTPUT_FILE}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
