#!/usr/bin/env python3
"""Read-only remote host probe payload for SSH or Brev exec."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import socket
import subprocess
from typing import Any


def run(name: str, command: list[str], timeout: int = 30) -> tuple[str, dict[str, Any]]:
    entry: dict[str, Any] = {"command": command}
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        entry.update(
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    except FileNotFoundError as exc:
        entry.update({"returncode": 127, "stdout": "", "stderr": str(exc)})
    except subprocess.TimeoutExpired as exc:
        entry.update(
            {
                "returncode": 124,
                "stdout": exc.stdout or "",
                "stderr": f"timed out after {timeout}s",
            }
        )
    return name, entry


commands = [
    ("hostname_fqdn", ["hostname", "-f"]),
    ("os_release", ["cat", "/etc/os-release"]),
    ("kernel", ["uname", "-a"]),
    ("lscpu", ["lscpu"]),
    ("meminfo", ["cat", "/proc/meminfo"]),
    ("home_dir", ["sh", "-lc", "printf %s \"$HOME\""]),
    ("disk", ["df", "-B1", "-P", "/", "/home", "/var/lib/docker", "/var/lib/containerd"]),
    ("nvidia_smi_table", ["nvidia-smi"]),
    ("nvidia_smi_list", ["nvidia-smi", "-L"]),
    (
        "nvidia_smi_gpu_query_extended",
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version,compute_cap,mig.mode.current",
            "--format=csv,noheader,nounits",
        ],
    ),
    (
        "nvidia_smi_gpu_query_basic",
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
    ),
    (
        "nvidia_smi_compute_apps",
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
    ),
    ("nvidia_smi_xml", ["nvidia-smi", "-q", "-x"], 60),
    ("nvidia_smi_topology", ["nvidia-smi", "topo", "-m"]),
    ("nvidia_smi_p2p_nvlink", ["nvidia-smi", "topo", "-p2p", "n"]),
    ("nvidia_driver_proc", ["cat", "/proc/driver/nvidia/version"]),
    ("docker_version", ["docker", "version", "--format", "{{json .}}"]),
    ("docker_runtimes", ["docker", "info", "--format", "{{json .Runtimes}}"]),
    ("containerd_version", ["containerd", "--version"]),
    ("podman_version", ["podman", "--version"]),
    ("nvidia_ctk_version", ["nvidia-ctk", "--version"]),
    ("nvidia_container_cli_version", ["nvidia-container-cli", "--version"]),
    (
        "listening_ports",
        [
            "sh",
            "-lc",
            "command -v ss >/dev/null 2>&1 && ss -H -ltnup || netstat -ltnup",
        ],
    ),
    (
        "inference_processes",
        [
            "sh",
            "-lc",
            "ps -eo pid,user,comm | grep -Eis 'vllm|sglang|trtllm|tensorrt|llama-server|ollama|text-generation' | grep -Ev 'grep -Eis' || true",
        ],
    ),
]

result: dict[str, Any] = {
    "schema_version": "nvidia-host-probe/raw-v1",
    "collector": "probe_remote_host_payload.py",
    "collected_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "remote_python": platform.python_version(),
    "remote_hostname": socket.gethostname(),
    "remote_user": os.environ.get("USER"),
    "commands": {},
}
for item in commands:
    name, command, *timeout = item
    key, payload = run(name, command, timeout[0] if timeout else 30)
    result["commands"][key] = payload

json.dump(result, fp=os.sys.stdout, indent=2, sort_keys=True)
os.sys.stdout.write("\n")
