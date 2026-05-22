#!/usr/bin/env python3
"""Normalize read-only probe evidence into host_facts.json."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from common_io import as_int, write_json


MIB = 1024**2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", help="raw JSON emitted by probe_remote_host.sh")
    parser.add_argument("--out", help="write normalized host facts JSON here")
    args = parser.parse_args()

    raw = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    commands = raw.get("commands", {})
    gpus = parse_gpus(commands)
    driver_version = next((gpu.get("driver_version") for gpu in gpus if gpu.get("driver_version")), None)
    cuda_hint = find_cuda_version(command_stdout(commands, "nvidia_smi_table"))
    memory_reporting = describe_memory_reporting(commands, gpus)
    facts = {
        "schema_version": "nvidia-host-facts/v1",
        "fact_scope": "read_only_discovery",
        "source": {
            "collector": raw.get("collector"),
            "collected_at_utc": raw.get("collected_at_utc"),
            "remote_hostname": raw.get("remote_hostname"),
            "remote_user": raw.get("remote_user"),
            "command_status": command_status(commands),
        },
        "host": {
            "hostname": first_line(command_stdout(commands, "hostname_fqdn"))
            or raw.get("remote_hostname"),
            "home_dir": first_line(command_stdout(commands, "home_dir")),
            "os": parse_os_release(command_stdout(commands, "os_release")),
            "kernel": command_stdout(commands, "kernel").strip(),
            "cpu": parse_lscpu(command_stdout(commands, "lscpu")),
            "memory": parse_meminfo(command_stdout(commands, "meminfo")),
            "disk": parse_disk(command_stdout(commands, "disk")),
        },
        "nvidia": {
            "available": bool(gpus),
            "driver_version": driver_version,
            "cuda_version_reported_by_driver": cuda_hint,
            "gpus": gpus,
            "memory_reporting": memory_reporting,
            "mig": {
                "states": [
                    {
                        "uuid": gpu.get("uuid"),
                        "index": gpu.get("index"),
                        "mode": gpu.get("mig_mode"),
                    }
                    for gpu in gpus
                    if gpu.get("mig_mode")
                ],
                "list_output": command_stdout(commands, "nvidia_smi_list").strip(),
            },
            "topology": {
                "matrix": command_stdout(commands, "nvidia_smi_topology").strip(),
                "p2p_nvlink": command_stdout(commands, "nvidia_smi_p2p_nvlink").strip(),
            },
            "visible_gpu_workloads": parse_gpu_workloads(
                command_stdout(commands, "nvidia_smi_compute_apps")
            ),
            "driver_proc_version": command_stdout(commands, "nvidia_driver_proc").strip(),
            "compatibility_hints": compatibility_hints(driver_version, cuda_hint),
        },
        "containers": parse_containers(commands),
        "network": {
            "listening_ports": parse_ports(command_stdout(commands, "listening_ports")),
            "active_inference_processes": parse_process_lines(
                command_stdout(commands, "inference_processes")
            ),
        },
        "observations": observations(commands, gpus, memory_reporting),
    }
    write_json(facts, args.out)


def command_stdout(commands: dict[str, Any], name: str) -> str:
    return str(commands.get(name, {}).get("stdout") or "")


def command_status(commands: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "command": entry.get("command"),
            "returncode": entry.get("returncode"),
            "stderr": str(entry.get("stderr") or "").strip()[:400],
        }
        for name, entry in commands.items()
    }


def command_ok(commands: dict[str, Any], name: str) -> bool:
    return commands.get(name, {}).get("returncode") == 0


def parse_gpus(commands: dict[str, Any]) -> list[dict[str, Any]]:
    extended = command_stdout(commands, "nvidia_smi_gpu_query_extended")
    rows = csv_rows(extended) if command_ok(commands, "nvidia_smi_gpu_query_extended") else []
    fields = [
        "index",
        "name",
        "uuid",
        "vram_total_mib",
        "vram_free_mib",
        "driver_version",
        "compute_capability",
        "mig_mode",
    ]
    if not rows:
        rows = csv_rows(command_stdout(commands, "nvidia_smi_gpu_query_basic"))
        fields = fields[:6]
    gpus = []
    for row in rows:
        if len(row) < len(fields):
            continue
        gpu = dict(zip(fields, row))
        total_mib = numeric_mib(gpu.pop("vram_total_mib", None))
        free_mib = numeric_mib(gpu.pop("vram_free_mib", None))
        gpu["index"] = as_int(gpu.get("index"), -1)
        gpu["vram_total_bytes"] = total_mib * MIB
        gpu["vram_free_bytes"] = free_mib * MIB
        for key in ("compute_capability", "mig_mode"):
            if key in gpu and unavailable(gpu[key]):
                gpu[key] = None
        gpus.append(gpu)
    return gpus


def csv_rows(text: str) -> list[list[str]]:
    return [[cell.strip() for cell in row] for row in csv.reader(text.splitlines()) if row]


def numeric_mib(value: Any) -> int:
    match = re.search(r"\d+", str(value or ""))
    return int(match.group(0)) if match else 0


def unavailable(value: Any) -> bool:
    return str(value or "").strip().lower() in {
        "",
        "n/a",
        "[n/a]",
        "not supported",
        "[not supported]",
    }


def describe_memory_reporting(
    commands: dict[str, Any], gpus: list[dict[str, Any]]
) -> dict[str, Any]:
    table = command_stdout(commands, "nvidia_smi_table")
    query_text = "\n".join(
        [
            command_stdout(commands, "nvidia_smi_gpu_query_extended"),
            command_stdout(commands, "nvidia_smi_gpu_query_basic"),
        ]
    )
    has_framebuffer_facts = any(
        as_int(gpu.get("vram_total_bytes"), 0) > 0 or as_int(gpu.get("vram_free_bytes"), 0) > 0
        for gpu in gpus
    )
    no_framebuffer_values = bool(gpus) and not has_framebuffer_facts
    table_reports_unsupported = "memory-usage" in table.lower() and "not supported" in table.lower()
    query_reports_unavailable = any(
        marker in query_text.lower() for marker in ("[n/a]", "not supported")
    )
    gb10_inventory_hint = any("gb10" in str(gpu.get("name") or "").lower() for gpu in gpus)
    unified_memory_hint = bool(
        no_framebuffer_values
        and table_reports_unsupported
        and query_reports_unavailable
        and gb10_inventory_hint
    )
    if has_framebuffer_facts:
        mode = "dedicated_framebuffer"
    elif unified_memory_hint:
        mode = "unified_system_memory_hint"
    else:
        mode = "unknown"

    evidence = []
    if no_framebuffer_values:
        evidence.append("Normalized GPU rows did not expose nonzero framebuffer total/free facts.")
    if table_reports_unsupported:
        evidence.append("nvidia-smi table reported Memory-Usage as Not Supported.")
    if query_reports_unavailable:
        evidence.append("nvidia-smi GPU memory query fields reported unavailable values.")
    if gb10_inventory_hint:
        evidence.append("GPU inventory includes a GB10 device name.")
    return {
        "mode": mode,
        "framebuffer_memory_facts_available": has_framebuffer_facts,
        "unified_memory_hint": unified_memory_hint,
        "system_memory_budget_eligible": unified_memory_hint,
        "evidence": evidence,
    }


def find_cuda_version(text: str) -> str | None:
    match = re.search(r"CUDA Version:\s*([0-9.]+)", text)
    return match.group(1) if match else None


def parse_os_release(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        result[key.lower()] = value.strip().strip('"')
    return result


def parse_lscpu(text: str) -> dict[str, Any]:
    raw: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            raw[key.strip().lower()] = value.strip()
    return {
        "architecture": raw.get("architecture"),
        "model_name": raw.get("model name"),
        "logical_cpus": as_int(raw.get("cpu(s)"), 0),
        "sockets": as_int(raw.get("socket(s)"), 0),
        "cores_per_socket": as_int(raw.get("core(s) per socket"), 0),
    }


def parse_meminfo(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"([^:]+):\s+(\d+)\s+kB", line)
        if match:
            values[match.group(1)] = int(match.group(2)) * 1024
    return {
        "total_bytes": values.get("MemTotal", 0),
        "available_bytes": values.get("MemAvailable", 0),
        "swap_total_bytes": values.get("SwapTotal", 0),
    }


def parse_disk(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("Filesystem"):
            continue
        fields = line.split()
        if len(fields) < 6:
            continue
        rows.append(
            {
                "filesystem": fields[0],
                "size_bytes": as_int(fields[1], 0),
                "used_bytes": as_int(fields[2], 0),
                "available_bytes": as_int(fields[3], 0),
                "mount": fields[-1],
            }
        )
    return rows


def parse_gpu_workloads(text: str) -> list[dict[str, Any]]:
    rows = []
    for cells in csv_rows(text):
        if len(cells) >= 4:
            rows.append(
                {
                    "pid": as_int(cells[0], 0),
                    "process_name": cells[1],
                    "used_memory_bytes": numeric_mib(cells[2]) * MIB,
                    "gpu_uuid": cells[3],
                }
            )
    return rows


def parse_containers(commands: dict[str, Any]) -> dict[str, Any]:
    runtimes_text = command_stdout(commands, "docker_runtimes").strip()
    return {
        "docker": {
            "available": command_ok(commands, "docker_version"),
            "version_output": command_stdout(commands, "docker_version").strip(),
            "runtimes_output": runtimes_text,
            "nvidia_runtime_hint": "nvidia" in runtimes_text.lower(),
        },
        "containerd": {
            "available": command_ok(commands, "containerd_version"),
            "version_output": command_stdout(commands, "containerd_version").strip(),
        },
        "podman": {
            "available": command_ok(commands, "podman_version"),
            "version_output": command_stdout(commands, "podman_version").strip(),
        },
        "nvidia_container_toolkit": {
            "available": command_ok(commands, "nvidia_ctk_version")
            or command_ok(commands, "nvidia_container_cli_version"),
            "nvidia_ctk_version": command_stdout(commands, "nvidia_ctk_version").strip(),
            "nvidia_container_cli_version": command_stdout(
                commands, "nvidia_container_cli_version"
            ).strip(),
        },
    }


def parse_ports(text: str) -> list[dict[str, str]]:
    ports = []
    for line in text.splitlines():
        fields = line.split()
        local = next(
            (field for field in fields if re.search(r":\d+$", field) and "users:" not in field),
            "",
        )
        ports.append(
            {
                "protocol": fields[0] if fields else "",
                "local": local,
                "evidence": line.strip(),
            }
        )
    return ports


def parse_process_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def compatibility_hints(driver_version: str | None, cuda_hint: str | None) -> list[str]:
    hints = []
    if driver_version:
        hints.append(f"NVIDIA driver reported by nvidia-smi: {driver_version}.")
    if cuda_hint:
        hints.append(
            f"nvidia-smi reports CUDA API compatibility up to {cuda_hint}; verify the selected container image against driver compatibility before apply."
        )
    hints.append(
        "A host CUDA toolkit is not sufficient evidence for container compatibility; check driver, runtime image, and NVIDIA Container Toolkit together."
    )
    return hints


def observations(
    commands: dict[str, Any],
    gpus: list[dict[str, Any]],
    memory_reporting: dict[str, Any],
) -> list[str]:
    result = []
    if not gpus:
        result.append("No GPU query rows were normalized from nvidia-smi.")
    if memory_reporting.get("mode") == "unified_system_memory_hint":
        result.append(
            "Dedicated framebuffer memory facts were unavailable; normalized facts allow a conservative unified-memory system budget."
        )
    if not command_ok(commands, "docker_version"):
        result.append("Docker was unavailable or inaccessible during read-only discovery.")
    if not command_ok(commands, "nvidia_ctk_version") and not command_ok(
        commands, "nvidia_container_cli_version"
    ):
        result.append("NVIDIA Container Toolkit commands were not visible during discovery.")
    return result


def first_line(text: str) -> str | None:
    return next((line.strip() for line in text.splitlines() if line.strip()), None)


if __name__ == "__main__":
    main()
