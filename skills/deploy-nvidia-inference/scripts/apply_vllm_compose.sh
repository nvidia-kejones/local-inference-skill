#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s --compose FILE --env FILE [--connection-file FILE | --ssh-target HOST] [--remote-dir DIR] [--state-out FILE] --apply --allow-model-downloads [--replace-existing]\n' "$0" >&2
}

ssh_target=
connection_file=
compose_file=
env_file=
remote_dir=.local/share/codex-inference/nvidia-inference
state_out=applied_deployment_state.json
apply=0
allow_downloads=0
replace_existing=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ssh-target) ssh_target=$2; shift 2 ;;
    --connection-file) connection_file=$2; shift 2 ;;
    --compose) compose_file=$2; shift 2 ;;
    --env) env_file=$2; shift 2 ;;
    --remote-dir) remote_dir=$2; shift 2 ;;
    --state-out) state_out=$2; shift 2 ;;
    --apply) apply=1; shift ;;
    --allow-model-downloads) allow_downloads=1; shift ;;
    --replace-existing) replace_existing=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [[ -z "$compose_file" || -z "$env_file" ]]; then
  usage
  exit 2
fi
if [[ -n "$connection_file" && -n "$ssh_target" ]]; then
  printf 'pass only one connection source: --connection-file or --ssh-target\n' >&2
  exit 2
fi
if [[ -z "$connection_file" && -z "$ssh_target" ]]; then
  printf 'one connection source is required: --connection-file or --ssh-target\n' >&2
  exit 2
fi
if [[ "$apply" -ne 1 || "$allow_downloads" -ne 1 ]]; then
  printf 'refusing remote write: pass both --apply and --allow-model-downloads after reviewing the plan\n' >&2
  exit 3
fi
if [[ ! -f "$compose_file" || ! -f "$env_file" ]]; then
  printf 'compose and env files must exist locally\n' >&2
  exit 2
fi
if [[ "$compose_file" == -* || "$env_file" == -* ]]; then
  printf 'compose and env file arguments must not start with a hyphen\n' >&2
  exit 2
fi
if [[ ! "$remote_dir" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  printf 'remote dir must be a simple path relative to the SSH login home\n' >&2
  exit 2
fi
if [[ "$remote_dir" == /* || "$remote_dir" == -* || "$remote_dir" =~ (^|/)\.\.(/|$) ]]; then
  printf 'remote dir must stay relative to the SSH login home without traversal components\n' >&2
  exit 2
fi

remote_compose=$remote_dir/docker-compose.yaml
remote_env=$remote_dir/deployment.env
remote_payload_dir="$(cd "$(dirname "$0")" && pwd)"
if [[ -n "$connection_file" ]]; then
  eval "$(python3 "$remote_payload_dir/remote_connection.py" --connection-file "$connection_file" --shell-env)"
fi

remote_exec() {
  local remote_command=$1
  if [[ -n "$connection_file" ]]; then
    if [[ "$TRANSPORT" == "brev" ]]; then
      command -v brev >/dev/null 2>&1 || {
        printf 'brev CLI is required for Brev connections\n' >&2
        exit 127
      }
      local brev_args=(brev exec "$BREV_INSTANCE" "$remote_command")
      "${brev_args[@]}"
      return
    fi
    ssh "${SSH_OPTIONS[@]}" "$SSH_TARGET" "$remote_command"
    return
  fi
  ssh "$ssh_target" "$remote_command"
}

remote_copy() {
  local src=$1
  local dest=$2
  if [[ -n "$connection_file" ]]; then
    if [[ "$TRANSPORT" == "brev" ]]; then
      command -v brev >/dev/null 2>&1 || {
        printf 'brev CLI is required for Brev connections\n' >&2
        exit 127
      }
      local brev_dest="${BREV_INSTANCE}:${dest}"
      brev copy "$src" "$brev_dest"
      return
    fi
    scp "${SSH_OPTIONS[@]}" "$src" "$SSH_TARGET:$dest"
    return
  fi
  scp "$src" "$ssh_target:$dest"
}

remote_exists=$(remote_exec "test -e '$remote_compose' && printf existing || true")
if [[ "$remote_exists" == "existing" && "$replace_existing" -ne 1 ]]; then
  printf 'remote compose file already exists; capture rollback state and pass --replace-existing only after review\n' >&2
  exit 4
fi

remote_exec "umask 077 && mkdir -p '$remote_dir' && chmod 700 '$remote_dir'"
remote_copy "$compose_file" "$remote_compose"
remote_copy "$env_file" "$remote_env"
vllm_image=$(awk -F= '$1=="VLLM_IMAGE"{print substr($0, length($1)+2); exit}' "$env_file")
hf_cache_path=$(awk -F= '$1=="HF_CACHE"{print substr($0, length($1)+2); exit}' "$env_file")
nvidia_visible_devices=$(awk -F= '$1=="NVIDIA_VISIBLE_DEVICES"{print substr($0, length($1)+2); exit}' "$env_file")
if [[ -n "${vllm_image:-}" ]]; then
  remote_exec "docker pull '$vllm_image'"
fi
if [[ -n "${hf_cache_path:-}" && "$hf_cache_path" == /* ]]; then
  remote_exec "umask 077 && mkdir -p '$hf_cache_path' && chmod 700 '$hf_cache_path'"
fi
remote_exec "chmod 600 '$remote_compose' '$remote_env'"
remote_exec "docker compose --env-file '$remote_env' -f '$remote_compose' config >/dev/null"
remote_exec "docker compose --env-file '$remote_env' -f '$remote_compose' up -d"
compose_ps=$(remote_exec "docker compose --env-file '$remote_env' -f '$remote_compose' ps")

python3 - "$state_out" "$compose_file" "$env_file" "$replace_existing" "$remote_dir" "$remote_compose" "$remote_env" "$compose_ps" "${hf_cache_path:-}" "${nvidia_visible_devices:-}" "${vllm_image:-}" <<'PY'
from __future__ import annotations

import datetime as dt
import json
import shlex
import sys

state_out, compose_file, env_file, replace_existing, remote_dir, remote_compose, remote_env, compose_ps, hf_cache_path, nvidia_visible_devices, vllm_image = sys.argv[1:]
state = {
    "schema_version": "nvidia-applied-deployment-state/v1",
    "applied_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "deployment_module": "vllm-compose-v1",
    "remote_dir": remote_dir,
    "local_rendered_files": {
        "compose": compose_file,
        "environment": env_file,
    },
    "replace_existing": replace_existing == "1",
    "commands": [
        "remote mkdir",
        "remote copy compose",
        "remote copy env",
        "docker pull pinned vllm image" if vllm_image else "vllm image not provided",
        "remote mkdir shared hf cache" if hf_cache_path else "shared hf cache not requested",
        "docker compose config",
        "docker compose up -d",
    ],
    "remote_compose_ps": compose_ps,
    "rollback_command": f"docker compose --env-file {shlex.quote(remote_env)} -f {shlex.quote(remote_compose)} down",
}
if hf_cache_path:
    state["shared_hf_cache_path"] = hf_cache_path
if nvidia_visible_devices:
    state["nvidia_visible_devices"] = nvidia_visible_devices
if vllm_image:
    state["vllm_image"] = vllm_image
with open(state_out, "w", encoding="utf-8") as handle:
    json.dump(state, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

printf 'wrote %s\n' "$state_out"
