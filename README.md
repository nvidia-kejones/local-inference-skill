# local-inference

Agent Skills for local and remote inference work.

## Skills

### `deploy-nvidia-inference`

Use this Agent Skill to discover a single remote NVIDIA Linux host over SSH, recommend host-aware model/runtime pairs for workloads, render a safe deployment plan, apply the vLLM Docker Compose baseline explicitly, and verify an OpenAI-compatible inference endpoint.

Supported recommendation/deployment targets:

- vLLM
- SGLang
- TensorRT-LLM / `trtllm-serve`
- llama.cpp
- Ollama

The v1 scripted apply path is vLLM Compose. The other runtimes have runtime guidance and deployment module boundaries so they can be added without weakening discovery, planning, fit estimation, or verification.

## Install

Install the skill with the `skills` CLI from any supported agent project:

```bash
npx skills add https://github.com/nvidia-kejones/local-inference.git \
  --skill deploy-nvidia-inference
```

From a local clone, use the repo path instead:

```bash
npx skills add "$PWD" \
  --skill deploy-nvidia-inference
```

To inspect what the CLI will discover before installing:

```bash
npx skills add https://github.com/nvidia-kejones/local-inference.git --list
```

The skill lives under `skills/deploy-nvidia-inference`, which is a standard `skills` CLI discovery path. Use `--agent <agent>` when the target agent is known, and add `--global` for a user-level install instead of a project install.

## Upgrade

Refresh an installed copy after pulling new changes from this repo:

```bash
npx skills update
```

Use scope flags when you want to limit the refresh to a specific install scope:

```bash
npx skills update -g
npx skills update -p
```

If you installed from a local clone, update the clone first and then rerun `npx skills update` from the same project or global context.

## Use

Ask the agent to use the skill with a concrete host and goal:

```text
Use the deploy-nvidia-inference skill to discover user@gpu-host over SSH and recommend the best model/runtime pairs this host can serve for interactive chat, code assistance, long-context RAG, and structured agent workloads.
```

For one deployment workflow:

```text
Use the deploy-nvidia-inference skill to inspect user@gpu-host, build a workload profile for an 8K-context code assistant with four concurrent sequences, recommend a model/runtime pair, render a safe deployment plan, and stop before apply.
```

### Claude Code

Install only for Claude Code:

```bash
npx skills add https://github.com/nvidia-kejones/local-inference.git \
  --skill deploy-nvidia-inference \
  --agent claude-code
```

Add `--global` to make that a user-level Claude Code install. From a local clone, manual installation is also possible:

```bash
mkdir -p "$HOME/.claude/skills"
cp -R skills/deploy-nvidia-inference "$HOME/.claude/skills/"
```

Claude Code can select the skill from the request or invoke it explicitly:

```text
/deploy-nvidia-inference Discover user@gpu-host over SSH and recommend host-aware model/runtime pairs for interactive chat and code assistance.
```

### Codex

Install only for Codex:

```bash
npx skills add https://github.com/nvidia-kejones/local-inference.git \
  --skill deploy-nvidia-inference \
  --agent codex
```

Add `--global` to make that a user-level Codex install. From a local clone, manual installation remains a fallback:

```bash
mkdir -p "${CODEX_HOME:-$HOME/.codex}/skills"
cp -R skills/deploy-nvidia-inference "${CODEX_HOME:-$HOME/.codex}/skills/"
```

Invoke the skill explicitly in Codex:

```text
Use $deploy-nvidia-inference to discover user@gpu-host over SSH and recommend the best model/runtime pairs this host can serve for interactive chat, code assistance, long-context RAG, and structured agent workloads.
```

The skill keeps recommendation and deployment state separate. During use it aims to produce:

- `outputs/deploy-nvidia-inference/<run-id>/host_probe.raw.json`
- `outputs/deploy-nvidia-inference/<run-id>/host_facts.json`
- `outputs/deploy-nvidia-inference/<run-id>/workload_profile.yaml`
- `outputs/deploy-nvidia-inference/<run-id>/candidate_set.json`
- `outputs/deploy-nvidia-inference/<run-id>/candidate_scorecard.json`
- `outputs/deploy-nvidia-inference/<run-id>/use_case_profiles.json` when comparing named workload profiles
- `outputs/deploy-nvidia-inference/<run-id>/use_case_recommendations.json` when comparing named workload profiles
- `outputs/deploy-nvidia-inference/<run-id>/deployment_plan.yaml`
- `outputs/deploy-nvidia-inference/<run-id>/applied_deployment_state.json` only after explicit apply
- `outputs/deploy-nvidia-inference/<run-id>/verification_report.json`

Each run directory should keep its raw probe, workload assumptions, candidate set, recommendation outputs, rendered plan, and verification reports together. `outputs/` is gitignored because those files can carry host inventory and deployment evidence. Discovery is read-only. Installs, model downloads, service changes, deployment writes, firewall changes, and endpoint exposure must be explicit apply decisions. Endpoints bind to `127.0.0.1` by default unless external exposure is requested and reviewed.

## Direct Script Flow

The skill normally drives these helpers, but the core read-only path can also be run from the skill directory:

```bash
cd skills/deploy-nvidia-inference
run_dir=../../outputs/deploy-nvidia-inference/$(date -u +%Y%m%dT%H%M%SZ)-gpu-host
mkdir -p "$run_dir"
scripts/probe_remote_host.sh user@gpu-host > "$run_dir/host_probe.raw.json"
python3 scripts/normalize_host_facts.py "$run_dir/host_probe.raw.json" \
  --out "$run_dir/host_facts.json"
python3 scripts/recommend_use_cases.py \
  --host "$run_dir/host_facts.json" \
  --profiles assets/use_case_profiles.example.json \
  --candidates "$run_dir/candidate_set.json" \
  --out "$run_dir/use_case_recommendations.json"
```

Build each run's `candidate_set.json` from current primary runtime/model documentation and model metadata before scoring. The bundled candidate examples are schemas and test shapes, not permanent model recommendations.
