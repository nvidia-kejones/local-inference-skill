# vLLM Baseline

V1 implements vLLM as the first scripted deployment module because it is a practical Hugging Face/CUDA serving baseline with an OpenAI-compatible server path.

## Implemented Path

1. Build a current candidate with:
   - `runtime: vllm`
   - a pinned `model_revision`
   - `deployment.container_image` pinned to an image digest
   - `deployment.tensor_parallel_size`
   - fit metadata for weights and KV cache
2. Render:

```bash
python3 scripts/render_deployment_plan.py \
  --host host_facts.json \
  --workload workload_profile.yaml \
  --candidate selected_candidate.json \
  --out deployment_plan.yaml \
  --compose-out docker-compose.yaml \
  --env-out deployment.env \
  --ssh-target user@host
```

3. Review `deployment_plan.yaml`, `docker-compose.yaml`, and `deployment.env`.
4. Apply explicitly:

```bash
scripts/apply_vllm_compose.sh \
  --ssh-target user@host \
  --compose docker-compose.yaml \
  --env deployment.env \
  --state-out applied_deployment_state.json \
  --apply \
  --allow-model-downloads
```

The Compose template binds `127.0.0.1` unless the plan renderer is given another bind address. It should also pin `NVIDIA_VISIBLE_DEVICES` from the selected fit GPUs instead of requesting all GPUs.
If the deployment includes more than one model endpoint, the planner should choose a shared host Hugging Face cache path so the containers reuse already-downloaded weights.
When pre-pulling an image to warm the cache, use the exact pinned image reference from `deployment.container_image` or the rendered `VLLM_IMAGE` env value. Do not warm with a nearby tag.

## Checks Before Apply

- Refresh vLLM Docker, OpenAI-compatible serving, supported model, quantization, and engine-argument docs.
- Verify the host driver and selected image can work together.
- Review whether `--tensor-parallel-size`, max model length, GPU-memory utilization, quantization flags, and trust-remote-code policy are appropriate.
- Record any Hugging Face token handling and cache path choice without placing secrets in reports. For multi-endpoint deployments, use a shared host cache path and record it in the plan. Treat the rendered environment file as a secret once a token is added; the renderer and apply path set restrictive file permissions, but reports should keep only paths and commands.
- Record the selected GPU device list from the fit result. Do not use `gpus: all` in the Compose baseline.
- If you warm the image cache before `docker compose up`, the warmup command must target the exact pinned image digest.
- If replacing a live service, capture its start command, config, image/binary revision, port, cache path, and smoke-test result first.

## Baseline Boundaries

The module renders and applies one Compose stack. It does not configure authentication, external proxying, firewall rules, observability, autoscaling, multi-host parallelism, or vLLM tuning beyond reviewed candidate args.
