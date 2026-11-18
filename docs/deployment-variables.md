# Deployment variables

The [deployment guide](deployment.md) creates a reusable Bash file for local paths. Source it in each terminal. The method loader expands `${NAME}` inside YAML scalar values at load time. The runner receives episode, run directory, arm and concurrency settings as CLI arguments. [Config loader](../agos/lite_core.py) · [Runner](../benchmark/run_parallel_eval.py)

| Name | Example or default | Read by / purpose |
|---|---|---|
| `AGOS_WORK` | `$HOME/agos-work` | Guide only. Writable data disk for downloads, simulator tree, models, logs and runs. |
| `CONDA_BASE` | `$HOME/miniforge3` | Guide only. Existing conda installations can use the absolute path printed by `conda info --base`. |
| `AGOS_REPO` | `$AGOS_WORK/AGOS-Bench` | Guide only. Public checkout containing `pyproject.toml` and the package directories. |
| `AGOS_DATA_ROOT` | `$AGOS_WORK/data` | Parent of the extracted split directories. Passed to `--data-root` or read from the runner environment. A training corpus uses the `trajectories-train1100/` volume under this root. |
| `AGOS_SIM_ROOT` | `$AGOS_WORK/sim/CarlaAir-v0.1.7` | Simulator installation containing `CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping`. `--sim-root` overrides it. Leave unset for a CPU-only dry-run without an engine installation. |
| `AGOS_OUTPUT_ROOT` | `$AGOS_WORK/runs` | Writable parent under which guide commands choose separate `--run-dir` values. Training commands choose separate output subdirectories. |
| `AGOS_ASSET_ROOT` | The release's `assets/` directory | YAML road-graph registry and bundled assets. The package resolves this by default. An override must contain the same navigation tree. |
| `MODEL_ROOT` | `$AGOS_WORK/models` | Guide only. Stores local DINO, SigLIP2, SAM2 and optional Qwen model snapshots. |
| `AGOS_BASE_MODEL` | `$MODEL_ROOT/Qwen3.5-4B` | Passed to `training.train --base` for local weights, tokenizer and processor. |
| `AGOS_SIDECAR_URL` | `http://127.0.0.1:8201` | Base URL for GSAM and UGV views. The agent config appends `/v1/gsam` and `/v1/ugv_views`. |
| `AGOS_VLM_URL` | `http://127.0.0.1:8114/v1/chat/completions` | Full image-capable chat-completions URL used for decisions. Change it when switching to a remote API. |
| `AGOS_VLM_MODEL` | `qwen3.5-4b` | Exact served model ID, shared across roles by the default agent configs. |
| `AGOS_UAV_MODEL`, `AGOS_UGV_MODEL` | `uav`, `ugv` | Required by the split-adapter example YAML. Both aliases must be served on the configured chat-completions URL. |
| `AGENT_CONFIG` | `$AGOS_REPO/configs/examples/agos_b_api.yaml` or `$AGOS_REPO/configs/examples/agos_b_split.yaml` | Guide only. Passed to `--config` when set. The selected file's arm must match `--arm`. |
| `VLLM_API_KEY` | Set in the service/worker shell as appropriate | Bearer credential named by `vlm.api_key_env`. The loopback example sets `local` in both processes. Enter a provider key in the evaluation terminal when using a remote API. |
| `UGV_ACTION_SPACE` | Leave unset for released tasks | An existing shell override takes precedence over the agent configuration. All 2,125 published plans use `sparse_nav_v1`. Run `unset UGV_ACTION_SPACE` before the offline preflight if you previously selected another action space. |
| `PYTHONNOUSERSITE` | `1` | Keeps a per-user Python install from shadowing the active conda client SDK. |

`PROFILE=local` and `PROFILE=api` in the guide are shell variables passed to `--profile`. They set six or ten slots by default. `--workers N` selects a smaller count. `--native-profile` independently selects the optional native client guard and verified engine. [Native profile implementation](../carlaair/engine/profile.py)

**中文。** 数据根是 `test300/` 等解压目录的父级。引擎根是 `CarlaUE4/` 的父级。输出根下每个实验用独立目录。感知地址填基址，模型地址填完整接口路径。并发 profile、模型路由和 native profile 是三个独立参数。密钥只在所属进程的 shell 中设置。