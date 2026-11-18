# AGOS agents

The four paper agents use the same `cooperative_search` method, visual proposal services, action menu and 30 m confirmation protocol. The arm selects the coverage planner and decision rhythm. The table uses the final joint-coverage × event-waking experiment on Unseen-Test300 with Qwen3.5-4B.

| Paper condition | Arm and configuration | Coverage | Wake rhythm | Identity threshold | Successes / 300 | Served YAML SHA-256 prefix |
|---|---|---|---|---:|---:|---|
| Baseline-Agent | [baseline_a](configs/baseline_a.yaml) | Fixed | Observation | 0.72 | 92 | `fa40c25d65682076` |
| Baseline + joint coverage | [agos_a](configs/agos_a.yaml) | Joint | Observation | 0.72 | 89 | `25e0d705d4400cb0` |
| Baseline + event waking | [agos_c](configs/agos_c.yaml) | Fixed | Event | 0.55 | 103 | `f67cfde28f394161` |
| AGOS-Agent | [agos_b](configs/agos_b.yaml) | Joint | Event | 0.55 | 113 | `93d3f7fd0d8eb2f7` |

The SHA prefixes identify the original served YAMLs. The public copies resolve paths and service addresses at load time. The `agos_c` threshold comes from the deployed event-only run. Event waking includes both the rhythm and its tuned threshold. All four arms enable the same v13 confirmation prompt and native confirmation images.

## Running a paper arm

Install the package, extract the task volumes under `AGOS_DATA_ROOT`, and configure the engine and inference services as described in the [deployment guide](../carlaair/README.md). From the repository root, use the arm name from the table in `--arm`. The benchmark selects its YAML by name. A dry run checks the inputs before launching the engine.

```bash
python -m benchmark.run_parallel_eval --split test300 --arm agos_b \
  --episode Town03/lite_easy_00001 --run-dir "$AGOS_OUTPUT_ROOT/agos-b-check" --dry-run
python -m benchmark.run_parallel_eval --split test300 --arm agos_b \
  --episode Town03/lite_easy_00001 --run-dir "$AGOS_OUTPUT_ROOT/agos-b-check" \
  --sim-root "$AGOS_SIM_ROOT" --profile local --workers 1
```

All four methods use an OpenAI-compatible chat-completions endpoint. `AGOS_VLM_URL` is the full URL, `AGOS_VLM_MODEL` is the served model name, and the client reads authentication from `VLLM_API_KEY`. `AGOS_SIDECAR_URL` is the perception service base URL. The agent calls `/v1/gsam` and `/v1/ugv_views` under that base. The defaults target local services on ports 8114 and 8201. `AGOS_ASSET_ROOT` can redirect the bundled calibration and six map graphs to a relocated copy.

For a hosted API, set the URL, model and key, and select `--profile api`. The [AGOS-B API configuration](../configs/examples/agos_b_api.yaml) omits the local Qwen chat-template extension. For a different arm, remove `vlm.chat_template_kwargs` from a copy of that arm's YAML if the provider requires it. Use `--config` to select the copy.

For a jointly trained adapter, set `AGOS_VLM_MODEL` to its served alias. For role-specific adapters, serve both on the same endpoint and use `--arm agos_b --config configs/examples/agos_b_split.yaml`. Set `AGOS_UAV_MODEL` and `AGOS_UGV_MODEL` to their corresponding served aliases. Every decision and verification call selects its client from the observation role. Without role overrides, both roles share the configured client. See the [split example](../configs/examples/agos_b_split.yaml) and [training tools](../training/README.md).

## Simulator deployment check

Use `--arm menu_oracle` to exercise the legal navigation menu and the private deployment checks using the [scripted configuration](configs/menu_oracle.yaml). The oracle uses the plan's privileged target and certified ground route and makes zero model calls. Its output is for simulator QA. The four paper methods use public observations for their decisions.
