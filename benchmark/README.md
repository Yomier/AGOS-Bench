# Evaluate AGOS-Bench

Run commands from the repository root. Extract the task volumes into one `AGOS_DATA_ROOT`. The release indexes 1,100 train, 275 validation, 300 unseen-test and 450 seen-test episodes. A task is identified by its split, town and episode. Each plan and both reference images are bound to their published SHA-256 values. [Dataset](../dataset/README.md) · [CarlaAir setup](../carlaair/README.md) · [Agent configurations](../agents/README.md)

## One episode

```bash
python -m benchmark.run_parallel_eval --data-root "$AGOS_DATA_ROOT" \
  --split test300 --episode Town03/lite_easy_00001 --arm menu_oracle \
  --workers 1 --run-dir "$AGOS_OUTPUT_ROOT/menu-smoke" --dry-run
```

`--dry-run` reads the actual plan, verifies its two PNG certificates and the geometry calibration, loads the arm configuration, and resolves the map in the road-graph registry. If `AGOS_SIM_ROOT` or `--sim-root` is supplied, the engine file must also exist. For offline data-only checks, leave it unset. A dry run produces no experiment directory.

On a configured Linux simulator host, set `AGOS_SIM_ROOT`, use the same command without `--dry-run`, and inspect the output. `menu_oracle` exercises the real observation, action and adjudication loop with a privileged scripted policy. Its final `vlm_calls` and `vlm_http_attempts` must both be zero. Its task outcome is a deployment check. Use an agent arm for a model score.

## Model evaluation

Start the perception service and configure the selected model endpoint. The release arm names are `agos_b` (AGOS-Agent), `baseline_a` (Baseline-Agent), `agos_a` and `agos_c`. `--config PATH` selects a compatible custom agent YAML. Each run directory belongs to a single arm, resolved config, backend profile, engine binary and set of task SHAs.

```bash
# All 300 unseen-test tasks, local inference, up to six simulator slots.
python -m benchmark.run_parallel_eval --data-root "$AGOS_DATA_ROOT" \
  --split test300 --arm agos_b --profile local --run-dir "$AGOS_OUTPUT_ROOT/local-test"

# All 2,125 tasks across four splits, remote model endpoint, up to ten local slots.
python -m benchmark.run_parallel_eval --data-root "$AGOS_DATA_ROOT" \
  --split all --arm agos_b --config configs/examples/agos_b_api.yaml \
  --profile api --run-dir "$AGOS_OUTPUT_ROOT/api-all"
```

Select a subset with repeated `--episode Town/episode` arguments in one split. `--workers` lowers concurrency. A shared cold-start gate spaces engine launches by 30 seconds, including replacements. `--stagger-seconds` adjusts that interval. Every slot has a private engine, AirSim settings, camera readiness probe and episode child. The default episode wall limit is 9,000 seconds.

## Results and repair

| Output in the run directory | Contents |
| --- | --- |
| `manifest.json` | Exact task list, configuration hashes, arm, profile and engine identity |
| `summary.json` | Successes, valid failures, coverage, fair SR and pending task keys |
| `episodes/<split>/<town>/<episode>/terminal.json` | Selected attempt, classification, vision counts, raw result path and SHA |
| `episodes/<split>/<town>/<episode>/attempt-NNNN/` | Raw `*_result.json`, method trace, worker and engine logs, job and score receipts |
| `slots/slot-XX/status.json` and `resources.jsonl` | Live stage, owned PIDs, ports and per-host load/GPU samples |

A correct adjudicated CONFIRM yields success. Wrong CONFIRM, reaching the decision or model-call limit, and method-side navigation stall or patrol loop yield valid failure. Runtime, setup, wall-time and unknown failures stay pending. An attempt with more visual errors than successful visual observations is pending even if the step limit was reached. Recorded pre-terminal infrastructure faults and incomplete fault audits also exclude the attempt. Post-terminal engine cleanup faults are recorded separately. SR is `successes / (successes + valid failures)` and coverage is the scored fraction of the selection. With no valid terminals SR is `null`.

The scheduler retries pending tasks up to `--max-attempts` per invocation (default 2). To finish a partial batch, fix the facility and repeat the same command with `--resume`. Previous attempts remain on disk. Resume verifies the manifest and locked result SHAs and schedules only pending tasks. Exit 0 indicates all selected tasks scored, including valid model failures. Exit 2 reports a preparation error, exit 3 reports remaining pending tasks or a failed engine shutdown, and exit 130 indicates interruption. A failed shutdown can occur after a valid terminal and therefore coexist with full coverage. Read `stop_failed` in the slot status before resuming.

Run `python -m benchmark.paper_metrics "$AGOS_OUTPUT_ROOT/local-test"` to compute the nine paper metrics from the locked episode results. The metric guide lists units, support counts and exit codes. [Scoring a run](METRICS.md)

For CPU-only harness regression tests run `python -m pytest -q benchmark/tests`.
