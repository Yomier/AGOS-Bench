# Assemble released tasks and generate custom episodes

Run these commands from the repository root. Set `AGOS_DATA_ROOT` to the root of the extracted task volume. Source the deployment session file to set `AGOS_OUTPUT_ROOT` to your experiment directory. Both commands require a fresh output directory. [Data and checksums](../dataset/README.md)

## Copy a verified subset

`subset` selects addresses from the published split index. It verifies the plan and two reference PNGs against the release table, checks their embedded certificates, and copies the original bytes. The output preserves the directory layout used by the benchmark and includes the selected rows in `tasks.csv`. Repeat `--episode` to select more tasks from the same split. Add `--dry-run` to verify the selection without writing files.

```bash
python -m dataset.generation.subset --data-root "$AGOS_DATA_ROOT" \
  --split test300 --episode Town03/lite_easy_00001 --output "$AGOS_OUTPUT_ROOT/test-subset"

python -m benchmark.run_parallel_eval --data-root "$AGOS_OUTPUT_ROOT/test-subset" \
  --split test300 --episode Town03/lite_easy_00001 --arm menu_oracle \
  --workers 1 --run-dir "$AGOS_OUTPUT_ROOT/subset-check" --dry-run
```

## Generate a custom ground-start episode

`derive` takes a verified parent task, preserves its search region, target, distractors, aerial start and reference photographs, and moves the ground vehicle to a different exact port on the frozen road graph. The parent photographs retain their original capture metadata. The seed selects among eligible landings on the parent's recorded route. Candidates must be inside the region, outside the confirmation radius and more than 10 m from each distractor. Mid and Hard retain a minimum 100 m ground-start distance from the target. Some parents have no eligible landing. Several seeds may select the same landing.

```bash
python -m dataset.generation.derive --data-root "$AGOS_DATA_ROOT" \
  --split test300 --episode Town03/lite_easy_00001 --seed 37 \
  --output "$AGOS_OUTPUT_ROOT/derived-37" --dry-run
```

The offline proposal checks the parent bytes, calibration, references, frozen road graph and control graph. For the published parent above, seed 37 selects port `1566:1:2:2` and episode `derived_9bf26cae4e13d828_37`. The proposal contains no simulator observations or acceptance receipt.

On a Linux workstation with the matching CarlaAir client and engine, set `AGOS_SIM_ROOT` and run the same command without `--dry-run`. `--slot 0` is the default and shares the benchmark's local slot lock. The tool uses two fresh engines in sequence and requires no model endpoint. [Engine setup](../carlaair/README.md)

1. The first engine deploys the parent vehicles and the new ground start. It checks the teammate's appearance, records the new start camera view, searches for the shortest confirmable route, physically replays its actions and validates the resulting plan and reference certificates.
2. The second engine redeploys the scene, independently recomputes the oracle, compares the route, deployment and legal confirmation evidence, and writes a replay receipt. Both engines must exit successfully before the owner publishes the episode.

A successful run has `accepted.json`, `episode/<derived-id>_plan.json`, both reference images under `episode/images/`, and engine evidence under `generate/` and `certify/`. The receipt binds the output plan and engine SHA-256. The plan records its parent split, episode and published SHA-256. Failed runs retain their candidate files and logs for inspection. Use a fresh directory for another attempt.

Custom episodes carry the dataset ID `AGOS_CUSTOM_GROUND_START_V1`. Their independent replay verifies the new ground-start variant. Difficulty labels describe the parent stratum. The four paper splits, their full scene sampling and the original quota certification have separate provenance. Only addresses in the published split index enter the official benchmark.

## Verification coverage

The real Unseen-Test300 example above was checked offline through subset assembly, benchmark admission, corruption rejection and graph-based proposal selection. Live generation and second-engine replay still require validation on a CarlaAir Linux host. To repeat the local tests with an extracted volume, run `python -m pytest -q dataset/generation/tests` with `AGOS_DATA_ROOT` set. [Subset CLI](../dataset/generation/subset.py) · [Generator](../dataset/generation/derive.py) · [Simulator worker](../dataset/generation/worker.py)
