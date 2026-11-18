---
license: cc-by-4.0
pretty_name: AGOS-Dataset
tags:
  - robotics
  - simulation
  - vision-language
---

# AGOS-Dataset

AGOS-Dataset contains 2,125 simulated cooperative vehicle-search tasks in CarlaAir and expert observations and actions for the 1,100 training tasks. Each task pairs an aerial robot with a ground vehicle. Both use reference images to identify a vehicle in a simulated town.

## Task sets

| Volume | Tasks | Contents |
| --- | ---: | --- |
| `taskset-train1100` | 1,100 | Training tasks linked to the final expert trajectory census |
| `taskset-validation275` | 275 | Validation tasks in five training towns |
| `taskset-test300` | 300 | Unseen-town test tasks in Town03, with 100 per difficulty |
| `taskset-seentest450` | 450 | Seen-town test tasks, with 150 per difficulty |

The split index lists the episode keys and the SHA-256 digest of the selection manifest for each split. Seen-Test450 uses 450 slots from a certified pool of 523. Its 73 reserve slots are excluded. Train1100 uses the 11 November 2026 audited roster, including its 17 revised source trajectories. Sixteen of these revisions changed the episode key. [Split index](agos-tasksets-splits-v1.json)

A task is identified by its original plan SHA-256. All 2,125 original hashes are distinct. Train1100 and Val275 share 216 map/episode key pairs, each with different plans. All other split pairs have disjoint keys. Join on `source_plan_sha256` or on `(split, episode_key)`. The latter also identifies both reference images. [Task index](tasks.csv)

Each task archive extracts into its own split directory. For example, the first Unseen-Test300 task uses these three files under the data root.

```text
test300/dataset/Town03/lite_easy_00001_plan.json
test300/dataset/Town03/images/lite_easy_00001_target_reference.png
test300/dataset/Town03/images/lite_easy_00001_teammate_reference.png
```

Each split also includes its own `tasks.csv` and `manifest.json`. In the index, `plan_sha256` covers the published JSON bytes. `source_plan_sha256` covers the original plan. Published plans refer to calibration and navigation assets using paths relative to the AGOS-Bench repository root. The `navigation_binding_sha256` and `navigation_certificate_sha256` columns identify the embedded navigation oracle. The publisher recomputed both canonical JSON hashes for every task and checked both reference image hashes against the original plan.

## Expert trajectories

The `trajectories-train1100` archive extracts into a directory of the same name. It contains `rows.jsonl`, `episodes.csv`, and `images/<sha256>.png`. The 27,720 rows contain 82,331 image references to 45,843 distinct images. Of these rows, 23,529 belong to the aerial robot and 4,191 to the ground vehicle. The trajectory outcomes were 910 successful confirmations, 177 budget terminations, and 13 navigation stalls.

Each JSONL row has `messages`, `images` and `meta`. Resolve image paths relative to the trajectory directory. `meta.episode_key` and `meta.call_id` identify a call within its episode. When joining to `episodes.csv`, normalize the simulator map spellings `Town04_Opt` and `Town05_Opt` to `Town04` and `Town05`. The CSV links each episode to the training plan by `source_plan_sha256`. The published JSONL preserves the audited row bytes, SHA-256 `24c64ec1b60a74c3d9512772f30b40cae95f0a431c8d38757fe24bbb3a01904c`.

These are simulator expert demonstrations. The historical trained models consumed 6,014 rows after length filtering from an earlier 8,212-row input. Training on the complete 27,720-row release is a distinct training setting.

## Access and verification

Five archives are prepared locally. External hosting is pending. The separate `certification` volume is on hold while original independent replay evidence for 300 Seen-Test450 tasks is recovered. The four task archives contain the embedded navigation certificates in every plan. The trajectory and task archives can be used independently of the separate replay volume.

From the AGOS-Bench repository root, place the archives in `data/`, verify them against the bundled checksums, and extract the desired archives into the same directory. `--volume all` checks every available volume and reports the certification hold. [Release checksums](SHA256SUMS)

```bash
python dataset/verify.py --directory data --volume all
tar -xzf data/taskset-test300.tar.gz -C data
```

Once hosting is available, the download tool fetches the selected archive from `Yomier/AGOS-Dataset` on Hugging Face and verifies it against the checksums bundled with this release. Install `huggingface_hub` in the environment used for the download. `--revision` accepts a Hub commit or tag.

```bash
python -m pip install huggingface_hub
python dataset/download.py --directory data --volume taskset-test300
```

## License and citation

The dataset is distributed under [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). Credit AGOS-Dataset and its authors when redistributing the tasks or demonstrations. The accompanying code has its own license in the AGOS-Bench repository.

Citation metadata is maintained with the code release. The arXiv identifier is pending (`TODO`).
