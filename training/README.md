# Train AGOS-Agent adapters

Use the public expert trajectories to train one mixed adapter or two independent role-specific adapters. All commands below run from the repository root. The conversion uses Python's standard library. GPU training uses a local Qwen3.5-4B checkpoint with its processor, a CUDA GPU with bfloat16 support, and the `train` extra. Obtain the base model separately. The dataset and simulator each have installation instructions. [Dataset access](../dataset/README.md) · [CarlaAir setup](../carlaair/README.md)

## Build the training files offline

Download, verify and unpack the `trajectories-train1100` volume using the dataset instructions. Set `AGOS_DATA_ROOT` to the directory that now contains `trajectories-train1100/`. Set `AGOS_OUTPUT_ROOT` to a writable directory for your outputs. A corpus or training run requires a new destination directory.

```bash
python -m training.corpus \
  --input "$AGOS_DATA_ROOT/trajectories-train1100" \
  --output "$AGOS_OUTPUT_ROOT/corpus"
```

The converter reads `rows.jsonl` in order, checks the system/user/assistant messages, matches the role in the deployed user prompt to the role in metadata, and verifies every content-addressed PNG. It retains the original line bytes and image order. Its three JSONL files share one `images/` directory. It hard-links images on the same filesystem where supported and otherwise copies them. Budget about 13 GB of additional image storage for a cross-filesystem conversion.

| Output | Rows for the full public volume | Use |
|---|---:|---|
| `mixed.jsonl` | 27,720 | One adapter for both agents |
| `uav.jsonl` | 23,529 | Aerial agent adapter |
| `ugv.jsonl` | 4,191 | Ground agent adapter |

Check `manifest.json` for 1,100 episodes, 45,843 unique images, per-file counts and SHA-256 hashes. When joining trajectory metadata to the task catalog, map the simulator town names `Town04_Opt` and `Town05_Opt` to `Town04` and `Town05`. Both role files together contain exactly the mixed file's rows. SHA-256 for `mixed.jsonl` matches that of the extracted `rows.jsonl`.

## Train one mixed adapter

Install GPU-enabled PyTorch and its compatible torchvision build for your CUDA environment, then install the training extra. Set `AGOS_BASE_MODEL` to the local model directory containing Qwen3.5-4B weights, tokenizer, processor and chat template. The model loader reads local files. [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)

```bash
export AGOS_BASE_MODEL="${MODEL_ROOT:?source the deployment session file}/Qwen3.5-4B"
test -r "$AGOS_BASE_MODEL/config.json" || exit 1
python -m pip install -e '.[train]'
python -m training.train \
  --base "$AGOS_BASE_MODEL" --data "$AGOS_OUTPUT_ROOT/corpus/mixed.jsonl" \
  --out "$AGOS_OUTPUT_ROOT/mixed" --seed 20261027 --max-sequence-tokens 5120
```

## Train independent aerial and ground adapters

```bash
for role in uav ugv; do
  python -m training.train \
    --base "$AGOS_BASE_MODEL" --data "$AGOS_OUTPUT_ROOT/corpus/$role.jsonl" \
    --out "$AGOS_OUTPUT_ROOT/$role" --seed 20261027 --max-sequence-tokens 5120 || exit 1
done
```

Each invocation starts from the same base. Each run tokenizes its own inputs with thinking disabled, skips whole examples above 5,120 multimodal tokens, and makes one pass over admitted rows. Both commands use LoRA rank 8, alpha 16 and dropout 0 on the language model's q/v projections in zero-based layers 3, 7, 11, 15, 19, 23, 27 and 31. Training uses AdamW at `1e-4`, zero weight decay, gradient clipping at 1, microbatch 1, accumulation 2 and assistant-only loss weighted by supervised token count. It decodes RGB at the supplied resolution before the saved processor's resizing. For a shorter run, `--max-steps N` limits optimizer steps after token admission.

`tokenization.json` records admitted and skipped row indices. `steps.jsonl` records each optimizer step and its input row indices. `result.json` is written after saving `adapter/` and `processor/` and checking the reloaded adapter's loss. It reports actual consumption, input hash and library versions.

For mixed evaluation, set `AGOS_VLM_MODEL` to the served mixed adapter name and use `--arm agos_b` with the standard AGOS-Agent configuration. For split evaluation, serve both adapters on the same `AGOS_VLM_URL`, set `AGOS_UAV_MODEL` and `AGOS_UGV_MODEL` to the corresponding served adapter names, and pass `--arm agos_b --config configs/examples/agos_b_split.yaml` to the evaluation runner. [Evaluation instructions](../benchmark/README.md)

## What the paper trained on

| Training input | Supplied | Admitted and consumed | Optimizer steps |
|---|---:|---:|---:|
| Mixed, 301 episodes | 8,212 | 6,014 | 3,007 |
| UAV partition of those episodes | 7,039 | 5,297 | 2,649 |
| UGV partition of those episodes | 1,173 | 717 | 359 |

The recorded runs used the same 6,014 examples across the two regimes. They completed one epoch below a 4,096-step ceiling. Their receipts record PyTorch 2.13.0, Transformers 5.16.1 and PEFT 0.21.2. The historical input used its own prompt preparation and, where the underlying panels were available, rebuilt ground-confirmation collages. The current public release provides a later, cleaned bank with 1,100 episodes and 27,720 examples. Training on this bank is a new experiment. Its admission counts, adapters and scores depend on the supplied data and processor. The example commands omit the historical 4,096-step ceiling so the larger bank can complete an epoch. Historical training inputs and adapters are outside this release.

The public trajectory volume contains 820 `expert_confirm` rows. All 820 were captured with the generic decision system prompt (`SYSTEM`), byte-identical to the system prompt in the volume's 22,376 decision rows. The current primary-decision prompt adds six lines, including guidance on matching both color and vehicle model and a limit on repeated `STOP_OBSERVE` actions. The corpus converter preserves the recorded prompts byte for byte. For fresh captures, the collector uses the deployed `confirm_verify` system prompt selected by the same configuration. With `verify_system_v2: true` in the supplied AGOS-Agent configuration, this selects the dedicated ground-confirmation prompt (`CONFIRM_SYSTEM`).

## Collect new expert examples

The published trajectories can be used entirely offline. Generating new simulated tasks, certifying them by execution, and collecting fresh expert traces require CarlaAir. The following collection flow uses the released, previously certified Train1100 tasks. Configure the runtime and perception service using the benchmark instructions. `AGOS_SIM_ROOT` supplies the simulator installation, and `--simulator-root` overrides it. If your build uses a native guard profile, pass its path with `--native-profile`. [Evaluation runner](../benchmark/README.md) · [Simulator and perception service](../carlaair/README.md)

```bash
python -m training.collect \
  --config agents/configs/agos_b.yaml --data-root "$AGOS_DATA_ROOT" \
  --output "$AGOS_OUTPUT_ROOT/collect-one" --workers 1 \
  --episode Town01/lite_easy_00001

python -m training.freeze \
  --run "$AGOS_OUTPUT_ROOT/collect-one/run" --output "$AGOS_OUTPUT_ROOT/fresh-trajectories"
python -m training.corpus \
  --input "$AGOS_OUTPUT_ROOT/fresh-trajectories" --output "$AGOS_OUTPUT_ROOT/fresh-corpus"
```

For an offline plan check, run `training.collect` with `--dry-run` and a separate new output directory. Repeat `--episode` to select several Train1100 tasks, or omit it to collect the split. The collector writes an expert-enabled config, records images under `captures/` and calls the same evaluation runner used for deployment. Keep the capture directory in place until freezing finishes because each trace references its original frames.

The freezer follows the runner's locked terminal record for each task. It reads the executed decision, proposal-verification and ground-confirmation calls, verifies the image digests in slot order and maps executable expert actions to the public answer format. It accepts clean successes, budget exhaustion and navigation stalls. Its manifest counts contributing episodes and excluded terminal records. Check those counts against your requested batch. Freezing, inspecting status, verifying image hashes and converting the corpus are offline operations.
