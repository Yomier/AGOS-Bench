# CarlaAir simulation and serving

AGOS uses the Linux CarlaAir v0.1.7 engine, its matching CARLA Python extension and the AirSim Python client. CarlaAir is an MIT-licensed upstream project by Louis Zeng and contributors. Obtain the Linux release and use the setup instructions in its archive. Use the Python 3.10 client bundled with that release. A PyPI CARLA wheel can load a different ABI. [CarlaAir upstream](https://github.com/louiszengCN/CarlaAir) · [v0.1.7 source release](https://github.com/louiszengCN/CarlaAir/releases/tag/v0.1.7) · [Pinned Linux binary distribution](https://huggingface.co/tianlezeng/CarlaAIr-v0.1.7/resolve/041872adade4b417280c2cf307cb9d9621c1fa2b/CarlaAir-v0.1.7.zip)

Follow the [Ubuntu installation and deployment guide](../docs/deployment.md) for the pinned archive checksum, environment setup, Python client installation and smoke-test commands. [Runtime variables](../docs/deployment-variables.md)

## Engine and client

Set `AGOS_SIM_ROOT` to the extracted simulator directory containing `CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping`. Use a non-root account with GPU access and write access to your run directory. Activate the matching Python environment before invoking the harness. Confirm that the CARLA binding resolves to the expected installation, and select a working Vulkan ICD for the engine environment.

```bash
python -c 'import carla, airsim; print(carla.__file__, airsim.__file__)'
test -x "$AGOS_SIM_ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
python -m benchmark.run_parallel_eval --help
```

The benchmark launches the shipping binary directly, pins the map at process startup, uses Epic offscreen rendering and `-norhithread`, and supplies private HOME/XDG directories and an AirSim settings file. Town03, Town04 and Town05 use the `_Opt` assets. The readiness child checks the server version and map, obtains an actual RGB frame, and pings AirSim. It permits at most one `load_world` per engine lifetime. Each attempt gets a new engine. Both readiness and evaluation children have independent deadlines enforced outside their processes.

The default upstream deployment and the v13 native build have different engine identities. Upstream Linux shipping SHA-256 is `cbc11fed19fd3cbf70b927aee3540ddafafda400016f4de8c5c8959c6b3c6ef3`. The paper build used a patched engine and a build-specific client Stop guard. If you already have the matched binaries, bind their paths on the target machine and pass `--native-profile` to the same benchmark CLI.

```bash
python -m carlaair.engine.profile --engine "$AGOS_SIM_ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" \
  --libcarla "$LIBCARLA_FILE" --guard "$STOP_GUARD_FILE" --output "$NATIVE_PROFILE"
```

The profile tool reports `v13-matched` only for the exact three historical digests. The evaluator restarts its interpreter with the guard preloaded, verifies the mapped library and active Stop symbol in client processes, and removes the preload from UE. Stock installations leave `--native-profile` unset and use the upstream deployment identity. [Native identity checks](engine/profile.py)

## Perception and local inference

Use separate serving environments as appropriate for PyTorch/Transformers and vLLM. Source the session file from the deployment guide in each service terminal. It sets `MODEL_ROOT` to the directory containing the downloaded models. The v13 perception service loads GroundingDINO-base, SigLIP2-base-patch16-224 and SAM2.1-hiera-small. The vehicle proposal endpoints use the frozen query and scoring policy. The six-view evidence endpoint does not use segmentation. [Perception implementation](perception/gsam.py)

```bash
python -m carlaair.perception.gsam --detector-model "${MODEL_ROOT:?source the deployment session file}/grounding-dino-base" \
  --encoder-model "$MODEL_ROOT/siglip2-base-patch16-224" --sam-model "$MODEL_ROOT/sam2.1-hiera-small" --port 8201
# In another terminal
curl --fail http://127.0.0.1:8201/health
```

For local Qwen3.5-4B, launch a foreground OpenAI-compatible server in the vLLM environment. The release defaults to port 8114, BF16, 16,384-token context, 16 scheduler sequences, eager mode and GPU memory utilization 0.30. Choose the serving environment for your model. The evaluator verifies the requested local model IDs through `/v1/models`. For role-specific adapters pass `--adapter NAME=DIRECTORY` twice and select the split-agent config. Each adapter directory must contain `adapter_config.json`. `--dry-run` prints the command without loading the model. [Serving launcher](serving/vllm.py)

```bash
python -m carlaair.serving.vllm --model-dir "${MODEL_ROOT:?source the deployment session file}/Qwen3.5-4B" --served-model-name qwen3.5-4b
```

Set `AGOS_VLM_URL` to the full chat-completions URL, `AGOS_VLM_MODEL` to its model ID, `AGOS_SIDECAR_URL` to the local perception service base URL, and use `VLLM_API_KEY` if the model endpoint requires a token. With a remote OpenAI-compatible model select `--profile api`; simulator slots remain on the local GPU host. Minimal-Agent needs only the model endpoint. The four cooperative-search arms also need the local perception service. `menu_oracle` uses neither service. [Evaluation commands and repair](../benchmark/README.md)

## Slot monitoring and shutdown

The `local` and `api` profiles permit 6 and 10 workers respectively. Slot index `i` owns five TCP ports: RPC `15500+10i`, streaming `RPC+1`, internal `RPC+2`, AirSim `31050+10i` and reserved traffic manager `18000+10i`. Linux abstract socket locks prevent concurrent runs from claiming the same slot. The supervisor checks IPv4/IPv6 port availability before launch and starts each process with a private session. The run directory records engine/worker PIDs, stage and timestamp alongside a resource sample every 15 seconds.

Interrupt the benchmark owner with Ctrl+C or SIGTERM. It stops issuing work, terminates the owned child and sends SIGTERM to its own engine, waiting up to 45 seconds. If the engine remains alive, the slot reports `stop_failed` with its PID and stops the run. Inspect the slot state and engine log. Do not invoke a global process-name kill, send SIGKILL to a CarlaUE4 engine or start the upstream convenience launcher in a shared multi-slot run. Restart the foreground model and perception services separately when required.
