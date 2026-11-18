# Deploy AGOS-Bench on Ubuntu

Use this guide on the machine that will render the UAV and UGV cameras. A remote model endpoint moves inference off that machine. CarlaAir runs on the simulator host. Baseline-Agent, AGOS-Agent and their ablations also use the perception service on that host. The English and Chinese instructions share the same Bash blocks. Start with one episode before allocating a split. [Runtime variables](deployment-variables.md)

**中文路线。** 在仿真主机上依次执行各节 Bash 块。先完成驱动、Vulkan 和 Python 自检，再用单集 `menu_oracle` 检查仿真全过程。评测 Minimal-Agent 时可跳过第 7 节的感知服务，直接配置多模态模型。AGOS-Agent、Baseline-Agent 及其消融配置需要同时启动感知服务。每节的中文说明给出命令执行位置和通过条件。

## 1. Give an ordinary Linux account access to the GPU

Use Ubuntu 22.04 x86_64 on bare metal or a VM with a passed-through NVIDIA GPU and graphics support. Upstream also lists Ubuntu 20.04. On a managed host with a working driver, start at the post-reboot checks. Upstream budgets about 8 GB of VRAM for a single Town10HD demo. The local inference model, detector, image encoder and mask model need additional VRAM, as does each concurrent engine. Measure on the actual host before choosing the worker count. [Ubuntu driver installation](https://ubuntu.com/server/docs/how-to/graphics/install-nvidia-drivers/) · [CarlaAir system requirements](https://huggingface.co/tianlezeng/CarlaAIr-v0.1.7) · [Town10HD quick start](https://github.com/louiszengCN/CarlaAir/blob/main/CarlaAir_Release/guide/Quick-Start.md)

**中文。** 以下从新装 Ubuntu 22.04 x86_64、有 sudo 权限的普通账户开始。已有厂商驱动的主机只安装工具并做重启后的检查。先看显存、主存和数据盘余量，再定并发。引擎 ZIP 与解压文件合计约 25.2 GB，数据卷、模型、环境和运行日志另计。

```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common vulkan-tools libvulkan1 pciutils \
  git curl unzip jq file iproute2 build-essential ca-certificates \
  libgl1 libglib2.0-0 libxrandr2 libxi6 libxinerama1 libxcursor1 libxkbcommon-x11-0
uname -m
lspci | grep -i nvidia
ubuntu-drivers devices
sudo ubuntu-drivers install
for group in video render; do
  if getent group "$group" >/dev/null; then sudo usermod -aG "$group" "$(id -un)"; fi
done
sudo reboot
```

Log in again as the same account. Inspect the physical device name in `vulkaninfo` and check that a fresh process can initialize CUDA. A display warning from `vulkaninfo` over SSH can coexist with a working physical device. A listing with only llvmpipe indicates software rendering. In a container the host must expose the NVIDIA graphics/Vulkan and compute libraries and device nodes to the account.

```bash
id
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free --format=csv
vulkaninfo --summary
python3 - <<'PY'
import ctypes
status = ctypes.CDLL('libcuda.so.1').cuInit(0)
print('cuInit:', status)
raise SystemExit(status != 0)
PY
free -h
df -h "$HOME"
```

## 2. Install conda and the official Linux simulator

Choose a writable data disk. The examples use `$HOME/agos-work`; to move it, set `AGOS_WORK` before creating the environment file. This file contains paths only. Source it in every new Bash terminal used below. The fresh-install commands use Miniforge. For an existing conda install, skip the installer and point `CONDA_BASE` in the file to `conda info --base`. [Miniforge release and checksum](https://github.com/conda-forge/miniforge/releases/tag/26.9.4-2)

**中文。** 用普通账号在数据盘建工作目录。环境文件不存 API 密钥。准备多个终端时，每个终端都先 `source "$HOME/.config/agos-bench/session.sh"`。如果已有 conda，把 `CONDA_BASE` 改为已有安装目录并跳过 Miniforge 安装块。

```bash
export AGOS_WORK="${AGOS_WORK:-$HOME/agos-work}"
mkdir -p "$HOME/.config/agos-bench" "$AGOS_WORK"
{
  printf 'export AGOS_WORK=%q\n' "$AGOS_WORK"
  cat <<'ENV'
export CONDA_BASE="$HOME/miniforge3"
export AGOS_REPO="$AGOS_WORK/AGOS-Bench"
export AGOS_SIM_ROOT="$AGOS_WORK/sim/CarlaAir-v0.1.7"
export AGOS_DATA_ROOT="$AGOS_WORK/data"
export AGOS_OUTPUT_ROOT="$AGOS_WORK/runs"
export MODEL_ROOT="$AGOS_WORK/models"
export PYTHONNOUSERSITE=1
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  export PATH="$CONDA_BASE/condabin:$PATH"
fi
ENV
} > "$HOME/.config/agos-bench/session.sh"
source "$HOME/.config/agos-bench/session.sh"
mkdir -p "$AGOS_WORK"/{downloads,sim,logs} "$AGOS_DATA_ROOT" "$AGOS_OUTPUT_ROOT" "$MODEL_ROOT"
cd "$AGOS_WORK/downloads"
```

```bash
MF=Miniforge3-26.9.4-2-Linux-x86_64.sh
curl -fL --retry 3 -o "$MF" "https://github.com/conda-forge/miniforge/releases/download/26.9.4-2/$MF"
printf '%s  %s\n' bbaf16b3897da24c4be50315934ba74848f617b4495982841d6a0a60e0edbbdf "$MF" | sha256sum -c - && \
  bash "$MF" -b -p "$CONDA_BASE"
source "$HOME/.config/agos-bench/session.sh"
conda --version
```

The official HF file is a ZIP with a **single `CarlaAir-v0.1.7/` root**, 33,274 entries and 18,335,602,720 uncompressed bytes. It contains the Linux Shipping executable, UE assets, AirSim settings, the custom CPython 3.10 SDK tarball and both upstream setup and test scripts. The pinned revision and expected archive digest below come from HF. Keep ZIP and extracted tree on the same Linux filesystem with execution enabled or put the extracted tree on a separate executable mount. [Pinned upstream file](https://huggingface.co/tianlezeng/CarlaAIr-v0.1.7/blob/041872adade4b417280c2cf307cb9d9621c1fa2b/CarlaAir-v0.1.7.zip)

**中文。** 下载官方 Linux ZIP，校验输出应为 `OK` 后才解压。`AGOS_SIM_ROOT` 指向内含 `CarlaUE4/` 的那一层。归档中的客户端扩展要求 Linux x86_64 上的 CPython 3.10。

```bash
cd "$AGOS_WORK/downloads"
ZIP=CarlaAir-v0.1.7.zip
URL=https://huggingface.co/tianlezeng/CarlaAIr-v0.1.7/resolve/041872adade4b417280c2cf307cb9d9621c1fa2b/CarlaAir-v0.1.7.zip
curl -fL --retry 3 -C - -o "$ZIP" "$URL"
printf '%s  %s\n' d6c3f369af7b1048dd3848bc4ff1d10ffffdd4c909daf132966d470600816cf8 "$ZIP" | sha256sum -c - && \
  unzip -qn "$ZIP" -d "$AGOS_WORK/sim"
cd "$AGOS_SIM_ROOT"
file CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping
test -x CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping && \
  test -s env_setup/carla_python_module.tar.gz && \
  test -r AirSimConfig/settings.json
```

The upstream script reuses the conda environment `carlaAir` and extracts the bundled `carla` and `carla.libs` into its site-packages. Use a fresh environment with that name. AirSim's source distribution imports NumPy, msgpack-RPC and OpenCV during metadata generation. Install these first using the same interpreter, then let the upstream script install its bundled client. [Upstream setup script](https://github.com/louiszengCN/CarlaAir/blob/main/env_setup/setup_env.sh) · [AirSim 1.8.1 packaging](https://github.com/microsoft/AirSim/blob/v1.8.1/PythonClient/setup.py)

```bash
conda create -y -n carlaAir python=3.10 pip
conda activate carlaAir
python -m pip install 'setuptools<82' wheel numpy 'opencv-contrib-python<5' \
  'msgpack-rpc-python==0.4.1' 'tornado==4.5.3'
python -m pip install --no-build-isolation 'airsim==1.8.1'
cd "$AGOS_SIM_ROOT"
bash env_setup/setup_env.sh
conda activate carlaAir
bash env_setup/test_env.sh
python - <<'PY'
import sys, airsim, carla, carla.libcarla, numpy, pygame, PIL
print(sys.version, carla.__file__, carla.libcarla.__file__, sep='\n')
assert sys.version_info[:2] == (3, 10)
print('client:', carla.Client('127.0.0.1', 2000).get_client_version())
PY
```

The first upstream test runs with the engine stopped. Expect successful imports and warnings about idle ports 2000 and 41451. For an optional standalone test **on an otherwise idle, dedicated graphical machine**, run `./CarlaAir.sh Town03 --fg` from the unpacked directory. Its startup script terminates other matching UE processes. In a second `carlaAir` terminal run the following checks, then press Ctrl+C in the demo terminal. On a shared/headless server skip the demo and use the one-worker benchmark smoke below. The benchmark starts the Shipping binary itself.

**中文。** 首次测试只验证环境，2000/41451 未监听是预期。独占图形主机可用上游 `--fg` 演示，第二终端执行下面的双协议断言，最后回到演示终端 Ctrl+C。上游启动器会清理同名进程，共享机器或无显示器主机直接走后文单 worker 冒烟，切勿与评测并行运行演示。

```bash
source "$HOME/.config/agos-bench/session.sh"
conda activate carlaAir
cd "$AGOS_SIM_ROOT"
bash env_setup/test_env.sh
python - <<'PY'
import carla, airsim
client = carla.Client('127.0.0.1', 2000)
client.set_timeout(10)
print(client.get_world().get_map().name)
assert airsim.MultirotorClient(ip='127.0.0.1', port=41451, timeout_value=10).ping()
PY
```

## 3. Record which native runtime you have

The public ZIP is sufficient to install the stock CarlaAir runtime. The following checks identify the two binaries actually installed. Historical production used a patched UE executable, the same CPython extension, a separate Linux Stop guard and an admitted native profile with `-norhithread`. The guard protects the Python client lifecycle. An engine inheriting a client-only `LD_PRELOAD` has the wrong runtime. Let the benchmark manage child environments.

| SHA-256 identity | Expected digest |
|---|---|
| Stock Shipping executable in the official ZIP | `cbc11fed19fd3cbf70b927aee3540ddafafda400016f4de8c5c8959c6b3c6ef3` |
| Bundled `carla/libcarla.cpython-310-x86_64-linux-gnu.so`, also used in production | `4795659108e845fa1e8c5f2af0749477813ea2a895f00470402c431f0e66508d` |
| Historical patched Shipping executable | `84ab7d701fe458ed8333210722a628a1204f3d6b1c9c661b7bde61f49e5f0d15` |
| Historical `libagos_stop_strand.so` | `ef08bc2d39b203efad1fd485adaa7ec327f0eee20e358a454ea727a7b415a318` |
| Original admitted profile JSON, including its historical paths | `0141730ddce3ec7fcd4a52402b346b0b6ab9ba0cc230b6176eb81ef75a278c8c` |

```bash
conda activate carlaAir
ue="$AGOS_SIM_ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping"
libcarla=$(python -c 'import carla.libcarla as m; print(m.__file__)')
sha256sum "$ue" "$libcarla"
```

**中文。** 官方包的引擎 SHA 以 `cbc11f` 开头，生产引擎以 `84ab7d` 开头，二者共用 SHA 以 `479565` 开头的 libcarla。只有已取得且验证过生产 UE 和 guard 的用户才能选择相应 native 运行。重新生成或迁移路径后的 JSON 有自己的 SHA；保留本机工件、flags 与实际进程的身份记录。公开 ZIP 不含这两件生产工件。以下默认使用 stock 模式，请在结果中注明。

## 4. Clone the benchmark and install its client

Clone the project from its GitHub repository. If you already have the release checkout, set `AGOS_REPO` to its absolute path in your session file and skip the clone. Use one terminal as the evaluation terminal for the remaining sections. The optional service terminals have their own setup blocks.

**中文。** 以下命令克隆正式项目地址。已有本地检出时，设好 `AGOS_REPO` 后直接执行第二块。后续以该目录为工作目录，评测终端始终使用 `carlaAir` 环境。

```bash
source "$HOME/.config/agos-bench/session.sh"
git clone https://github.com/Yomier/AGOS-Bench.git "$AGOS_REPO"
```

```bash
source "$HOME/.config/agos-bench/session.sh"
conda activate carlaAir
cd "$AGOS_REPO"
python -m pip install -e '.[simulator,dev,dataset]'
python -m pip check
python -c 'import agos, carla.libcarla, airsim; print(agos.__file__, carla.libcarla.__file__)'
python -m benchmark.run_parallel_eval --help
python -m pytest -q
git rev-parse HEAD
```

For the stock installation, initialize this Bash array in the evaluation terminal. Every subsequent invocation below inherits this choice.

```bash
native=()
```

If you already possess the three historical native artifacts, put the patched UE in a **separate complete simulator installation**, switch `AGOS_SIM_ROOT` to that installation, then use this optional block in the evaluation terminal. Compare the three file digests with section 3. The printed identity `v13-matched` checks all three configured file hashes. The live run separately checks the native client mappings. Do not export a shell-wide `LD_PRELOAD`. [Native profile implementation](../carlaair/engine/profile.py)

```bash
read -erp 'Complete simulator root with the patched engine: ' AGOS_SIM_ROOT
export AGOS_SIM_ROOT
read -erp 'Absolute path to the admitted Stop guard .so: ' GUARD_SO
profile_file="$AGOS_WORK/native-$(date -u +%Y%m%dT%H%M%SZ).json"
python -m carlaair.engine.profile \
  --engine "$AGOS_SIM_ROOT/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" \
  --libcarla "$(python -c 'import carla.libcarla as m; print(m.__file__)')" \
  --guard "$GUARD_SO" --output "$profile_file" && native=(--native-profile "$profile_file")
```

## 5. Obtain and check the data volumes

One test volume is enough for the smoke. Keep the **split name, town, episode ID and plan hash** together. Episode IDs can repeat in different towns and splits. Four task volumes contain 2,125 episodes. The `certification` volume contains independent replay evidence for all 2,125 tasks. `trajectories-train1100` holds the 1,100 expert trajectories and 27,720 labeled samples used to construct a new training corpus. [Dataset release and availability](../dataset/README.md) · [Corpus and adapter training](../training/README.md)

| Volume argument | Split for the runner | Episodes |
|---|---|---:|
| `taskset-test300` | `test300` (held-out Town03) | 300 |
| `taskset-seentest450` | `seentest450` (seen cities) | 450 |
| `taskset-validation275` | `validation275` | 275 |
| `taskset-train1100` | `train1100` | 1,100 |

Six archives are available in the offline delivery. External hosting is pending. To verify a full delivery, run `python dataset/verify.py --volume all --directory "$DELIVERY"` from the repository root with `DELIVERY` set to the directory holding the archives. Expect six `OK` lines. A missing archive, missing checksum or digest mismatch returns a nonzero exit status. For the first smoke, copy just the test volume into your data root as shown below. [Checksum list](../dataset/SHA256SUMS)

**中文。** 本地交付含四个任务卷、一个覆盖全部 2,125 项任务的独立认证卷，以及一个专家轨迹卷。`--volume all` 严格校验这六卷。首次冒烟只需要 `taskset-test300`。有离线文件时执行下面的复制块；已有文件可略过复制。HF 发布后也可用下一块下载，二选一。两条路线都用本次代码随附的摘要校验，通过后才解压。

```bash
cd "$AGOS_REPO"
volume=taskset-test300
read -erp 'Directory containing the delivered .tar.gz volumes: ' DELIVERY
cp -n "$DELIVERY/$volume.tar.gz" "$AGOS_DATA_ROOT/"
```

**After HF publication**, the alternative is to download the same volume from the dataset repository. To bind a Hub revision, add `--revision` and its commit hash to the download command.

```bash
cd "$AGOS_REPO"
volume=taskset-test300
python dataset/download.py --volume "$volume" --directory "$AGOS_DATA_ROOT"
```

For either route, expect `OK  taskset-test300.tar.gz`, followed by the selected plan's digest. This archive's SHA-256 is `b3c314bb12a47d947f884d34ddbeb276fb96931b5d22147810271cd790faf127`.

```bash
python dataset/verify.py --volume "$volume" --directory "$AGOS_DATA_ROOT" && \
  tar -xzf "$AGOS_DATA_ROOT/$volume.tar.gz" -C "$AGOS_DATA_ROOT"
EPISODE=$(jq -er '.splits.UnseenTest300.episode_keys[0]' dataset/agos-tasksets-splits-v1.json)
PLAN="$AGOS_DATA_ROOT/test300/dataset/${EPISODE}_plan.json"
test -s "$PLAN" && sha256sum "$PLAN"
```

Repeat with the appropriate volume to use another split. The data root passed to the runner is the **parent** of `test300/`, `seentest450/`, etc. A complete split also contains the assets referenced by each plan. If using an offline transfer with a different release, obtain its matching checksum file from that release and pass `--checksums` to `dataset/verify.py`. Preserve the archive digest, the dataset revision and the code commit with your results.

## 6. Run one episode with zero model API calls

Run this in the evaluation terminal after closing the standalone demo. `--dry-run` checks the selected data and configuration without an engine or model request. Removing `--dry-run` launches a real engine, connects both clients and runs the oracle policy. `menu_oracle` is a deployment diagnostic. For a model-based arm, configure an image-capable model endpoint. The cooperative-search arms also call the perception service. [Evaluation entry point](../benchmark/run_parallel_eval.py)

**中文。** 用上节从真实索引取到的键做冒烟，`test300` 对应 Town03。先做离线演练，再用不同的运行目录跑真实单集；这两步均无需 API 密钥、模型或 GSAM。查看真实结果、worker 日志与引擎退出情况后再进行后续模型实验。

```bash
cd "$AGOS_REPO"
SMOKE="$AGOS_OUTPUT_ROOT/smoke-$(date -u +%Y%m%dT%H%M%SZ)"
smoke_args=(--data-root "$AGOS_DATA_ROOT" --split test300 --episode "$EPISODE"
            --arm menu_oracle --workers 1 "${native[@]}")
python -m benchmark.run_parallel_eval "${smoke_args[@]}" --run-dir "$SMOKE-plan" --dry-run && \
  python -m benchmark.run_parallel_eval "${smoke_args[@]}" --run-dir "$SMOKE"
jq . "$SMOKE/summary.json"
python - "$SMOKE" "$EPISODE" <<'PY'
import json, sys
from pathlib import Path
folder = Path(sys.argv[1]) / 'episodes' / 'test300' / sys.argv[2]
terminal = json.loads((folder / 'terminal.json').read_text())
assert terminal['outcome'] in ('success', 'failure'), terminal
result = json.loads((folder / terminal['attempt'] / terminal['result']).read_text())
print(terminal['outcome'], 'VLM calls', result['vlm_calls'], 'HTTP attempts', result['vlm_http_attempts'])
assert result['vlm_calls'] == 0 and result['vlm_http_attempts'] == 0
PY
```

For a CPU-only check on a separate machine, install `.[dev,dataset]` in Python 3.10, verify/extract the test volume, and run the dry-run command with `AGOS_SIM_ROOT` unset and `native=()`. A JSON line with `tasks: 1`, `slots: 1`, `mode: menu_oracle`, `native: upstream` and `engine_path_checked: false` confirms plan/reference/config/registry validation there. The subsequent live command requires the Linux GPU host.

## 7. Start the shared perception service

The shared service loads Grounding DINO, SigLIP2 and SAM2 at startup. For the four cooperative-search arms, the UAV proposal screen requests `segment=False`. The UGV ranks detected crops from its camera views using reference-image similarity. The optional `segment=True` service check below requests a mask when the detector proposes a box. [Agent perception path](../agos/methods/cooperative_search/perception.py)

Run one shared service on the simulator host. Minimal-Agent users can skip this section and proceed to section 8. The following pinned upstream revisions provide reproducible downloads. The detector revision matches the historical launch command. Record the actual files you used if comparing model or runtime variants. [Grounding DINO Base](https://huggingface.co/IDEA-Research/grounding-dino-base) · [SigLIP2 Base](https://huggingface.co/google/siglip2-base-patch16-224) · [SAM2.1 Hiera Small](https://huggingface.co/facebook/sam2.1-hiera-small)

**中文。** 新开感知终端，使用独立 Python 环境。服务启动时加载 DINO、SigLIP2 和 SAM2.1 三套模型。正式 UAV 筛选关闭分割，UGV 对检测框裁图做相似度排序。下文开启分割只用于额外的接口自检。PyTorch 与 torchvision 需匹配驱动支持的 CUDA 版本。下面的安装后断言同时检查 CUDA 和 torchvision 的 CUDA NMS 算子；如失败，按该机器对应的安装命令重装二者，然后再启动服务。 [PyTorch wheel selector](https://pytorch.org/get-started/locally/)

```bash
source "$HOME/.config/agos-bench/session.sh"
cd "$AGOS_REPO"
conda create -y -n agos-perception python=3.12 pip
conda activate agos-perception
python -m pip install -e '.[perception,dataset]'
python - <<'PY'
import torch, torchvision
from torchvision.ops import nms
print(torch.__version__, torchvision.__version__, torch.version.cuda)
assert torch.cuda.is_available()
boxes = torch.tensor([[0., 0., 4., 4.]], device='cuda')
assert nms(boxes, torch.ones(1, device='cuda'), 0.5).tolist() == [0]
PY
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(os.environ['MODEL_ROOT'])
for repo, revision in [
    ('IDEA-Research/grounding-dino-base', '12bdfa3120f3e7ec7b434d90674b3396eccf88eb'),
    ('google/siglip2-base-patch16-224', '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'),
    ('facebook/sam2.1-hiera-small', 'ee5bba1d82bb8749febdf90f45e84b687142ba03'),
]:
    snapshot_download(repo_id=repo, revision=revision, local_dir=root / repo.split('/')[-1])
PY
```

```bash
set -o pipefail
python -u -m carlaair.perception.gsam --host 127.0.0.1 --port 8201 --device cuda \
  --detector-model "$MODEL_ROOT/grounding-dino-base" \
  --encoder-model "$MODEL_ROOT/siglip2-base-patch16-224" \
  --sam-model "$MODEL_ROOT/sam2.1-hiera-small" \
  2>&1 | tee "$AGOS_WORK/logs/gsam-$(date -u +%Y%m%dT%H%M%SZ).log"
```

Leave this terminal running. In the evaluation terminal, probe a real image through the detector/encoder and request segmentation. An empty proposal list is a valid detector output for some images. A 400 response, `status: error`, missing model data or a CPU device needs investigation before ramp-up. [Perception service implementation](../carlaair/perception/gsam.py)

```bash
export AGOS_SIDECAR_URL=http://127.0.0.1:8201
curl -fsS "$AGOS_SIDECAR_URL/health" | jq -e '.ok and (.device | startswith("cuda"))'
python - <<'PY'
import base64, os
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
root = Path(os.environ['AGOS_DATA_ROOT']) / 'test300' / 'dataset'
image = next(p for p in sorted(root.rglob('*')) if p.suffix.lower() in {'.png','.jpg','.jpeg'})
buf = BytesIO()
Image.open(image).convert('RGB').save(buf, 'PNG')
frame = base64.b64encode(buf.getvalue()).decode('ascii')
r = requests.post(os.environ['AGOS_SIDECAR_URL'] + '/v1/gsam', timeout=120,
                  json={'image_b64':frame,'target_reference_b64':frame,'target_description':'vehicle','segment':True})
r.raise_for_status()
body = r.json()
print(image.name, body['status'], len(body['proposals']), body['latency_ms'])
assert body['status'] in {'ok','empty'}
PY
```

## 8. Choose how the agent obtains multimodal decisions

Choose **one** model route. OpenAI-compatible chat completions must accept `image_url` messages with inline image data. Use `minimal_agent` for the direct MLLM control, `agos_b` for AGOS-Agent or `baseline_a` for Baseline-Agent. The two other public arms vary coverage or wake-up scheduling. For a fine-tuned variant, use a trained adapter and match its served name to the client. The base model and LoRA checkpoints are separate downloads or training outputs. [Agent configurations](../agents/README.md) · [Training and adapter routing](../training/README.md)

### A. Local Qwen3.5-4B

Use a third terminal on the simulator host or a dedicated inference GPU. The historical zero-shot service used `vllm 0.28.0`, `transformers 5.16.1` and `torch 2.13.0+cu130`, bf16, context 16,384, 16 sequences, GPU memory utilization 0.30 and eager execution. Local vLLM defaults to loopback port 8114. The VRAM fraction is a fraction of a whole GPU. On a smaller device, it may be too small even to load the weights. Record any change to `--gpu-memory-utilization`, then recheck capacity for the engines. [vLLM launcher](../carlaair/serving/vllm.py) · [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-4B) · [vLLM installation](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)

**中文。** 本地模型路线在第三个终端执行。历史环境为独立 Python 3.12，须有匹配的 CUDA 驱动和 BF16 GPU。使用已发布模型目录可跳过下载，保留 tokenizer、processor 和权重索引。`--dry-run` 可先检查 shell 命令。实际启动后该终端保持前台，评测终端也要导出同一密钥。

```bash
source "$HOME/.config/agos-bench/session.sh"
cd "$AGOS_REPO"
conda create -y -n agos-vllm python=3.12 pip
conda activate agos-vllm
python -m pip install 'vllm==0.28.0' 'transformers==5.16.1'
python -m pip install -e '.[dataset]'
python - <<'PY'
import os, torch, transformers, vllm
from huggingface_hub import snapshot_download
print(vllm.__version__, transformers.__version__, torch.__version__, torch.version.cuda)
assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
snapshot_download('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
                  local_dir=os.path.join(os.environ['MODEL_ROOT'], 'Qwen3.5-4B'))
PY
export VLLM_API_KEY=local
set -o pipefail
python -m carlaair.serving.vllm --model-dir "$MODEL_ROOT/Qwen3.5-4B" --dry-run && \
  python -u -m carlaair.serving.vllm --model-dir "$MODEL_ROOT/Qwen3.5-4B" \
  2>&1 | tee "$AGOS_WORK/logs/vllm-$(date -u +%Y%m%dT%H%M%SZ).log"
```

In the evaluation terminal, use the model's served name and verify that `/v1/models` returns it. The following block selects the zero-shot base. For trained weights follow the optional adapter instructions immediately below it.

```bash
PROFILE=local
unset AGENT_CONFIG AGOS_UAV_MODEL AGOS_UGV_MODEL
export AGOS_VLM_URL=http://127.0.0.1:8114/v1/chat/completions
export AGOS_VLM_MODEL=qwen3.5-4b
export VLLM_API_KEY=local
curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:8114/v1/models | \
  jq -e --arg id "$AGOS_VLM_MODEL" 'any(.data[]; .id == $id)'
```

**Optional trained adapters.** Use your own completed training outputs. The [training guide](../training/README.md) uses `$AGOS_OUTPUT_ROOT/{mixed,uav,ugv}/adapter`. Stop your own base-only server in its terminal before starting a replacement. For mixed training, add `--adapter "mixed=$AGOS_OUTPUT_ROOT/mixed/adapter"` to the local launcher and set `AGOS_VLM_MODEL=mixed` in the evaluation terminal. For split training, pass both adapters to one service as shown below. The role-specific YAML selects a served adapter for every aerial or ground decision. A CUDA training job and these inference services compete for VRAM; finish training before starting evaluation on the same card.

**中文。** 零样本无需 LoRA。mixed 模式两角色用同一模型名；split 模式在同一端口加载两套 rank-8 adapter，再在评测终端导出两个被服务的角色名并指定 split 配置。下面两块只用于 split。训练权重目录来自读者自己的训练输出。

```bash
# vLLM terminal, after stopping the base-only service
python -u -m carlaair.serving.vllm --model-dir "$MODEL_ROOT/Qwen3.5-4B" \
  --adapter "uav=$AGOS_OUTPUT_ROOT/uav/adapter" --adapter "ugv=$AGOS_OUTPUT_ROOT/ugv/adapter" \
  2>&1 | tee "$AGOS_WORK/logs/vllm-split-$(date -u +%Y%m%dT%H%M%SZ).log"
```

```bash
# Evaluation terminal, split route only
export AGOS_UAV_MODEL=uav AGOS_UGV_MODEL=ugv
AGENT_CONFIG="$AGOS_REPO/configs/examples/agos_b_split.yaml"
curl -fsS -H "Authorization: Bearer $VLLM_API_KEY" http://127.0.0.1:8114/v1/models | \
  jq -e --arg u "$AGOS_UAV_MODEL" --arg g "$AGOS_UGV_MODEL" '([.data[].id] | index($u) != null and index($g) != null)'
```

### B. Remote image-capable chat completions

Skip the local vLLM installation. In the evaluation terminal enter the provider's **full chat-completions endpoint**, deployed model identifier and credential. The harness `api` profile changes the worker count. The URL selects the service. A remote batch sends camera/reference images and prompts to that provider and may incur inference charges. Keep the detector running on the simulator host when evaluating a cooperative-search arm.

**中文。** 远端模式只把 MLLM 放在远端。向服务方获取可接收图像的完整 chat/completions 地址、准确模型名和密钥，在评测终端输入。`--profile api` 本身不会切换模型地址。先用下节的一次真实图像调用核对鉴权和多模态支持。

```bash
PROFILE=api
AGENT_CONFIG="$AGOS_REPO/configs/examples/agos_b_api.yaml"
read -rp 'Full HTTPS chat completions URL: ' AGOS_VLM_URL
read -rp 'Deployed image-capable model ID: ' AGOS_VLM_MODEL
read -rsp 'Provider API key: ' VLLM_API_KEY
printf '\n'
export AGOS_VLM_URL AGOS_VLM_MODEL VLLM_API_KEY
```

### Select Minimal-Agent

For the paper configuration, use local Qwen3.5-4B from route A. In the evaluation terminal select the following arm and configuration **before** the model probe and pilot commands in section 9. For a hosted endpoint that rejects `chat_template_kwargs`, use a copy of the Minimal-Agent YAML with that optional key removed and record the changed configuration with the run.

**中文。** 复现论文的 Minimal-Agent 时选择本地 Qwen3.5-4B。先运行下面两行，再执行第 9 节的图像请求和单槽试跑；这条路径无需启动第 7 节的感知模型。

```bash
ARM=minimal_agent
AGENT_CONFIG="$AGOS_REPO/agents/configs/minimal_agent.yaml"
```

## 9. Check model responses and measure worker capacity

In the **evaluation terminal**, send a red image through each selected decision route. The shared configuration sends one request. The split configuration sends one to the UAV adapter and one to the UGV adapter. Check the served names and answers in the output. A hosted service may bill these requests. The subsequent one-worker pilot exercises the full agent prompts.

```bash
cd "$AGOS_REPO"
python - "${AGENT_CONFIG:-agents/configs/agos_b.yaml}" <<'PY'
import base64, io, os, sys, requests
from PIL import Image
from agos.lite_core import load_config
vlm = load_config(sys.argv[1])['vlm']
if 'uav_model' in vlm or 'ugv_model' in vlm:
    routes = [('UAV', vlm['uav_model']), ('UGV', vlm['ugv_model'])]
else:
    routes = [('shared', vlm['model'])]
img = io.BytesIO()
Image.new('RGB', (224,224), 'red').save(img, 'PNG')
uri = 'data:image/png;base64,' + base64.b64encode(img.getvalue()).decode('ascii')
question = [{'type':'text','text':'Name the single color in the image in one word.'},
            {'type':'image_url','image_url':{'url':uri}}]
for role, model in routes:
    payload = {'model':model,'messages':[{'role':'user','content':question}],
               'max_tokens':vlm['max_output_tokens'],'temperature':vlm['temperature']}
    if 'chat_template_kwargs' in vlm:
        payload['chat_template_kwargs'] = vlm['chat_template_kwargs']
    response = requests.post(vlm['api_url'], timeout=vlm['timeout_s'],
        headers={'Authorization':'Bearer '+os.environ[vlm['api_key_env']]}, json=payload)
    response.raise_for_status()
    content = response.json()['choices'][0]['message']['content']
    assert isinstance(content, str) and content.strip(), f'{role} {model}: empty answer'
    print(role, model, content)
PY
```

Run independent short pilots at **1 → 2 → 4 → 6** workers with local inference, or **1 → 2 → 4 → 6 → 8 → 10** with a remote API. These are operator-selected stages for measuring capacity. The public runner caps the profiles at six and ten respectively, staggers cold starts by 30 seconds and gives each active slot a fresh engine per attempt. Run the next stage only after inspecting the current one. Using a fresh run directory per stage keeps timing and outcomes attributable to a specific concurrency level. GPU memory measured under one town or one model can change with the others. [Slot configuration](../carlaair/engine/slot.py) · [Runner arguments](../benchmark/run_parallel_eval.py)

**中文。** 共享模型发一次真实图像请求；分角色模式将同一张图分别发给 UAV、UGV 两个实际服务的模型名，各验一次，期望回答红色。选定 `PROFILE=local` 或 `api` 后定义下列函数。逐档手动执行，1 个 worker 用 1 集，后续每档选至少与 worker 数一样多的不同集，才会真的并发。等本档收卷后检查 summary、视觉轨迹、模型鉴权、剩余显存和延迟，再到下一档。遇到设施异常先查原因，保留发生时的 attempt。

```bash
cd "$AGOS_REPO"
ARM=${ARM:-agos_b}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
common=(--data-root "$AGOS_DATA_ROOT" --split test300 --arm "$ARM" --profile "$PROFILE" "${native[@]}")
if [ -n "${AGENT_CONFIG:-}" ]; then common+=(--config "$AGENT_CONFIG"); fi
pilot() {
  local count=$1 key dir="$AGOS_OUTPUT_ROOT/pilot-$STAMP-$PROFILE-$ARM-w$1"
  local -a selected=()
  while IFS= read -r key; do selected+=(--episode "$key"); done < <(
    jq -r --argjson n "$count" '.splits.UnseenTest300.episode_keys[:$n][]' dataset/agos-tasksets-splits-v1.json
  )
  (( ${#selected[@]} == 2 * count )) || return 2
  python -m benchmark.run_parallel_eval "${common[@]}" "${selected[@]}" --workers "$count" \
    --stagger-seconds 30 --max-attempts 1 --run-dir "$dir" || return $?
  jq -e '.coverage == 1 and (.pending | length == 0)' "$dir/summary.json"
}
pilot 1
```

At each step run `pilot 2`, then `pilot 4`, then `pilot 6` in the same terminal. For `PROFILE=api`, continue with `pilot 8` and `pilot 10` if there is sufficient GPU/CPU memory and the endpoint accepts the offered load. Use the largest worker count that passed on this host. Model failure can be a valid outcome; the pilot is checking for working cameras, actual service calls, normal task termination and clean engine exits. Inspect live slot status during the run. With zero-based slot `i`, CARLA RPC occupies `15500 + 10*i`, streaming and internal ports use RPC+1 and RPC+2, AirSim uses `31050 + 10*i` and the reserved TM port is `18000 + 10*i`. Check both IPv4 and IPv6 listeners, and check the host's ephemeral range if bind failures occur.

```bash
# Monitoring terminal, while a pilot or the full run is active
source "$HOME/.config/agos-bench/session.sh"
read -erp 'Full path to the active run: ' ACTIVE_RUN
jq '{total,successes,failures,coverage,pending_count:(.pending|length)}' "$ACTIVE_RUN/summary.json"
find "$ACTIVE_RUN/slots" -name status.json -exec jq '{slot,stage,ports,engine_pid,child_pid}' {} \;
tail -n 3 "$ACTIVE_RUN/resources.jsonl"
ss -ltnp
cat /proc/sys/net/ipv4/ip_local_port_range
```

## 10. Start the split and read the results

The machine is ready for the selected worker count once the verified archive and references load, the actual simulator passes the dual-RPC/camera probe, the zero-API oracle has an adjudicated terminal, the perception and image-model probes return usable responses, and the chosen model pilot finishes with full coverage and stopped slots. Keep the same model ID, endpoint, data, native runtime and arm configuration throughout a run. The released arm YAML sets the method budget to 80 team steps and 1,000 model calls. The default wall timeout is 9,000 seconds per episode. Each scheduled slot reserves the host's ports, so run one batch at a time on the simulator host.

**中文。** 在评测终端输入这台机器经过阶跃验证的并发数，对整个 split 先做 dry-run（应显示 300 个 test300 任务及预期 profile、模型臂），再启动实际任务。独立实验使用新目录。切换模型、臂或 split 后重新做小批检查；不要覆盖已有运行。

```bash
read -rp 'Highest worker count verified on this host: ' WORKERS
RUN="$AGOS_OUTPUT_ROOT/$ARM-test300-$PROFILE-$(date -u +%Y%m%dT%H%M%SZ)"
batch=("${common[@]}" --workers "$WORKERS" --stagger-seconds 30 --episode-timeout 9000 --run-dir "$RUN")
python -m benchmark.run_parallel_eval "${batch[@]}" --dry-run && \
  python -m benchmark.run_parallel_eval "${batch[@]}"
```

The summary's `sr` is a fraction, `successes / (successes + failures)`. `coverage` is `(successes + failures) / total`. An empty scored denominator yields a null SR. The paper-metrics command verifies the locked result hashes, reports normalized metrics in **percent**, navigation error (`NE`) in meters and decisions (`DS`) in calls per episode. Undefined or unsupported metrics are null. `menu_oracle` has no model decision cost to report. [Scoring rules](../benchmark/scoring.py) · [Manuscript metrics](../benchmark/paper_metrics.py)

| Inspect | What it records |
|---|---|
| `manifest.json` | Selected split/town/episode keys and plan hashes, arm/config and native identity |
| `summary.json` | Successes, method failures, current pending task keys, SR and coverage |
| `resources.jsonl`, `slots/slot-XX/status.json` | Host load, GPU use, slot PID, ports and lifecycle stage |
| `episodes/<split>/<town>/<episode>/terminal.json` | Most recent scored attempt, decision and result hash |
| `attempt-0001/`, `attempt-0002/`, … under that episode | `job.json`, `worker.log`, `engine.log`, score, result and traces produced by that attempt |

```bash
jq '{total,successes,failures,coverage,sr,pending_count:(.pending|length)}' "$RUN/summary.json"
python -m benchmark.paper_metrics "$RUN"
```

A successful ground confirmation must identify the correct object under Confirm30, within 30 m horizontally, with line of sight and visible in the stopping views. A false confirmation, exhausted step/call budget, navigation stall or patrol loop can yield a **scored method failure**. A simulator crash, camera timeout, service failure, unrecognized termination or visually contaminated run is **pending**. Pending episodes stay outside SR until repaired. A rising completed-episode count alone can hide bad camera/model service responses. Inspect per-attempt traces, model calls and logged `vision.status` counts. The public scorer treats `vision_errors > vision_ok` as contaminated. Publication-quality coverage requires zero pending episodes. [Scoring implementation](../benchmark/scoring.py)

## 11. Repair pending episodes and stop services

Read the first pending attempt before restarting. Live runner exit codes are `0` for complete evaluation, `2` for a setup/input failure, `3` for unresolved pending/infrastructure interruption, and `130` for a handled interrupt. If an episode wrote a valid result and its engine subsequently failed to stop, coverage can already be 1 while the runner returns `3`. Inspect `slots/slot-XX/status.json` for `stop_failed` and verify the recorded engine has exited before another dispatch. The separate paper-metrics command exits `2` if coverage is incomplete. Correct the facility problem, then resume using the same original selection, run directory, model endpoint, YAML and native identity. Existing valid successes **and valid failures** stay locked by hash. Regenerating the manifest or deleting `terminal.json` breaks that protection. The default runner attempts pending tasks up to twice per invocation; later invocations create new attempt directories.

**中文。** 先在原目录里查看 pending 的首个 attempt，修复仿真或服务问题。对同一 `batch` 加 `--resume`，只补缺失和设施 pending，成功与有效方法失败均保留。补跑结束应看到 `coverage=1`、`pending=[]`，再计算正式指标。另开 shell 时，重新设置原路径、模型、profile、split、arm 和 `RUN`；不要再次执行生成新时间戳的行。

```bash
jq -r '.pending[]' "$RUN/summary.json"
first=$(jq -r '.pending[0] // empty' "$RUN/summary.json")
if [ -n "$first" ]; then
  find "$RUN/episodes/$first" -maxdepth 2 -type f | sort
  test ! -f "$RUN/episodes/$first/terminal.json" || jq . "$RUN/episodes/$first/terminal.json"
fi
```

```bash
python -m benchmark.run_parallel_eval "${batch[@]}" --resume && \
  python -m benchmark.paper_metrics "$RUN"
```

To interrupt, press **Ctrl+C in the evaluation terminal first**. Let the supervisor close its own episode child and UE process. Check each created slot's final `stage` and investigate `stop_failed` against the recorded PID, command line and engine log. An old PID can be reused. When engines have exited, press Ctrl+C in the **vLLM** and **GSAM** foreground terminals you started. For the remote route, only the local GSAM service belongs to this setup. Keep run directories, source revisions, model revisions and logs. Use `ss -ltnp` and `nvidia-smi` to confirm resource release.

**中文。** 停服顺序为评测前台 Ctrl+C → 等各 slot `stopped` → 分别在自己启动的 vLLM、GSAM 前台 Ctrl+C → 查端口、GPU 和日志。远端服务归提供方管理。全局杀同名 UE 会影响其他作业并破坏现场。

## Troubleshooting / 故障排查

| Symptom / 现象 | What to check and how to recover / 处理 |
|---|---|
| `nvidia-smi` fails or `cuInit` returns nonzero | Check the loaded driver and reboot after a driver upgrade. Re-run the two checks and `vulkaninfo --summary` as the actual evaluation account. 有 NVML 输出仍需确认 CUDA 新进程可用。 |
| Vulkan shows only llvmpipe, or engine cannot create a device | Inspect NVIDIA's ICD in `/usr/share/vulkan/icd.d` or `/etc/vulkan/icd.d`, device permissions and the current login groups. For containers verify graphics libraries/device passthrough. 看到软件设备时不要发评测。 |
| `import carla`/`airsim` fails or reports the wrong ABI | Use `conda activate carlaAir`, Python 3.10, `PYTHONNOUSERSITE=1`, and the bundled `env_setup` payload. Inspect `carla.libcarla.__file__` and `ldd` on that extension. Rebuild the AirSim sdist after its build-time dependencies have been installed. 客户端应来自官方 ZIP，避免通用 PyPI carla 覆盖。 |
| ZIP hash mismatch / 数据卷哈希不符 | Fetch the correct immutable revision into a fresh download location and verify again. For AGOS volumes select the checksum file matching the delivered archive. 不能靠改 checksum 放行。 |
| Unknown episode / missing plan or reference image | Inspect the split index, the `split/dataset/TownXX/` layout and tar contents. Select `Town03` for `test300`. Re-extract the verified volume with all references. |
| Dry-run passes but engine fails to launch | Inspect `attempt-*/engine.log`, execute bit, ELF architecture and missing `ldd` libraries. Use a non-root owner and an executable filesystem. Headless evaluation uses `-RenderOffScreen` and real Vulkan rendering. |
| Repeated `checking_camera_and_rpc` or `waiting_for_ports` | Read `worker.log` and all five ports from slot status. Identify the process using each TCP listener. End your own old run gracefully or wait; check RAM/VRAM and the correct map. 单独开放 RPC 不能证明 AirSim 和相机可用。 |
| 401/403/404, empty completion or image-format error | Confirm key in the **worker's shell**, exact model ID, complete URL, and provider support for image data and the request fields in the selected YAML. Send the image probe, then run the one-worker pilot again. |
| HTTP 429, slow responses, OOM or driver reset under load | Stop at the last working concurrency stage. Inspect per-attempt timestamps, endpoint quotas, `resources.jsonl`, memory left after loading the three perception models and local vLLM. Retry pending tasks after recovery. |
| Sidecar HTTP 200 yet many `vision:error` / stale successes | Inspect its timestamped logs, `/v1/gsam` and `/v1/ugv_views` responses, actual model paths/device and the attempt trace. Restart the owning service and use the same frozen weights for repair. |
| Native-profile mismatch, stale lock or existing run | Check the three full hashes and exact simulator root, runtime environment, active owner and original `manifest.json`. Resume the correct run; use a new directory for a changed identity. |

The upstream archive contents and selected ELF hashes can be checked from the public distribution. Shell syntax, CLI help and offline data checks exercise different portions of the deployment. Live driver, UE, perception, multimodal inference and concurrency admission require running the probes and pilots on the target Linux GPU host.
