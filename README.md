# AGOS-Bench

**Embodied air-ground cooperative object search.** A UAV and a ground vehicle receive one reference image of a vehicle and a map of the search region. The UAV searches from above. The ground vehicle must reach the candidate, see it in its cameras and confirm that it matches the reference. Both vehicles act on their own schedules in one CARLA-Air world.

| Start here | What you can do |
| --- | --- |
| [AGOS-Bench](benchmark/README.md) | Run episodes, inspect their outcomes and compute benchmark scores. |
| [AGOS-Dataset](dataset/README.md) | Select a split, verify the downloaded files and locate expert demonstrations. |
| [Agentic methods](agents/README.md) | Run AGOS-Agent, Baseline-Agent or either of the two single-factor ablations. |
| [CARLA-Air integration](carlaair/README.md) | Connect an upstream simulator installation, configure model services and manage concurrent episodes. |

### Start an evaluation

1. On an Ubuntu GPU machine, install the driver, the official CARLA-Air v0.1.7 Linux release and its matching Python SDK. The step-by-step guide includes Vulkan, camera and model-service checks. [Deployment guide](docs/deployment.md) · [Upstream CARLA-Air](https://github.com/louiszengCN/CarlaAir) · [v0.1.7 distribution](https://huggingface.co/tianlezeng/CarlaAIr-v0.1.7)
2. Activate the simulator environment, install this checkout with `python -m pip install -e '.[dev,dataset]'`, and obtain the `taskset-test300` archive. Verify its SHA-256 and extract it under `$AGOS_DATA_ROOT`. [Access and verification](dataset/README.md#access-and-verification)
3. Run the command below from this checkout. It validates one real Town03 task, its reference images and the agent configuration offline. With `$AGOS_SIM_ROOT` set and a GPU ready, remove `--dry-run` to execute one model-free simulation. Then choose the local or API profile in the deployment guide.

```bash
python -m benchmark.run_parallel_eval --data-root "$AGOS_DATA_ROOT" --split test300 \
  --episode Town03/lite_easy_00001 --arm menu_oracle --workers 1 \
  --run-dir "$AGOS_OUTPUT_ROOT/first-episode" --dry-run
```

### Task splits and demonstrations

| Split | Episodes | Experimental role |
| --- | ---: | --- |
| Train | 1,100 | Training tasks with expert trajectories |
| Validation | 275 | Model and method development |
| Seen test | 450 | New tasks in the five training cities |
| Unseen test | 300 | Held-out city, Town03 |

The four splits contain **2,125 task instances**. The 1,100 training episodes also provide **27,720 labeled examples** with camera frames and expert actions. A town and episode name can occur in more than one split, so keep the split and plan identity when joining records. The [training tools](training/README.md) explain how to build training inputs from the released trajectories. The paper's reported fine-tuning experiment used 8,212 examples from 301 episodes, of which 6,014 passed sequence-length filtering.

### Methods and scoring

AGOS-Agent assigns coverage jointly and wakes its multimodal model on events such as new candidate evidence. Baseline-Agent uses fixed coverage and calls the model at coverage decision points. Their perception and verification components are shared. The two ablations vary coverage and wake-up policy one factor at a time.

Success requires a ground `MATCH` verdict, horizontal distance at most 30 m, line of sight and target visibility in the ground vehicle’s stopping views. A false confirmation, exhausted budget or method-side stall counts as a failure. Infrastructure failures remain pending until rerun and stay outside the scored denominator.

### 中文入口

在这个任务中，无人机和地面车依据一张参考图协同寻找目标车辆。最终确认必须由地面车完成。首次部署请从[部署指南](docs/deployment.md)开始。数据分卷、校验和训练轨迹的字段说明见[数据说明](dataset/README.md)。

### Citation and licenses

The accompanying manuscript is *Towards Embodied Air-Ground Cooperative Object Search: Benchmark, Dataset and Agentic Method*. The arXiv identifier is pending. Software citation metadata is available in [CITATION.cff](CITATION.cff).

AGOS-Bench code is distributed under [Apache-2.0](LICENSE). AGOS-Dataset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). CARLA-Air and the model packages are available from their respective upstream authors.
