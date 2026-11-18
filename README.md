# AGOS-Bench

**Embodied air-ground cooperative object search.** A UAV and a ground vehicle receive one reference image of a vehicle and a map of the search region. The UAV searches from above. The ground vehicle must reach the candidate, see it in its cameras and confirm that it matches the reference. Both vehicles act on their own schedules in one CARLA-Air world.

| Start here | What you can do |
| --- | --- |
| [AGOS-Bench](benchmark/README.md) | Run episodes, inspect their outcomes and compute benchmark scores. |
| [AGOS-Dataset](dataset/README.md) | Select a split, verify the downloaded files and locate expert demonstrations. |
| [Agentic methods](agents/README.md) | Run AGOS-Agent, Baseline-Agent or either of the two single-factor ablations. |
| [CARLA-Air integration](carlaair/README.md) | Connect an upstream simulator installation, configure model services and manage concurrent episodes. |

### Start an evaluation

Follow the [deployment guide](docs/deployment.md) from a fresh Linux machine. Start with one episode to check the simulator and the dataset before selecting the local-model or API concurrency profile. The simulator release comes from the CARLA-Air authors. [Upstream CARLA-Air](https://github.com/louiszengCN/CarlaAir) · [v0.1.7 distribution](https://huggingface.co/tianlezeng/CarlaAIr-v0.1.7)

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

The accompanying manuscript is *Towards Embodied Air-Ground Cooperative Object Search: Benchmark, Dataset and Agentic Method*. Software citation metadata is available in [CITATION.cff](CITATION.cff).

AGOS-Bench code is distributed under [Apache-2.0](LICENSE). AGOS-Dataset is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). CARLA-Air and the model packages are available from their respective upstream authors.
