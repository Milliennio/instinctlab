# Project Instinct

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Overview

This repository is the environment side of [Project-Instinct](https://project-instinct.github.io/).

We aim at industralize Reinforcement Learning for Humanoid (legged robots) whole-body control.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.
- `Unified Ecosystem` This repository is a part of the Project-Instinct ecosystem, which includes the [instinct_rl](https://github.com/project-instinct/instinct_rl) and [instinct_onboard](https://github.com/project-instinct/instinct_onboard) repositories.
    - The core design of this ecosystem is to treat each experiment as a standalone structured folder, which start with a timestamp as a unique identifier.
    - Adding `--exportonnx` flag to the `play.py` script will export the policy as an ONNX model. After that, you should directly copy the logdir to the robot computer and use the `instinct_onboard` workflow to run the policy on the real robot.

**Keywords:** extension, template, isaaclab

## Warning
This codebase is under [CC BY-NC 4.0 license](LICENSE), with inherited license in IsaacLab. You may not use the material for commercial purposes, e.g., to make demos to advertise your commercial products or wrap the code for your own commercial purposes.

## Contributing
See our [Contributor Agreement](CONTRIBUTOR_AGREEMENT.md) for contribution guidelines. By contributing or submitting a pull request, you agree to transfer copyright ownership of your contributions to the project maintainers.

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for a list of acknowledged contributors.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) and **Switch to 5.1.0 version**. We recommend using the conda installation as it simplifies calling Python scripts from the terminal. At the time of release, the IsaacLab commit we are using is `37ddf626871758333d6ed89cf64ad702aef127d0` on Jan 30 2026.

- Install Instinct-RL by following the [installation guide](https://github.com/project-instinct/instinct_rl/blob/main/README.md).
    TL; DR;
    ```bash
    git clone https://github.com/project-instinct/instinct_rl.git
    python -m pip install -e instinct_rl
    ```

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

    ```bash
    # Option 1: HTTPS
    git clone https://github.com/project-instinct/instinctlab.git

    # Option 2: SSH
    git clone git@github.com:project-instinct/instinctlab.git
    ```

- Using a python interpreter that has Isaac Lab installed, install the library

    ```bash
    python -m pip install -e source/instinctlab
    ```

- To run with `instinct-rl`, you can use the following command after installing [instinct-rl](https://github.com/project-instinct/instinct_rl):

    ```bash
    python scripts/instinct_rl/train.py --task=Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --headless
    ```

## Documentation of Critical Components

- [Instinct-RL Documentation](https://github.com/project-instinct/instinct_rl/blob/main/README.md)
- [InstinctLab Documentation](https://github.com/project-instinct/instinctlab/blob/main/DOCS.md)

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory. The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse. This helps in indexing all the python modules for intelligent suggestions while writing code.


## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

To make the `pre-commit` run automatically on every commit, you can use the following command in your repository:

```bash
pre-commit install
```

## Train your own projects

***To preserve your code development and progress. PLEASE create your own repository as an individual project by referring to https://isaac-sim.github.io/IsaacLab/main/source/overview/own-project/index.html***

And copy `scripts/instinct_rl` to your own repository.

### Or you are just to stubborn and want to fork and directly modify the code in this repo.

- Please create a new folder in the `source/instinctlab/instinctlab/tasks` directory. The name of the folder should be your project name. Inside the folder, DO add `__init__.py` in each level of the subfolders. (Many people tend to forget this step and could not find the supposely registered tasks.)

- We inherit the manager based RL env from IsaacLab to add new features. DO use `instinctlab.envs:InstinctRlEnv` as the entry_point in the `gym.register` call. For example, if you want to add a new task, you can use the following code:

```python
import gymnasium as gym
from . import agents
task_entry = "instinctlab.tasks.shadowing.perceptive.config.g1"
gym.register(
    id="Instinct-Perceptive-Shadowing-G1-Play-v0",
    entry_point="instinctlab.envs:InstinctRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.perceptive_shadowing_cfg:G1PerceptiveShadowingEnvCfg_PLAY",
        "instinct_rl_cfg_entry_point": f"{agents.__name__}.instinct_rl_ppo_cfg:G1PerceptiveShadowingPPORunnerCfg",
    },
)
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/instinctlab"
    ]
}
```
obs_format to build EncoderMoEActorCritic: {'policy': {'base_ang_vel': (24,), 'projected_gravity': (24,), 'velocity_commands': (24,), 'joint_pos': (232,), 'joint_vel': (232,), 'actions': (232,), 'depth_image': (8, 18, 32)}, 'critic': {'base_lin_vel': (24,), 'base_ang_vel': (24,), 'projected_gravity': (24,), 'velocity_commands': (24,), 'joint_pos': (232,), 'joint_vel': (232,), 'actions': (232,), 'depth_image': (8, 18, 32)}, 'amp_policy': {'projected_gravity': (30,), 'joint_pos_rel': (290,), 'joint_vel': (290,), 'base_lin_vel': (30,), 'base_ang_vel': (30,)}, 'amp_reference': {'projected_gravity': (30,), 'joint_pos_rel': (290,), 'joint_vel': (290,), 'base_lin_vel': (30,), 'base_ang_vel': (30,)}}
Actor MLP: MoeLayer(
  (act_fn): ELU(alpha=1.0)
  (gate): Sequential(
    (0): Linear(in_features=896, out_features=4, bias=True)
  )
  (experts): ModuleList(
    (0-3): 4 x Sequential(
      (0): Linear(in_features=896, out_features=256, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=256, out_features=128, bias=True)
      (3): ELU(alpha=1.0)
      (4): Linear(in_features=128, out_features=64, bias=True)
      (5): ELU(alpha=1.0)
      (6): Linear(in_features=64, out_features=29, bias=True)
    )
  )
)
Critic MLP: MoeLayer(
  (act_fn): ELU(alpha=1.0)
  (gate): Sequential(
    (0): Linear(in_features=920, out_features=4, bias=True)
  )
  (experts): ModuleList(
    (0-3): 4 x Sequential(
      (0): Linear(in_features=920, out_features=256, bias=True)
      (1): ELU(alpha=1.0)
      (2): Linear(in_features=256, out_features=128, bias=True)
      (3): ELU(alpha=1.0)
      (4): Linear(in_features=128, out_features=64, bias=True)
      (5): ELU(alpha=1.0)
      (6): Linear(in_features=64, out_features=1, bias=True)
    )
  )
)
Actor Encoder: ParallelLayer(1 blocks): ModuleDict(
  (depth_encoder): Conv2dHeadModel(
    (conv): Conv2dModel(
      (conv): Sequential(
        (0): Conv2d(8, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
      )
    )
    (head): MlpModel(
      (model): Sequential(
        (0): Linear(in_features=2304, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=128, bias=True)
        (5): ReLU()
      )
    )
  )
)
Critic Encoder: ParallelLayer(1 blocks): ModuleDict(
  (depth_encoder): Conv2dHeadModel(
    (conv): Conv2dModel(
      (conv): Sequential(
        (0): Conv2d(8, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
      )
    )
    (head): MlpModel(
      (model): Sequential(
        (0): Linear(in_features=2304, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=128, bias=True)
        (5): ReLU()
      )
    )
  )
)
Discriminator Network Structure: Discriminator(
  (model): MlpModel(
    (model): Sequential(
      (0): Linear(in_features=670, out_features=1024, bias=True)
      (1): ReLU()
      (2): Linear(in_features=1024, out_features=512, bias=True)
      (3): ReLU()
      (4): Linear(in_features=512, out_features=1, bias=True)
    )
  )
)
[INFO]: Loading model checkpoint from: /home/you/instinct_rl/instinctlab/logs/instinct_rl/g1_parkour/0414/model_60000.pt

1. 观察空间 (Observation Format)
网络将输入数据分为了不同的段（Segments），供不同的网络分支使用：

Policy 输入 (Actor)：
向量数据：包含 base_ang_vel (24), projected_gravity (24), velocity_commands (24), joint_pos (232), joint_vel (232), actions (232)。
向量总维度：24 + 24 + 24 + 232 + 232 + 232 = 768 维。
图像数据：depth_image (8, 18, 32)，通常表示 8 帧/通道的深度图，分辨率为 18x32。
Critic 输入 (Value)：
向量数据：相比 Policy，多了一个 base_lin_vel (24)，因为在真实环境中机器人的线速度往往是不可直接观测的（属于特权信息），但在训练时 Critic 可以使用。
向量总维度：768 + 24 = 792 维。
图像数据：同样的 depth_image (8, 18, 32)。
AMP 相关输入：amp_policy 和 amp_reference 用于对抗性运动先验（WASABI/AMP）的判别器训练，这部分数据不直接进入 Actor/Critic 的主干。
2. 编码器网络 (Encoder)输入：depth_image (8, 18, 32)
因为输入中包含了三维的深度图像 depth_image，普通的 MLP 无法很好处理，所以你使用了 ParallelLayer 架构来分离并单独处理图像数据。Actor 和 Critic 各自拥有一个结构相同的 Encoder：

Conv2d 层：
输入通道为 8，输出通道为 4。
使用 3x3 卷积核，步长 1，填充 1，因此特征图大小不变（依然是 18x32）。
展平后的维度：4 * 18 * 32 = 2304 维。
Head MLP 层：
将 2304 维的卷积特征进一步压缩并提取高层语义。
网络层级：2304 -> 256 -> 256 -> 128。
最终输出：一个 128 维的深度图像潜在特征（Latent Vector）。
3. Actor 网络 (策略网络)
Actor 负责根据当前状态输出机器人的动作指令。它的主干是一个 混合专家层（MoeLayer）。

输入特征融合：Actor 会自动将前面计算出的 768 维向量数据，与 Encoder 算出的 128 维深度图像特征拼接（Concatenate）在一起。
输入总维度：768 + 128 = 896 维。
MoE 门控网络 (Gate)：
包含一个线性层 Linear(896, 4)，它的作用是根据当前的 896 维输入特征，为接下来的 4 个“专家”分配权重（通常经过 Softmax）。
专家网络 (Experts)：
一共有 4 个并行的专家网络。
每个专家的结构为：896 -> 256 -> 128 -> 64 -> 29 (激活函数为 ELU)。
最终输出：每个专家输出 29 维数据（正好对应你的 Unitree G1 29自由度机器人的动作空间）。网络会将这 4 个 29 维的输出按照 Gate 给出的权重进行加权求和，得到最终的 29 维动作均值。
4. Critic 网络 (价值网络)
Critic 负责评估当前状态的价值（Value），即预测未来能拿到多少 Reward。结构与 Actor 类似，但输入和输出不同。

输入特征融合：792 维的特权向量数据 + 128 维深度特征。
输入总维度：792 + 128 = 920 维。
MoE 门控网络 (Gate)：
同样是一个线性层 Linear(920, 4)，预测 4 个专家的权重。
专家网络 (Experts)：
包含 4 个并行的专家网络。
每个专家的结构为：920 -> 256 -> 128 -> 64 -> 1 (激活函数为 ELU)。
最终输出：每个专家评估出一个标量（1维 Value），加权求和后得出最终的状态价值。


我想修改配置文件，将深度图的处理从现在的 Conv2d 换成 Transformer,更改思路：
定位网络配置文件：强化学习智能体的网络结构（如 Policy/Critic 的 Encoder 配置）通常在类似于 G1ParkourPPORunnerCfg 的 Agent 配置文件中通过字典（dict）来指定。
修改 Encoder 的映射类：将配置字典中对应深度图 Encoder 的 class_name 从 "Conv2dHeadModel" 替换为 "TransformerHeadModel"。
调整配套参数：
移除属于卷积的超参：如 channels, kernel_sizes, strides, paddings, hidden_sizes 等。
添加属于 Transformer 的超参：如 num_heads, num_layers, d_model, dim_feedforward 以及输出降维方式 output_selection。
底层数据流匹配（自动完成）：根据你代码库中 ParallelLayer 的逻辑，当你把 class_name 指定为 TransformerHeadModel 时，temporal=True 会被触发，底层会自动将原本展平的 1D 历史观测数据恢复成 (Batch, Sequence_Length, Feature_Size) 的 3D 张量，你不需要去手动修改底层的观测张量重组代码。


ython source/instinctlab/instinctlab/tasks/parkour/scripts/collect_and_vis_depth.py --task Instinct-Shadowing-WholeBody-Plane-G1-Play-v0 --load_run <your_run_name>


transformer encoder:instinct_rl_amp_cfg_transformer

Actor Encoder: ParallelLayer(1 blocks): ModuleDict(
  (depth_encoder): TransformerHeadModel(
    (input_layer): MlpModel(
      (model): Sequential(
        (0): Linear(in_features=576, out_features=256, bias=True)
        (1): ReLU()
      )
    )
    (tf_encoder): TransformerEncoder(
      (layers): ModuleList(
        (0): TransformerEncoderLayer(
          (self_attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
          )
          (linear1): Linear(in_features=256, out_features=512, bias=True)
          (dropout): Dropout(p=0.1, inplace=False)
          (linear2): Linear(in_features=512, out_features=256, bias=True)
          (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
          (dropout1): Dropout(p=0.1, inplace=False)
          (dropout2): Dropout(p=0.1, inplace=False)
        )
      )
      (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    )
    (output_layer): MlpModel(
      (model): Sequential(
        (0): Linear(in_features=256, out_features=128, bias=True)
        (1): ReLU()
      )
    )
  )
)


之前的模型：cnn_encoder:instinct_rl_amp_cfg_cnn.py

Actor Encoder: ParallelLayer(1 blocks): ModuleDict(
  (depth_encoder): Conv2dHeadModel(
    (conv): Conv2dModel(
      (conv): Sequential(
        (0): Conv2d(8, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
      )
    )
    (head): MlpModel(
      (model): Sequential(
        (0): Linear(in_features=2304, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=256, bias=True)
        (3): ReLU()
        (4): Linear(in_features=256, out_features=128, bias=True)
        (5): ReLU()
      )
    )
  )
)



在实际部署（实机）的代码中，对 RealSense 相机获取的原始深度图进行了一系列预处理，以尽可能对齐仿真环境中的观测。根据你代码库中的 /home/you/instinct_rl/instinct_onboard/instinct_onboard/agents/parkour_agent.py 文件（特别是 refresh_depth_frame 和相关函数），主要进行了以下 8 个步骤 的处理：

1. 缩放 (Resize)
python
depth_image = cv2.resize(depth_image_np, self.output_resolution, interpolation=cv2.INTER_NEAREST)
将 RealSense 原始的高分辨率深度图（如 480x270）直接降采样到仿真配置的初始目标分辨率（例如 64x36）。 注意：这里使用了 cv2.INTER_NEAREST（最近邻插值）。这种插值方式不平滑，它直接抽取最近的一个像素点的值。这在保留物体边缘深度突变的同时，也会导致明显的“锯齿”和边缘像素跳动，这是引发纯 Transformer 策略失效的关键因素之一。

2. 图像裁剪 (Crop)
python
if hasattr(self, "crop_region"):
    # ...
    depth_image = depth_image[x1 : shape[0] - x2, y1 : shape[1] - y2]
将降采样后的图像边缘裁剪掉（例如上下左右各裁掉一部分），这通常用于切除相机边缘的畸变区域或视野盲区。裁剪后的最终尺寸就是输入给网络的 (18, 32)。

3. 缺失值修补 (Inpainting / Hole Filling)
python
mask = (depth_image < 0.2).astype(np.uint8)
depth_image = cv2.inpaint(depth_image, mask, 3, cv2.INPAINT_NS)
由于 RealSense 等深度相机在面对反光、极近距离（小于 0.2 米）或遮挡时，会出现无效的“黑洞”（深度值为 0 或极小）。 代码中将深度小于 0.2 米的区域视为无效掩码，并使用 OpenCV 的 Navier-Stokes 算法 (cv2.INPAINT_NS) 根据周围像素推断并填补这些空洞。

4. 盲区遮罩 (Blind Spot Masking - 可选)
python
if hasattr(self, "blind_spot_crop"):
    # ... 将指定的四周边缘区域强制设为 0
如果在仿真中配置了盲区，这里会将图像边缘特定宽度的区域像素值强行清零（模拟机器人本体遮挡）。

5. 高斯模糊滤波 (Gaussian Blur - 可选)
python
if hasattr(self, "gaussian_kernel_size"):
    depth_image = cv2.GaussianBlur(depth_image, self.gaussian_kernel_size, self.gaussian_sigma...)
如果配置中启用了模糊，这里会对深度图施加高斯模糊，用来平滑真实相机产生的高频早点（Noise）。

6. 截断与归一化 (Clipping & Normalization)
python
filt_m = np.clip(depth_image, self.depth_range[0], self.depth_range[1])
filt_norm = (filt_m - self.depth_range[0]) / (self.depth_range[1] - self.depth_range[0])
截断：将深度限制在物理上有意义的范围内（比如 [0.0, 1.5] 米），超出范围的值被“削平”。
归一化：将深度值线性映射到 [0.0, 1.0] 的区间内，这也是神经网络最容易消化的输入范围。
7. 时序历史缓存 (History Buffering & Downsampling)
在获得单帧归一化的深度图后，它会被放入一个环形缓冲区（CircularBuffer）中。 由于真实相机的帧率（如 60 Hz）可能与仿真的控制频率不同，_get_depth_image_downsample_obs 会按照算好的索引 self.depth_obs_indices，从中均匀抽取 8 帧历史数据，完成时间的降频对齐。

8. 维度重组 (Reshape)
python
depth_obs = (
    self._get_single_obs_term(self.depth_obs_names[0])
    .reshape(1, -1, self.depth_height, self.depth_width)
    .astype(np.float32)
)
最终在 step() 中，这 8 帧图像被转化为 (Batch=1, History=8, Height=18, Width=32) 的 4D 浮点张量，然后送入 ONNX 模型。

与 Transformer 模型失效的关系总结： 上述流程中，最近邻插值（Nearest Neighbor Resize） 和 空洞修补（Inpainting） 会在图像空间上引入局部的不规则形变和边缘偏移（且每一帧的偏移可能都在抖动）。

CNN 由于拥有局部的卷积核和池化层，能够自动容忍这种“1~2个像素的空间抖动”和“局部斑块变形”。
Transformer（当前使用的展平方案） 对输入向量的位置绝对敏感。实机上的这些处理产生的像素偏移，在 Transformer 眼里就是“完全没有见过的输入特征排布”，进而导致了遇到台阶无法跨越的问题。