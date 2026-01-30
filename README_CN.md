# RoboChallenge-Table30-Calibration（中文说明）

[中文](README_CN.md) | [English](README.md)

本项目提供了一套用于 [RoboChallenge/Table30](https://huggingface.co/datasets/RoboChallenge/Table30) 数据集的**后标定（post-hoc calibration）**完整工作流。它支持将机器人 3D 坐标精确投影到相机 2D 图像坐标，从而为 Table30 基准中多样化的操作任务提供视觉反馈与数据对齐能力。

## 目录

- [概览](#overview)
- [快速开始](#quickstart)
- [安装](#installation)
- [支持的机器人与视角](#supported-robots--views)
- [数据集目录结构](#dataset-layout)
- [工作流程](#workflow)
  - [1. 配置](#1-configuration)
  - [2. 生成标注任务](#2-generate-label-tasks)
  - [3. 标注](#3-labeling)
  - [4. 构建标注后的数据集](#4-build-labeled-dataset)
  - [5. 标定](#5-calibration)
- [推理](#inference)
  - [用法](#usage)
  - [双臂支持（Aloha）](#dual-arm-support-aloha)
- [数学模型与细节](#mathematical-models--details)
  - [1. Project2D（有理函数模型）](#1-project2d-rational-function-model)
  - [2. PnP（Perspective-n-Point）](#2-pnp-perspective-n-point)
  - [3. 坐标变换与 Delta 应用](#3-coordinate-transformation--delta-application)
- [结果](#results)
- [排错指南](#troubleshooting)
- [项目结构](#project-structure)

## Overview

**RoboChallenge-Table30-Calibration** 的核心目标是为 Table30 数据集提供准确的相机标定参数。通过将机器人基坐标系下的 3D 点（例如夹爪指尖）映射到相机视角下的 2D 像素坐标，本工具集可用于：

- **视觉反馈**：在相机画面上叠加机器人状态信息。
- **数据验证**：检查录制的机器人状态与视觉观测之间的对齐程度。
- **下游任务**：为需要精确 2D-3D 对应关系的学习算法提供基础能力。

代码库专门面向 RoboChallenge/Table30 数据集中的 30 个操作任务与 4 种机器人形态（Aloha、ARX5、Franka、UR5）进行适配。

## Quickstart

1. 在 [config.py] 中将 `DATA_ROOT` 设置为你本地的 Table30 目录。
2. 生成标注任务：

```bash
python3 gen_label_tasks.py
```

3. 启动标注 UI，并打开 `http://localhost:5312`：

```bash
python3 serve.py
```

4. 构建用于拟合的 `processed_data.pkl`：

```bash
python3 read_data.py
```

5. 拟合并导出标定参数：

```bash
python3 main.py --method project2d
python3 main.py --method pnp
```

## Installation

请确保已安装 Python 3.10+。安装所需依赖：

```bash
pip install flask opencv-python numpy pyyaml scipy av tqdm
```

## Supported Robots & Views

系统支持多种机器人配置与相机视角，均在 `config.py` 中定义：

**机器人：**
- **Aloha**：双臂机器人（左/右臂）。
- **ARX5**：单臂机器人。
- **Franka**：单臂机器人（四元数位姿）。
- **UR5**：单臂机器人（四元数位姿）。

**视角：**
- **固定相机**：`global_view`、`side_view`。
- *说明*：本项目当前聚焦于固定相机标定。若需要腕部相机（第一人称视角）支持，可在代码层面做少量修改实现。

## Dataset Layout

本仓库默认 Table30 数据集目录结构如下（与 [config.py]、[gen_label_tasks.py]、[read_data.py] 的遍历逻辑一致）：

```text
DATA_ROOT/
  <task_name>/
    meta/task_info.json                # 包含 robot_id（例如 "arx5_9"、"aloha_1" 等）
    data/
      episode_000000/
        meta/episode_meta.json         # 包含帧数
        videos/<view_video>.mp4        # 每种机器人对应的视角视频映射在 config.py（VIDEO_CONFIG）中定义
        states/*.jsonl                 # read_data.py 使用的机器人状态日志
      episode_000001/
        ...
```

如果你的数据布局不同，请相应修改 [config.py]（尤其是 `VIDEO_CONFIG` 与 `get_data_dir()`）。

## Workflow

### 1. Configuration

开始之前，你**必须**在 `config.py` 中更新 `DATA_ROOT` 变量，使其指向你本地的 Table30 数据集目录。

```python
# config.py
DATA_ROOT = '/path/to/your/table30/dataset/' 
```

### 2. Generate Label Tasks

通过扫描数据目录生成标注任务。该过程会从录制的 episode 中随机抽取若干帧用于人工标注。

```bash
python3 gen_label_tasks.py
```

### 3. Labeling

启动基于 Web 的标注服务：

```bash
python3 serve.py
```

- 在浏览器中打开 `http://localhost:5312`。
- 选择任务（task）与视角（view）。
- 将鼠标移动到目标位置，按数字键 `1..N` 放置关键点（相对于图像宽/高归一化到 `[0, 1]`）。
- 点击 `Clear All Labels` 清空当前帧的标注。
- 使用 `A/W/←/↑` 查看上一帧，使用 `D/S/→/↓` 查看下一帧。
- 标注结果会自动保存到对应的 `table30/<task>/<view>/` 目录下的 `results.json`。

关键点数量：
- 单臂机器人（ARX5/Franka/UR5）：2 个点（`point0`、`point1`）
- Aloha：4 个点（`point0`..`point3`）= 左臂指尖 + 右臂指尖（名称在 `config.py` 中配置）

标注文件：

- `tasks.json`：待标注帧列表

```json
[
  {"path": "episode_000000", "frameid": 123},
  {"path": "episode_000001", "frameid": 45}
]
```

- `results.json`：以 `<episode>/videos/<video>.mp4_<frameid>` 为键的字典，存储归一化关键点

```json
{
  "episode_000000/videos/global_realsense_rgb.mp4_123": {
    "point0": [0.5123, 0.4139],
    "point1": [0.5589, 0.4128]
  }
}
```

![标注界面 - 首页](assets/home.png)
![标注界面 - 任务](assets/label.png)

### 4. Build Labeled Dataset

将 `tasks.json` + `results.json` + 机器人状态日志转换为每个任务的 `processed_data.pkl`，供拟合代码使用：

```bash
python3 read_data.py
```

生成路径：

```text
table30/<task_name>/<view_name>/processed_data.pkl
```

### 5. Calibration

运行标定流水线，使用标注数据拟合投影模型。

**使用有理函数模型（推荐）：**
```bash
python3 main.py --method project2d
```

**使用 PnP 模型：**
```bash
python3 main.py --method pnp
```

该过程会：
1.  读取 `processed_data.pkl`。
2.  优化模型参数（Project2D 系数，或 PnP 外参 + delta 偏移）。
3.  在 `project2d/results/` 或 `pnp/results/` 中生成验证用的图像与视频。
4.  将参数导出到 `project2d/parameters.yaml` 或 `pnp/extrinsics.yaml`。

如果 `pnp/intrinsics.yaml` 缺失，或其中未包含当前 task/view，PnP 流水线会使用黄金分割搜索估计相机内参（fx、fy、cx、cy），将结果写回 `pnp/intrinsics.yaml`，然后继续进行外参拟合。

## Inference

`infer.py` 脚本演示了如何使用已标定的参数将 3D 点投影到 2D 图像坐标。它会自动识别机器人类型，并同时支持单臂与双臂配置。

### Usage

```bash
python3 infer.py --task <task_name> --view <view_name> --method <method>
```

**参数说明：**
- `--task`：任务名称（例如 `arrange_flowers`、`clean_dining_table`）。
- `--view`：相机视角（例如 `global_view`、`side_view`）。
- `--method`：推理方法（`pnp` 或 `project2d`）。

### Dual-Arm Support (Aloha)

对于 Aloha 这类双臂机器人，推理引擎会自动同时处理两条机械臂。

**示例命令：**
```bash
python3 infer.py --task clean_dining_table --view global_view --method project2d
```

**输出示例：**
```text
--- Inference Example for clean_dining_table (global_view) ---
Method: project2d
Robot: aloha

Input Robot State (2 arms):
  Arm 1: Pose=[0.3  0.2  0.2  0.   1.57 0.  ], Gripper=0.04
  Arm 2: Pose=[ 0.3  -0.2   0.2   0.    1.57  0.  ], Gripper=0.04

Processing Arm 1 (Left)...
  Input Pose: [0.3  0.2  0.2  0.   1.57 0.  ]
  Calculating 3D Endpoints...
  Tip 1 (3D): [0.50086817 0.225964   0.23377097]
  Tip 2 (3D): [0.50086817 0.174036   0.23377097]
  Projecting to 2D Image Coordinates (Project2D)...
  Tip 1 (Normalized): (0.5115, 0.4139)
  Tip 2 (Normalized): (0.5589, 0.4128)

Processing Arm 2 (Right)...
  ...
```

### Single-Arm Example (ARX5)

```bash
python3 infer.py --task arrange_flowers --view global_view --method pnp
```

**输出示例：**
```text
--- Inference Example for arrange_flowers (global_view) ---
Method: pnp
Robot: arx5

Input Robot State (1 arms):
  Arm 1: Pose=[0.3  0.   0.2  0.   1.57 0.  ], Gripper=0.04
...
  Tip 1 (2D pixel): (433.90, 213.02)
  Tip 2 (2D pixel): (469.42, 212.72)
```

## Mathematical Models & Details

### 1. Project2D (Rational Function Model)

该模型使用有理多项式函数，将 3D 点 $(x, y, z)$ 映射到 2D 归一化像素坐标 $(u, v)$。它是一种数据驱动的方法，可在不显式引入相机内参的前提下，隐式处理镜头畸变与透视效应。

**公式：**

$$
u = \frac{p_0 x + p_1 y + p_2 z + p_6}{1 + p_3 x + p_4 y + p_5 z}
$$

$$
v = \frac{q_0 x + q_1 y + q_2 z + q_6}{1 + q_3 x + q_4 y + q_5 z}
$$

其中：
- $(x, y, z)$ 为机器人基坐标系下的 3D 特征点。
- $(u, v)$ 为归一化像素坐标（0 到 1）。
- $p_0, \dots, p_6$ 与 $q_0, \dots, q_6$ 为待学习的参数。

**YAML 结构（`project2d/parameters.yaml`）：**

```yaml
task_name:
  robot: robot_name
  view_name:
    regvec:
      - [p0, q0]  # Row 0
      - ...
      - [p6, q6]  # Row 6
    delta: [dx, dy, dz]
```

*   **`regvec`**：一个 $7 \times N$ 矩阵。
    *   **单臂**：$7 \times 2$（第 0 列为 $u$ 参数，第 1 列为 $v$ 参数）。
    *   **双臂**：$7 \times 4$（第 0-1 列：左臂 $u,v$；第 2-3 列：右臂 $u,v$）。
*   **`delta`**：偏移参数 `[dx, dy, dz]`。

### 2. PnP (Perspective-n-Point)

该方法使用经典的小孔相机模型，估计相机相对于机器人基坐标系的 6D 位姿（旋转 $\mathbf{R}$、平移 $\mathbf{t}$）。

**公式：**

$$
s \left[\begin{array}{c} u \\ v \\ 1 \end{array}\right] = \mathbf{K} \left( \mathbf{R} \left[\begin{array}{c} x \\ y \\ z \end{array}\right] + \mathbf{t} \right)
$$

其中：
- $\mathbf{K}$ 为相机内参矩阵（焦距 $f_x, f_y$ 与主点 $c_x, c_y$）。
- $\mathbf{R}$（旋转矩阵）与 $\mathbf{t}$（平移向量）为相机外参。

**YAML 结构（`pnp/extrinsics.yaml`）：**

```yaml
task_name:
  robot: robot_name
  view_name:
    rvec: [...] # Flattened rotation vectors
    tvec: [...] # Flattened translation vectors
    delta: [dx, dy, dz]
    resolution: [width, height]
```

### 3. Coordinate Transformation & Delta Application

在应用任意投影模型之前，需要将机器人的末端执行器位姿（通常是腕部/法兰）转换到具体的 3D 特征点（例如夹爪指尖）。这一步会使用 `delta` 偏移参数完成。

**通用公式：**

$$
\mathbf{P} = \mathbf{p}_{ee} + \mathbf{R}_{pose} \cdot \mathbf{offset}
$$

其中：
- $\mathbf{p}_{ee}$ 为末端执行器的位置。
- $\mathbf{R}_{pose}$ 为由机器人位姿（欧拉角或四元数）得到的旋转矩阵。
- $\mathbf{offset}$ 由 `delta`（$dx, dy, dz$）与夹爪宽度（$w$）共同确定。

**不同机器人的 offset 定义：**

#### Aloha（双臂）
**左臂：**

$$
\begin{aligned}
\mathbf{P}_{\text{left}} &= \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ \frac{w}{2} + dy \\ dz \end{bmatrix} \\
\mathbf{P}_{\text{right}} &= \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ -\frac{w}{2} - dy \\ dz \end{bmatrix}
\end{aligned}
$$

**右臂：**

$$
\begin{aligned}
\mathbf{P}_{\text{left}} &= \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ \frac{w}{2} + dy \\ dz \end{bmatrix} \\
\mathbf{P}_{\text{right}} &= \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ -\frac{w}{2} - dy \\ dz \end{bmatrix}
\end{aligned}
$$

#### ARX5
$$
\begin{aligned}
\mathbf{P}_{1} &= \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ -\frac{w}{2} - dy \\ dz \end{bmatrix} \\
\mathbf{P}_{2} &= \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ \frac{w}{2} + dy \\ dz \end{bmatrix}
\end{aligned}
$$

#### Franka
$$
\begin{aligned}
\mathbf{P}_{1} &= \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ \frac{w}{2} + dy \\ dz \end{bmatrix} \\
\mathbf{P}_{2} &= \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ -\frac{w}{2} - dy \\ dz \end{bmatrix}
\end{aligned}
$$

#### UR5
*（注意坐标轴交换）*

$$
\begin{aligned}
\mathbf{P}_{1} &= \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} \frac{w}{2} + dy \\ -dx \\ dz \end{bmatrix} \\
\mathbf{P}_{2} &= \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} -\frac{w}{2} - dy \\ -dx \\ dz \end{bmatrix}
\end{aligned}
$$

## Results

该标定方法在不同任务与机器人平台上都能达到较高精度。典型的归一化重投影误差通常远低于 1.0%，可确保 3D 机器人状态与 2D 相机视角之间的精确对齐。

### Extrinsics Evaluation Report

| Task Name | View | Robot | Project2D Error | Project2D Plot | PnP Error | PnP Plot |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| arrange_flowers | global_view | arx5 | 0.011328 | [View Plot](project2d/results/arrange_flowers/global_view/project2d_fit.png) | 0.020123 | [View Plot](pnp/results/arrange_flowers/global_view/pnp_fit.png) |
| arrange_flowers | side_view | arx5 | 0.009229 | [View Plot](project2d/results/arrange_flowers/side_view/project2d_fit.png) | 0.014245 | [View Plot](pnp/results/arrange_flowers/side_view/pnp_fit.png) |
| arrange_fruits_in_basket | global_view | ur5 | 0.006856 | [View Plot](project2d/results/arrange_fruits_in_basket/global_view/project2d_fit.png) | 0.016555 | [View Plot](pnp/results/arrange_fruits_in_basket/global_view/pnp_fit.png) |
| arrange_paper_cups | global_view | arx5 | 0.007222 | [View Plot](project2d/results/arrange_paper_cups/global_view/project2d_fit.png) | 0.009693 | [View Plot](pnp/results/arrange_paper_cups/global_view/pnp_fit.png) |
| arrange_paper_cups | side_view | arx5 | 0.004157 | [View Plot](project2d/results/arrange_paper_cups/side_view/project2d_fit.png) | 0.011333 | [View Plot](pnp/results/arrange_paper_cups/side_view/pnp_fit.png) |
| clean_dining_table | global_view | aloha | 0.014808 | [View Plot](project2d/results/clean_dining_table/global_view/project2d_fit.png) | 0.011188 | [View Plot](pnp/results/clean_dining_table/global_view/pnp_fit.png) |
| fold_dishcloth | global_view | arx5 | 0.004898 | [View Plot](project2d/results/fold_dishcloth/global_view/project2d_fit.png) | 0.008899 | [View Plot](pnp/results/fold_dishcloth/global_view/pnp_fit.png) |
| fold_dishcloth | side_view | arx5 | 0.005601 | [View Plot](project2d/results/fold_dishcloth/side_view/project2d_fit.png) | 0.009432 | [View Plot](pnp/results/fold_dishcloth/side_view/pnp_fit.png) |
| hang_toothbrush_cup | global_view | ur5 | 0.006777 | [View Plot](project2d/results/hang_toothbrush_cup/global_view/project2d_fit.png) | 0.009427 | [View Plot](pnp/results/hang_toothbrush_cup/global_view/pnp_fit.png) |
| make_vegetarian_sandwich | global_view | aloha | 0.012185 | [View Plot](project2d/results/make_vegetarian_sandwich/global_view/project2d_fit.png) | 0.015294 | [View Plot](pnp/results/make_vegetarian_sandwich/global_view/pnp_fit.png) |
| move_objects_into_box | global_view | franka | 0.016472 | [View Plot](project2d/results/move_objects_into_box/global_view/project2d_fit.png) | 0.020058 | [View Plot](pnp/results/move_objects_into_box/global_view/pnp_fit.png) |
| move_objects_into_box | side_view | franka | 0.016452 | [View Plot](project2d/results/move_objects_into_box/side_view/project2d_fit.png) | 0.015302 | [View Plot](pnp/results/move_objects_into_box/side_view/pnp_fit.png) |
| open_the_drawer | global_view | arx5 | 0.006765 | [View Plot](project2d/results/open_the_drawer/global_view/project2d_fit.png) | 0.005606 | [View Plot](pnp/results/open_the_drawer/global_view/pnp_fit.png) |
| open_the_drawer | side_view | arx5 | 0.004772 | [View Plot](project2d/results/open_the_drawer/side_view/project2d_fit.png) | 0.006514 | [View Plot](pnp/results/open_the_drawer/side_view/pnp_fit.png) |
| place_shoes_on_rack | global_view | arx5 | 0.005776 | [View Plot](project2d/results/place_shoes_on_rack/global_view/project2d_fit.png) | 0.008937 | [View Plot](pnp/results/place_shoes_on_rack/global_view/pnp_fit.png) |
| place_shoes_on_rack | side_view | arx5 | 0.003354 | [View Plot](project2d/results/place_shoes_on_rack/side_view/project2d_fit.png) | 0.008737 | [View Plot](pnp/results/place_shoes_on_rack/side_view/pnp_fit.png) |
| plug_in_network_cable | global_view | aloha | 0.003502 | [View Plot](project2d/results/plug_in_network_cable/global_view/project2d_fit.png) | 0.005957 | [View Plot](pnp/results/plug_in_network_cable/global_view/pnp_fit.png) |
| pour_fries_into_plate | global_view | aloha | 0.004096 | [View Plot](project2d/results/pour_fries_into_plate/global_view/project2d_fit.png) | 0.005583 | [View Plot](pnp/results/pour_fries_into_plate/global_view/pnp_fit.png) |
| press_three_buttons | global_view | franka | 0.010446 | [View Plot](project2d/results/press_three_buttons/global_view/project2d_fit.png) | 0.011783 | [View Plot](pnp/results/press_three_buttons/global_view/pnp_fit.png) |
| press_three_buttons | side_view | franka | 0.008143 | [View Plot](project2d/results/press_three_buttons/side_view/project2d_fit.png) | 0.008816 | [View Plot](pnp/results/press_three_buttons/side_view/pnp_fit.png) |
| put_cup_on_coaster | global_view | arx5 | 0.005664 | [View Plot](project2d/results/put_cup_on_coaster/global_view/project2d_fit.png) | 0.008178 | [View Plot](pnp/results/put_cup_on_coaster/global_view/pnp_fit.png) |
| put_cup_on_coaster | side_view | arx5 | 0.003307 | [View Plot](project2d/results/put_cup_on_coaster/side_view/project2d_fit.png) | 0.007817 | [View Plot](pnp/results/put_cup_on_coaster/side_view/pnp_fit.png) |
| put_opener_in_drawer | global_view | aloha | 0.006352 | [View Plot](project2d/results/put_opener_in_drawer/global_view/project2d_fit.png) | 0.004611 | [View Plot](pnp/results/put_opener_in_drawer/global_view/pnp_fit.png) |
| put_pen_into_pencil_case | global_view | aloha | 0.004148 | [View Plot](project2d/results/put_pen_into_pencil_case/global_view/project2d_fit.png) | 0.006877 | [View Plot](pnp/results/put_pen_into_pencil_case/global_view/pnp_fit.png) |
| scan_QR_code | global_view | aloha | 0.008651 | [View Plot](project2d/results/scan_QR_code/global_view/project2d_fit.png) | 0.009677 | [View Plot](pnp/results/scan_QR_code/global_view/pnp_fit.png) |
| search_green_boxes | global_view | arx5 | 0.021506 | [View Plot](project2d/results/search_green_boxes/global_view/project2d_fit.png) | 0.019801 | [View Plot](pnp/results/search_green_boxes/global_view/pnp_fit.png) |
| search_green_boxes | side_view | arx5 | 0.004553 | [View Plot](project2d/results/search_green_boxes/side_view/project2d_fit.png) | 0.008160 | [View Plot](pnp/results/search_green_boxes/side_view/pnp_fit.png) |
| set_the_plates | global_view | ur5 | 0.006966 | [View Plot](project2d/results/set_the_plates/global_view/project2d_fit.png) | 0.015899 | [View Plot](pnp/results/set_the_plates/global_view/pnp_fit.png) |
| shred_scrap_paper | global_view | ur5 | 0.005149 | [View Plot](project2d/results/shred_scrap_paper/global_view/project2d_fit.png) | 0.012577 | [View Plot](pnp/results/shred_scrap_paper/global_view/pnp_fit.png) |
| sort_books | global_view | ur5 | 0.008996 | [View Plot](project2d/results/sort_books/global_view/project2d_fit.png) | 0.014859 | [View Plot](pnp/results/sort_books/global_view/pnp_fit.png) |
| sort_electronic_products | global_view | arx5 | 0.009639 | [View Plot](project2d/results/sort_electronic_products/global_view/project2d_fit.png) | 0.014282 | [View Plot](pnp/results/sort_electronic_products/global_view/pnp_fit.png) |
| sort_electronic_products | side_view | arx5 | 0.049210 | [View Plot](project2d/results/sort_electronic_products/side_view/project2d_fit.png) | 0.019808 | [View Plot](pnp/results/sort_electronic_products/side_view/pnp_fit.png) |
| stack_bowls | global_view | aloha | 0.006792 | [View Plot](project2d/results/stack_bowls/global_view/project2d_fit.png) | 0.008455 | [View Plot](pnp/results/stack_bowls/global_view/pnp_fit.png) |
| stack_color_blocks | global_view | ur5 | 0.006822 | [View Plot](project2d/results/stack_color_blocks/global_view/project2d_fit.png) | 0.013320 | [View Plot](pnp/results/stack_color_blocks/global_view/pnp_fit.png) |
| stick_tape_to_box | global_view | aloha | 0.004974 | [View Plot](project2d/results/stick_tape_to_box/global_view/project2d_fit.png) | 0.008202 | [View Plot](pnp/results/stick_tape_to_box/global_view/pnp_fit.png) |
| sweep_the_rubbish | global_view | aloha | 0.009326 | [View Plot](project2d/results/sweep_the_rubbish/global_view/project2d_fit.png) | 0.009971 | [View Plot](pnp/results/sweep_the_rubbish/global_view/pnp_fit.png) |
| turn_on_faucet | global_view | aloha | 0.006290 | [View Plot](project2d/results/turn_on_faucet/global_view/project2d_fit.png) | 0.003837 | [View Plot](pnp/results/turn_on_faucet/global_view/pnp_fit.png) |
| turn_on_light_switch | global_view | arx5 | 0.004081 | [View Plot](project2d/results/turn_on_light_switch/global_view/project2d_fit.png) | 0.006065 | [View Plot](pnp/results/turn_on_light_switch/global_view/pnp_fit.png) |
| turn_on_light_switch | side_view | arx5 | 0.006027 | [View Plot](project2d/results/turn_on_light_switch/side_view/project2d_fit.png) | 0.006053 | [View Plot](pnp/results/turn_on_light_switch/side_view/pnp_fit.png) |
| water_potted_plant | global_view | arx5 | 0.005694 | [View Plot](project2d/results/water_potted_plant/global_view/project2d_fit.png) | 0.016435 | [View Plot](pnp/results/water_potted_plant/global_view/pnp_fit.png) |
| water_potted_plant | side_view | arx5 | 0.008216 | [View Plot](project2d/results/water_potted_plant/side_view/project2d_fit.png) | 0.015948 | [View Plot](pnp/results/water_potted_plant/side_view/pnp_fit.png) |
| wipe_the_table | global_view | arx5 | 0.007016 | [View Plot](project2d/results/wipe_the_table/global_view/project2d_fit.png) | 0.010697 | [View Plot](pnp/results/wipe_the_table/global_view/pnp_fit.png) |
| wipe_the_table | side_view | arx5 | 0.004354 | [View Plot](project2d/results/wipe_the_table/side_view/project2d_fit.png) | 0.009210 | [View Plot](pnp/results/wipe_the_table/side_view/pnp_fit.png) |

### Potential Sources of Error

造成标定误差的因素可能包括：

1.  **时间不同步**：视频帧与机器人状态日志可能无法做到完全同步，导致画面中机器人位置与记录状态存在偏差。
2.  **标注噪声**：人工标注关键点（例如夹爪指尖）不可避免会引入误差，从而影响模型拟合。
3.  **数据分布偏移**：Table30 数据集中许多任务存在严重遮挡，因此可标注的数据可能集中在机器人可见的某些区域或角落，进而产生分布偏移并影响对整个工作空间的泛化。
4.  **机械/安装不稳定**：机器人运动时可能出现抖动，或采集过程中相机略有位移，都会造成记录状态与真实物理配置不一致。
5.  **PnP 内参未知**：若相机内参未知且从同一批标注数据估计，得到的外参可能会引入额外偏差或方差。

为提升标定精度，建议**标注更多且覆盖更广的帧**，尽量覆盖不同的机器人位姿与工作空间位置。

## Troubleshooting

- `No tasks found for <task>/<view>. Run gen_label_tasks.py first.`：请确认 `DATA_ROOT` 配置正确，并已执行 `python3 gen_label_tasks.py`。
- UI 中图像为空/黑屏：请检查 `config.py` 中视频文件名映射（`VIDEO_CONFIG`）是否与你的数据集一致。
- 未生成 `processed_data.pkl`：请确认 `results.json` 中包含标注，且对应 episode 下存在 `states/*.jsonl`。
- `pip install av` 失败：PyAV 可能依赖系统的 FFmpeg 开发库；请通过系统包管理器安装后重试。
- 端口冲突：`serve.py` 默认使用端口 `5312`；如有需要可在 [serve.py] 中修改。

## Project Structure

```
.
├── config.py             # 全局配置（任务、机器人、视角）
├── gen_label_tasks.py    # 生成标注任务脚本
├── infer.py              # 推理脚本（主要用法入口）
├── main.py               # 标定流水线入口
├── serve.py              # Web 标注服务器
├── pnp/                  # PnP 方法实现
│   ├── extrinsics.yaml   # 标定后的 PnP 参数
│   └── intrinsics.yaml   # 相机内参
├── project2d/            # Project2D 方法实现
│   ├── parameters.yaml   # 标定后的 Project2D 参数
│   └── guide.md          # Project2D 详细指南
└── templates/            # 标注工具的 HTML 模板
```
