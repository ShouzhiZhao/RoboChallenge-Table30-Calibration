# Project2d Parameters Usage Guide

This guide explains how to use the `project2d/parameters.yaml` file to project 3D points from the robot frame to 2D image coordinates.

## 1. Mathematical Model

The projection from a 3D point $(x, y, z)$ to a 2D pixel coordinate $(u, v)$ is modeled using a rational function:

$$
u = \frac{p_0 x + p_1 y + p_2 z + p_6}{1 + p_3 x + p_4 y + p_5 z}
$$

$$
v = \frac{q_0 x + q_1 y + q_2 z + q_6}{1 + q_3 x + q_4 y + q_5 z}
$$

Where:
- $(x, y, z)$ is the 3D feature point in the robot base frame (after applying `delta` adjustments).
- $(u, v)$ are the normalized pixel coordinates (0 to 1).
- $p_0, \dots, p_6$ are the parameters for the $u$ coordinate.
- $q_0, \dots, q_6$ are the parameters for the $v$ coordinate.

## 2. YAML Structure

The `project2d/parameters.yaml` file contains the parameters for each task and view.

```yaml
task_name:
  robot: robot_name
  view_name:
    regvec:
      - [row0_col0, row0_col1, ...]
      - ...
      - [row6_col0, row6_col1, ...]
    delta: [dx, dy, dz]
```

- **`regvec`**: A $7 \times N$ matrix containing the projection parameters.
  - Rows 0-6 correspond to indices 0-6 of the parameters ($p_i$ or $q_i$).
  - **Single-arm robots (e.g., ARX5, UR5, Franka)**: `regvec` is $7 \times 2$.
    - Column 0: Parameters for $u$ ($p_0 \dots p_6$).
    - Column 1: Parameters for $v$ ($q_0 \dots q_6$).
  - **Dual-arm robots (e.g., Aloha)**: `regvec` is $7 \times 4$.
    - Column 0: Left Arm $u$.
    - Column 1: Left Arm $v$.
    - Column 2: Right Arm $u$.
    - Column 3: Right Arm $v$.

- **`delta`**: A list `[dx, dy, dz]` used to adjust the gripper position to the actual feature point (e.g., finger tip).

## 3. Coordinate Transformation & Delta Application

Before applying the projection model, the robot's end-effector pose must be converted to the specific 3D feature point using `delta`. The transformation depends on the robot type.

**Notation:**
- $\mathbf{p}_{ee} = [x, y, z]^T$: Robot end-effector position.
- $\mathbf{R}$: Rotation matrix derived from robot pose.
- $w$: Gripper width.
- $\mathbf{\delta} = [dx, dy, dz]^T$: Delta parameters.
- $\mathbf{P}$: The resulting 3D feature point.

The general transformation formula is:

$$
\mathbf{P} = \mathbf{p}_{ee} + \mathbf{R} \cdot \mathbf{offset}
$$

The $\mathbf{offset}$ vector varies by robot type:

### Aloha (Dual Arm)
**Left Arm:**
$$
\mathbf{P}_{left} = \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ w/2 + dy \\ dz \end{bmatrix}, \quad
\mathbf{P}_{right} = \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ -w/2 - dy \\ dz \end{bmatrix}
$$

**Right Arm:**
$$
\mathbf{P}_{left} = \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ w/2 + dy \\ dz \end{bmatrix}, \quad
\mathbf{P}_{right} = \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ -w/2 - dy \\ dz \end{bmatrix}
$$

### ARX5
$$
\mathbf{P}_{1} = \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ -w/2 - dy \\ dz \end{bmatrix}, \quad
\mathbf{P}_{2} = \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ w/2 + dy \\ dz \end{bmatrix}
$$

### Franka
$$
\mathbf{P}_{1} = \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ w/2 + dy \\ dz \end{bmatrix}, \quad
\mathbf{P}_{2} = \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} dx \\ -w/2 - dy \\ dz \end{bmatrix}
$$

### UR5
*(Note the axis swap)*
$$
\mathbf{P}_{1} = \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} w/2 + dy \\ -dx \\ dz \end{bmatrix}, \quad
\mathbf{P}_{2} = \mathbf{p}_{ee} + \mathbf{R} \cdot \begin{bmatrix} -w/2 - dy \\ -dx \\ dz \end{bmatrix}
$$

## 4. Usage Example (Python)

```python
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

def project_point(xyz_point, params_u, params_v):
    """
    Projects a 3D point to 2D using the rational function model.
    """
    x, y, z = xyz_point
    
    # u = (p0*x + p1*y + p2*z + p6) / (1 + p3*x + p4*y + p5*z)
    num_u = params_u[0]*x + params_u[1]*y + params_u[2]*z + params_u[6]
    den_u = 1 + params_u[3]*x + params_u[4]*y + params_u[5]*z
    u = num_u / den_u
    
    # v = (q0*x + q1*y + q2*z + q6) / (1 + q3*x + q4*y + q5*z)
    num_v = params_v[0]*x + params_v[1]*y + params_v[2]*z + params_v[6]
    den_v = 1 + params_v[3]*x + params_v[4]*y + params_v[5]*z
    v = num_v / den_v
    
    return u, v

def load_extrinsics(yaml_path, task_name, view_name):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    task_data = data.get(task_name)
    if not task_data:
        raise ValueError(f"Task {task_name} not found")
        
    view_data = task_data.get(view_name)
    if not view_data:
        raise ValueError(f"View {view_name} not found for task {task_name}")
        
    return np.array(view_data['regvec']), np.array(view_data['delta'])

# Example Usage
if __name__ == "__main__":
    # Load parameters
    regvec, delta = load_extrinsics('extrinsics.yaml', 'arrange_flowers', 'global_view')
    
    # Assume we have robot state
    # Example for ARX5
    robot_pos = np.array([0.3, -0.1, 0.2])  # x, y, z
    robot_rpy = np.array([0.1, 0.0, 1.5])   # roll, pitch, yaw
    gripper_width = 0.05
    
    rotmat = Rotation.from_euler('xyz', robot_rpy).as_matrix()
    dx, dy, dz = delta
    
    # Compute feature point (Point 1 for ARX5)
    # P = xyz + rotmat * [dx, -gripper/2 - dy, dz]
    offset = np.array([dx, -gripper_width/2 - dy, dz])
    point_3d = robot_pos + rotmat.dot(offset)
    
    # Get parameters for u and v (Column 0 and 1 for single arm)
    params_u = regvec[:, 0]
    params_v = regvec[:, 1]
    
    u, v = project_point(point_3d, params_u, params_v)
    
    print(f"Projected coordinates (normalized): u={u:.4f}, v={v:.4f}")
    # To get pixel coordinates, multiply by image width and height
    # px = u * image_width
    # py = v * image_height
```

## Appendix: Extrinsics Evaluation Report
| Task Name | View | Robot | Mean Reprojection Error (Normalized) | Plot |
| :--- | :--- | :--- | :--- | :--- |
| arrange_flowers | global_view | arx5 | 0.011274 | [View Plot](extrinsics/arrange_flowers/global_view/project2d_fit.png) |
| arrange_flowers | side_view | arx5 | 0.009229 | [View Plot](extrinsics/arrange_flowers/side_view/project2d_fit.png) |
| arrange_fruits_in_basket | global_view | ur5 | 0.006856 | [View Plot](extrinsics/arrange_fruits_in_basket/global_view/project2d_fit.png) |
| arrange_paper_cups | global_view | arx5 | 0.007222 | [View Plot](extrinsics/arrange_paper_cups/global_view/project2d_fit.png) |
| arrange_paper_cups | side_view | arx5 | 0.004157 | [View Plot](extrinsics/arrange_paper_cups/side_view/project2d_fit.png) |
| clean_dining_table | global_view | aloha | 0.014808 | [View Plot](extrinsics/clean_dining_table/global_view/project2d_fit.png) |
| fold_dishcloth | global_view | arx5 | 0.004898 | [View Plot](extrinsics/fold_dishcloth/global_view/project2d_fit.png) |
| fold_dishcloth | side_view | arx5 | 0.005601 | [View Plot](extrinsics/fold_dishcloth/side_view/project2d_fit.png) |
| hang_toothbrush_cup | global_view | ur5 | 0.006777 | [View Plot](extrinsics/hang_toothbrush_cup/global_view/project2d_fit.png) |
| make_vegetarian_sandwich | global_view | aloha | 0.012185 | [View Plot](extrinsics/make_vegetarian_sandwich/global_view/project2d_fit.png) |
| move_objects_into_box | global_view | franka | 0.016472 | [View Plot](extrinsics/move_objects_into_box/global_view/project2d_fit.png) |
| move_objects_into_box | side_view | franka | 0.016452 | [View Plot](extrinsics/move_objects_into_box/side_view/project2d_fit.png) |
| open_the_drawer | global_view | arx5 | 0.006765 | [View Plot](extrinsics/open_the_drawer/global_view/project2d_fit.png) |
| open_the_drawer | side_view | arx5 | 0.004772 | [View Plot](extrinsics/open_the_drawer/side_view/project2d_fit.png) |
| place_shoes_on_rack | global_view | arx5 | 0.005776 | [View Plot](extrinsics/place_shoes_on_rack/global_view/project2d_fit.png) |
| place_shoes_on_rack | side_view | arx5 | 0.003354 | [View Plot](extrinsics/place_shoes_on_rack/side_view/project2d_fit.png) |
| plug_in_network_cable | global_view | aloha | 0.003502 | [View Plot](extrinsics/plug_in_network_cable/global_view/project2d_fit.png) |
| pour_fries_into_plate | global_view | aloha | 0.004096 | [View Plot](extrinsics/pour_fries_into_plate/global_view/project2d_fit.png) |
| press_three_buttons | global_view | franka | 0.010446 | [View Plot](extrinsics/press_three_buttons/global_view/project2d_fit.png) |
| press_three_buttons | side_view | franka | 0.008143 | [View Plot](extrinsics/press_three_buttons/side_view/project2d_fit.png) |
| put_cup_on_coaster | global_view | arx5 | 0.005664 | [View Plot](extrinsics/put_cup_on_coaster/global_view/project2d_fit.png) |
| put_cup_on_coaster | side_view | arx5 | 0.003307 | [View Plot](extrinsics/put_cup_on_coaster/side_view/project2d_fit.png) |
| put_opener_in_drawer | global_view | aloha | 0.006352 | [View Plot](extrinsics/put_opener_in_drawer/global_view/project2d_fit.png) |
| put_pen_into_pencil_case | global_view | aloha | 0.004148 | [View Plot](extrinsics/put_pen_into_pencil_case/global_view/project2d_fit.png) |
| scan_QR_code | global_view | aloha | 0.008651 | [View Plot](extrinsics/scan_QR_code/global_view/project2d_fit.png) |
| search_green_boxes | global_view | arx5 | 0.021506 | [View Plot](extrinsics/search_green_boxes/global_view/project2d_fit.png) |
| search_green_boxes | side_view | arx5 | 0.004553 | [View Plot](extrinsics/search_green_boxes/side_view/project2d_fit.png) |
| set_the_plates | global_view | ur5 | 0.006966 | [View Plot](extrinsics/set_the_plates/global_view/project2d_fit.png) |
| shred_scrap_paper | global_view | ur5 | 0.005149 | [View Plot](extrinsics/shred_scrap_paper/global_view/project2d_fit.png) |
| sort_books | global_view | ur5 | 0.008996 | [View Plot](extrinsics/sort_books/global_view/project2d_fit.png) |
| sort_electronic_products | global_view | arx5 | 0.009639 | [View Plot](extrinsics/sort_electronic_products/global_view/project2d_fit.png) |
| sort_electronic_products | side_view | arx5 | 0.023651 | [View Plot](extrinsics/sort_electronic_products/side_view/project2d_fit.png) |
| stack_bowls | global_view | aloha | 0.006792 | [View Plot](extrinsics/stack_bowls/global_view/project2d_fit.png) |
| stack_color_blocks | global_view | ur5 | 0.006822 | [View Plot](extrinsics/stack_color_blocks/global_view/project2d_fit.png) |
| stick_tape_to_box | global_view | aloha | 0.004974 | [View Plot](extrinsics/stick_tape_to_box/global_view/project2d_fit.png) |
| sweep_the_rubbish | global_view | aloha | 0.009326 | [View Plot](extrinsics/sweep_the_rubbish/global_view/project2d_fit.png) |
| turn_on_faucet | global_view | aloha | 0.006290 | [View Plot](extrinsics/turn_on_faucet/global_view/project2d_fit.png) |
| turn_on_light_switch | global_view | arx5 | 0.004081 | [View Plot](extrinsics/turn_on_light_switch/global_view/project2d_fit.png) |
| turn_on_light_switch | side_view | arx5 | 0.006027 | [View Plot](extrinsics/turn_on_light_switch/side_view/project2d_fit.png) |
| water_potted_plant | global_view | arx5 | 0.005694 | [View Plot](extrinsics/water_potted_plant/global_view/project2d_fit.png) |
| water_potted_plant | side_view | arx5 | 0.008216 | [View Plot](extrinsics/water_potted_plant/side_view/project2d_fit.png) |
| wipe_the_table | global_view | arx5 | 0.007016 | [View Plot](extrinsics/wipe_the_table/global_view/project2d_fit.png) |
| wipe_the_table | side_view | arx5 | 0.004354 | [View Plot](extrinsics/wipe_the_table/side_view/project2d_fit.png) |
