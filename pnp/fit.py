import cv2
import numpy as np
import yaml
import os
import pickle
import multiprocessing
import av
import json
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from project2d.fit import compute_features
from config import get_task_list, get_task_dir, get_robot_type, get_robot_name, get_arms_config, get_video_path, get_data_dir, get_view_name

# Global variables for multiprocessing
global_dataset = None
global_robot_type = None
global_arms_config = None
global_camera_matrix = None
global_dist_coeffs = None
global_width = None
global_height = None

def load_intrinsics(task_name, view_name):
    path = os.path.join(os.path.dirname(__file__), 'intrinsics.yaml')
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return None, None
        
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    if task_name not in data:
        # Try finding by robot type if task specific not found?
        # The yaml seems to be task specific (keys are task names)
        # But some tasks might share robot. 
        # For now, strictly follow yaml structure.
        print(f"Warning: Task {task_name} not found in intrinsics.yaml")
        return None, None
        
    task_data = data[task_name]
    if view_name not in task_data:
        print(f"Warning: View {view_name} not found for task {task_name} in intrinsics.yaml")
        return None, None
        
    cam_data = task_data[view_name]
    camera_matrix = np.array([
        [cam_data['fx'], 0, cam_data['cx']],
        [0, cam_data['fy'], cam_data['cy']],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.array(cam_data['coeffs'], dtype=np.float64)
    return camera_matrix, dist_coeffs

def save_intrinsics_yaml(task_name, view_name, camera_matrix, dist_coeffs, camera_name="Intel RealSense D435I"):
    path = os.path.join(os.path.dirname(__file__), 'intrinsics.yaml')
    if os.path.exists(path):
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    if task_name not in data:
        data[task_name] = {}

    robot_name = get_robot_name(task_name)
    if robot_name:
        data[task_name]['robot'] = robot_name

    if view_name not in data[task_name]:
        data[task_name][view_name] = {}

    data[task_name][view_name]['fx'] = float(camera_matrix[0, 0])
    data[task_name][view_name]['fy'] = float(camera_matrix[1, 1])
    data[task_name][view_name]['cx'] = float(camera_matrix[0, 2])
    data[task_name][view_name]['cy'] = float(camera_matrix[1, 2])
    data[task_name][view_name]['coeffs'] = [float(v) for v in dist_coeffs.flatten().tolist()]
    data[task_name][view_name]['name'] = camera_name

    sorted_data = dict(sorted(data.items()))
    with open(path, 'w') as f:
        yaml.dump(sorted_data, f, default_flow_style=None, sort_keys=False)

def build_camera_matrix(fx, fy, cx, cy):
    return np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

def get_image_resolution(task_name, view_name):
    # Find a video
    video_filename = get_video_path(task_name, view_name)
    if not video_filename:
        return None, None
        
    data_dir_base = get_data_dir(task_name)
    if not os.path.exists(data_dir_base):
        return None, None
        
    # Check first few episodes
    for entry in os.scandir(data_dir_base):
        if entry.is_dir() and entry.name.startswith('episode_'):
            video_path = os.path.join(entry.path, 'videos', video_filename)
            if os.path.exists(video_path):
                try:
                    container = av.open(video_path)
                    stream = container.streams.video[0]
                    width = stream.width
                    height = stream.height
                    container.close()
                    return width, height
                except Exception as e:
                    print(f"Error reading video {video_path}: {e}")
                    continue
    return None, None

def fit_pnp_model(dataset, robot_type, arms_config, camera_matrix, dist_coeffs, width, height):
    extrinsics = {}
    errors = []
    
    for arm in arms_config:
        Xs_3d = []
        Ys_2d = []
        
        # Collect Data
        for idx, kp in enumerate(arm['keypoints']):
            for item in dataset:
                if kp in item:
                    if arm['feature_key'] not in item: continue
                    feat = item[arm['feature_key']]
                    # feat has 6 elements: p1 (3), p2 (3)
                    if idx == 0:
                        xyzvec = feat[:3]
                    else:
                        xyzvec = feat[3:6]
                    Xs_3d.append(xyzvec)
                    # Denormalize points
                    u, v = item[kp]
                    Ys_2d.append([u * width, v * height])
        
        Xs_3d = np.array(Xs_3d, dtype=np.float64)
        Ys_2d = np.array(Ys_2d, dtype=np.float64)
        
        if len(Xs_3d) < 4:
            continue

        # Solve PnP
        try:
            success, rvec, tvec = cv2.solvePnP(Xs_3d, Ys_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        except Exception:
            success = False
            
        if not success:
            continue
            
        extrinsics[arm['name']] = {'rvec': rvec, 'tvec': tvec}
        
        # Calculate Error
        projected_points, _ = cv2.projectPoints(Xs_3d, rvec, tvec, camera_matrix, dist_coeffs)
        projected_points = projected_points.reshape(-1, 2)
        
        # Calculate normalized error for consistency with project2d
        # projected pixels -> normalized
        u_pred = projected_points[:, 0] / width
        v_pred = projected_points[:, 1] / height
        
        u_gt = Ys_2d[:, 0] / width
        v_gt = Ys_2d[:, 1] / height
        
        err = np.mean(np.sqrt((u_gt - u_pred)**2 + (v_gt - v_pred)**2))
        errors.append(err)
        
    if not errors: return float('inf'), extrinsics
    return np.mean(errors), extrinsics

def plot_results(dataset, extrinsics, arms_config, camera_matrix, dist_coeffs, width, height, output_img, output_json=None):
    """
    Generates and saves the visualization plot and error data for PnP.
    """
    plt.figure(figsize=(12, 8), dpi=300)
    plot_idx = 1
    total_plots = sum(len(arm['keypoints']) for arm in arms_config)
    
    error_stats = {}
    
    for arm in arms_config:
        print(f"Processing {arm['name']}...")
        
        if arm['name'] not in extrinsics:
            print(f"  No extrinsics found for {arm['name']}")
            continue
            
        rvec = extrinsics[arm['name']]['rvec']
        tvec = extrinsics[arm['name']]['tvec']
        
        for idx, kp in enumerate(arm['keypoints']):
            Xs_3d = []
            Ys_2d = []
            for item in dataset:
                if kp in item:
                    if arm['feature_key'] not in item: continue
                    feat = item[arm['feature_key']]
                    if idx == 0:
                        xyzvec = feat[:3]
                    else:
                        xyzvec = feat[3:6]
                    Xs_3d.append(xyzvec)
                    Ys_2d.append(item[kp]) # Normalized
            
            Xs_3d = np.array(Xs_3d, dtype=np.float64)
            Ys_2d = np.array(Ys_2d, dtype=np.float64)
            N = len(Xs_3d)
            
            if N == 0: continue

            # Predict
            projected_points, _ = cv2.projectPoints(Xs_3d, rvec, tvec, camera_matrix, dist_coeffs)
            projected_points = projected_points.reshape(-1, 2)
            
            # Normalize predicted points
            u_pred = projected_points[:, 0] / width
            v_pred = projected_points[:, 1] / height
            
            # Ys_2d is already normalized
            u_gt = Ys_2d[:, 0]
            v_gt = Ys_2d[:, 1]
            
            # Error
            err = np.sqrt((u_gt - u_pred)**2 + (v_gt - v_pred)**2)
            mean_err = np.mean(err)
            print(f"    {arm['names'][idx]} Error: {mean_err:.4f} (normalized)")
            
            error_stats[arm['names'][idx]] = float(mean_err)
            
            # Plot
            ax = plt.subplot(1, total_plots, plot_idx)
            plot_idx += 1
            ax.plot(u_gt, v_gt, 'r.', alpha=0.5, label='GT')
            for i in range(N):
                ax.plot([u_gt[i], u_pred[i]], [v_gt[i], v_pred[i]], 'g-', alpha=0.3)
            ax.plot(u_pred, v_pred, 'b.', alpha=0.5, label='Pred')
            ax.set_title(f"{arm['names'][idx]}\nErr={mean_err:.3f}")
            ax.set_xlim(0, 1)
            ax.set_ylim(1, 0)
            ax.legend()

    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Saved plot to {output_img}")
    plt.close()
    
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(error_stats, f, indent=2)
        print(f"Saved error stats to {output_json}")

def init_worker(dataset, robot_type, arms_config, camera_matrix, dist_coeffs, width, height):
    global global_dataset, global_robot_type, global_arms_config, global_camera_matrix, global_dist_coeffs, global_width, global_height
    global_dataset = dataset
    global_robot_type = robot_type
    global_arms_config = arms_config
    global_camera_matrix = camera_matrix
    global_dist_coeffs = dist_coeffs
    global_width = width
    global_height = height

def evaluate_delta_worker(dx, dy, dz):
    delta = np.array([dx, dy, dz])
    curr_dataset = compute_features(global_dataset, delta, global_robot_type)
    err, extrinsics = fit_pnp_model(curr_dataset, global_robot_type, global_arms_config, global_camera_matrix, global_dist_coeffs, global_width, global_height)
    return err, delta, extrinsics

def optimize_delta_parameters(raw_dataset, robot_type, arms_config, camera_matrix, dist_coeffs, width, height, search_profile=None):
    best_delta = np.array([0.0, 0.0, 0.0])
    min_err = float('inf')
    best_extrinsics = None

    if search_profile is None:
        search_profile = [
            {
                'name': "Stage 1: Coarse Search",
                'type': 'absolute',
                'x': (-0.1, 0.1, 21),
                'y': (-0.05, 0.05, 11),
                'z': (-1, 1, 201)
            },
            {'name': "Stage 2: Fine Search", 'type': 'relative', 'span': 0.02, 'num': 41},
            {'name': "Stage 3: Pro-Fine Search", 'type': 'relative', 'span': 0.002, 'num': 41},
            {'name': "Stage 4: Max-Fine Search", 'type': 'relative', 'span': 0.0002, 'num': 41},
            {'name': "Stage 5: Ultra-Fine Search", 'type': 'relative', 'span': 0.00002, 'num': 41}
        ]

    for profile in search_profile:
        if profile['type'] == 'absolute':
            x0, x1, xn = profile['x']
            y0, y1, yn = profile['y']
            z0, z1, zn = profile['z']
            deltas_x = np.linspace(x0, x1, xn)
            deltas_y = np.linspace(y0, y1, yn)
            deltas_z = np.linspace(z0, z1, zn)
        else:
            span = profile['span']
            num = profile['num']
            curr_x, curr_y, curr_z = best_delta
            deltas_x = np.linspace(curr_x - span, curr_x + span, num)
            deltas_y = np.linspace(curr_y - span, curr_y + span, num)
            deltas_z = np.linspace(curr_z - span, curr_z + span, num)

        tasks = [(dx, dy, dz) for dx in deltas_x for dy in deltas_y for dz in deltas_z]
        best_delta, min_err, best_extrinsics = execute_search_stage(profile['name'], tasks, raw_dataset, robot_type, arms_config, best_delta, min_err, best_extrinsics, camera_matrix, dist_coeffs, width, height)

    return best_delta, best_extrinsics, min_err

def golden_section_search(func, low, high, iterations=1000):
    phi = (np.sqrt(5) - 1) / 2
    c = high - phi * (high - low)
    d = low + phi * (high - low)
    fc = func(c)
    fd = func(d)

    for _ in range(iterations):
        if fc < fd:
            high = d
            d = c
            fd = fc
            c = high - phi * (high - low)
            fc = func(c)
        else:
            low = c
            c = d
            fc = fd
            d = low + phi * (high - low)
            fd = func(d)

    if fc < fd:
        return c, fc
    return d, fd

def estimate_intrinsics_and_extrinsics(raw_dataset, robot_type, arms_config, width, height, dist_coeffs=None):
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    max_dim = float(max(width, height))
    fx_bounds = (max(1.0, 0.3 * max_dim), 2.5 * max_dim)
    fy_bounds = (max(1.0, 0.3 * max_dim), 2.5 * max_dim)
    cx_bounds = (0.1 * width, 0.9 * width)
    cy_bounds = (0.1 * height, 0.9 * height)

    current = {
        'fx': 0.7 * max_dim,
        'fy': 0.7 * max_dim,
        'cx': width / 2.0,
        'cy': height / 2.0
    }

    quick_profile = [
        {
            'name': "Stage 1: Coarse Search",
            'type': 'absolute',
            'x': (-0.08, 0.08, 17),
            'y': (-0.04, 0.04, 9),
            'z': (-0.6, 0.6, 121)
        },
        {'name': "Stage 2: Fine Search", 'type': 'relative', 'span': 0.02, 'num': 21}
    ]

    def evaluate(params):
        camera_matrix = build_camera_matrix(params['fx'], params['fy'], params['cx'], params['cy'])
        delta, extrinsics, err = optimize_delta_parameters(
            raw_dataset,
            robot_type,
            arms_config,
            camera_matrix,
            dist_coeffs,
            width,
            height,
            search_profile=quick_profile
        )
        return err, delta, extrinsics, camera_matrix

    best_err, best_delta, best_extrinsics, best_camera_matrix = evaluate(current)

    for _ in range(2):
        for key, bounds in [('fx', fx_bounds), ('fy', fy_bounds), ('cx', cx_bounds), ('cy', cy_bounds)]:
            low, high = bounds
            def objective(val):
                trial = dict(current)
                trial[key] = val
                err, _, _, _ = evaluate(trial)
                return err

            best_val, _ = golden_section_search(objective, low, high, iterations=1000)
            current[key] = best_val
            curr_err, curr_delta, curr_extrinsics, curr_camera_matrix = evaluate(current)
            if curr_err < best_err:
                best_err = curr_err
                best_delta = curr_delta
                best_extrinsics = curr_extrinsics
                best_camera_matrix = curr_camera_matrix

    return best_camera_matrix, dist_coeffs, best_delta, best_extrinsics

def execute_search_stage(stage_name, tasks, raw_dataset, robot_type, arms_config, best_delta, min_err, best_extrinsics, camera_matrix, dist_coeffs, width, height):
    print(f"{stage_name}...")
    
    with multiprocessing.Pool(initializer=init_worker, initargs=(raw_dataset, robot_type, arms_config, camera_matrix, dist_coeffs, width, height)) as pool:
        results = pool.starmap(evaluate_delta_worker, tasks)
    
    for err, delta, extrinsics in results:
        if err < min_err:
            min_err = err
            best_delta = delta
            best_extrinsics = extrinsics
            
    print(f"{stage_name} Best Delta: {best_delta} (Err: {min_err:.6f})")
    return best_delta, min_err, best_extrinsics

def pnp_fit(view_name):
    task_list = get_task_list(view_name)
    print(f"\n{'='*50}")
    print(f"Starting PnP Fit for View: {view_name}")
    print(f"Found {len(task_list)} tasks")
    print(f"{'='*50}\n")

    for task_name in task_list:
        print(f"\n{'-'*30}")
        print(f"Processing Task: {task_name}")
        print(f"{'-'*30}")
        task_dir = get_task_dir(task_name, view_name)
        if not task_dir: continue
        
        input_pkl = os.path.join(task_dir, 'processed_data.pkl')
        if not os.path.exists(input_pkl):
            print(f"Skipping {task_name}: processed_data.pkl not found")
            continue
    
        robot_type = get_robot_type(task_name)
        arms_config = get_arms_config(task_name)
        
        camera_matrix, dist_coeffs = load_intrinsics(task_name, view_name)

        width, height = get_image_resolution(task_name, view_name)
        if width is None:
            continue
            
        print(f"Processing {task_name} (Resolution: {width}x{height})")
        
        with open(input_pkl, 'rb') as f:
            raw_dataset = pickle.load(f)
            
        if not raw_dataset:
            print("  Dataset empty.")
            continue
            
        output_fit = os.path.join('pnp', 'results', task_name, view_name, 'pnp_fit.pkl')
        output_img = os.path.join('pnp', 'results', task_name, view_name, 'pnp_fit.png')
        output_json = os.path.join('pnp', 'results', task_name, view_name, 'pnp_fit_error.json')
        os.makedirs(os.path.dirname(output_fit), exist_ok=True)
        
        if camera_matrix is None:
            print(f"Intrinsics missing for {task_name} {view_name}, estimating with golden search...")
            camera_matrix, dist_coeffs, _, _ = estimate_intrinsics_and_extrinsics(
                raw_dataset, robot_type, arms_config, width, height
            )
            save_intrinsics_yaml(task_name, view_name, camera_matrix, dist_coeffs)
            best_delta, best_extrinsics, _ = optimize_delta_parameters(
                raw_dataset, robot_type, arms_config, camera_matrix, dist_coeffs, width, height
            )
        else:
            best_delta, best_extrinsics, _ = optimize_delta_parameters(
                raw_dataset, robot_type, arms_config, camera_matrix, dist_coeffs, width, height
            )
    
        # Final visualization with best delta
        final_dataset = compute_features(raw_dataset, best_delta, robot_type)
        plot_results(final_dataset, best_extrinsics, arms_config, camera_matrix, dist_coeffs, width, height, output_img, output_json)
        
        # Save results
        save_data = {'extrinsics': best_extrinsics, 'delta': best_delta, 'width': width, 'height': height}
        with open(output_fit, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Saved PnP fit results to {output_fit}")

def main():
    view_names = get_view_name()
    for view_name in view_names:
        pnp_fit(view_name)

if __name__ == '__main__':
    main()
