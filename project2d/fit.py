import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
from scipy.spatial.transform import Rotation
import json
from config import get_task_list, get_task_dir, get_robot_type, get_arms_config, get_view_name

def euler_to_rotmat(rpy):
    return Rotation.from_euler('xyz', rpy).as_matrix()

def quaternion_to_rotmat(q):
    return Rotation.from_quat(q).as_matrix()

def compute_features(dataset, delta, robot_type):
    """
    Computes 3D feature points for each item in the dataset based on a given delta (dx, dy, dz).
    """
    new_dataset = []
    dx, dy, dz = delta[0], delta[1], delta[2]
    
    for item in dataset:
        if robot_type == 'aloha':
            new_item = item.copy()
            
            # Left Arm
            if 'left_pose' in item:
                l_pose = item['left_pose']
                l_xyz = np.array(l_pose[:3])
                l_rotmat = euler_to_rotmat(l_pose[3:])
                l_gripper = item['left_gripper_width']
                
                l_p1 = l_xyz + l_rotmat.dot(np.float32([dx, l_gripper/2 + dy, dz]))
                l_p2 = l_xyz + l_rotmat.dot(np.float32([dx, -l_gripper/2 - dy, dz]))
                
                feat1 = np.hstack([l_p1, l_p2])
                new_item['feature1'] = feat1
            
            # Right Arm
            if 'right_pose' in item:
                r_pose = item['right_pose']
                r_xyz = np.array(r_pose[:3])
                r_rotmat = euler_to_rotmat(r_pose[3:])
                r_gripper = item['right_gripper_width']
                
                r_p1 = r_xyz + r_rotmat.dot(np.float32([dx, r_gripper/2 + dy, dz]))
                r_p2 = r_xyz + r_rotmat.dot(np.float32([dx, -r_gripper/2 - dy, dz]))
                
                feat2 = np.hstack([r_p1, r_p2])
                new_item['feature2'] = feat2
            
            new_dataset.append(new_item)
  
        elif robot_type == 'arx5':
            pose = item['pose']
            gripper_width = item['gripper_width']
            xyz = np.array(pose[:3])
            rpy = np.array(pose[3:])
            rotmat = euler_to_rotmat(rpy)
            
            est_point1 = xyz + rotmat.dot(np.float32([dx, -gripper_width/2 - dy, dz]))
            est_point2 = xyz + rotmat.dot(np.float32([dx, gripper_width/2 + dy, dz]))
            
            feat = np.hstack([est_point1, est_point2])
            
            new_item = item.copy()
            new_item['feature'] = feat 
            new_dataset.append(new_item)

        elif robot_type == 'franka':
            pose = item['pose']
            gripper_width = item['gripper_width']
            xyz = np.array(pose[:3])
            rotmat = quaternion_to_rotmat(pose[3:])
            
            est_point1 = xyz + rotmat.dot(np.float32([dx, gripper_width/2 + dy, dz]))
            est_point2 = xyz + rotmat.dot(np.float32([dx, -gripper_width/2 - dy, dz]))
            
            feat = np.hstack([est_point1, est_point2])
            
            new_item = item.copy()
            new_item['feature'] = feat 
            new_dataset.append(new_item)
        
        elif robot_type == 'ur5':
            pose = item['pose']
            gripper_width = item['gripper_width']
            xyz = np.array(pose[:3])
            rotmat = quaternion_to_rotmat(pose[3:])
            
            est_point1 = xyz + rotmat.dot(np.float32([gripper_width/2 + dy, -dx, dz]))
            est_point2 = xyz + rotmat.dot(np.float32([-gripper_width/2 - dy, -dx, dz]))
            
            feat = np.hstack([est_point1, est_point2])
            
            new_item = item.copy()
            new_item['feature'] = feat 
            new_dataset.append(new_item)
    return new_dataset

def fit_model(dataset, robot_type, arms_config):
    """
    Fits the projection model for all arms and returns the regression vector and mean error.
    """
    # 3D points input -> 7 params per coordinate (c1*x + c2*y + c3*z + c4) / (c5*x + c6*y + c7*z + 1)
    if robot_type == 'aloha':
        regvec = np.zeros((7, 4))
    else:
        regvec = np.zeros((7, 2))

    errors = []
    
    for arm in arms_config:
        Xs_joint = []
        Ys_joint = []
        
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
                    Xs_joint.append(xyzvec)
                    Ys_joint.append(item[kp])
        
        Xs_joint = np.array(Xs_joint)
        Ys_joint = np.array(Ys_joint)
        
        if len(Xs_joint) == 0: continue

        # Fit U
        u = Ys_joint[:, 0]
        # A_u: [X, Y, Z, -X*u, -Y*u, -Z*u, 1]
        A_u = np.hstack([Xs_joint, -Xs_joint * u[:, None], np.ones((len(Xs_joint), 1))])
        params_u, _, _, _ = np.linalg.lstsq(A_u, u, rcond=None)
        
        # Fit V
        v = Ys_joint[:, 1]
        # A_v: [X, Y, Z, -X*v, -Y*v, -Z*v, 1]
        A_v = np.hstack([Xs_joint, -Xs_joint * v[:, None], np.ones((len(Xs_joint), 1))])
        params_v, _, _, _ = np.linalg.lstsq(A_v, v, rcond=None)
        
        # Store params
        regvec[:, arm['reg_cols'][0]] = params_u
        regvec[:, arm['reg_cols'][1]] = params_v
        
        # Calculate Error for this arm
        # u = (p0*x + p1*y + p2*z + p6) / (1 + p3*x + p4*y + p5*z)
        u_pred = (Xs_joint @ params_u[:3] + params_u[6]) / (1 + Xs_joint @ params_u[3:6])
        v_pred = (Xs_joint @ params_v[:3] + params_v[6]) / (1 + Xs_joint @ params_v[3:6])
        
        err = np.mean(np.sqrt((Ys_joint[:, 0] - u_pred)**2 + (Ys_joint[:, 1] - v_pred)**2))
        errors.append(err)
        
    if not errors: return float('inf'), regvec
    return np.mean(errors), regvec

def plot_results(dataset, regvec, arms_config, output_img, output_json=None):
    """
    Generates and saves the visualization plot and error data.
    """
    plt.figure(figsize=(12, 8), dpi=300)
    plot_idx = 1
    total_plots = sum(len(arm['keypoints']) for arm in arms_config)
    
    error_stats = {}

    for arm in arms_config:
        print(f"Processing {arm['name']}...")
        
        params_u = regvec[:, arm['reg_cols'][0]]
        params_v = regvec[:, arm['reg_cols'][1]]
        
        for idx, kp in enumerate(arm['keypoints']):
            Xs = []
            Ys = []
            for item in dataset:
                if kp in item:
                    if arm['feature_key'] not in item: continue
                    feat = item[arm['feature_key']]
                    if idx == 0:
                        xyzvec = feat[:3]
                    else:
                        xyzvec = feat[3:6]
                    Xs.append(xyzvec)
                    Ys.append(item[kp])
            
            Xs = np.array(Xs)
            Ys = np.array(Ys)
            N = len(Xs)
            
            if N == 0: continue

            # Predict
            u_pred = (Xs @ params_u[:3] + params_u[6]) / (1 + Xs @ params_u[3:6])
            v_pred = (Xs @ params_v[:3] + params_v[6]) / (1 + Xs @ params_v[3:6])
            
            # Error
            err = np.sqrt((Ys[:, 0] - u_pred)**2 + (Ys[:, 1] - v_pred)**2)
            mean_err = np.mean(err)
            print(f"    {arm['names'][idx]} Error: {mean_err:.4f} (normalized)")
            
            error_stats[arm['names'][idx]] = float(mean_err)

            # Plot
            ax = plt.subplot(1, total_plots, plot_idx)
            plot_idx += 1
            ax.plot(Ys[:, 0], Ys[:, 1], 'r.', alpha=0.5, label='GT')
            for i in range(N):
                ax.plot([Ys[i, 0], u_pred[i]], [Ys[i, 1], v_pred[i]], 'g-', alpha=0.3)
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

def init_worker(dataset, robot_type, arms_config):
    global global_dataset, global_robot_type, global_arms_config
    global_dataset = dataset
    global_robot_type = robot_type
    global_arms_config = arms_config

def evaluate_delta_worker(dx, dy, dz):
    delta = np.array([dx, dy, dz])
    curr_dataset = compute_features(global_dataset, delta, global_robot_type)
    err, regvec = fit_model(curr_dataset, global_robot_type, global_arms_config)
    return err, delta, regvec

def execute_search_stage(stage_name, tasks, raw_dataset, robot_type, arms_config, best_delta, min_err, best_regvec):
    print(f"{stage_name}...")
    print(f"Refining with {len(tasks)} tasks...")
    
    with multiprocessing.Pool(initializer=init_worker, initargs=(raw_dataset, robot_type, arms_config)) as pool:
        results = pool.starmap(evaluate_delta_worker, tasks)
    
    for err, delta, regvec in results:
        if err < min_err:
            min_err = err
            best_delta = delta
            best_regvec = regvec
            
    print(f"{stage_name} Best: {best_delta} (Err: {min_err:.6f})")
    return best_delta, min_err, best_regvec

def optimize_delta_parameters(raw_dataset, robot_type, arms_config):
    best_delta = np.array([0.0, 0.0, 0.0])
    min_err = float('inf')
    best_regvec = None

    # Stage 1: Coarse Search
    deltas_x = np.linspace(-0.1, 0.1, 21)  # step 0.01
    deltas_y = np.linspace(-0.05, 0.05, 11)  # step 0.01
    deltas_z = np.linspace(-1, 1, 201)       # step 0.01
    tasks = [(dx, dy, dz) for dx in deltas_x for dy in deltas_y for dz in deltas_z]
    
    best_delta, min_err, best_regvec = execute_search_stage("Stage 1: Coarse Search", tasks, raw_dataset, robot_type, arms_config, best_delta, min_err, best_regvec)

    # Stages 2-5
    search_configs = [
        ("Stage 2: Fine Search", 0.02, 41),
        ("Stage 3: Pro-Fine Search", 0.002, 41),
        ("Stage 4: Max-Fine Search", 0.0002, 41),
        ("Stage 5: Ultra-Fine Search", 0.00002, 41)
    ]

    for name, span, num in search_configs:
        curr_x, curr_y, curr_z = best_delta
        deltas_x = np.linspace(curr_x - span, curr_x + span, num)
        deltas_y = np.linspace(curr_y - span, curr_y + span, num)
        deltas_z = np.linspace(curr_z - span, curr_z + span, num)
        tasks = [(dx, dy, dz) for dx in deltas_x for dy in deltas_y for dz in deltas_z]
        best_delta, min_err, best_regvec = execute_search_stage(name, tasks, raw_dataset, robot_type, arms_config, best_delta, min_err, best_regvec)

    return best_delta, best_regvec

def project2d_fit(view_name):
    task_list = get_task_list(view_name)
    print(f"\n{'='*50}")
    print(f"Starting Project2D Fit for View: {view_name}")
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
            continue
    
        robot_type = get_robot_type(task_name)
        arms_config = get_arms_config(task_name)
        
        with open(input_pkl, 'rb') as f:
            raw_dataset = pickle.load(f)
            
        if not raw_dataset:
            print("  Dataset empty.")
            continue
            
        output_fit = os.path.join('project2d', 'results', task_name, view_name, 'project2d_fit.pkl')
        output_img = os.path.join('project2d', 'results', task_name, view_name, 'project2d_fit.png')
        output_json = os.path.join('project2d', 'results', task_name, view_name, 'project2d_fit_error.json')
        os.makedirs(os.path.dirname(output_img), exist_ok=True)
        
        # Optimization Loop
        best_delta, best_regvec = optimize_delta_parameters(raw_dataset, robot_type, arms_config)
    
        # Final visualization with best delta
        final_dataset = compute_features(raw_dataset, best_delta, robot_type)
        plot_results(final_dataset, best_regvec, arms_config, output_img, output_json)
        
        # Save results
        save_data = {'regvec': best_regvec, 'delta': best_delta}
        with open(output_fit, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Saved fit results to {output_fit}")

def main():
    view_names = get_view_name()
    for view_name in view_names:
        project2d_fit(view_name)

if __name__ == '__main__':
    main()
