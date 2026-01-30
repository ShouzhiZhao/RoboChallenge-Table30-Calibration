import av
import cv2
import numpy as np
import pickle
import json
import os
from project2d.fit import quaternion_to_rotmat, euler_to_rotmat
from pnp.fit import load_intrinsics
from config import (
    get_task_list, 
    get_task_dir, 
    get_robot_type, 
    get_video_path,
    get_arms_config,
    get_data_dir,
    get_view_name,
)

def extract_endpoints_pnp(ee, gripper, extrinsics, delta, robot_type, arm_name, camera_matrix, dist_coeffs):
    xyz = ee[:3]
    dx, dy, dz = delta[0], delta[1], delta[2]
    
    if robot_type == 'arx5':
        rotmat = euler_to_rotmat(ee[3:])
        est_point1 = xyz + rotmat.dot(np.float32([dx, -gripper/2 - dy, dz]))
        est_point2 = xyz + rotmat.dot(np.float32([dx, gripper/2 + dy, dz]))
    elif robot_type == 'aloha':
        rotmat = euler_to_rotmat(ee[3:])
        est_point1 = xyz + rotmat.dot(np.float32([dx, gripper/2 + dy, dz]))
        est_point2 = xyz + rotmat.dot(np.float32([dx, -gripper/2 - dy, dz]))
    elif robot_type == 'franka':
        rotmat = quaternion_to_rotmat(ee[3:])
        est_point1 = xyz + rotmat.dot(np.float32([dx, gripper/2 + dy, dz]))
        est_point2 = xyz + rotmat.dot(np.float32([dx, -gripper/2 - dy, dz]))
    elif robot_type == 'ur5':
        rotmat = quaternion_to_rotmat(ee[3:])
        est_point1 = xyz + rotmat.dot(np.float32([gripper/2 + dy, -dx, dz]))
        est_point2 = xyz + rotmat.dot(np.float32([-gripper/2 - dy, -dx, dz]))

    # Project
    if arm_name not in extrinsics:
        return np.zeros((2, 2))
        
    rvec = extrinsics[arm_name]['rvec']
    tvec = extrinsics[arm_name]['tvec']
    
    points_3d = np.array([est_point1, est_point2], dtype=np.float64)
    projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    
    return projected_points

def pnp_view(view_name):
    task_list = get_task_list(view_name)
    print(f"\n{'='*50}")
    print(f"Starting PnP View Visualization for View: {view_name}")
    print(f"Found {len(task_list)} tasks")
    print(f"{'='*50}\n")
    
    for task_name in task_list:
        print(f"\n{'-'*30}")
        print(f"Processing Task: {task_name}")
        print(f"{'-'*30}")
        task_dir = get_task_dir(task_name, view_name)
        if not task_dir: continue
        
        fit_pkl_path = os.path.join('pnp', 'results', task_name, view_name, 'pnp_fit.pkl')
        if not os.path.exists(fit_pkl_path):
            continue
            
        with open(fit_pkl_path, 'rb') as f:
            data = pickle.load(f)
            extrinsics = data.get('extrinsics')
            delta = data.get('delta')
            
        robot_type = get_robot_type(task_name)
        arms_config = get_arms_config(task_name)
        
        # Load intrinsics
        camera_matrix, dist_coeffs = load_intrinsics(task_name, view_name)
        if camera_matrix is None:
            print(f"Intrinsics not found: {task_name}")
            continue

        # Find one episode to visualize
        tasks_json = os.path.join(task_dir, 'tasks.json')
        found_episode = None
        if os.path.exists(tasks_json):
            with open(tasks_json, 'r') as f:
                tasks = json.load(f)
                if tasks:
                    found_episode = tasks[0]['path']
        
        if not found_episode:
            data_dir_base = get_data_dir(task_name)
            if os.path.exists(data_dir_base):
                for ep in os.listdir(data_dir_base):
                    if ep.startswith('episode_'):
                        found_episode = ep
                        break
        
        if not found_episode:
            continue
            
        print(f"Visualizing Episode: {found_episode} from Task: {task_name}")
        
        episode_dir = os.path.join(get_data_dir(task_name), found_episode)
        video_path = os.path.join(episode_dir, 'videos', get_video_path(task_name, view_name))
        
        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
            
        # Read states
        states = []
        try:
            if robot_type == 'aloha':
                left_path = os.path.join(episode_dir, 'states', 'left_states.jsonl')
                right_path = os.path.join(episode_dir, 'states', 'right_states.jsonl')
                if os.path.exists(left_path) and os.path.exists(right_path):
                    with open(left_path, 'r') as f_l, open(right_path, 'r') as f_r:
                        for l_line, r_line in zip(f_l, f_r):
                            l_st = json.loads(l_line)
                            r_st = json.loads(r_line)
                            states.append({'left': l_st, 'right': r_st})
            else:
                state_path = os.path.join(episode_dir, 'states', 'states.jsonl')
                if os.path.exists(state_path):
                    with open(state_path, 'r') as f:
                        for line in f:
                            states.append(json.loads(line))
        except Exception as e:
            print(f"Error reading states: {e}")
            continue
            
        if not states:
            print(f"No states loaded for {found_episode}")
            continue

        # Visualize
        output_video_path = os.path.join('pnp', 'results', task_name, view_name, 'pnp_view.mp4')
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        
        container = av.open(video_path)
        stream = container.streams.video[0]
        width = stream.width
        height = stream.height
        
        output_container = av.open(output_video_path, mode='w')
        out_stream = output_container.add_stream('h264', rate=stream.average_rate)
        out_stream.width = width
        out_stream.height = height
        out_stream.pix_fmt = 'yuv420p'
        
        for idx, frame in enumerate(container.decode(video=0)):
            if idx >= len(states): break
            
            img = frame.to_rgb().to_ndarray()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            state = states[idx]
            colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]

            # Prediction
            if robot_type == 'aloha':
                arm1_cfg = arms_config[0]
                preds1 = extract_endpoints_pnp(np.array(state['left']['ee_pose_rpy']), state['left']['gripper'], extrinsics, delta, robot_type, arm1_cfg['name'], camera_matrix, dist_coeffs)
                arm2_cfg = arms_config[1]
                preds2 = extract_endpoints_pnp(np.array(state['right']['ee_pose_rpy']), state['right']['gripper'], extrinsics, delta, robot_type, arm2_cfg['name'], camera_matrix, dist_coeffs)
                
                # Draw
                cv2.circle(img, (int(preds1[0][0]), int(preds1[0][1])), 5, colors[0], -1)
                cv2.circle(img, (int(preds1[1][0]), int(preds1[1][1])), 5, colors[1], -1)
                cv2.circle(img, (int(preds2[0][0]), int(preds2[0][1])), 5, colors[2], -1)
                cv2.circle(img, (int(preds2[1][0]), int(preds2[1][1])), 5, colors[3], -1)

            elif robot_type in ['arx5', 'franka', 'ur5']:
                arm_cfg = arms_config[0]
                if robot_type == 'arx5':
                    pose = np.array(state['end_effector_pose'])
                    gripper = state['gripper_width']
                elif robot_type == 'franka':
                    pose = np.array(state['ee_positions'])
                    gripper = state['gripper_width'][0]
                else: # ur5
                    pose = np.array(state['ee_positions'])
                    gripper = (255 - state['gripper']) / 255.0 * 0.085
                    
                preds = extract_endpoints_pnp(pose, gripper, extrinsics, delta, robot_type, arm_cfg['name'], camera_matrix, dist_coeffs)
                
                cv2.circle(img, (int(preds[0][0]), int(preds[0][1])), 5, colors[0], -1)
                cv2.circle(img, (int(preds[1][0]), int(preds[1][1])), 5, colors[1], -1)
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out_frame = av.VideoFrame.from_ndarray(img_rgb, format='rgb24')
            for packet in out_stream.encode(out_frame):
                output_container.mux(packet)
            
        for packet in out_stream.encode():
            output_container.mux(packet)
        output_container.close()
        print(f"Saved video to {output_video_path}")

def main():
    view_names = get_view_name()
    for view_name in view_names:
        pnp_view(view_name)

if __name__ == '__main__':
    main()