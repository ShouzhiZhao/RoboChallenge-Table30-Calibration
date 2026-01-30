import av
import cv2
import numpy as np
import pickle
import json
import os
from .fit import quaternion_to_rotmat, euler_to_rotmat
from config import (
    get_task_list, 
    get_task_dir, 
    get_robot_type, 
    get_video_path,
    get_arms_config,
    get_data_dir,
    get_view_name,
)

def extract_endpoints(ee, gripper, reg, delta, robot_type, reg_cols):
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

    col_u = reg_cols[0]
    col_v = reg_cols[1]

    # u = (p0*x + p1*y + p2*z + p6) / (1 + p3*x + p4*y + p5*z)
    
    # est_point1
    u1 = (est_point1.dot(reg[:3, col_u]) + reg[6, col_u]) / (1 + est_point1.dot(reg[3:6, col_u]))
    v1 = (est_point1.dot(reg[:3, col_v]) + reg[6, col_v]) / (1 + est_point1.dot(reg[3:6, col_v]))
    
    # est_point2
    u2 = (est_point2.dot(reg[:3, col_u]) + reg[6, col_u]) / (1 + est_point2.dot(reg[3:6, col_u]))
    v2 = (est_point2.dot(reg[:3, col_v]) + reg[6, col_v]) / (1 + est_point2.dot(reg[3:6, col_v]))

    preds = np.float32([
        [u1, v1],
        [u2, v2],
    ])
    return preds

def project2d_view(view_name):
    task_list = get_task_list(view_name)
    print(f"\n{'='*50}")
    print(f"Starting Project2D View Visualization for View: {view_name}")
    print(f"Found {len(task_list)} tasks")
    print(f"{'='*50}\n")
    
    for task_name in task_list:
        print(f"\n{'-'*30}")
        print(f"Processing Task: {task_name}")
        print(f"{'-'*30}")
        task_dir = get_task_dir(task_name, view_name)
        if not task_dir: continue
        
        fit_pkl_path = os.path.join('project2d', 'results', task_name, view_name, 'project2d_fit.pkl')
        if not os.path.exists(fit_pkl_path):
            continue
            
        with open(fit_pkl_path, 'rb') as f:
            data = pickle.load(f)
            models = (data.get('regvec'), data.get('delta'))
        
        regvec, delta = models
        
        # Find one episode to visualize
        tasks_json = os.path.join(task_dir, 'tasks.json')
        
        found_episode = None
        if os.path.exists(tasks_json):
            with open(tasks_json, 'r') as f:
                tasks = json.load(f)
                if tasks:
                    found_episode = tasks[0]['path']
        
        if not found_episode:
            # Fallback: check data dir
            data_dir_base = get_data_dir(task_name, view_name)
            if os.path.exists(data_dir_base):
                for ep in os.listdir(data_dir_base):
                    if ep.startswith('episode_'):
                        found_episode = ep
                        break
        
        if not found_episode:
            continue
            
        print(f"Visualizing Episode: {found_episode} from Task: {task_name}")
        
        episode_dir = os.path.join(get_data_dir(task_name), found_episode)
        video_filename = get_video_path(task_name, view_name)
        video_file = os.path.join(episode_dir, 'videos', video_filename)
        
        if not os.path.exists(video_file):
            print(f"Video not found: {video_file}")
            continue
            
        container = av.open(video_file)
        robot_type = get_robot_type(task_name)
        arms_config = get_arms_config(task_name)
        
        # Read states
        states = []
        try:
            if robot_type == 'aloha':
                left_states_file = os.path.join(episode_dir, 'states', 'left_states.jsonl')
                right_states_file = os.path.join(episode_dir, 'states', 'right_states.jsonl')
                if os.path.exists(left_states_file) and os.path.exists(right_states_file):
                    with open(left_states_file, 'r') as f:
                        left_lines = f.readlines()
                    with open(right_states_file, 'r') as f:
                        right_lines = f.readlines()
                    for l, r in zip(left_lines, right_lines):
                        states.append({'left': json.loads(l), 'right': json.loads(r)})
            elif robot_type in ['arx5', 'franka', 'ur5']:
                 states_file = os.path.join(episode_dir, 'states', 'states.jsonl')
                 if os.path.exists(states_file):
                     with open(states_file, 'r') as f:
                         for line in f:
                             states.append(json.loads(line))
        except Exception as e:
            print(f"Error reading states: {e}")
            continue
            
        if not states:
            print("No states loaded.")
            continue

        # Visualize (create video)
        output_video_path = os.path.join('project2d', 'results', task_name, view_name, 'project2d_view.mp4')
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
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
                preds1 = extract_endpoints(np.array(state['left']['ee_pose_rpy']), state['left']['gripper'], regvec, delta, robot_type, reg_cols=arm1_cfg['reg_cols'])
                arm2_cfg = arms_config[1]
                preds2 = extract_endpoints(np.array(state['right']['ee_pose_rpy']), state['right']['gripper'], regvec, delta, robot_type, reg_cols=arm2_cfg['reg_cols'])
                
                # Draw
                cv2.circle(img, (int(preds1[0][0]*width), int(preds1[0][1]*height)), 5, colors[0], -1)
                cv2.circle(img, (int(preds1[1][0]*width), int(preds1[1][1]*height)), 5, colors[1], -1)
                cv2.circle(img, (int(preds2[0][0]*width), int(preds2[0][1]*height)), 5, colors[2], -1)
                cv2.circle(img, (int(preds2[1][0]*width), int(preds2[1][1]*height)), 5, colors[3], -1)

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
                preds = extract_endpoints(pose, gripper, regvec, delta, robot_type, reg_cols=arm_cfg['reg_cols'])
                
                cv2.circle(img, (int(preds[0][0]*width), int(preds[0][1]*height)), 5, colors[0], -1)
                cv2.circle(img, (int(preds[1][0]*width), int(preds[1][1]*height)), 5, colors[1], -1)
            
            # out.write(img)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            out_frame = av.VideoFrame.from_ndarray(img_rgb, format='rgb24')
            for packet in out_stream.encode(out_frame):
                output_container.mux(packet)
            
        # out.release()
        for packet in out_stream.encode():
            output_container.mux(packet)
        output_container.close()
        print(f"Saved video to {output_video_path}")

def main():
    view_names = get_view_name()
    for view_name in view_names:
        project2d_view(view_name)

if __name__ == '__main__':
    main()
