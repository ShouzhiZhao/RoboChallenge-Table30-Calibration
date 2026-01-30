import json
import pickle
import os
import tqdm
from config import (
    get_data_dir,
    get_task_dir,
    get_task_list, 
    get_robot_type, 
    get_video_path,
    get_view_name
)

def process_data(view_name):
    datasets = {} # task_name -> list of items
    task_list = get_task_list(view_name)

    # Iterate over all tasks
    for task_name in task_list:
        robot_type = get_robot_type(task_name)
        
        task_dir = get_task_dir(task_name, view_name)
        tasks_path = os.path.join(task_dir, 'tasks.json')
        results_path = os.path.join(task_dir, 'results.json')

        if not os.path.exists(results_path) or not os.path.exists(tasks_path):
            continue
        
        print(f"Processing task: {task_name}; view: {view_name}")
        with open(tasks_path, 'r') as f:
            tasks = json.load(f)
        with open(results_path, 'r') as f:
            results = json.load(f)
            
        data_dir_base = get_data_dir(task_name)
        video_filename = get_video_path(task_name, view_name)
        
        if task_name not in datasets:
            datasets[task_name] = []
            
        for task in tqdm.tqdm(tasks, desc=f"  Loading {task_name}", leave=False):
            path = task['path'] # episode_XXXXXX
            frameid = task['frameid']
            
            # Construct key
            video_rel_path = os.path.join(path, 'videos', video_filename)
            res_key = f"{video_rel_path}_{frameid}"
            
            if res_key not in results: continue
            labels = results[res_key]
            if not labels: continue

            try:
                # Read states
                if robot_type == "arx5":
                    states_path = os.path.join(data_dir_base, path, 'states', 'states.jsonl')
                    with open(states_path, 'r') as f:
                        lines = f.readlines()
                        state = json.loads(lines[frameid])
                    
                    # "end_effector_pose": [x, y, z, r, p, y] (6 floats)
                    pose = state['end_effector_pose']
                    gripper_width = state['gripper_width']    
                    datasets[task_name].append({
                        'pose': pose,
                        'gripper_width': gripper_width,
                        'path': path,
                        'frameid': frameid,
                        **labels
                    })
                elif robot_type == "aloha":
                    left_states_path = os.path.join(data_dir_base, path, 'states', 'left_states.jsonl')
                    right_states_path = os.path.join(data_dir_base, path, 'states', 'right_states.jsonl')
                    with open(left_states_path, 'r') as f:
                        left_lines = f.readlines()
                        left_state = json.loads(left_lines[frameid])
                    with open(right_states_path, 'r') as f:
                        right_lines = f.readlines()
                        right_state = json.loads(right_lines[frameid])
                    
                    # "ee_pose_quaternion": [x, y, z, qx, qy, qz, qw] (7 floats)
                    left_pose = left_state['ee_pose_rpy']
                    left_gripper_width = left_state['gripper']
                    right_pose = right_state['ee_pose_rpy']
                    right_gripper_width = right_state['gripper']
                    datasets[task_name].append({
                        'left_pose': left_pose,
                        'left_gripper_width': left_gripper_width,
                        'right_pose': right_pose,
                        'right_gripper_width': right_gripper_width,
                        'path': path,
                        'frameid': frameid,
                        **labels
                    })
                elif robot_type == "franka":
                    states_path = os.path.join(data_dir_base, path, 'states', 'states.jsonl')
                    with open(states_path, 'r') as f:
                        lines = f.readlines()
                        state = json.loads(lines[frameid])
                    
                    # "ee_positions": [x, y, z, qx, qy, qz, qw]
                    pose = state['ee_positions']
                    gripper_width = state['gripper_width'][0]
                    datasets[task_name].append({
                        'pose': pose,
                        'gripper_width': gripper_width,
                        'path': path,
                        'frameid': frameid,
                        **labels
                    })
                elif robot_type == "ur5":
                    states_path = os.path.join(data_dir_base, path, 'states', 'states.jsonl')
                    with open(states_path, 'r') as f:
                        lines = f.readlines()
                        state = json.loads(lines[frameid])
                    
                    # "ee_positions": [x, y, z, qx, qy, qz, qw]
                    pose = state['ee_positions']
                    gripper_width = (255 - state['gripper']) / 255.0 * 0.085
                    datasets[task_name].append({
                        'pose': pose,
                        'gripper_width': gripper_width,
                        'path': path,
                        'frameid': frameid,
                        **labels
                    })
                else:
                    print(f"Unknown robot type: {robot_type}")
                    continue
            except Exception as e:
                print(f"Error reading state for {task_name}, {path}, frame {frameid}: {e}")
                continue
                
    # Save datasets
    for task_name, data in datasets.items():
        if not data: continue
        
        output_dir = get_task_dir(task_name, view_name)
        os.makedirs(output_dir, exist_ok=True)
        output_pkl = os.path.join(output_dir, 'processed_data.pkl')
        
        with open(output_pkl, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {len(data)} labeled frames to {output_pkl}")

def main():
    view_names = get_view_name()
    for view_name in view_names:
        process_data(view_name)

if __name__ == '__main__':
    main()
