import json
import os
import random
import tqdm
from config import get_task_list, get_task_dir, get_data_dir, get_view_name

def gen_label_tasks(view_name):
    print(f'Scanning all files for view {view_name}...')
    task_list = get_task_list(view_name)
    for task in task_list:
        all_episodes = []
        data_dir_base = get_data_dir(task)
        if not os.path.exists(data_dir_base):
            print(f"Error: Data directory {data_dir_base} does not exist.")
            continue
            
        for entry in os.scandir(data_dir_base):
            if entry.is_dir() and entry.name.startswith('episode_'):
                all_episodes.append(entry.name)
                
        all_episodes.sort()
        print(f"Found {len(all_episodes)} episodes for {task}.")
        
        selected = []
        # For each episode, pick 1 frame.
        for ep in tqdm.tqdm(all_episodes, desc=f"Processing {task}"):
            meta_path = os.path.join(data_dir_base, ep, 'meta', 'episode_meta.json')
            if not os.path.exists(meta_path):
                continue

            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except Exception as e:
                print(f"Error reading {meta_path}: {e}")
                continue
                
            total_frames = meta.get('frames', 0)
            if total_frames <= 0:
                continue
                
            # Randomly pick 1 frame
            frameid = random.randint(0, total_frames - 1)
            selected.append({'path': ep, 'frameid': frameid})

        print(f"Generated {len(selected)} tasks for {task}")
        if selected:
            print("Sample task:", selected[0])
        
        task_dir = get_task_dir(task, view_name)
        if task_dir:
            output_file = os.path.join(task_dir, 'tasks.json')
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(selected, f, indent=2)
            print(f"Tasks saved to {output_file}")
        else:
            print(f"Skipping task {task} (unknown robot)")


if __name__ == '__main__':
    view_names = get_view_name()
    for view_name in view_names:
        gen_label_tasks(view_name)
