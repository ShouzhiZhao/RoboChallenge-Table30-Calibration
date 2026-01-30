import json
import os

# Configuration
view_names = ['global_view', 'side_view']
DATA_ROOT = '/dexmal-table30/release/20251230/' # you need to change this to your own data root
TASK_ROOT = 'table30'

POINT_CONFIG = {
    'aloha': {
        'point_names': ['left tip left', 'left tip right', 'right tip left', 'right tip right']
    },
    'default': {
        'point_names': ['tip left', 'tip right']
    }
}

ARMS_CONFIG = {
    'aloha': [
        {
            'name': 'Arm 1 (Left)',
            'keypoints': ['point0', 'point1'],
            'names': ['left tip left', 'left tip right'],
            'feature_key': 'feature1',
            'reg_cols': [0, 1]
        },
        {
            'name': 'Arm 2 (Right)',
            'keypoints': ['point2', 'point3'],
            'names': ['right tip left', 'right tip right'],
            'feature_key': 'feature2',
            'reg_cols': [2, 3]
        }
    ],
    'default': [
        {
            'name': 'Arm 1',
            'keypoints': ['point0', 'point1'],
            'names': ['tip left', 'tip right'],
            'feature_key': 'feature',
            'reg_cols': [0, 1]
        }
    ]
}

VIDEO_CONFIG = {
    'arx5': {
        'wrist_view': 'arm_realsense_rgb.mp4',
        'global_view': 'global_realsense_rgb.mp4',
        'side_view': 'right_realsense_rgb.mp4'
    },
    'ur5': {
        'wrist_view': 'handeye_realsense_rgb.mp4',
        'global_view': 'global_realsense_rgb.mp4'
    },
    'franka': {
        'wrist_view': 'handeye_realsense_rgb.mp4',
        'global_view': 'main_realsense_rgb.mp4',
        'side_view': 'side_realsense_rgb.mp4'
    },
    'aloha': {
        'left_wrist_view': 'cam_wrist_left_rgb.mp4',
        'global_view': 'cam_high_rgb.mp4',
        'right_wrist_view': 'cam_wrist_right_rgb.mp4'
    }
}


all_tasks_dirs = sorted(os.listdir(DATA_ROOT))
task_list = {view_name: [] for view_name in view_names}
task_meta_cache = {}

for task_name in all_tasks_dirs:
    task_dir = os.path.join(DATA_ROOT, task_name)
    if os.path.isdir(task_dir):
        meta_path = os.path.join(task_dir, 'meta', 'task_info.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            r_id = meta.get('robot_id', '').lower()
            for view_name in view_names:
                if view_name in VIDEO_CONFIG.get(r_id.split('_')[0], {}):
                    task_list[view_name].append(task_name)
            if r_id == 'arx5_3':
                r_id = 'arx5_9' # arx5_3 is a duplicate of arx5_9
            task_meta_cache[task_name] = {
                'robot_name': r_id,
                'robot_type': r_id.split('_')[0]
            }


def get_robot_name(task_name):
    return task_meta_cache.get(task_name, {}).get('robot_name')

def get_robot_type(task_name):
    return task_meta_cache.get(task_name, {}).get('robot_type')

def get_save_dir(task_name, view_name):
    return os.path.join(task_name, view_name)

def get_task_dir(task_name, view_name):
    return os.path.join(TASK_ROOT, task_name, view_name)

def get_point_names(task_name):
    r_type = get_robot_type(task_name)
    if r_type == 'aloha':
        return POINT_CONFIG['aloha']['point_names']
    return POINT_CONFIG['default']['point_names']

def get_arms_config(task_name):
    r_type = get_robot_type(task_name)
    if r_type == 'aloha':
        return ARMS_CONFIG['aloha']
    return ARMS_CONFIG['default']

def get_video_path(task_name, view_name):
    r_type = get_robot_type(task_name)
    if r_type in VIDEO_CONFIG:
        return VIDEO_CONFIG[r_type].get(view_name)
    return None

def get_data_dir(task_name):
    return os.path.join(DATA_ROOT, task_name, 'data')

def get_task_list(view_name):
    return task_list.get(view_name, [])

def get_view_name():
    return view_names

