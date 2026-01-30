import flask
import io
import json
import av
import cv2
import numpy as np
import os
from config import (
    get_task_dir,
    get_point_names, 
    get_video_path, 
    get_view_name, 
    get_data_dir, 
    get_task_list
)

app = flask.Flask(__name__)

# Configuration
# Global dictionaries to store data for all tasks
ALL_TASKS = {}
ALL_RESULTS = {}
ALL_DATA_DIRS = {}
ALL_RESULTS_FILES = {}

# Initialize for all tasks in task_list
TASKS_VIEWS = {} # task_name -> list of views
for view_name in get_view_name():
    t_list = get_task_list(view_name)
    for t_name in t_list:
        if t_name not in TASKS_VIEWS:
            TASKS_VIEWS[t_name] = []
        if view_name not in TASKS_VIEWS[t_name]:
            TASKS_VIEWS[t_name].append(view_name)
            
        # Data directory (videos)
        data_dir_base = get_data_dir(t_name)
        ALL_DATA_DIRS[t_name] = data_dir_base
        
        # Label files
        # We need to construct key as task_name/view_name
        key = f"{t_name}/{view_name}"
        task_dir = get_task_dir(t_name, view_name)
        
        if task_dir:
            tasks_file = os.path.join(task_dir, 'tasks.json')
            results_file = os.path.join(task_dir, 'results.json')
            ALL_RESULTS_FILES[key] = results_file

            # Load tasks
            if os.path.exists(tasks_file):
                with open(tasks_file, 'r') as f:
                    ALL_TASKS[key] = json.load(f)
            else:
                ALL_TASKS[key] = []

            # Load results
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    ALL_RESULTS[key] = json.load(f)
            else:
                ALL_RESULTS[key] = {}
        else:
            print(f"Warning: Could not determine label dir for {key}")
            ALL_TASKS[key] = []
            ALL_RESULTS[key] = {}

def save_results(key):
    if key not in ALL_RESULTS or key not in ALL_RESULTS_FILES:
        return
    results_file = ALL_RESULTS_FILES[key]
    # Ensure directory exists
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    with open(results_file, 'w') as f:
        json.dump(ALL_RESULTS[key], f, indent=2)

@app.route('/setlabel/<task_name>/<view_name>/<int:frameid>/<path:path>', methods = ['POST'])
def set_label(task_name, view_name, frameid, path):
    key = f"{task_name}/{view_name}"
    if key not in ALL_RESULTS:
        return flask.jsonify({'success': False, 'error': 'Invalid task/view'}), 404
        
    label = flask.request.get_json(force = True)
    # Save to RESULTS
    # Key: path + "_" + frameid
    res_key = f"{path}_{frameid}"
    ALL_RESULTS[key][res_key] = label
    save_results(key)
    print(f"Saved label for {key}, {path} frame {frameid}: {label}")
    return flask.jsonify({'success': True})

@app.route('/label/<task_name>/<view_name>/<int:taskid>')
def label_frame(task_name, view_name, taskid):
    key = f"{task_name}/{view_name}"
    if key not in ALL_TASKS:
        return f"Task {task_name} view {view_name} not found.", 404
        
    tasks = ALL_TASKS[key]
    results = ALL_RESULTS[key]
    
    if not tasks:
        return f"No tasks found for {key}. Run gen_label_tasks.py first.", 404
        
    if taskid >= len(tasks):
        return "Task ID out of range", 404
    task = tasks[taskid]
    
    # Task path is 'episode_XXXXXX'
    ep_path = task['path']
    frameid = task['frameid']
    
    # Determine video path for this task
    video_filename = get_video_path(task_name, view_name)
    if not video_filename:
        return f"Could not determine video path for {key}", 500

    # Construct video path relative to data_dir_base
    video_rel_path = os.path.join(ep_path, 'videos', video_filename)
    
    # Check if we have existing labels
    res_key = f"{video_rel_path}_{frameid}"
    labeled = results.get(res_key, {})
    
    # robot_name = get_robot_name(task_name)
    point_names = get_point_names(task_name)

    info = {
        'task_name' : task_name,
        'view_name': view_name,
        'taskid' : taskid,
        'frameid' : frameid,
        'path' : video_rel_path, # This relative path is passed to client
        'labeled' : labeled,
        'point_names': point_names,
        'task_list': list(ALL_TASKS.keys()) # Pass full task list keys
    }
    return flask.render_template('label.html', info = info)


def read_frame(task_name, rel_path, idx):
    if task_name not in ALL_DATA_DIRS:
        print(f"Data dir for {task_name} not found")
        return np.zeros((480, 640, 3), dtype=np.uint8)

    data_dir_base = ALL_DATA_DIRS[task_name]
    full_path = os.path.join(data_dir_base, rel_path)
    
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        return np.zeros((480, 640, 3), dtype=np.uint8) # Return black frame if not found

    try:
        with av.open(full_path) as container:
            stream = container.streams.video[0]
            target_pts = int(idx / stream.time_base / stream.average_rate)
            
            container.seek(target_pts, stream=stream, backward = True)
            
            frame_data = None
            for _, frame in enumerate(container.decode(stream)):
                if frame.pts >= target_pts:
                    frame_data = frame
                    break
            
            if frame_data:
                img = frame_data.to_rgb().to_ndarray()
                # OpenCV uses BGR
                return img[:, :, ::-1].copy()
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8)
                
    except Exception as e:
        print(f"Error reading frame {idx} from {full_path}: {e}")
        return np.zeros((480, 640, 3), dtype=np.uint8)

@app.route('/image/<task_name>/<int:idx>/<path:path>')
def get_frame(task_name, idx, path):
    frame = read_frame(task_name, path, idx)
    content = cv2.imencode('.jpg', frame)[1].tobytes()
    return flask.send_file(io.BytesIO(content), mimetype = 'image/jpeg')

@app.route('/task/<task_name>')
def task_views(task_name):
    if task_name not in TASKS_VIEWS:
        return f"Task {task_name} not found", 404
    
    views = TASKS_VIEWS[task_name]
    # Calculate counts for each view
    task_counts = {}
    for v in views:
        key = f"{task_name}/{v}"
        if key in ALL_TASKS:
            task_counts[key] = len(ALL_TASKS[key])
        else:
            task_counts[key] = 0
            
    return flask.render_template('views.html', 
                               task_name=task_name, 
                               views=views,
                               task_counts=task_counts)

@app.route('/')
def index():
    if not TASKS_VIEWS:
         return "No tasks found", 404
    
    task_list = sorted(list(TASKS_VIEWS.keys()))
    return flask.render_template('index.html', 
                               task_list=task_list, 
                               tasks_views=TASKS_VIEWS)

@app.errorhandler(404)
def page_not_found(e):
    return flask.redirect(flask.url_for('index'))

if __name__ == '__main__':
    # Use 5312 to avoid conflict
    app.run(host='0.0.0.0', port=5312, threaded = True, debug = True)
