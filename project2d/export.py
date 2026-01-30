
import os
import yaml
import pickle
from config import get_task_list, get_robot_name, get_view_name

def export_parameters(view_name, data):
    task_list = get_task_list(view_name)
    
    for task_name in task_list:
        # Load fit results
        fit_pkl = os.path.join('project2d', 'results', task_name, view_name, 'project2d_fit.pkl')
        if not os.path.exists(fit_pkl):
            continue
            
        with open(fit_pkl, 'rb') as f:
            fit_data = pickle.load(f)
            
        regvec = fit_data['regvec']
        delta = fit_data['delta']
        
        # Structure for YAML
        # task_name:
        #   robot: robot_name
        #   view_name:
        #     regvec: list of lists
        #     delta: list
        
        if task_name not in data:
            data[task_name] = {}
        
        # Add robot name
        data[task_name]['robot'] = get_robot_name(task_name)
            
        if view_name not in data[task_name]:
            data[task_name][view_name] = {}
            
        # Convert numpy arrays to lists
        data[task_name][view_name]['regvec'] = regvec.tolist()
        data[task_name][view_name]['delta'] = delta.tolist()

def main():
    print(f"\n{'='*50}")
    print("Starting Project2D Parameters Export")
    print(f"{'='*50}\n")
    
    data = {}
    view_names = get_view_name()
    for view_name in view_names:
        print(f"Exporting parameters for view: {view_name}")
        export_parameters(view_name, data)
        
    # Save to yaml
    # Sort keys for consistent output
    sorted_data = dict(sorted(data.items()))
    output_yaml = os.path.join('project2d', 'parameters.yaml')
    if os.path.exists(output_yaml):
        print(f"Removing existing file: {output_yaml}")
        os.remove(output_yaml)

    with open(output_yaml, 'w') as f:
        yaml.dump(sorted_data, f, default_flow_style=None, sort_keys=False)
    
    print(f"\n{'='*50}")
    print(f"Exported parameters to {output_yaml}")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    main()
