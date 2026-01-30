
import os
import yaml
import pickle
import numpy as np
from config import get_task_list, get_robot_name, get_arms_config, get_view_name

def export_extrinsics(view_name, data):
    task_list = get_task_list(view_name)
    
    for task_name in task_list:
        # Load fit results
        fit_pkl = os.path.join('pnp', 'results', task_name, view_name, 'pnp_fit.pkl')
        if not os.path.exists(fit_pkl):
            continue
            
        with open(fit_pkl, 'rb') as f:
            fit_data = pickle.load(f)
            
        extrinsics_dict = fit_data.get('extrinsics', {})
        delta = fit_data.get('delta')
        width = fit_data.get('width')
        height = fit_data.get('height')
        
        # Get arms config to ensure order
        arms_config = get_arms_config(task_name)
        
        # Structure for YAML: concatenated rvec and tvec lists
        # rvec: [r1_0, r1_1, r1_2, r2_0, r2_1, r2_2, ...]
        # tvec: [t1_0, t1_1, t1_2, t2_0, t2_1, t2_2, ...]
        
        all_rvec = []
        all_tvec = []
        
        for arm_cfg in arms_config:
            arm_name = arm_cfg['name']
            if arm_name in extrinsics_dict:
                params = extrinsics_dict[arm_name]
                rvec = params['rvec'].flatten().tolist() if isinstance(params['rvec'], np.ndarray) else params['rvec']
                tvec = params['tvec'].flatten().tolist() if isinstance(params['tvec'], np.ndarray) else params['tvec']
                
                all_rvec.extend(rvec)
                all_tvec.extend(tvec)
            else:
                print(f"Warning: Extrinsics for {arm_name} missing in {task_name}, padding with zeros.")
                all_rvec.extend([0.0, 0.0, 0.0])
                all_tvec.extend([0.0, 0.0, 0.0])

        if task_name not in data:
            data[task_name] = {}
        
        # Add robot name
        data[task_name]['robot'] = get_robot_name(task_name)
            
        if view_name not in data[task_name]:
            data[task_name][view_name] = {}
            
        # Write flat lists
        data[task_name][view_name]['rvec'] = all_rvec
        data[task_name][view_name]['tvec'] = all_tvec
        
        # Clean up old format if present
        if 'extrinsics' in data[task_name][view_name]:
            del data[task_name][view_name]['extrinsics']
        
        if delta is not None:
            data[task_name][view_name]['delta'] = delta.tolist() if isinstance(delta, np.ndarray) else delta
            
        if width is not None and height is not None:
            data[task_name][view_name]['resolution'] = [width, height]
        
def main():
    print(f"\n{'='*50}")
    print("Starting PnP Extrinsics Export")
    print(f"{'='*50}\n")
        
    data = {}
    view_names = get_view_name()
    for view_name in view_names:
        print(f"Exporting extrinsics for view: {view_name}")
        export_extrinsics(view_name, data)
    
    output_yaml = os.path.join('pnp', 'extrinsics.yaml')
    if os.path.exists(output_yaml):
        print(f"Removing existing file: {output_yaml}")
        os.remove(output_yaml)

    # Save to yaml
    sorted_data = dict(sorted(data.items()))
    with open(output_yaml, 'w') as f:
        yaml.dump(sorted_data, f, default_flow_style=None, sort_keys=False)
    
    print(f"\n{'='*50}")
    print(f"Exported extrinsics to {output_yaml}")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    main()
