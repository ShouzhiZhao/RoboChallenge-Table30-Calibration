import yaml
import os
import numpy as np
import cv2
import argparse
from scipy.spatial.transform import Rotation
from project2d.fit import quaternion_to_rotmat, euler_to_rotmat
from pnp.fit import load_intrinsics
from config import get_robot_type

def project_point_pnp(point_3d, rvec, tvec, camera_matrix, dist_coeffs):
    """
    Projects a 3D point to 2D image coordinates using PnP parameters.
    """
    points_3d = np.array([point_3d], dtype=np.float64)
    projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
    return projected_points.reshape(-1, 2)[0]

def project_point_project2d(point_3d, params_u, params_v):
    """
    Projects a 3D point to 2D image coordinates using Rational Function Model (Project2D).
    Formula: u = (P0*x + P1*y + P2*z + P6) / (1 + P3*x + P4*y + P5*z)
    """
    x, y, z = point_3d
    # params: [p0, p1, p2, p3, p4, p5, p6]
    
    num_u = params_u[0]*x + params_u[1]*y + params_u[2]*z + params_u[6]
    den_u = 1 + params_u[3]*x + params_u[4]*y + params_u[5]*z
    u = num_u / den_u
    
    num_v = params_v[0]*x + params_v[1]*y + params_v[2]*z + params_v[6]
    den_v = 1 + params_v[3]*x + params_v[4]*y + params_v[5]*z
    v = num_v / den_v
    
    return np.array([u, v])

def load_pnp_params(task_name, view_name):
    path = os.path.join(os.path.dirname(__file__), 'pnp', 'extrinsics.yaml')
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return None, None
        
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        
    if task_name not in data:
        print(f"Error: Task {task_name} not found in extrinsics.yaml")
        return None, None
        
    task_data = data[task_name]
    if view_name not in task_data:
        print(f"Error: View {view_name} not found for task {task_name}")
        return None, None
        
    view_data = task_data[view_name]
    
    # Updated: extrinsics is now flattened lists in 'rvec' and 'tvec'
    rvec_list = view_data.get('rvec', [])
    tvec_list = view_data.get('tvec', [])
    
    extrinsics_parsed = []
    
    # Chunk into groups of 3
    if rvec_list and tvec_list and len(rvec_list) == len(tvec_list):
        num_arms = len(rvec_list) // 3
        for i in range(num_arms):
            rvec = np.array(rvec_list[i*3 : (i+1)*3], dtype=np.float64)
            tvec = np.array(tvec_list[i*3 : (i+1)*3], dtype=np.float64)
            extrinsics_parsed.append({'rvec': rvec, 'tvec': tvec})
            
    delta = np.array(view_data.get('delta', [0, 0, 0]))
    return extrinsics_parsed, delta

def load_project2d_params(task_name, view_name):
    path = os.path.join(os.path.dirname(__file__), 'project2d', 'parameters.yaml')
    if not os.path.exists(path):
        print(f"Error: {path} not found")
        return None, None
        
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        
    if task_name not in data:
        print(f"Error: Task {task_name} not found in parameters.yaml")
        return None, None
        
    task_data = data[task_name]
    if view_name not in task_data:
        print(f"Error: View {view_name} not found for task {task_name}")
        return None, None
        
    view_data = task_data[view_name]
    
    regvec = np.array(view_data.get('regvec', []))
    delta = np.array(view_data.get('delta', [0, 0, 0]))
    
    return regvec, delta

def get_endpoint_position(pose, gripper, delta, robot_type):
    """
    Calculate the 3D position of the gripper endpoints (tips) in robot base frame,
    applying the calibrated delta offset.
    """
    xyz = pose[:3]
    dx, dy, dz = delta
    
    if robot_type == 'arx5':
        rotmat = euler_to_rotmat(pose[3:])
        p1 = xyz + rotmat.dot(np.float32([dx, -gripper/2 - dy, dz]))
        p2 = xyz + rotmat.dot(np.float32([dx, gripper/2 + dy, dz]))
        return p1, p2
        
    elif robot_type == 'aloha':
        rotmat = euler_to_rotmat(pose[3:])
        p1 = xyz + rotmat.dot(np.float32([dx, gripper/2 + dy, dz]))
        p2 = xyz + rotmat.dot(np.float32([dx, -gripper/2 - dy, dz]))
        return p1, p2
        
    elif robot_type == 'franka':
        rotmat = quaternion_to_rotmat(pose[3:])
        p1 = xyz + rotmat.dot(np.float32([dx, gripper/2 + dy, dz]))
        p2 = xyz + rotmat.dot(np.float32([dx, -gripper/2 - dy, dz]))
        return p1, p2
        
    elif robot_type == 'ur5':
        rotmat = quaternion_to_rotmat(pose[3:])
        p1 = xyz + rotmat.dot(np.float32([gripper/2 + dy, -dx, dz]))
        p2 = xyz + rotmat.dot(np.float32([-gripper/2 - dy, -dx, dz]))
        return p1, p2
    
    return None, None

def run_pnp_inference(task_name, view_name, robot_type, dummy_pose, dummy_gripper):
    print("Loading PnP parameters...")
    camera_matrix, dist_coeffs = load_intrinsics(task_name, view_name)
    extrinsics_list, delta = load_pnp_params(task_name, view_name)
    
    if camera_matrix is None or extrinsics_list is None:
        print("Failed to load PnP parameters.")
        return

    print(f"Loaded extrinsics count: {len(extrinsics_list)}")
    print(f"Loaded delta: {delta}")
    
    print("\nCalculating 3D Endpoints...")
    p1_3d, p2_3d = get_endpoint_position(dummy_pose, dummy_gripper, delta, robot_type)
    if p1_3d is None:
        print(f"Unknown robot type: {robot_type}")
        return

    print(f"  Tip 1 (3D): {p1_3d}")
    print(f"  Tip 2 (3D): {p2_3d}")
    
    print("\nProjecting to 2D Image Coordinates (PnP)...")
    
    # Use first set of extrinsics for demonstration (Arm 1)
    if not extrinsics_list:
        print("No extrinsics found.")
        return
        
    # For dual arm robots, we might need to select which extrinsics to use
    # Here we just demo with the first one
    params = extrinsics_list[0]
    rvec = params['rvec']
    tvec = params['tvec']
    
    uv1 = project_point_pnp(p1_3d, rvec, tvec, camera_matrix, dist_coeffs)
    uv2 = project_point_pnp(p2_3d, rvec, tvec, camera_matrix, dist_coeffs)
    
    print(f"  Tip 1 (2D pixel): ({uv1[0]:.2f}, {uv1[1]:.2f})")
    print(f"  Tip 2 (2D pixel): ({uv2[0]:.2f}, {uv2[1]:.2f})")

def run_project2d_inference(task_name, view_name, robot_type, dummy_pose, dummy_gripper):
    print("Loading Project2D parameters...")
    regvec, delta = load_project2d_params(task_name, view_name)
    
    if regvec is None:
        print("Failed to load Project2D parameters.")
        return
        
    print(f"Loaded regvec shape: {regvec.shape}")
    print(f"Loaded delta: {delta}")
    
    print("\nCalculating 3D Endpoints...")
    p1_3d, p2_3d = get_endpoint_position(dummy_pose, dummy_gripper, delta, robot_type)
    if p1_3d is None:
        print(f"Unknown robot type: {robot_type}")
        return

    print(f"  Tip 1 (3D): {p1_3d}")
    print(f"  Tip 2 (3D): {p2_3d}")
    
    print("\nProjecting to 2D Image Coordinates (Project2D)...")
    
    # Column 0 is U, Column 1 is V for the first arm
    params_u1 = regvec[:, 0]
    params_v1 = regvec[:, 1]
    
    uv1 = project_point_project2d(p1_3d, params_u1, params_v1)
    uv2 = project_point_project2d(p2_3d, params_u1, params_v1)
    
    print(f"  Tip 1 (Normalized): ({uv1[0]:.4f}, {uv1[1]:.4f})")
    print(f"  Tip 2 (Normalized): ({uv2[0]:.4f}, {uv2[1]:.4f})")
    
    try:
        # Try to get resolution from PnP file to scale back to pixels
        pnp_path = os.path.join(os.path.dirname(__file__), 'pnp', 'extrinsics.yaml')
        if os.path.exists(pnp_path):
            with open(pnp_path, 'r') as f:
                pnp_data = yaml.safe_load(f)
                if task_name in pnp_data and view_name in pnp_data[task_name]:
                    res = pnp_data[task_name][view_name].get('resolution')
                    if res:
                        w, h = res
                        print(f"  Tip 1 (Pixel, est {w}x{h}): ({uv1[0]*w:.2f}, {uv1[1]*h:.2f})")
                        print(f"  Tip 2 (Pixel, est {w}x{h}): ({uv2[0]*w:.2f}, {uv2[1]*h:.2f})")
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description='Inference Example')
    parser.add_argument('--method', type=str, default='pnp', choices=['pnp', 'project2d'],
                        help='Inference method: pnp or project2d')
    parser.add_argument('--task', type=str, default='arrange_flowers',
                        help='Task name')
    parser.add_argument('--view', type=str, default='global_view',
                        help='View name (global_view, side_view, etc.)')
    
    args = parser.parse_args()
    
    task_name = args.task
    view_name = args.view
    
    # Auto-detect robot type
    robot_type = get_robot_type(task_name)
    if not robot_type:
        print(f"Warning: Could not detect robot type for {task_name}, defaulting to arx5")
        robot_type = 'arx5'
    
    print(f"--- Inference Example for {task_name} ({view_name}) ---")
    print(f"Method: {args.method}")
    print(f"Robot: {robot_type}")
    
    # Dummy Robot State
    # robot_state list of tuples: [(pose, gripper), ...]
    robot_state = []
    
    if robot_type == 'aloha':
        # Aloha has 2 arms
        # Left Arm (Arm 1)
        pose_left = np.array([0.3, 0.2, 0.2, 0.0, 1.57, 0.0])
        gripper_left = 0.04
        
        # Right Arm (Arm 2)
        pose_right = np.array([0.3, -0.2, 0.2, 0.0, 1.57, 0.0])
        gripper_right = 0.04
        
        robot_state.append((pose_left, gripper_left))
        robot_state.append((pose_right, gripper_right))
        
    elif robot_type in ['franka', 'ur5']:
        # Quaternion robots
        pose = np.array([0.3, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0])
        gripper = 0.04
        robot_state.append((pose, gripper))
        
    else: # arx5, etc.
        pose = np.array([0.3, 0.0, 0.2, 0.0, 1.57, 0.0])
        gripper = 0.04
        robot_state.append((pose, gripper))
    
    print(f"\nInput Robot State ({len(robot_state)} arms):")
    for i, (p, g) in enumerate(robot_state):
        print(f"  Arm {i+1}: Pose={p}, Gripper={g}")
    
    if args.method == 'pnp':
        run_pnp_inference(task_name, view_name, robot_type, robot_state)
    elif args.method == 'project2d':
        run_project2d_inference(task_name, view_name, robot_type, robot_state)
    
    print("\nInference Complete.")

if __name__ == '__main__':
    main()
