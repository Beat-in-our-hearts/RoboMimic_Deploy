from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
import numpy as np
import yaml
from common.utils import FSMCommand, progress_bar
import onnx
import onnxruntime
import torch
import os
import math


class SONIC_HUMAN_Mimic(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_GAE
        self.name_str = "sonic_human_mimic"
        self.counter_step = 0
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "SONIC_HUMAN_Mimic.yaml")
        
        # Check if config file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please ensure SONIC_HUMAN_Mimic.yaml exists in the config directory."
            )
        
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.onnx_path = os.path.join(current_dir, "model", config["onnx_path"])
            self.motion_path = os.path.join(current_dir, "motion", config["motion_path"])
            
            # Check if model file exists
            if not os.path.exists(self.onnx_path):
                raise FileNotFoundError(
                    f"ONNX model file not found: {self.onnx_path}\n"
                    f"Please place the ONNX model file (e.g., policy.onnx) in:\n"
                    f"  {os.path.join(current_dir, 'model/')}\n"
                    f"Expected filename from config: {config['onnx_path']}"
                )
            
            # Check if motion file exists
            if not os.path.exists(self.motion_path):
                raise FileNotFoundError(
                    f"Motion data file not found: {self.motion_path}\n"
                    f"Please place the motion NPZ file in:\n"
                    f"  {os.path.join(current_dir, 'motion/lafan1/')}\n"
                    f"Expected filename from config: {config['motion_path']}"
                )
            
            self._load_motion()

            # load policy
            self.onnx_model = onnx.load(self.onnx_path)
            self.ort_session = onnxruntime.InferenceSession(self.onnx_path)
            print("Loaded GaeMimic ONNX model from ", self.onnx_path)
            
            input = self.ort_session.get_inputs()
            self.input_name = []
            for i, inpt in enumerate(input):
                self.input_name.append(inpt.name)
            print("Input names: ", self.input_name)
            
            output = self.ort_session.get_outputs()
            self.output_name = []
            for i, outpt in enumerate(output):
                self.output_name.append(outpt.name)
            print("Output names: ", self.output_name)
            
            self.metadata = self._load_metadata()
            self.kps_lab = self.metadata["joint_stiffness"]
            self.kds_lab = self.metadata["joint_damping"]
            self.default_angles_lab = self.metadata["default_joint_pos"]
            
            self.lab_joint_names = self.metadata["joint_names"]
            self.lab_body_names = self.metadata["body_names"]
            
            mj_joint_names = config["mj_joint_names"]
            self.mj2lab = [mj_joint_names.index(joint) for joint in self.lab_joint_names]

            self.motion_anchor_body_name = self.metadata["motion_anchor_body_name"]
            self.motion_anchor_id = self.lab_body_names.index(self.motion_anchor_body_name)
            
            observation_dims = [round(dims) for dims in self.metadata["observation_dims"]] # list[float]
            self.num_obs = sum(observation_dims)
            
            self.num_actions = len(self.metadata["action_scale"])
            self.action_scale = np.array(self.metadata["action_scale"], dtype=np.float32)
            
            # init datas
            self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.obs = np.zeros(self.num_obs)
            self.action = np.zeros(self.num_actions)
            
            self.ref_joint_pos = np.zeros(self.num_actions*10, dtype=np.float32)
            self.ref_joint_vel = np.zeros(self.num_actions*10, dtype=np.float32)
            self.ref_anchor_ori_w = np.zeros(4, dtype=np.float32)
            
            print("kp_lab: ", self.kps_lab)
            print("kd_lab: ", self.kds_lab)
            print("default_angles_lab: ", self.default_angles_lab)
            print("mj_joint_names: ", self.lab_joint_names)
            print("action_scale_lab: ", self.action_scale)
            print("mj2lab", self.mj2lab)
            
            print("GaeMimic policy initializing ...")
    
    def _load_motion(self):
        """
        Load motion data from the specified motion path.
        
        IMPORTANT: The start pose must be static, and end pose must be static as well.
        
        - joint_pos: (num_frames, num_joints) joint positions
        - joint_vel: (num_frames, num_joints) joint velocities
        - body_pos_w: (num_frames, num_bodies, 3) body positions
        - body_quat_w: (num_frames, num_bodies, 4) body quaternions
        - body_lin_vel_w: (num_frames, num_bodies, 3) body linear velocities
        - body_ang_vel_w: (num_frames, num_bodies, 3) body angular velocities
        """
        data = np.load(self.motion_path)
        self.motion_data = {
            "pose_body": data["pose_body"],
            "root_orient": data["root_orient"],
            "trans": data["trans"],
        }
        self.motion_length = self.motion_data["pose_body"].shape[0]
        self.motion_data["root_quat_w"] = self._axis_angle_to_quaternion(self.motion_data["root_orient"])
        
    def _load_metadata(self):
        metadata = {}
        
        for prop in self.onnx_model.metadata_props:
            key = prop.key
            value = prop.value
            
            # Try to parse as list (comma-separated values)
            if "," in value:
                try:
                    # Try parsing as float list first
                    parsed_list = [float(x.strip()) for x in value.split(",")]
                    metadata[key] = parsed_list
                except ValueError:
                    # If parsing as float fails, keep as string list
                    metadata[key] = [x.strip() for x in value.split(",")]
            else:
                # Try to parse as single value
                try:
                    metadata[key] = float(value)
                except ValueError:
                    # Keep as string if not a number
                    metadata[key] = value
        
        return metadata

    def enter(self):

        self.counter_step = 0

        observation = {}
        for i, input_name in enumerate(self.input_name):
            observation[input_name] = np.zeros((1, self.ort_session.get_inputs()[i].shape[1]), dtype=np.float32)


        outputs_result = self.ort_session.run(None, observation)
        self.action = outputs_result[0]

        self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(self.num_obs)
        
        self.action = np.zeros(self.num_actions)
        pass
       
    @staticmethod 
    def quat_mul(q1, q2):
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
        # perform multiplication
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)
        return np.array([w, x, y, z])
        
    @staticmethod 
    def matrix_from_quat(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
        ])

    @staticmethod 
    def yaw_quat(q):
        w, x, y, z = q
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
        return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    
    @staticmethod
    def euler_single_axis_to_quat(angle, axis, degrees=False):
        if degrees:
            angle = np.radians(angle)
        
        half_angle = angle * 0.5
        cos_half = np.cos(half_angle)
        sin_half = np.sin(half_angle)
        
        if isinstance(axis, str):
            if axis.lower() == 'x':
                return np.array([cos_half, sin_half, 0.0, 0.0])
            elif axis.lower() == 'y':
                return np.array([cos_half, 0.0, sin_half, 0.0])
            elif axis.lower() == 'z':
                return np.array([cos_half, 0.0, 0.0, sin_half])
            else:
                raise ValueError("axis must be 'x', 'y', 'z' or a 3D unit vector")
        else:
            axis = np.array(axis, dtype=np.float32)
            axis_norm = np.linalg.norm(axis)
            if axis_norm == 0:
                raise ValueError("axis vector cannot be zero")
            axis = axis / axis_norm
            
            w = cos_half
            x = sin_half * axis[0]
            y = sin_half * axis[1]
            z = sin_half * axis[2]
            
            return np.array([w, x, y, z])
    
    @staticmethod
    def _euler_xyz_to_rotation_matrix(euler_xyz):
        """
        Convert Euler angles (radians) in XYZ order to rotation matrix.
        R_total = R_z(z) @ R_y(y) @ R_x(x)
        
        Args:
            euler_xyz: np.array([x_angle, y_angle, z_angle]) in radians
        
        Returns:
            3x3 rotation matrix
        """
        x, y, z = euler_xyz
        
        # Rotation around X axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(x), -np.sin(x)],
            [0, np.sin(x), np.cos(x)]
        ], dtype=np.float64)
        
        # Rotation around Y axis
        Ry = np.array([
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)]
        ], dtype=np.float64)
        
        # Rotation around Z axis
        Rz = np.array([
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Combined rotation: R = Rz @ Ry @ Rx
        R = Rz @ Ry @ Rx
        return R
    
    @staticmethod
    def _rotation_matrix_to_quaternion(R):
        """
        Convert 3x3 rotation matrix to quaternion (wxyz).
        
        Args:
            R: 3x3 rotation matrix
        
        Returns:
            np.array([w, x, y, z]) quaternion in wxyz order
        """
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([w, x, y, z], dtype=np.float64)
    
    @staticmethod
    def _quaternion_multiply(q1, q2):
        """
        Multiply two quaternions: q_result = q1 * q2
        
        Args:
            q1: np.array([w1, x1, y1, z1])
            q2: np.array([w2, x2, y2, z2])
        
        Returns:
            np.array([w, x, y, z]) result quaternion
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        return np.array([w, x, y, z], dtype=np.float64)
    
    @staticmethod
    def _axis_angle_to_quaternion(axis_angle):
        """
        Convert axis-angle representation to quaternion with rotation correction.
        
        SMPL Unity model uses Y-up coordinate system, but we need to correct for
        the fixed rotation: R_fix = R_x(-90°) * R_z(-90°)
        
        Final quaternion: q_final = q_fix * q_aa
        
        Args:
            axis_angle: shape (N, 3) or (3,) - axis-angle in radians
        
        Returns:
            quaternion: shape (N, 4) or (4,) - quaternion in wxyz order
        """
        axis_angle = np.asarray(axis_angle, dtype=np.float64)
        is_single = axis_angle.ndim == 1
        
        if is_single:
            axis_angle = axis_angle[np.newaxis, :]  # (1, 3)
        
        # Step 1: Standard axis-angle to quaternion conversion
        angles = np.linalg.norm(axis_angle, axis=1, keepdims=True)  # (N, 1)
        
        small_angle_mask = angles[:, 0] < 1e-6
        
        # Compute half angles
        half_angles = angles / 2.0
        
        # Compute quaternion components
        w = np.cos(half_angles[:, 0])
        
        # Avoid division by zero
        safe_angles = np.where(small_angle_mask[:, np.newaxis], 1.0, angles)
        axes = axis_angle / safe_angles
        
        sin_half = np.sin(half_angles)
        xyz = axes * sin_half
        
        # For small angles, use Taylor approximation
        xyz_small = axis_angle * 0.5
        xyz = np.where(small_angle_mask[:, np.newaxis], xyz_small, xyz)
        w = np.where(small_angle_mask, 1.0, w)
        
        q_aa = np.concatenate([w[:, np.newaxis], xyz], axis=1)  # (N, 4)
        
        # Step 2: Compute correction rotation: R_fix = R_x(-90°) * R_z(-90°)
        euler_fix = np.array([math.radians(-90), 0, math.radians(-90)], dtype=np.float64)
        R_fix = SONIC_HUMAN_Mimic._euler_xyz_to_rotation_matrix(euler_fix)
        q_fix = SONIC_HUMAN_Mimic._rotation_matrix_to_quaternion(R_fix)
        
        # Step 3: Apply correction: q_final = q_fix * q_aa
        q_final = np.zeros_like(q_aa)
        for i in range(q_aa.shape[0]):
            q_final[i] = SONIC_HUMAN_Mimic._quaternion_multiply(q_fix, q_aa[i])
        
        if is_single:
            return q_final[0]
        else:
            return q_final
    
    @staticmethod
    def compute_projected_gravity(quat_w, gravity_w=None):
        if gravity_w is None:
            gravity_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        
        w, x, y, z = quat_w
        
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ], dtype=np.float32)
        
        projected_gravity_b = R.T @ gravity_w
        
        return projected_gravity_b

    def get_human_command(self):
        # get 10 frame, 0.1s interval
        step = np.arange(0, 50, 5) + self.counter_step
        step = np.clip(step, 0, self.motion_length - 1)
        
        # (10, num_joints) 
        command = self.motion_data["pose_body"][step]
        return command.reshape(1, -1)    
        
    def run(self):
        robot_quat = self.state_cmd.base_quat
        gravity_ori = self.compute_projected_gravity(robot_quat)
        
        qj = self.state_cmd.q[self.mj2lab]
        qj = (qj - self.default_angles_lab)

        base_troso_yaw = qj[2]
        base_troso_roll = qj[5]
        base_troso_pitch = qj[8]
        
        quat_yaw = self.euler_single_axis_to_quat(base_troso_yaw, 'z', degrees=False)
        quat_roll = self.euler_single_axis_to_quat(base_troso_roll, 'x', degrees=False)
        quat_pitch = self.euler_single_axis_to_quat(base_troso_pitch, 'y', degrees=False)
        temp1 = self.quat_mul(quat_roll, quat_pitch)
        temp2 = self.quat_mul(quat_yaw, temp1)
        robot_quat = self.quat_mul(robot_quat, temp2)
        
        self.ref_anchor_ori_w = self.motion_data["root_quat_w"][self.counter_step]
        print(f"[INFO] {self.counter_step}: {self.ref_anchor_ori_w}")

        if(self.counter_step < 2):
            init_to_anchor = self.matrix_from_quat(self.yaw_quat(self.ref_anchor_ori_w))
            world_to_anchor = self.matrix_from_quat(self.yaw_quat(robot_quat))
            self.init_to_world = world_to_anchor @ init_to_anchor.T
            print("self.init_to_world: ", self.init_to_world)
            self.counter_step += 1
            return

        motion_anchor_ori_b = self.matrix_from_quat(robot_quat).T @ self.init_to_world @ self.matrix_from_quat(self.ref_anchor_ori_w)

        ang_vel = self.state_cmd.ang_vel
        
        dqj = self.state_cmd.dq[self.mj2lab]
        
        # command motion_anchor_ori_b base_ang_vel projected_gravity joint_pos joint_vel previous_action
        observation = {}
        observation[self.input_name[0]] = self.get_human_command().astype(np.float32)
        observation[self.input_name[1]] = motion_anchor_ori_b[:,:2].reshape(1, -1).astype(np.float32)
        observation[self.input_name[2]] = ang_vel.reshape(1, -1).astype(np.float32)
        observation[self.input_name[3]] = gravity_ori.reshape(1, -1).astype(np.float32)
        observation[self.input_name[4]] = qj.reshape(1, -1).astype(np.float32)
        observation[self.input_name[5]] = dqj.reshape(1, -1).astype(np.float32)
        observation[self.input_name[6]] = self.action.reshape(1, -1).astype(np.float32)
        
        outputs_result = self.ort_session.run(None, observation)

        # update action
        self.action = outputs_result[0]
        
        target_dof_pos_lab = self.action * self.action_scale + self.default_angles_lab
        target_dof_pos_mj = np.zeros(29)
        target_dof_pos_mj[self.mj2lab] = target_dof_pos_lab.squeeze(0)
        
        self.policy_output.actions = target_dof_pos_mj
        self.policy_output.kps[self.mj2lab] = self.kps_lab
        self.policy_output.kds[self.mj2lab] = self.kds_lab
        
        # update motion phase
        self.counter_step += 1
        
        if self.counter_step >= self.motion_length - 1:
            self.state_cmd.skill_cmd = FSMCommand.LOCO

    def exit(self):
        self.action.fill(0.0)
        self.counter_step = 0
        print("exited")

    
    def checkChange(self):
        if(self.state_cmd.skill_cmd == FSMCommand.LOCO):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_COOLDOWN
        elif(self.state_cmd.skill_cmd == FSMCommand.PASSIVE):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.PASSIVE
        elif(self.state_cmd.skill_cmd == FSMCommand.POS_RESET):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE
        else:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.SKILL_GAE
