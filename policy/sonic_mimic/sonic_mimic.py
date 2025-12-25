from common.path_config import PROJECT_ROOT
from common.math_utils import *
from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput
import numpy as np
import yaml
from common.utils import FSMCommand, progress_bar
import onnx
import onnxruntime
import torch
import os


class SONIC_Mimic(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.SKILL_SONIC
        self.name_str = "sonic_mimic"
        self.counter_step = 0
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "SONIC_Mimic.yaml")
        
        # Check if config file exists
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please ensure SONIC_Mimic.yaml exists in the config directory."
            )
        
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.mode = config.get("mode", "robot")  # default to "robot" mode
            self.onnx_path = os.path.join(current_dir, "model", config["onnx_path"][self.mode])
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
            
            #
            self.cmd_interval = config["cmd_interval"]
            self.cmd_window_len = config["cmd_window_len"]

            # load policy
            self.onnx_model = onnx.load(self.onnx_path)
            self.ort_session = onnxruntime.InferenceSession(self.onnx_path)
            print("Loaded SONIC Mimic ONNX model from ", self.onnx_path)
            
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
            self.action = np.zeros(self.num_actions)
            
            print("kp_lab: ", self.kps_lab)
            print("kd_lab: ", self.kds_lab)
            print("default_angles_lab: ", self.default_angles_lab)
            print("mj_joint_names: ", self.lab_joint_names)
            print("action_scale_lab: ", self.action_scale)
            print("mj2lab", self.mj2lab)
            
            print("Sonic Mimic policy initializing ...")
    
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
            "joint_pos": data["joint_pos"],
            "joint_vel": data["joint_vel"],
            "body_pos_w": data["body_pos_w"],
            "body_quat_w": data["body_quat_w"],
            "body_lin_vel_w": data["body_lin_vel_w"],
            "body_ang_vel_w": data["body_ang_vel_w"],
            "smplx_pose_body": data["smplx_pose_body"],
            "robot_keypoints_trans": data["robot_keypoints_trans"].reshape(data["joint_pos"].shape[0], -1),
            "robot_keypoints_rot": data["robot_keypoints_rot"].reshape(data["joint_pos"].shape[0], -1),
        }
        self.motion_length = self.motion_data["joint_pos"].shape[0]
        
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
        self.action = np.zeros(self.num_actions)
        pass

    def get_robot_command(self):
        # get `num=window_len` with `interval=cmd_interval` starting from `counter_step`
        step = np.arange(0, self.cmd_window_len * self.cmd_interval, self.cmd_interval) + self.counter_step
        step = np.clip(step, 0, self.motion_length - 1)
        
        # (window_len, num_joints) 
        ref_joint_pos = self.motion_data["joint_pos"][step] # (window_len, num_joints)
        ref_joint_vel = self.motion_data["joint_vel"][step] # (window_len, num_joints)
        
        command = np.concatenate((ref_joint_pos, ref_joint_vel), axis=-1)
        return command.reshape(1, -1)
    
    def get_human_command(self):
        # get `num=window_len` with `interval=cmd_interval` starting from `counter_step`
        step = np.arange(0, self.cmd_window_len * self.cmd_interval, self.cmd_interval) + self.counter_step
        step = np.clip(step, 0, self.motion_length - 1)
        
        # (window_len, 21*6)
        command = self.motion_data["smplx_pose_body"][step] # (window_len, 21*6)
        
        return command.reshape(1, -1)
    
    def get_keypoints_command(self):
        # get `num=window_len` with `interval=cmd_interval` starting from `counter_step`
        step = np.arange(0, self.cmd_window_len * self.cmd_interval, self.cmd_interval) + self.counter_step
        step = np.clip(step, 0, self.motion_length - 1)
        
        # (window_len, num_keypoints*9)
        ref_keypoints_trans = self.motion_data["robot_keypoints_trans"][step] # (window_len, num_keypoints*3)
        ref_keypoints_rot = self.motion_data["robot_keypoints_rot"][step] # (window_len, num_keypoints*6)
        
        command = np.concatenate((ref_keypoints_trans, ref_keypoints_rot), axis=-1)
        return command.reshape(1, -1)
        
    def run(self):
        robot_quat = self.state_cmd.base_quat
        gravity_ori = self.state_cmd.gravity_ori
        
        qj = self.state_cmd.q[self.mj2lab]
        qj = (qj - self.default_angles_lab)
        self.qj_obs[:] = qj

        base_troso_yaw = qj[2]
        base_troso_roll = qj[5]
        base_troso_pitch = qj[8]
        
        quat_yaw = euler_single_axis_to_quat(base_troso_yaw, 'z', degrees=False)
        quat_roll = euler_single_axis_to_quat(base_troso_roll, 'x', degrees=False)
        quat_pitch = euler_single_axis_to_quat(base_troso_pitch, 'y', degrees=False)
        temp1 = quat_mul(quat_roll, quat_pitch)
        temp2 = quat_mul(quat_yaw, temp1)
        robot_quat = quat_mul(robot_quat, temp2)
        
        ref_anchor_ori_w = self.motion_data["body_quat_w"][self.counter_step, self.motion_anchor_id]

        if(self.counter_step < 2):
            init_to_anchor = matrix_from_quat(yaw_quat(ref_anchor_ori_w))
            world_to_anchor = matrix_from_quat(yaw_quat(robot_quat))
            self.init_to_world = world_to_anchor @ init_to_anchor.T
            print("self.init_to_world: ", self.init_to_world)
            self.counter_step += 1
            return

        motion_anchor_ori_b = matrix_from_quat(robot_quat).T @ self.init_to_world @ matrix_from_quat(ref_anchor_ori_w)

        ang_vel = self.state_cmd.ang_vel
        
        dqj = self.state_cmd.dq[self.mj2lab]
        self.dqj_obs[:] = dqj
        
        # command motion_anchor_ori_b base_ang_vel projected_gravity joint_pos joint_vel previous_action
        observation = {}
        if self.mode == "robot":
            observation[self.input_name[0]] = self.get_robot_command().astype(np.float32)
        elif self.mode == "human":
            observation[self.input_name[0]] = self.get_human_command().astype(np.float32)
        elif self.mode == "keypoints":
            observation[self.input_name[0]] = self.get_keypoints_command().astype(np.float32)
        
        observation[self.input_name[1]] = motion_anchor_ori_b[:,:2].reshape(1, -1).astype(np.float32)
        observation[self.input_name[2]] = ang_vel.reshape(1, -1).astype(np.float32)
        observation[self.input_name[3]] = gravity_ori.reshape(1, -1).astype(np.float32)
        observation[self.input_name[4]] = self.qj_obs.reshape(1, -1).astype(np.float32)
        observation[self.input_name[5]] = self.dqj_obs.reshape(1, -1).astype(np.float32)
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
            return FSMStateName.SKILL_SONIC
