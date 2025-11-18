# Place motion NPZ files here
# Expected files:
# - walk1_subject2.npz (or update motion_path in config/GAE_Mimic.yaml)
# 
# Motion data format should include:
# - joint_pos: (num_frames, num_joints)
# - joint_vel: (num_frames, num_joints)
# - body_pos_w: (num_frames, num_bodies, 3)
# - body_quat_w: (num_frames, num_bodies, 4)
# - body_lin_vel_w: (num_frames, num_bodies, 3)
# - body_ang_vel_w: (num_frames, num_bodies, 3)
