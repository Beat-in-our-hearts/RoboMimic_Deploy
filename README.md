<div align="center">
  <h1 align="center">RoboMimic Deploy - Enhanced Edition</h1>
  <p align="center">
    <a href="README_zh.md">🇨🇳 中文</a> | <span>🌎 English</span>
  </p>
</div>

<p align="center">
  <strong>A multi-policy robot deployment framework based on state-switching mechanism for Unitree G1 robot (29-DoF), with enhanced GAE_Mimic motion tracking capabilities.</strong>
</p>

---

## 📢 About This Fork

This repository is an enhanced version based on the excellent work from [ccrpRepo/RoboMimic_Deploy](https://github.com/ccrpRepo/RoboMimic_Deploy). 

### 🙏 Acknowledgments

We sincerely thank the original authors for their outstanding contribution to the robotics community. Their work has provided a solid foundation for multi-policy deployment on humanoid robots.

**Original Repository**: [https://github.com/ccrpRepo/RoboMimic_Deploy](https://github.com/ccrpRepo/RoboMimic_Deploy)

---

## ✨ What's New - GAE_Mimic Integration

We have integrated **GAE_Mimic** (Generalized Action Encoding Mimic), a state-of-the-art motion tracking and imitation learning policy, into this deployment framework.

### 🎯 GAE_Mimic Features

- **Motion Retargeting**: Advanced motion tracking from reference trajectories
- **Generalized Encoding**: Robust motion representation using quaternion-based transformations
- **Real-time Execution**: ONNX model inference for efficient deployment
- **Multi-motion Support**: Compatible with various motion datasets (LAFAN1, etc.)

### 🚀 Quick Start - GAE_Mimic

#### 1. File Preparation

Before running GAE_Mimic, you need to manually place the following files:

**ONNX Model**:
- Location: `policy/gae_mimic/model/`
- Default filename: `policy.onnx`

**Motion Data**:
- Location: `policy/gae_mimic/motion/lafan1/`
- Default filename: `walk1_subject2.npz`
- Required data format:
  - `joint_pos`: (num_frames, num_joints)
  - `joint_vel`: (num_frames, num_joints)
  - `body_pos_w`: (num_frames, num_bodies, 3)
  - `body_quat_w`: (num_frames, num_bodies, 4)
  - `body_lin_vel_w`: (num_frames, num_bodies, 3)
  - `body_ang_vel_w`: (num_frames, num_bodies, 3)

#### 2. Triggering GAE_Mimic

**In MuJoCo Simulation**:
```bash
python deploy_mujoco/deploy_mujoco.py
# Press B + L1 on Xbox controller
```

**Keyboard Control (No Joystick)**:
```bash
python deploy_mujoco/deploy_mujoco_keyboard_input.py
# Type command: b+l1
```

**On Real Robot**:
```bash
python deploy_real/deploy_real.py
# Press B + L1 on controller
```

#### 3. Keyboard Commands Reference

| Command | Function | Description |
|---------|----------|-------------|
| `l3` | Passive mode | Damping protection |
| `start` | Position reset | Reset to default pose |
| `a+r1` | Locomotion mode | Walking mode |
| `x+r1` | Skill 1 | Dance |
| `y+r1` | Skill 2 | KungFu |
| `b+r1` | Skill 3 | Kick |
| `y+l1` | Skill 4 | BeyondMimic |
| `b+l1` | **Skill GAE** | **GAE_Mimic** ⭐ |
| `vel x y z` | Set velocity | e.g., `vel 0.5 0 0.2` |
| `exit` | Exit program | Quit simulation |

---

## 📚 Documentation

For detailed installation, configuration, and usage instructions, please refer to:

- **[English Tutorial](TUTORIAL.md)** - Complete setup and operation guide
- **[中文教程](TUTORIAL_zh.md)** - 完整安装和使用指南
- **[GAE_Mimic Migration Notes](policy/gae_mimic/MIGRATION_NOTES.md)** - Technical details of GAE_Mimic integration

---

## 🏗️ Project Structure

```
RoboMimic_Deploy/
├── policy/
│   ├── passive/              # Passive damping mode
│   ├── fixedpose/            # Fixed position reset
│   ├── loco_mode/            # Locomotion policy
│   ├── dance/                # Charleston dance
│   ├── kungfu/               # Martial arts motion
│   ├── kungfu2/              # Alternative kungfu
│   ├── kick/                 # Kick motion
│   ├── beyond_mimic/         # BeyondMimic tracking
│   └── gae_mimic/            # ⭐ GAE_Mimic tracking (NEW)
│       ├── config/           # Configuration files
│       ├── model/            # ONNX models (user-provided)
│       └── motion/           # Motion data (user-provided)
├── FSM/                      # Finite State Machine controller
├── deploy_mujoco/            # MuJoCo simulation deployment
├── deploy_real/              # Real robot deployment
└── common/                   # Shared utilities
```

---

## 🛠️ Supported Policies

| Policy Name | Description | Status |
|-------------|-------------|--------|
| **PassiveMode** | Damping protection mode | ✅ Stable |
| **FixedPose** | Position control reset | ✅ Stable |
| **LocoMode** | Stable walking | ✅ Stable |
| **Dance** | Charleston dance | ✅ Verified on real robot |
| **KungFu** | Martial arts | ⚠️ Simulation only |
| **KungFu2** | Alternative kungfu | ⚠️ Simulation only |
| **Kick** | Kick motion | ⚠️ Simulation only |
| **BeyondMimic** | Motion tracking | ⚠️ Experimental |
| **GAE_Mimic** | Advanced motion tracking | ⭐ NEW |

---

## ⚠️ Important Notes

### Robot Compatibility
- This framework is designed for **Unitree G1 robots with 3-DOF waist**
- If waist fixing bracket is installed, unlock it following official instructions
- **Remove hands** to avoid collision during dance movements

### Safety Guidelines
1. **Test in simulation first** before deploying on real robot
2. Press `F1` or `Select` for emergency stop (Passive Mode)
3. Charleston Dance (R1+X) is the most stable policy for real robot
4. Other motions are **recommended for simulation only**

### Known Limitations
- Not compatible with Orin NX platform directly (use unitree_sdk2 + ROS instead)
- Mimic policies may not guarantee 100% success on complex terrains
- Manual stabilization may be needed at dance start/end

---

## 🔧 Installation

### Quick Start

```bash
# Create virtual environment
conda create -n robomimic python=3.8
conda activate robomimic

# Install PyTorch
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# Clone repository
git clone https://github.com/YOUR_USERNAME/RoboMimic_Deploy.git
cd RoboMimic_Deploy

# Install dependencies
pip install numpy==1.20.0
pip install onnx onnxruntime

# Install Unitree SDK
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

For detailed instructions, see [TUTORIAL.md](TUTORIAL.md).

---

## 🎥 Video Tutorial

[Watch the video tutorial](https://www.bilibili.com/video/BV1VTKHzSE6C/?vd_source=713b35f59bdf42930757aea07a44e7cb#reply114743994027967)

---

## 📝 License

This project maintains the same license as the original repository.

---

## 🤝 Contributing

We welcome contributions! If you find issues or want to add new features, please:

1. Fork the repository
2. Create your feature branch
3. Submit a pull request

---

## 📧 Contact

For questions or support regarding:
- **Original framework**: Refer to [ccrpRepo/RoboMimic_Deploy](https://github.com/ccrpRepo/RoboMimic_Deploy)
- **GAE_Mimic enhancement**: Open an issue in this repository

---

<div align="center">
  <sub>Built with ❤️ based on the outstanding work from ccrpRepo</sub>
</div>
