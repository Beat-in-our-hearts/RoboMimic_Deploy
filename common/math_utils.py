import numpy as np

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions.
    Args:
        q1: First quaternion as a numpy array [w, x, y, z].
        q2: Second quaternion as a numpy array [w, x, y, z].
    Returns:
        The resulting quaternion as a numpy array [w, x, y, z].
    """
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

def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """Convert a quaternion to a rotation matrix.
    Args:
        q: Quaternion as a numpy array [w, x, y, z].
    Returns:
        A 3x3 rotation matrix as a numpy array.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ])

def yaw_quat(q: np.ndarray) -> np.ndarray:
    """Extract yaw component from a quaternion.
    Args:
        q: Quaternion as a numpy array [w, x, y, z].
    Returns:
        A quaternion representing only the yaw rotation as a numpy array [w, x, y, z].
    """
    w, x, y, z = q
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


def euler_single_axis_to_quat(angle: float, axis: str | np.ndarray, degrees: bool = False) -> np.ndarray:
    """Convert a single-axis Euler rotation to a quaternion.
    Args:
        angle: Rotation angle around the specified axis.
        axis: Axis of rotation ('x', 'y', 'z') or a 3D unit vector.
        degrees: If True, the angle is in degrees; otherwise, in radians.
    Returns:
        A quaternion as a numpy array [w, x, y, z].
    """
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
    
def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """Calculate the gravity orientation vector from a quaternion.
    Args:
        quaternion: A quaternion as a numpy array [w, x, y, z].
    Returns:
        A 3D gravity orientation vector as a numpy array.
    """
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation

def compute_projected_gravity(quat_w: np.ndarray, gravity_w: np.ndarray = None) -> np.ndarray:
    """Compute the projected gravity vector in the body frame given the world quaternion.
    Args:
        quat_w: Quaternion representing the body's orientation in the world frame as a numpy array [w, x, y, z].
        gravity_w: Gravity vector in the world frame as a numpy array. Defaults to [0, 0, -1] if None.
    Returns:
        Projected gravity vector in the body frame as a numpy array.
    """
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