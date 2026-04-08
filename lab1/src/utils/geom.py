import math
import numpy as np
from typing import Tuple

def wrap_pi(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (theta + math.pi) % (2 * math.pi) - math.pi

def se2_compose(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Compose SE(2):  T = A ⊕ B
    a = (x,y,theta), b = (dx,dy,dtheta) measured in frame of A.
    """
    ax, ay, ath = a
    bx, by, bth = b
    c = math.cos(ath)
    s = math.sin(ath)
    x = ax + c * bx - s * by
    y = ay + s * bx + c * by
    th = wrap_pi(ath + bth)
    return x, y, th

def invert_measurement(m: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Invert SE(2) relative transform z_ij to z_ji.
    If z_ij = (dx,dy,dth), then z_ji = inv(z_ij).
    """
    dx, dy, dth = m
    c = math.cos(dth)
    s = math.sin(dth)
    # inv rotation = -dth, inv translation = -R(-dth) * t = -R^T * t
    inv_dx = -(c * dx + s * dy)
    inv_dy = -(-s * dx + c * dy)
    inv_dth = wrap_pi(-dth)
    return inv_dx, inv_dy, inv_dth

def se2_apply(pose: Tuple[float, float, float], pts: np.ndarray) -> np.ndarray:
    """Apply SE2 pose (x,y,theta) to Nx2 points."""
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (pts @ R.T) + np.array([x, y], dtype=np.float64)

def pano_xy_to_u_v(x: float, y: float, W: int, H: int) -> Tuple[float, float]:
    u = ((x + 0.5) / W - 0.5) * 2.0 * math.pi
    v = -((y + 0.5) / H - 0.5) * math.pi
    return u, v

def ray_from_uv(u: float, v: float) -> Tuple[float, float, float]:
    cu, su = math.cos(u), math.sin(u)
    cv, sv = math.cos(v), math.sin(v)
    return (cv * cu, cv * su, sv)

def intersect_with_z_plane(dir3: Tuple[float, float, float], z_plane: float = -1.0) -> Tuple[float, float, float]:
    dz = dir3[2]
    if abs(dz) < 1e-8:
        return None
    t = z_plane / dz
    if t <= 0:
        return None
    return (t * dir3[0], t * dir3[1], t * dir3[2])
