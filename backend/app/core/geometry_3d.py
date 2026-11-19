"""3D Computer Vision & Metric Geometry engine for MyLeenTo3.

Implements exact coordinate transformations between Google Gemini's camera frame
(X-Right, Y-Forward/depth, Z-Up), OpenCV pinhole projection frame (X-Right,
Y-Down, Z-Forward), and Three.js right-handed world frame (X-Right, Y-Up,
Z-Backward), along with exact 2D/BEV/3D OBB IoU, 15-axis SAT collision,
surface clearance optimization, and Kay-Kajiya ray-OBB occlusion analysis.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

# Chirality-preserving basis change Rx(-90 deg): det == +1.0
# Gemini (X-Right, Y-Forward, Z-Up) -> Three.js (X-Right, Y-Up, Z-Backward)
BASIS_GEMINI_TO_THREE: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=float,
)

# View matrix Rx(+90 deg): Gemini (X-Right, Y-Forward, Z-Up) -> OpenCV (X-Right, Y-Down, Z-Forward)
R_VIEW_GEMINI_TO_OPENCV: np.ndarray = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=float,
)

OBB_EDGES: list[list[int]] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]


def euler_deg_to_quaternion(
    roll_deg: float, pitch_deg: float, yaw_deg: float
) -> tuple[float, float, float, float]:
    """Convert intrinsic ZYX Euler angles (roll, pitch, yaw) in degrees to (qx, qy, qz, qw)."""
    r = math.radians(float(roll_deg))
    p = math.radians(float(pitch_deg))
    y = math.radians(float(yaw_deg))

    sr, sp, sy = math.sin(r / 2.0), math.sin(p / 2.0), math.sin(y / 2.0)
    cr, cp, cz = math.cos(r / 2.0), math.cos(p / 2.0), math.cos(y / 2.0)

    qx = sr * cp * cz - cr * sp * sy
    qy = cr * sp * cz + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cz
    qw = cr * cp * cz + sr * sp * sy

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm > 0.0:
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    return (float(qx), float(qy), float(qz), float(qw))


def quaternion_to_rotation_matrix(
    q: Sequence[float],
) -> np.ndarray:
    """Convert unit quaternion (qx, qy, qz, qw) to a 3x3 rotation matrix."""
    qx, qy, qz, qw = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0.0:
        return np.eye(3, dtype=float)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    xw, yw, zw = qx * qw, qy * qw, qz * qw

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - zw), 2.0 * (xz + yw)],
            [2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - xw)],
            [2.0 * (xz - yw), 2.0 * (yz + xw), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 proper rotation matrix to canonical unit quaternion (qx, qy, qz, qw)."""
    mat = np.asarray(R, dtype=float)
    if mat.shape != (3, 3):
        raise ValueError(f"Expected (3, 3) rotation matrix, got {mat.shape}")

    tr = float(mat[0, 0] + mat[1, 1] + mat[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (mat[2, 1] - mat[1, 2]) / s
        qy = (mat[0, 2] - mat[2, 0]) / s
        qz = (mat[1, 0] - mat[0, 1]) / s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = math.sqrt(max(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2], 1e-16)) * 2.0
        qw = (mat[2, 1] - mat[1, 2]) / s
        qx = 0.25 * s
        qy = (mat[0, 1] + mat[1, 0]) / s
        qz = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = math.sqrt(max(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2], 1e-16)) * 2.0
        qw = (mat[0, 2] - mat[2, 0]) / s
        qx = (mat[0, 1] + mat[1, 0]) / s
        qy = 0.25 * s
        qz = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1], 1e-16)) * 2.0
        qw = (mat[1, 0] - mat[0, 1]) / s
        qx = (mat[0, 2] + mat[2, 0]) / s
        qy = (mat[1, 2] + mat[2, 1]) / s
        qz = 0.25 * s

    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm > 0.0:
        qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    # Canonical hemisphere sign for determinism
    if qw < 0.0 or (
        abs(qw) < 1e-12
        and (
            qx < 0.0
            or (abs(qx) < 1e-12 and (qy < 0.0 or (abs(qy) < 1e-12 and qz < 0.0)))
        )
    ):
        qx, qy, qz, qw = -qx, -qy, -qz, -qw

    return (float(qx), float(qy), float(qz), float(qw))


def compute_fov_v_deg(fov_h_deg: float, width: int, height: int) -> float:
    """Compute vertical field of view in degrees from horizontal FOV and aspect ratio."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive integers")
    half_h_rad = math.radians(float(fov_h_deg)) / 2.0
    return float(2.0 * math.degrees(math.atan(math.tan(half_h_rad) * (height / width))))


def _box_params(
    box_3d: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack box_3d into (center[3], half_extents[3], R_gem[3x3])."""
    if len(box_3d) != 9:
        raise ValueError(f"box_3d must have 9 elements, got {len(box_3d)}")
    x, y, z, dx, dy, dz, roll, pitch, yaw = [float(v) for v in box_3d]
    center = np.array([x, y, z], dtype=float)
    half_extents = np.array([abs(dx) / 2.0, abs(dy) / 2.0, abs(dz) / 2.0], dtype=float)
    q = euler_deg_to_quaternion(roll, pitch, yaw)
    R_gem = quaternion_to_rotation_matrix(q)
    return center, half_extents, R_gem


def obb_corners_gemini(box_3d: Sequence[float]) -> np.ndarray:
    """Compute the 8 oriented bounding box corners in Gemini camera coordinates.

    Vertex ordering:
      0..3: top face (+dz/2) in perimeter loop
      4..7: bottom face (-dz/2) matching vertical pillars (0-4, 1-5, 2-6, 3-7)
    """
    center, half, R_gem = _box_params(box_3d)
    hx, hy, hz = float(half[0]), float(half[1]), float(half[2])

    raw = [
        [xi, yi, zi]
        for xi in (-hx, hx)
        for yi in (-hy, hy)
        for zi in (-hz, hz)
    ]
    ordered_local = np.array(
        [raw[1], raw[3], raw[7], raw[5], raw[0], raw[2], raw[6], raw[4]],
        dtype=float,
    )
    return (R_gem @ ordered_local.T).T + center


def gemini_to_threejs_pose(box_3d: Sequence[float]) -> dict[str, Any]:
    """Map a Gemini 9-DOF box_3d into Three.js position, quaternion, size, corners & heading."""
    center, half, R_gem = _box_params(box_3d)
    M = BASIS_GEMINI_TO_THREE

    position_three = (M @ center).tolist()
    R_three = M @ R_gem @ M.T
    quat_three = list(rotation_matrix_to_quaternion(R_three))

    dx, dy, dz = float(half[0] * 2.0), float(half[1] * 2.0), float(half[2] * 2.0)
    size_three = [dx, dz, dy]

    corners_gem = obb_corners_gemini(box_3d)
    corners_three = (M @ corners_gem.T).T.tolist()

    fwd = M @ R_gem @ np.array([0.0, 1.0, 0.0], dtype=float)
    fwd_norm = float(np.linalg.norm(fwd))
    if fwd_norm > 0.0:
        fwd = fwd / fwd_norm

    return {
        "position": [float(v) for v in position_three],
        "quaternion": [float(v) for v in quat_three],
        "size": size_three,
        "corners_three": [[float(c) for c in pt] for pt in corners_three],
        "forward_dir_three": [float(v) for v in fwd.tolist()],
    }


def project_obb_to_2d(
    box_3d: Sequence[float], width: int, height: int, fov_h_deg: float
) -> dict[str, Any]:
    """Project a Gemini 3D OBB onto a 2D pinhole image plane."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    center, half, R_gem = _box_params(box_3d)
    corners_gem = obb_corners_gemini(box_3d)

    f = float(width) / (2.0 * math.tan(math.radians(float(fov_h_deg)) / 2.0))
    cx = float(width) / 2.0
    cy = float(height) / 2.0

    def _project_point(pt_gem: np.ndarray) -> tuple[float, float, float]:
        pt_cv = R_VIEW_GEMINI_TO_OPENCV @ pt_gem
        raw_z = float(pt_cv[2])
        z_cv = max(raw_z, 0.05)
        u = f * (float(pt_cv[0]) / z_cv) + cx
        v = f * (float(pt_cv[1]) / z_cv) + cy
        return u, v, raw_z

    vertices_2d: list[list[float]] = []
    vertices_norm: list[list[float]] = []
    raw_zs: list[float] = []

    for corner in corners_gem:
        u, v, raw_z = _project_point(corner)
        vertices_2d.append([float(u), float(v)])
        vertices_norm.append([float((v / height) * 1000.0), float((u / width) * 1000.0)])
        raw_zs.append(raw_z)

    u_c, v_c, raw_z_c = _project_point(center)
    center_2d = [float(u_c), float(v_c)]

    forward_tip_gem = center + R_gem @ np.array([0.0, max(float(half[1]) * 1.2, 0.08), 0.0])
    u_tip, v_tip, _ = _project_point(forward_tip_gem)
    front_arrow_2d = [center_2d, [float(u_tip), float(v_tip)]]

    behind_camera = any(rz <= 0.05 for rz in raw_zs) or (raw_z_c <= 0.05)

    us = [pt[0] for pt in vertices_2d]
    vs = [pt[1] for pt in vertices_2d]
    min_u, max_u = min(us), max(us)
    min_v, max_v = min(vs), max(vs)

    # Top face horizontal center, anchored at highest projected vertex
    top_u = sum(vertices_2d[i][0] for i in range(4)) / 4.0
    label_anchor_2d = [float(top_u), float(min_v)]

    ymin = float(np.clip((min_v / height) * 1000.0, 0.0, 1000.0))
    xmin = float(np.clip((min_u / width) * 1000.0, 0.0, 1000.0))
    ymax = float(np.clip((max_v / height) * 1000.0, 0.0, 1000.0))
    xmax = float(np.clip((max_u / width) * 1000.0, 0.0, 1000.0))

    return {
        "vertices_2d": vertices_2d,
        "vertices_norm": vertices_norm,
        "edges": [list(e) for e in OBB_EDGES],
        "center_2d": center_2d,
        "label_anchor_2d": label_anchor_2d,
        "front_arrow_2d": front_arrow_2d,
        "box_2d_reproj": [ymin, xmin, ymax, xmax],
        "behind_camera": bool(behind_camera),
        "min_depth_m": float(min(raw_zs)),
        "max_depth_m": float(max(raw_zs)),
    }


# ---------------------------------------------------------------------------
# 2D Polygon / BEV IoU (Andrew's convex hull + Sutherland-Hodgman clipping)
# ---------------------------------------------------------------------------


def _cross2d(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull_ccw(points: np.ndarray) -> list[tuple[float, float]]:
    """Compute the strict CCW 2D convex hull of a set of 2D points."""
    pts = sorted({(round(float(p[0]), 12), round(float(p[1]), 12)) for p in points})
    if len(pts) <= 2:
        return pts

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross2d(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)

    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross2d(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def _polygon_area(poly: list[tuple[float, float]]) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    area2 = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) * 0.5


def _sutherland_hodgman_clip(
    subject_poly: list[tuple[float, float]],
    clip_poly: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Exact Sutherland-Hodgman clipping of subject_poly against CCW convex clip_poly."""
    if len(subject_poly) < 3 or len(clip_poly) < 3:
        return []

    output = list(subject_poly)
    n_clip = len(clip_poly)
    for i in range(n_clip):
        cp1 = clip_poly[i]
        cp2 = clip_poly[(i + 1) % n_clip]
        input_list = output
        output = []
        if not input_list:
            break

        s = input_list[-1]
        for e in input_list:
            s_in = _cross2d(cp1, cp2, s) >= -1e-12
            e_in = _cross2d(cp1, cp2, e) >= -1e-12

            def _intersect(
                p1: tuple[float, float], p2: tuple[float, float]
            ) -> tuple[float, float]:
                dcx, dcy = cp1[0] - cp2[0], cp1[1] - cp2[1]
                dpx, dpy = p1[0] - p2[0], p1[1] - p2[1]
                denom = dcx * dpy - dcy * dpx
                if abs(denom) < 1e-15:
                    return p2
                n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
                n2 = p1[0] * p2[1] - p1[1] * p2[0]
                x = (n1 * dpx - n2 * dcx) / denom
                y = (n1 * dpy - n2 * dcy) / denom
                return (x, y)

            if e_in:
                if not s_in:
                    output.append(_intersect(s, e))
                output.append(e)
            elif s_in:
                output.append(_intersect(s, e))
            s = e

    return output


def _xy_yaw_rect_ccw(box_3d: Sequence[float]) -> list[tuple[float, float]]:
    """Exact CCW 4-vertex XY footprint using (x, y, dx, dy, yaw)."""
    x, y, _z, dx, dy, _dz, _r, _p, yaw_deg = [float(v) for v in box_3d]
    hx, hy = abs(dx) / 2.0, abs(dy) / 2.0
    cy = math.cos(math.radians(yaw_deg))
    sy = math.sin(math.radians(yaw_deg))
    local_ccw = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    return [(x + cy * lx - sy * ly, y + sy * lx + cy * ly) for lx, ly in local_ccw]


def _bev_footprint_poly(box_3d: Sequence[float]) -> list[tuple[float, float]]:
    """Return CCW convex polygon representing the box's XY BEV footprint."""
    roll, pitch = float(box_3d[6]), float(box_3d[7])
    if abs(roll) < 1e-6 and abs(pitch) < 1e-6:
        return _xy_yaw_rect_ccw(box_3d)
    corners_xy = obb_corners_gemini(box_3d)[:, :2]
    return _convex_hull_ccw(corners_xy)


def compute_bev_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute exact Bird's-Eye-View (XY) IoU via Sutherland-Hodgman polygon clipping."""
    poly_a = _bev_footprint_poly(box_a)
    poly_b = _bev_footprint_poly(box_b)
    area_a = _polygon_area(poly_a)
    area_b = _polygon_area(poly_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    inter_poly = _sutherland_hodgman_clip(poly_a, poly_b)
    inter_area = _polygon_area(inter_poly)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(np.clip(inter_area / union_area, 0.0, 1.0))


# ---------------------------------------------------------------------------
# 15-Axis Separating Axis Theorem (SAT) & 3D OBB IoU
# ---------------------------------------------------------------------------


def obb_intersects_sat(box_a: Sequence[float], box_b: Sequence[float]) -> bool:
    """Exact 15-axis Separating Axis Theorem (SAT) intersection test for two 3D OBBs."""
    ca, ha, Ra = _box_params(box_a)
    cb, hb, Rb = _box_params(box_b)

    t = cb - ca
    axes_a = [Ra[:, i] for i in range(3)]
    axes_b = [Rb[:, i] for i in range(3)]

    candidate_axes: list[np.ndarray] = list(axes_a) + list(axes_b)
    for ua in axes_a:
        for ub in axes_b:
            cross = np.cross(ua, ub)
            norm = float(np.linalg.norm(cross))
            if norm > 1e-9:
                candidate_axes.append(cross / norm)

    for axis in candidate_axes:
        proj_a = sum(float(ha[i]) * abs(float(np.dot(axis, axes_a[i]))) for i in range(3))
        proj_b = sum(float(hb[i]) * abs(float(np.dot(axis, axes_b[i]))) for i in range(3))
        dist = abs(float(np.dot(t, axis)))
        if dist > proj_a + proj_b + 1e-9:
            return False
    return True


def compute_3d_iou(
    box_a: Sequence[float], box_b: Sequence[float], seed: int = 42
) -> float:
    """Compute 3D OBB IoU.

    Uses exact XY-polygon * Z-overlap prism formula when both boxes have
    |roll| < 2 deg and |pitch| < 2 deg; otherwise performs 15-axis SAT early-exit
    followed by deterministic stratified OBB Monte Carlo integration.
    """
    ca, ha, Ra = _box_params(box_a)
    cb, hb, Rb = _box_params(box_b)

    vol_a = float(8.0 * ha[0] * ha[1] * ha[2])
    vol_b = float(8.0 * hb[0] * hb[1] * hb[2])
    if vol_a <= 0.0 or vol_b <= 0.0:
        return 0.0

    # Upright prism fast-exact path
    if (
        abs(float(box_a[6])) < 2.0
        and abs(float(box_a[7])) < 2.0
        and abs(float(box_b[6])) < 2.0
        and abs(float(box_b[7])) < 2.0
    ):
        z_min = max(float(ca[2] - ha[2]), float(cb[2] - hb[2]))
        z_max = min(float(ca[2] + ha[2]), float(cb[2] + hb[2]))
        z_ov = max(0.0, z_max - z_min)
        if z_ov <= 0.0:
            return 0.0

        poly_a = _xy_yaw_rect_ccw(box_a)
        poly_b = _xy_yaw_rect_ccw(box_b)
        inter_area = _polygon_area(_sutherland_hodgman_clip(poly_a, poly_b))
        inter_vol = inter_area * z_ov
        union_vol = vol_a + vol_b - inter_vol
        if union_vol <= 0.0:
            return 0.0
        return float(np.clip(inter_vol / union_vol, 0.0, 1.0))

    # General 9-DOF path: exact 15-axis SAT early exit
    if not obb_intersects_sat(box_a, box_b):
        return 0.0

    # Exact identity shortcut
    if (
        np.allclose(ca, cb, atol=1e-9)
        and np.allclose(ha, hb, atol=1e-9)
        and np.allclose(Ra, Rb, atol=1e-9)
    ):
        return 1.0

    # Deterministic stratified sampling inside the smaller OBB (20^3 = 8,000 strata)
    if vol_a <= vol_b:
        c_s, h_s, R_s, v_s = ca, ha, Ra, vol_a
        c_o, h_o, R_o = cb, hb, Rb
    else:
        c_s, h_s, R_s, v_s = cb, hb, Rb, vol_b
        c_o, h_o, R_o = ca, ha, Ra

    n = 20
    rng = np.random.default_rng(int(seed))
    grid = np.meshgrid(
        np.arange(n, dtype=float),
        np.arange(n, dtype=float),
        np.arange(n, dtype=float),
        indexing="ij",
    )
    base = np.stack([g.ravel() for g in grid], axis=1)  # (8000, 3)
    jitter = rng.random(base.shape)
    u = (base + jitter) / float(n) - 0.5  # in [-0.5, 0.5]^3
    local_pts_s = u * (2.0 * h_s)

    world_pts = local_pts_s @ R_s.T + c_s
    local_pts_o = (world_pts - c_o) @ R_o  # R_o.T @ (world - c_o)

    inside = np.all(np.abs(local_pts_o) <= (h_o + 1e-12), axis=1)
    frac = float(np.mean(inside))
    inter_vol = frac * v_s
    union_vol = vol_a + vol_b - inter_vol
    if union_vol <= 0.0:
        return 0.0
    return float(np.clip(inter_vol / union_vol, 0.0, 1.0))


def compute_box_2d_iou(b1: Sequence[float], b2: Sequence[float]) -> float:
    """Compute 2D axis-aligned IoU for boxes in [ymin, xmin, ymax, xmax] format."""
    y1 = max(float(b1[0]), float(b2[0]))
    x1 = max(float(b1[1]), float(b2[1]))
    y2 = min(float(b1[2]), float(b2[2]))
    x2 = min(float(b1[3]), float(b2[3]))

    inter = max(0.0, y2 - y1) * max(0.0, x2 - x1)
    a1 = max(0.0, float(b1[2]) - float(b1[0])) * max(0.0, float(b1[3]) - float(b1[1]))
    a2 = max(0.0, float(b2[2]) - float(b2[0])) * max(0.0, float(b2[3]) - float(b2[1]))
    union = a1 + a2 - inter
    if union <= 0.0:
        return 0.0
    return float(np.clip(inter / union, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Pairwise Surface Clearance & Closest-Point Geometry
# ---------------------------------------------------------------------------


def _closest_points_between_obbs(
    box_a: Sequence[float], box_b: Sequence[float]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find exact closest points (p_a, p_b) in Gemini coordinates and Euclidean distance.

    Formulates the problem as a convex box-constrained QP over the 6 local OBB
    coordinates u in [-ha, ha], v in [-hb, hb] solved to machine precision via L-BFGS-B.
    """
    ca, ha, Ra = _box_params(box_a)
    cb, hb, Rb = _box_params(box_b)

    def _obj_and_grad(z: np.ndarray) -> tuple[float, np.ndarray]:
        u = z[:3]
        v = z[3:]
        pa = ca + Ra @ u
        pb = cb + Rb @ v
        diff = pa - pb
        val = 0.5 * float(np.dot(diff, diff))
        grad_u = Ra.T @ diff
        grad_v = -Rb.T @ diff
        return val, np.concatenate([grad_u, grad_v])

    # Warm start: project each center into the opposite OBB
    u0 = np.clip(Ra.T @ (cb - ca), -ha, ha)
    v0 = np.clip(Rb.T @ (ca - cb), -hb, hb)
    z0 = np.concatenate([u0, v0])

    bounds = [
        (-float(ha[0]), float(ha[0])),
        (-float(ha[1]), float(ha[1])),
        (-float(ha[2]), float(ha[2])),
        (-float(hb[0]), float(hb[0])),
        (-float(hb[1]), float(hb[1])),
        (-float(hb[2]), float(hb[2])),
    ]

    res = minimize(
        _obj_and_grad,
        z0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"ftol": 1e-15, "gtol": 1e-12, "maxiter": 200},
    )
    u_opt = np.clip(res.x[:3], -ha, ha)
    v_opt = np.clip(res.x[3:], -hb, hb)
    pa = ca + Ra @ u_opt
    pb = cb + Rb @ v_opt
    dist = float(np.linalg.norm(pa - pb))
    return pa, pb, dist


def compute_pairwise_geometry(
    box_a: Sequence[float], box_b: Sequence[float]
) -> dict[str, Any]:
    """Compute exact pairwise spatial relationships between two 9-DOF OBBs."""
    ca, _ha, _Ra = _box_params(box_a)
    cb, _hb, _Rb = _box_params(box_b)

    delta = cb - ca
    center_dist = float(np.linalg.norm(delta))
    colliding = obb_intersects_sat(box_a, box_b)
    pa_gem, pb_gem, min_dist = _closest_points_between_obbs(box_a, box_b)

    surface_clearance = 0.0 if colliding else float(max(0.0, min_dist))
    M = BASIS_GEMINI_TO_THREE
    pa_three = (M @ pa_gem).tolist()
    pb_three = (M @ pb_gem).tolist()

    return {
        "center_distance_m": center_dist,
        "surface_clearance_m": surface_clearance,
        "is_colliding": bool(colliding),
        "iou_3d": compute_3d_iou(box_a, box_b),
        "iou_bev": compute_bev_iou(box_a, box_b),
        "closest_point_a_three": [float(v) for v in pa_three],
        "closest_point_b_three": [float(v) for v in pb_three],
        "centroid_delta_m": [float(v) for v in delta.tolist()],
    }


# ---------------------------------------------------------------------------
# Kay-Kajiya Ray-OBB Slab Intersection & Scene Occlusion Matrix
# ---------------------------------------------------------------------------


def _ray_obb_slab_intersect(
    ray_origin_gem: np.ndarray,
    ray_dir_gem: np.ndarray,
    center: np.ndarray,
    half: np.ndarray,
    R_gem: np.ndarray,
) -> tuple[bool, float, float]:
    """Kay-Kajiya slab intersection in local OBB coordinates.

    Returns (hit, t_enter, t_exit) along P(t) = ray_origin + t * ray_dir.
    """
    o_loc = R_gem.T @ (ray_origin_gem - center)
    d_loc = R_gem.T @ ray_dir_gem

    t_min = -math.inf
    t_max = math.inf

    for axis in range(3):
        h = float(half[axis])
        o = float(o_loc[axis])
        d = float(d_loc[axis])
        if abs(d) < 1e-12:
            if o < -h or o > h:
                return False, 0.0, 0.0
        else:
            t1 = (-h - o) / d
            t2 = (h - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
            if t_min > t_max:
                return False, 0.0, 0.0

    return True, float(t_min), float(t_max)


def compute_occlusion_matrix(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute camera-origin ray-cast occlusion matrix across scene objects.

    Casts 9 deterministic rays from camera origin (0,0,0) to the 8 OBB corners
    plus centroid of every farther object, testing slab intersection against
    nearer objects' OBBs.
    """
    if len(objects) < 2:
        return []

    M = BASIS_GEMINI_TO_THREE
    origin = np.zeros(3, dtype=float)

    precomputed: list[dict[str, Any]] = []
    for idx, obj in enumerate(objects):
        box = obj["box_3d"]
        c, h, R = _box_params(box)
        corners = obb_corners_gemini(box)
        targets = np.vstack([corners, c.reshape(1, 3)])  # 9 target points
        precomputed.append(
            {
                "id": str(obj.get("id", f"obj_{idx}")),
                "label": str(obj.get("label", "object")),
                "center": c,
                "half": h,
                "R": R,
                "depth_y": float(c[1]),
                "targets": targets,
            }
        )

    results: list[dict[str, Any]] = []
    for i, occ in enumerate(precomputed):
        for j, tgt in enumerate(precomputed):
            if i == j:
                continue
            # Occluder must lie strictly closer along camera depth Y
            depth_gap = tgt["depth_y"] - occ["depth_y"]
            if depth_gap <= 1e-4:
                continue

            occluded_rays_three: list[dict[str, list[float]]] = []
            for pt in tgt["targets"]:
                dist_target = float(np.linalg.norm(pt))
                if dist_target <= 1e-6:
                    continue
                ray_dir = pt / dist_target
                hit, t_enter, t_exit = _ray_obb_slab_intersect(
                    origin, ray_dir, occ["center"], occ["half"], occ["R"]
                )
                if not hit:
                    continue
                # Must hit occluder strictly in front of camera and strictly before target
                if t_exit > 1e-4 and t_enter < dist_target - 1e-4:
                    t_hit = max(t_enter, 1e-4)
                    if t_hit < dist_target - 1e-4:
                        hit_pt_gem = ray_dir * t_hit
                        occluded_rays_three.append(
                            {
                                "origin": [0.0, 0.0, 0.0],
                                "hit": [float(v) for v in (M @ hit_pt_gem).tolist()],
                                "target": [float(v) for v in (M @ pt).tolist()],
                            }
                        )

            hit_count = len(occluded_rays_three)
            if hit_count > 0:
                results.append(
                    {
                        "occluder_id": occ["id"],
                        "occluder_label": occ["label"],
                        "occluded_id": tgt["id"],
                        "occluded_label": tgt["label"],
                        "occlusion_ratio": float(hit_count / 9.0),
                        "depth_gap_m": float(depth_gap),
                        "occluded_rays_three": occluded_rays_three,
                    }
                )

    results.sort(key=lambda r: (-r["occlusion_ratio"], r["depth_gap_m"], r["occluder_id"], r["occluded_id"]))
    return results
