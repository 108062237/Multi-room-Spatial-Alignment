#!/usr/bin/env python3
"""
Step 3: Pose Graph Optimization (PGO) using GTSAM Pose2.

Inputs:
  - edges_measurements.json (from Step 1)
  - initial_poses.json      (from Step 2)

Outputs:
  - optimized_poses.json
  - residual_report.json (before/after summary + per-edge residuals)

Usage:
  python src/03_optimize_pose_graph_gtsam.py \
    --edges  data/group/58472_Floor1/edges_measurements.json \
    --init   data/group/58472_Floor1/initial_poses.json \
    --out    data/group/58472_Floor1/optimized_poses.json \
    --report data/group/58472_Floor1/residual_report.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import gtsam

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.geom import wrap_pi


def load_json(p: Path) -> Dict[str, Any]:
    return json.loads(p.read_text())


def save_json(p: Path, data: Dict[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def pose_dict_to_pose2(d: Dict[str, float]) -> gtsam.Pose2:
    return gtsam.Pose2(float(d["x"]), float(d["y"]), float(d["theta"]))


def pose2_to_dict(p: gtsam.Pose2) -> Dict[str, float]:
    return {"x": float(p.x()), "y": float(p.y()), "theta": float(p.theta())}


def make_symbol_map(node_ids: List[str]) -> Dict[str, int]:
    node_ids_sorted = sorted(node_ids)
    return {nid: k for k, nid in enumerate(node_ids_sorted)}


def build_noise_model(sigma_xy: float, sigma_theta: float, use_robust: bool, huber_k: float):
    base = gtsam.noiseModel.Diagonal.Sigmas(np.array([sigma_xy, sigma_xy, sigma_theta], dtype=np.float64))
    if not use_robust:
        return base
    huber = gtsam.noiseModel.mEstimator.Huber.Create(huber_k)
    return gtsam.noiseModel.Robust.Create(huber, base)


def relative_error_pose2(pi: gtsam.Pose2, pj: gtsam.Pose2, measured_relative_pose: gtsam.Pose2) -> Tuple[float, float, float]:
    pred = pi.between(pj)  
    err = measured_relative_pose.between(pred)  
    return float(err.x()), float(err.y()), float(wrap_pi(err.theta()))


def align_umeyama(moving: Dict[str, gtsam.Pose2], fixed: Dict[str, gtsam.Pose2]) -> Dict[str, gtsam.Pose2]:
    """
    Aligns `moving` trajectories to `fixed` trajectories using Umeyama algorithm (SE2).
    This ensures RMSE represents structural deformation, not global gauge drift.
    """
    common_keys = list(set(moving.keys()) & set(fixed.keys()))
    if len(common_keys) < 2:
        # Not enough points, just anchor first node
        if len(common_keys) == 1:
            k = common_keys[0]
            T_align = fixed[k].between(moving[k]).inverse()
            return {nid: T_align.compose(p) for nid, p in moving.items()}
        return moving

    # Extract XY points
    P = np.array([[fixed[k].x(), fixed[k].y()] for k in common_keys]).T
    Q = np.array([[moving[k].x(), moving[k].y()] for k in common_keys]).T

    mu_P = np.mean(P, axis=1, keepdims=True)
    mu_Q = np.mean(Q, axis=1, keepdims=True)

    P_centered = P - mu_P
    Q_centered = Q - mu_Q

    H = Q_centered @ P_centered.T
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = mu_P - R @ mu_Q
    theta = math.atan2(R[1, 0], R[0, 0])

    T_align = gtsam.Pose2(float(t[0, 0]), float(t[1, 0]), float(theta))
    
    aligned_poses = {}
    for nid, p in moving.items():
        aligned_poses[nid] = T_align.compose(p)
    return aligned_poses


def compute_residual_stats(
    poses: Dict[str, gtsam.Pose2],
    edges: List[Dict[str, Any]],
    initial_poses: Optional[Dict[str, gtsam.Pose2]] = None
) -> Dict[str, Any]:
    
    # Optional Umeyama Alignment to prevent global drift inflating RMSE
    eval_poses = poses
    if initial_poses is not None:
        eval_poses = align_umeyama(poses, initial_poses)

    per_edge = []
    trans_errs = []
    rot_errs = []

    for e in edges:
        i, j = e["i"], e["j"]
        z = e["measurement"]
        measured_relative_pose = gtsam.Pose2(float(z["dx"]), float(z["dy"]), float(z["dtheta"]))

        pi = eval_poses[i]
        pj = eval_poses[j]
        ex, ey, eth = relative_error_pose2(pi, pj, measured_relative_pose)

        trans = math.sqrt(ex * ex + ey * ey)
        rot = abs(eth)

        per_edge.append({
            "i": i,
            "j": j,
            "residual": {"dx": ex, "dy": ey, "dtheta": eth},
            "trans_l2": trans,
            "rot_abs": rot,
            "meta": e.get("meta", {}),
        })
        trans_errs.append(trans)
        rot_errs.append(rot)

    trans_arr = np.array(trans_errs, dtype=np.float64) if trans_errs else np.zeros((0,))
    rot_arr = np.array(rot_errs, dtype=np.float64) if rot_errs else np.zeros((0,))

    def safe_stats(arr: np.ndarray):
        if arr.size == 0:
            return {"rmse": 0.0, "mean": 0.0, "median": 0.0, "max": 0.0}
        return {
            "rmse": float(np.sqrt(np.mean(arr ** 2))),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "translation": safe_stats(trans_arr),
        "rotation_rad": safe_stats(rot_arr),
        "per_edge": per_edge,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=str, required=True, help="edges_measurements.json")
    ap.add_argument("--init", type=str, required=True, help="initial_poses.json")
    ap.add_argument("--out", type=str, required=True, help="optimized_poses.json")
    ap.add_argument("--report", type=str, required=True, help="residual_report.json")
    ap.add_argument("--use_robust", action="store_true", help="Enable robust Huber kernel")
    ap.add_argument("--huber_k", type=float, default=None, help="Huber kernel parameter (Auto if None)")
    ap.add_argument("--prior_sigma_xy", type=float, default=1e-3, help="Prior sigma for root translation")
    ap.add_argument("--prior_sigma_theta_deg", type=float, default=1e-3, help="Prior sigma for root rotation")
    ap.add_argument("--lm_max_iters", type=int, default=100, help="Levenberg-Marquardt max iterations")
    ap.add_argument("--lm_lambda", type=float, default=1e-3, help="Initial LM lambda")
    ap.add_argument("--theta_priors", default=None, help="06 output json for relative Manhattan constraints")

    args = ap.parse_args()

    edges_path = Path(args.edges)
    init_path = Path(args.init)
    out_path = Path(args.out)
    report_path = Path(args.report)

    edges_data = load_json(edges_path)
    init_data = load_json(init_path)

    edges = edges_data.get("edges", [])
    if not edges:
        raise RuntimeError("edges_measurements.json contains no edges")

    root_id = init_data.get("root", "")
    poses_init_dict = init_data.get("poses", {})
    if not poses_init_dict:
        raise RuntimeError("initial_poses.json contains no poses")

    node_ids = sorted(list(poses_init_dict.keys()))
    if root_id == "" or root_id not in poses_init_dict:
        root_id = node_ids[0]

    id_to_idx = make_symbol_map(node_ids)

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()

    # Safely insert initial values
    for nid in node_ids:
        k = gtsam.symbol('x', id_to_idx[nid])
        p = pose_dict_to_pose2(poses_init_dict[nid])
        initial.insert(k, p)

    # Prior on root to fix gauge freedom
    prior_sigma_theta = math.radians(args.prior_sigma_theta_deg)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(
        [args.prior_sigma_xy, args.prior_sigma_xy, prior_sigma_theta], dtype=np.float64
    ))
    root_key = gtsam.symbol('x', id_to_idx[root_id])
    
    # Anchor root strictly to its Initial Pose provided by Dijkstra!
    root_pose = initial.atPose2(root_key)
    graph.add(gtsam.PriorFactorPose2(root_key, root_pose, prior_noise))

    # Optional Theta Priors from Layout (Manhattan Edge Constraints)
    th_map = {}
    if args.theta_priors:
        priors_data = load_json(Path(args.theta_priors))
        th_map = priors_data.get("theta_priors", {})
        print(f"[INFO] Loaded {len(th_map)} layout theta priors to Snap relative edges.")

    missing_nodes = 0
    for e in edges:
        i, j = e["i"], e["j"]
        
        # Security validation to avoid Segmentation Fault
        if i not in id_to_idx or j not in id_to_idx:
            missing_nodes += 1
            continue
            
        key_i = gtsam.symbol('x', id_to_idx[i])
        key_j = gtsam.symbol('x', id_to_idx[j])
        
        if not initial.exists(key_i) or not initial.exists(key_j):
            missing_nodes += 1
            print(f"[WARNING] Edge {i}->{j} omitted: Target missing from initialization scope.")
            continue

        meas = e["measurement"]
        measured_relative_pose = gtsam.Pose2(float(meas["dx"]), float(meas["dy"]), float(meas["dtheta"]))

        # Check if we should override dtheta with snapped Manhattan priors
        if th_map and i in th_map and j in th_map:
            th_i = float(th_map[i])
            th_j = float(th_map[j])
            
            # 取得原本靠 BFS 或熱點算出的相對角度
            original_dtheta = float(meas["dtheta"])
            
            # 計算應該要是 90 度的幾倍 (k)
            half_pi = math.pi / 2.0
            
            # 尋找最接近原始角度的 k 值
            # 公式推導: snapped_dtheta = th_i - th_j + k * (pi/2)
            # => k = round( (original_dtheta - th_i + th_j) / (pi/2) )
            k = round((original_dtheta - th_i + th_j) / half_pi)
            
            # 計算出完美對齊曼哈頓網格的相對旋轉角
            snapped_dtheta = wrap_pi(th_i - th_j + k * half_pi)

            # 覆寫測量值
            measured_relative_pose = gtsam.Pose2(
                float(meas["dx"]), 
                float(meas["dy"]), 
                snapped_dtheta
            )

        sig = e.get("noise_sigma", {})
        sigma_xy = float(sig.get("sigma_xy", 1.0))
        sigma_theta = float(sig.get("sigma_theta", math.radians(30.0)))

        # Dynamic Huber: Calculate initial residual using non-robust metric to gauge k
        huber_k = args.huber_k
        if args.use_robust and huber_k is None:
            # Baseline calculation for dynamic K finding
            err_x, err_y, err_th = relative_error_pose2(initial.atPose2(key_i), initial.atPose2(key_j), measured_relative_pose)
            baseline_rmse = math.sqrt((err_x**2 + err_y**2) / 2.0)
            huber_k = max(1.0, baseline_rmse * 1.5)  # Auto-tune Huber to 1.5x initial RMSE scale

        # Fallback if unactivated
        if huber_k is None:
            huber_k = 1.345

        model = build_noise_model(sigma_xy, sigma_theta, args.use_robust, huber_k)
        graph.add(gtsam.BetweenFactorPose2(key_i, key_j, measured_relative_pose, model))

    if missing_nodes > 0:
        print(f"\\n[CRITICAL WARNING] {missing_nodes} edges dropped! Graph topology may be fractured. GTSAM optimization might Singulate!\\n")

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(args.lm_max_iters)
    params.setlambdaInitial(args.lm_lambda)
    params.setVerbosityLM("SUMMARY")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    poses_opt: Dict[str, gtsam.Pose2] = {}
    poses_init: Dict[str, gtsam.Pose2] = {}
    for nid in node_ids:
        key = gtsam.symbol('x', id_to_idx[nid])
        poses_opt[nid] = result.atPose2(key)
        poses_init[nid] = initial.atPose2(key)

    # Calculate reports utilizing Umeyama alignment
    before = compute_residual_stats(poses_init, edges)
    after = compute_residual_stats(poses_opt, edges, initial_poses=poses_init)

    report = {
        "scene_id": edges_data.get("scene_id", ""),
        "root": root_id,
        "num_nodes": len(node_ids),
        "num_edges": len(edges),
        "missing_nodes_in_edges": missing_nodes,
        "optimizer": {
            "type": "LevenbergMarquardt",
            "max_iters": args.lm_max_iters,
            "lambda_initial": args.lm_lambda,
            "use_robust": bool(args.use_robust),
            "huber_k_auto": args.huber_k is None,
        },
        "before": {
            "translation": before["translation"],
            "rotation_rad": before["rotation_rad"],
        },
        "after_umeyama_aligned": {
            "translation": after["translation"],
            "rotation_rad": after["rotation_rad"],
        },
        "top_edges_after_by_translation": sorted(after["per_edge"], key=lambda d: -d["trans_l2"])[:min(5, len(after["per_edge"]))],
        "top_edges_after_by_rotation": sorted(after["per_edge"], key=lambda d: -d["rot_abs"])[:min(5, len(after["per_edge"]))],
    }

    out = {
        "scene_id": edges_data.get("scene_id", ""),
        "root": root_id,
        "num_nodes": len(node_ids),
        "poses": {nid: pose2_to_dict(poses_opt[nid]) for nid in sorted(poses_opt.keys())},
        "note": "Optimized poses via GTSAM Pose2 PGO. Angles in radians.",
    }

    save_json(out_path, out)
    save_json(report_path, report)

    print(f"[OK] wrote optimized poses -> {out_path}")
    print(f"[OK] wrote residual report -> {report_path}")
    print("Before (trans rmse, rot rmse):",
          report["before"]["translation"]["rmse"], report["before"]["rotation_rad"]["rmse"])
    print("After Aligned (trans rmse, rot rmse):",
          report["after_umeyama_aligned"]["translation"]["rmse"], report["after_umeyama_aligned"]["rotation_rad"]["rmse"])


if __name__ == "__main__":
    main()
