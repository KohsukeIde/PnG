"""
Step H: Pose Optimization Test with Real Data

Purpose: Verify that pose optimization converges correctly with:
1. Different score functions (loss, avg_cost, mass_aware, full_uot)
2. Epsilon annealing for collapse prevention
3. Starting from nearby initialization

Test progression:
- First: Small initial error (5-10 deg)
- Then: Medium error (30 deg)
- Finally: Large error (60 deg) with epsilon annealing
"""

import csv
import numpy as np
import torch
import sys
import os
from typing import Tuple, Dict, List

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from src.optimizer.optimal_transport_solver_torch import OptimalTransportSolver
from src.primitive.camera import Lie
from src.utils.colmap_utils import (
    load_cameras_from_colmap,
    load_images_from_colmap,
    quaternion_to_rotation_matrix,
)
from src.utils.gaussian_utils import load_gaussians


def load_colmap_cameras(scan_name: str = "scan63"):
    """Load camera data from COLMAP sparse reconstruction."""
    colmap_dir = os.path.join(project_root, f"data/DTU/{scan_name}/sparse/0")
    cameras = load_cameras_from_colmap(colmap_dir)
    images = load_images_from_colmap(colmap_dir)
    return cameras, images


def get_colmap_camera_params(cameras, images, image_name: str):
    """Get camera parameters for a given image from COLMAP data."""
    image_data = None
    for img_id, img in images.items():
        if img['name'] == image_name:
            image_data = img
            break

    if image_data is None:
        raise ValueError(f"Image {image_name} not found in COLMAP data")

    camera = cameras[image_data['camera_id']]
    K = camera.get_camera_matrix()

    R = quaternion_to_rotation_matrix(
        image_data['qw'], image_data['qx'],
        image_data['qy'], image_data['qz']
    )

    t = np.array([image_data['tx'], image_data['ty'], image_data['tz']])

    return {'K': K, 'R': R, 't': t, 'name': image_name}


def compute_relative_pose_wc(cam1: dict, cam2: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Compute relative pose (world-to-camera) from camera 1 to camera 2.

    COLMAP stores world-to-camera (w2c) pose: P_cam = R @ P_world + t.
    We compute relative pose T_12 such that: P_cam2 = R_12 @ P_cam1 + t_12.
    """
    # COLMAP w2c poses
    R1_w2c, t1_w2c = cam1['R'], cam1['t']
    R2_w2c, t2_w2c = cam2['R'], cam2['t']

    # Relative pose: cam1 -> cam2
    R_12 = R2_w2c @ R1_w2c.T
    t_12 = t2_w2c - R_12 @ t1_w2c
    t_12_norm = t_12 / (np.linalg.norm(t_12) + 1e-10)

    return R_12, t_12_norm


def invert_pose(R_wc: np.ndarray, t_wc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert world-to-camera pose to camera-to-world pose."""
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    return R_cw, t_cw


def compute_relative_pose_cw(cam1: dict, cam2: dict) -> Tuple[np.ndarray, np.ndarray]:
    """Compute relative pose (camera-to-world) from camera 1 to camera 2."""
    R_wc, t_wc = compute_relative_pose_wc(cam1, cam2)
    return invert_pose(R_wc, t_wc)


def rodrigues_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix using Rodrigues formula."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _format_float_for_path(value: float) -> str:
    """Format float for filesystem-safe path segments."""
    return f"{value:.3f}".replace(".", "p")


def load_debug_log(log_path: str) -> List[Dict[str, float]]:
    """Load debug log CSV into list of numeric dicts."""
    if not os.path.exists(log_path):
        return []
    rows: List[Dict[str, float]] = []
    with open(log_path, "r", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            parsed: Dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def summarize_debug_log(log_path: str, tail: int = 10) -> Dict[str, float]:
    """Summarize recent debug log entries with median stats."""
    rows = load_debug_log(log_path)
    if not rows:
        return {}
    if tail > 0 and len(rows) > tail:
        rows = rows[-tail:]

    def median(key: str) -> float:
        values = [r[key] for r in rows if key in r]
        if not values:
            return 0.0
        return float(np.median(values))

    transport_cost = median("Transport_Cost")
    cheirality_loss = median("Cheirality_Loss")
    ratio = cheirality_loss / transport_cost if transport_cost > 1e-12 else 0.0
    return {
        "rot_grad_norm_median": median("Rot_Grad_Norm"),
        "trans_grad_norm_median": median("Trans_Grad_Norm"),
        "transport_cost_median": transport_cost,
        "cheirality_loss_median": cheirality_loss,
        "cheirality_over_transport_median": ratio,
    }


def rotation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute rotation error in degrees."""
    R_diff = R1 @ R2.T
    trace = np.clip(np.trace(R_diff), -1, 3)
    angle = np.arccos((trace - 1) / 2)
    return np.degrees(angle)


def translation_error(t1: np.ndarray, t2: np.ndarray) -> float:
    """Compute translation direction error in degrees."""
    t1_norm = t1 / (np.linalg.norm(t1) + 1e-10)
    t2_norm = t2 / (np.linalg.norm(t2) + 1e-10)
    cos_angle = np.clip(np.dot(t1_norm, t2_norm), -1, 1)
    return np.degrees(np.arccos(cos_angle))


def perturb_pose_wc(
    R_gt_wc: np.ndarray, t_gt_wc: np.ndarray, rot_deg: float, trans_deg: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Perturb ground truth pose (world-to-camera) by given angles.

    Args:
        R_gt_wc: Ground truth rotation (world-to-camera).
        t_gt_wc: Ground truth translation direction (world-to-camera, unit norm).
        rot_deg: Rotation perturbation in degrees (around Y axis).
        trans_deg: Translation perturbation in degrees (around X axis).

    Returns:
        R_init_wc: Perturbed rotation (world-to-camera).
        t_init_wc: Perturbed translation direction (world-to-camera, unit norm).
    """
    # Rotation perturbation around Y axis (camera frame)
    angle_rad = np.radians(rot_deg)
    R_perturb = rodrigues_rotation(np.array([0, 1, 0]), angle_rad)
    R_init_wc = R_perturb @ R_gt_wc

    # Translation perturbation (optional, around X axis)
    if trans_deg > 0:
        t_perturb = rodrigues_rotation(np.array([1, 0, 0]), np.radians(trans_deg))
        t_init_wc = t_perturb @ t_gt_wc
        t_init_wc = t_init_wc / (np.linalg.norm(t_init_wc) + 1e-10)
    else:
        t_init_wc = t_gt_wc.copy()

    return R_init_wc, t_init_wc


def test_pose_optimization(
    idx1: int,
    idx2: int,
    init_rot_error_deg: float,
    score_type: str = "avg_cost",
    epsilon_annealing: bool = False,
    max_iter: int = 200,
    rot_lr: float = 1e-3,
    trans_lr: float = None,  # Default: 0.1 * rot_lr
    optimize_mode: str = "both",  # Step II: "both", "rotation_only", "translation_only"
    init_trans_error_deg: float = 0.0,  # Perturbation for translation (around X axis)
    lambda_cheirality: float = 0.0,
    cheirality_topk: int = 3,
    log_cheirality: bool = False,
    noise_model: str = "gaussian",
    huber_delta: float = 1.0,
    cauchy_c: float = 1.0,
    differentiable_transport: bool = False,
):
    """Run pose optimization test with specified settings."""
    print(f"\n{'='*70}")
    print(f"Test: Pair ({idx1}, {idx2}), Init Rot Error: {init_rot_error_deg}deg, Score: {score_type}")
    # Default trans_lr to 0.1 * rot_lr (prevents translation drift)
    if trans_lr is None:
        trans_lr = 0.1 * rot_lr
    print(f"Epsilon Annealing: {epsilon_annealing}")
    print(f"Learning rates: rot_lr={rot_lr}, trans_lr={trans_lr}")
    print(f"Optimize mode: {optimize_mode}")
    print(f"Cheirality: lambda={lambda_cheirality}, topk={cheirality_topk}, log_raw={log_cheirality}")
    print(f"Noise model: {noise_model}")
    print(f"Differentiable transport: {differentiable_transport}")
    if init_trans_error_deg != 0.0:
        print(f"Init Trans Error: {init_trans_error_deg}deg")
    print("=" * 70)

    # Load Gaussians
    data1 = load_gaussians(idx1)
    data2 = load_gaussians(idx2)

    g1 = data1['original_gaussians']
    g2 = data2['original_gaussians']
    ot_mass1 = data1.get('ot_mass', None)
    ot_mass2 = data2.get('ot_mass', None)

    # Load COLMAP camera poses
    cameras, images = load_colmap_cameras()
    cam1 = get_colmap_camera_params(cameras, images, f"{idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{idx2:04d}.png")
    K = cam1['K']

    # Ground truth relative pose (world-to-camera)
    # This is the convention used by solver: E = [t_wc]_x @ R_wc
    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)

    # Create perturbed initial pose (world-to-camera)
    R_init_wc, t_init_wc = perturb_pose_wc(R_gt_wc, t_gt_wc, init_rot_error_deg, init_trans_error_deg)

    init_rot_err = rotation_error(R_init_wc, R_gt_wc)
    # Translation error with sign ambiguity (epipolar geometry)
    init_trans_err = min(
        translation_error(t_init_wc, t_gt_wc),
        translation_error(-t_init_wc, t_gt_wc)
    )

    print(f"\nInitial errors (world-to-camera, unified convention):")
    print(f"  Rotation: {init_rot_err:.2f} deg")
    print(f"  Translation: {init_trans_err:.2f} deg")

    # OT settings from LOG.md
    sigma_epipolar = 400.0
    epsilon = 0.05
    rho = 0.5

    # Create solver
    solver = OptimalTransportSolver(
        gaussians1=g1,
        gaussians2=g2,
        k1=K,
        k2=K,
        device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0,
        lambda_cov=0.0,
        lambda_epipolar=1.0,
        sigma_epipolar=sigma_epipolar,
        epi_clip=None,
        ot_mass1=ot_mass1,
        ot_mass2=ot_mass2,
        lambda_cheirality=lambda_cheirality,
        cheirality_topk=cheirality_topk,
        noise_model=noise_model,
        huber_delta=huber_delta,
        cauchy_c=cauchy_c,
    )

    # Initialize solver with appropriate pose based on optimize_mode (Step VIII)
    #
    # Solver internal convention:
    #   T_cw = se3_to_SE3(se3_vec)  # camera-to-world (Rt matrix)
    #   R_wc = R_cw.T, t_wc = -R_cw.T @ t_cw  # world-to-camera
    #   F = _build_F_from_wc(R_wc, t_wc)  # E = [t_wc]_x @ R_wc
    #
    # For relative pose (R_wc, t_wc) where E = [t_wc]_x @ R_wc:
    #   T_cw needs: R_cw = R_wc.T, t_cw = -R_wc.T @ t_wc
    #
    # IMPORTANT: Use SE3_to_se3 for correct initialization!
    #   se3_to_SE3 returns t = V(w) @ u, so u != t
    #   SE3_to_se3 uses invV to recover u from t correctly

    # Step VIII: Proper initialization based on mode (in w2c convention)
    if optimize_mode == "rotation_only":
        # R perturbed, t fixed to GT (in w2c)
        R_wc_for_init = R_init_wc
        t_wc_for_init = t_gt_wc
        print(f"  Mode: rotation_only -> R from perturbed, t from GT (w2c)")
    elif optimize_mode == "translation_only":
        # R fixed to GT, t perturbed (in w2c)
        R_wc_for_init = R_gt_wc
        t_wc_for_init = t_init_wc
        print(f"  Mode: translation_only -> R from GT, t from perturbed (w2c)")
    else:  # "both"
        R_wc_for_init = R_init_wc
        t_wc_for_init = t_init_wc

    # Convert w2c to c2w for SE3 parametrization
    R_cw_init = R_wc_for_init.T
    t_cw_init = -R_wc_for_init.T @ t_wc_for_init

    # Build SE3 matrix and use SE3_to_se3 for correct initialization
    # This ensures that se3_to_SE3(se3_vec) produces exactly the intended Rt
    Rt_cw = np.concatenate([R_cw_init, t_cw_init[:, None]], axis=1)  # (3, 4)
    Rt_cw_tensor = torch.tensor(Rt_cw, dtype=torch.float32)
    se3_vec = solver.lie.SE3_to_se3(Rt_cw_tensor)  # [w, u] where t = V(w) @ u

    solver.rot_vec = torch.nn.Parameter(se3_vec[:3].clone())
    solver.trans_vec = torch.nn.Parameter(se3_vec[3:].clone())

    # Verify initialization (debug)
    with torch.no_grad():
        T_check = solver.lie.se3_to_SE3(se3_vec)
        R_check = T_check[:3, :3].numpy()
        t_check = T_check[:3, 3].numpy()
        R_wc_check = R_check.T
        t_wc_check = -R_check.T @ t_check
        t_wc_check /= (np.linalg.norm(t_wc_check) + 1e-10)
        init_verify_rot = rotation_error(R_wc_check, R_wc_for_init)
        init_verify_trans = translation_error(t_wc_check, t_wc_for_init)
        print(f"  Init verification: R_err={init_verify_rot:.4f}deg, t_err={init_verify_trans:.4f}deg")

    # Output directories (include key settings to avoid collisions)
    suffix_parts = []
    if optimize_mode != "both":
        suffix_parts.append(optimize_mode)
    if lambda_cheirality != 0.0 or log_cheirality:
        suffix_parts.append(f"cheirality_{_format_float_for_path(lambda_cheirality)}")
    if noise_model != "gaussian":
        suffix_parts.append(f"noise_{noise_model}")
    if init_trans_error_deg != 0.0:
        suffix_parts.append(f"tinit_{_format_float_for_path(init_trans_error_deg)}")
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    results_dir = os.path.join(
        project_root,
        "results",
        f"step_h_{idx1}_{idx2}_{init_rot_error_deg}deg_{score_type}{suffix}",
    )
    diagnostics_dir = os.path.join(results_dir, "diagnostics")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(diagnostics_dir, exist_ok=True)

    # Run optimization
    print(f"\nStarting optimization...")
    print(f"  Results: {results_dir}")

    try:
        loss_history = solver.optimize_with_SE3(
            max_iter=max_iter,
            rot_lr=rot_lr,
            trans_lr=trans_lr,
            momentum=0.9,
            tol=1e-8,
            diagnostics_dir=diagnostics_dir,
            differentiable_transport=differentiable_transport,
            sinkhorn_epsilon=epsilon,
            sinkhorn_rho=rho,
            score_type=score_type,
            lambda_kl=0.1,
            epsilon_annealing=epsilon_annealing,
            epsilon_start=0.2 if epsilon_annealing else epsilon,
            epsilon_end=epsilon,
            anneal_steps=50,
            optimize_mode=optimize_mode,  # Step II: R/t separation
            log_cheirality=log_cheirality,
        )

        # Extract final pose and convert to w2c convention (same as GT)
        with torch.no_grad():
            se3_vec_final = torch.cat([solver.rot_vec, solver.trans_vec])
            T_cw_final = solver.lie.se3_to_SE3(se3_vec_final)
            R_cw_final = T_cw_final[:3, :3].numpy()
            t_cw_final = T_cw_final[:3, 3].numpy()

            # Convert c2w to w2c: R_wc = R_cw.T, t_wc = -R_cw.T @ t_cw
            R_wc_final = R_cw_final.T
            t_wc_final = -R_cw_final.T @ t_cw_final
            t_wc_final = t_wc_final / (np.linalg.norm(t_wc_final) + 1e-10)

        final_rot_err = rotation_error(R_wc_final, R_gt_wc)
        # Translation error with sign ambiguity (epipolar geometry)
        final_trans_err = min(
            translation_error(t_wc_final, t_gt_wc),
            translation_error(-t_wc_final, t_gt_wc)
        )

        print(f"\nFinal errors (world-to-camera, unified convention):")
        print(f"  Rotation: {final_rot_err:.2f} deg (was {init_rot_err:.2f})")
        print(f"  Translation: {final_trans_err:.2f} deg (was {init_trans_err:.2f})")

        rot_improved = init_rot_err - final_rot_err
        trans_improved = init_trans_err - final_trans_err

        print(f"\nImprovement:")
        print(f"  Rotation: {rot_improved:.2f} deg {'(better)' if rot_improved > 0 else '(worse)'}")
        print(f"  Translation: {trans_improved:.2f} deg {'(better)' if trans_improved > 0 else '(worse)'}")

        return {
            'init_rot_err': init_rot_err,
            'init_trans_err': init_trans_err,
            'final_rot_err': final_rot_err,
            'final_trans_err': final_trans_err,
            'rot_improved': rot_improved,
            'trans_improved': trans_improved,
            'loss_history': loss_history,
            'converged': rot_improved > 0,
            'results_dir': results_dir,
            'diagnostics_dir': diagnostics_dir,
        }

    except Exception as e:
        print(f"\nOptimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_comprehensive_test():
    """Run comprehensive test suite."""
    print("=" * 70)
    print("Step H: Comprehensive Pose Optimization Test")
    print("=" * 70)

    # Test settings
    test_pair = (0, 10)  # Best pair from Step E/F

    # Test progression: small -> medium -> large initial error
    # Format: (init_error, score_type, epsilon_annealing, rot_lr, trans_lr)
    test_configs = [
        # Small error tests with adjusted learning rates
        (10, "avg_cost", False, 1e-3, 1e-4),
        (10, "mass_aware", False, 1e-3, 1e-4),
        # Medium error tests
        (30, "avg_cost", False, 1e-3, 1e-4),
        (30, "avg_cost", True, 1e-3, 1e-4),  # With annealing
        (30, "mass_aware", True, 1e-3, 1e-4),
        # Large error tests (need annealing)
        (60, "avg_cost", True, 1e-3, 1e-4),
        (60, "mass_aware", True, 1e-3, 1e-4),
    ]

    results = []
    for init_err, score_type, anneal, r_lr, t_lr in test_configs:
        result = test_pose_optimization(
            idx1=test_pair[0],
            idx2=test_pair[1],
            init_rot_error_deg=init_err,
            score_type=score_type,
            epsilon_annealing=anneal,
            max_iter=200,
            rot_lr=r_lr,
            trans_lr=t_lr,
        )
        if result:
            result['config'] = {
                'init_error': init_err,
                'score_type': score_type,
                'epsilon_annealing': anneal,
                'rot_lr': r_lr,
                'trans_lr': t_lr,
            }
            results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Init Err':>10} {'Score':>12} {'Anneal':>8} {'Final Rot':>10} {'Final Trans':>12} {'Status':>10}")
    print("-" * 70)

    for r in results:
        cfg = r['config']
        status = "OK" if r['converged'] else "FAIL"
        print(f"{cfg['init_error']:>10} {cfg['score_type']:>12} {str(cfg['epsilon_annealing']):>8} "
              f"{r['final_rot_err']:>10.2f} {r['final_trans_err']:>12.2f} {status:>10}")


def run_step_iv_cheirality_tests():
    """Step IV: cheirality loss sweep."""
    print("=" * 70)
    print("Step IV: Cheirality Constraint Tests")
    print("=" * 70)

    test_pair = (0, 10)
    init_errors = [30, 60]
    lambda_values = [0.0, 0.01]
    max_iter = int(os.getenv("STEP_IV_MAX_ITER", "120"))

    results = []
    for init_err in init_errors:
        for lam in lambda_values:
            result = test_pose_optimization(
                idx1=test_pair[0],
                idx2=test_pair[1],
                init_rot_error_deg=init_err,
                score_type="full_uot",
                epsilon_annealing=True,
                max_iter=max_iter,
                rot_lr=1e-3,
                trans_lr=1e-4,
                lambda_cheirality=lam,
                cheirality_topk=3,
                log_cheirality=True,
            )
            if result:
                debug_log_path = os.path.join(result["diagnostics_dir"], "gradient_debug.log")
                diag = summarize_debug_log(debug_log_path, tail=10)
                result['config'] = {
                    'init_error': init_err,
                    'lambda_cheirality': lam,
                }
                result['diagnostics'] = diag
                results.append(result)

    print("\n" + "=" * 70)
    print("Step IV Summary")
    print("=" * 70)
    print(f"{'Init Err':>10} {'lambda':>10} {'Final Rot':>10} {'Final Trans':>12} "
          f"{'Cheir/Cost':>12}")
    print("-" * 70)
    for r in results:
        cfg = r['config']
        ratio = r.get("diagnostics", {}).get("cheirality_over_transport_median", 0.0)
        print(f"{cfg['init_error']:>10} {cfg['lambda_cheirality']:>10.3f} "
              f"{r['final_rot_err']:>10.2f} {r['final_trans_err']:>12.2f} "
              f"{ratio:>12.4f}")


def run_step_v_robust_tests():
    """Step V: robust noise model tests."""
    print("=" * 70)
    print("Step V: Robust Noise Model Tests")
    print("=" * 70)

    test_pair = (0, 10)
    init_errors = [60]
    noise_models = ["gaussian", "cauchy"]
    max_iter = int(os.getenv("STEP_V_MAX_ITER", "120"))

    results = []
    for init_err in init_errors:
        for model in noise_models:
            result = test_pose_optimization(
                idx1=test_pair[0],
                idx2=test_pair[1],
                init_rot_error_deg=init_err,
                score_type="avg_cost",
                epsilon_annealing=True,
                max_iter=max_iter,
                rot_lr=1e-3,
                trans_lr=1e-4,
                noise_model=model,
                cauchy_c=1.0,
            )
            if result:
                result['config'] = {
                    'init_error': init_err,
                    'noise_model': model,
                }
                results.append(result)

    print("\n" + "=" * 70)
    print("Step V Summary")
    print("=" * 70)
    print(f"{'Init Err':>10} {'model':>10} {'Final Rot':>10} {'Final Trans':>12}")
    print("-" * 70)
    for r in results:
        cfg = r['config']
        print(f"{cfg['init_error']:>10} {cfg['noise_model']:>10} "
              f"{r['final_rot_err']:>10.2f} {r['final_trans_err']:>12.2f}")


def run_step_vi_two_stage_test():
    """Step VI: two-stage optimization (R-only -> t-only)."""
    print("=" * 70)
    print("Step VI: Two-Stage Optimization Test")
    print("=" * 70)

    idx1, idx2 = 0, 10
    init_rot_error_deg = 30
    init_trans_error_deg = 0.0
    score_type = "avg_cost"
    epsilon_annealing = True
    max_iter_stage1 = int(os.getenv("STEP_VI_STAGE1_MAX_ITER", "120"))
    max_iter_stage2 = int(os.getenv("STEP_VI_STAGE2_MAX_ITER", "120"))
    rot_lr = 1e-3
    trans_lr = 1e-4

    print(f"Pair ({idx1}, {idx2}), init_rot={init_rot_error_deg}deg")

    # Load Gaussians
    data1 = load_gaussians(idx1)
    data2 = load_gaussians(idx2)
    g1 = data1['original_gaussians']
    g2 = data2['original_gaussians']
    ot_mass1 = data1.get('ot_mass', None)
    ot_mass2 = data2.get('ot_mass', None)

    # Load COLMAP camera poses
    cameras, images = load_colmap_cameras()
    cam1 = get_colmap_camera_params(cameras, images, f"{idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{idx2:04d}.png")
    K = cam1['K']

    # Ground truth relative pose (world-to-camera) - Step VIII unified convention
    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)

    # Create perturbed initial pose (world-to-camera)
    R_init_wc, t_init_wc = perturb_pose_wc(R_gt_wc, t_gt_wc, init_rot_error_deg, init_trans_error_deg)

    init_rot_err = rotation_error(R_init_wc, R_gt_wc)
    init_trans_err = min(
        translation_error(t_init_wc, t_gt_wc),
        translation_error(-t_init_wc, t_gt_wc)
    )
    print(f"Initial errors (w2c): R={init_rot_err:.2f}deg, t={init_trans_err:.2f}deg")

    # OT settings from LOG.md
    sigma_epipolar = 400.0
    epsilon = 0.05
    rho = 0.5

    solver = OptimalTransportSolver(
        gaussians1=g1,
        gaussians2=g2,
        k1=K,
        k2=K,
        device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0,
        lambda_cov=0.0,
        lambda_epipolar=1.0,
        sigma_epipolar=sigma_epipolar,
        epi_clip=None,
        ot_mass1=ot_mass1,
        ot_mass2=ot_mass2,
    )

    # Initialize SE3 params using SE3_to_se3 (Step VIII fix)
    R_cw_init = R_init_wc.T
    t_cw_init = -R_init_wc.T @ t_init_wc
    Rt_cw = np.concatenate([R_cw_init, t_cw_init[:, None]], axis=1)
    Rt_cw_tensor = torch.tensor(Rt_cw, dtype=torch.float32)
    se3_vec = solver.lie.SE3_to_se3(Rt_cw_tensor)
    solver.rot_vec = torch.nn.Parameter(se3_vec[:3].clone())
    solver.trans_vec = torch.nn.Parameter(se3_vec[3:].clone())

    print("\nStage 1: rotation_only")
    solver.optimize_with_SE3(
        max_iter=max_iter_stage1,
        rot_lr=rot_lr,
        trans_lr=trans_lr,
        momentum=0.9,
        tol=1e-8,
        diagnostics_dir=os.path.join(project_root, "results", "step_vi_stage1", "diagnostics"),
        differentiable_transport=False,
        sinkhorn_epsilon=epsilon,
        sinkhorn_rho=rho,
        score_type=score_type,
        lambda_kl=0.1,
        epsilon_annealing=epsilon_annealing,
        epsilon_start=0.2 if epsilon_annealing else epsilon,
        epsilon_end=epsilon,
        anneal_steps=50,
        optimize_mode="rotation_only",
    )

    with torch.no_grad():
        se3_vec = torch.cat([solver.rot_vec, solver.trans_vec])
        T_cw = solver.lie.se3_to_SE3(se3_vec)
        R_cw_stage1 = T_cw[:3, :3].numpy()
        t_cw_stage1 = T_cw[:3, 3].numpy()
        R_wc_stage1 = R_cw_stage1.T
        t_wc_stage1 = -R_cw_stage1.T @ t_cw_stage1
        t_wc_stage1 = t_wc_stage1 / (np.linalg.norm(t_wc_stage1) + 1e-10)
    stage1_rot_err = rotation_error(R_wc_stage1, R_gt_wc)
    stage1_trans_err = min(
        translation_error(t_wc_stage1, t_gt_wc),
        translation_error(-t_wc_stage1, t_gt_wc)
    )
    print(f"Stage 1 errors (w2c): R={stage1_rot_err:.2f}deg, t={stage1_trans_err:.2f}deg")

    print("\nStage 2: translation_only")
    stage2_diag_dir = os.path.join(project_root, "results", "step_vi_stage2", "diagnostics")
    solver.optimize_with_SE3(
        max_iter=max_iter_stage2,
        rot_lr=rot_lr,
        trans_lr=trans_lr,
        momentum=0.9,
        tol=1e-8,
        diagnostics_dir=stage2_diag_dir,
        differentiable_transport=False,
        sinkhorn_epsilon=epsilon,
        sinkhorn_rho=rho,
        score_type=score_type,
        lambda_kl=0.1,
        epsilon_annealing=epsilon_annealing,
        epsilon_start=0.2 if epsilon_annealing else epsilon,
        epsilon_end=epsilon,
        anneal_steps=50,
        optimize_mode="translation_only",
    )

    with torch.no_grad():
        se3_vec = torch.cat([solver.rot_vec, solver.trans_vec])
        T_cw = solver.lie.se3_to_SE3(se3_vec)
        R_cw_final = T_cw[:3, :3].numpy()
        t_cw_final = T_cw[:3, 3].numpy()
        R_wc_final = R_cw_final.T
        t_wc_final = -R_cw_final.T @ t_cw_final
        t_wc_final = t_wc_final / (np.linalg.norm(t_wc_final) + 1e-10)
    final_rot_err = rotation_error(R_wc_final, R_gt_wc)
    final_trans_err = min(
        translation_error(t_wc_final, t_gt_wc),
        translation_error(-t_wc_final, t_gt_wc)
    )
    print(f"Stage 2 errors (w2c): R={final_rot_err:.2f}deg, t={final_trans_err:.2f}deg")

    debug_log_path = os.path.join(stage2_diag_dir, "gradient_debug.log")
    diag = summarize_debug_log(debug_log_path, tail=10)
    if diag:
        print(
            f"Stage 2 grad median (last 10): "
            f"rot={diag['rot_grad_norm_median']:.3e}, "
            f"trans={diag['trans_grad_norm_median']:.3e}"
        )


def optimize_rt_decoupled(
    solver: OptimalTransportSolver,
    R_wc_init: np.ndarray,
    t_wc_init: np.ndarray,
    R_wc_gt: np.ndarray,
    t_wc_gt: np.ndarray,
    *,
    score_type: str = "avg_cost",
    max_iter: int = 120,
    rot_lr: float = 1e-3,
    trans_lr: float = 1e-4,
    epsilon: float = 0.05,
    rho: float = 0.5,
    momentum: float = 0.9,
    optimize_mode: str = "both",  # "both", "rotation_only", "translation_only"
    differentiable_transport: bool = False,
) -> Dict[str, float]:
    """Optimize with decoupled (R, t_dir) parameters on w2c convention."""
    lie = Lie()
    R_wc_t = torch.tensor(R_wc_init, dtype=torch.float32)
    t_wc_t = torch.tensor(t_wc_init, dtype=torch.float32)
    rot_vec = torch.nn.Parameter(lie.SO3_to_so3(R_wc_t).clone())
    trans_vec = torch.nn.Parameter(t_wc_t.clone())

    if optimize_mode == "rotation_only":
        trans_vec.requires_grad = False
    elif optimize_mode == "translation_only":
        rot_vec.requires_grad = False

    params = []
    if rot_vec.requires_grad:
        params.append({"params": rot_vec, "lr": rot_lr})
    if trans_vec.requires_grad:
        params.append({"params": trans_vec, "lr": trans_lr})
    optimizer = torch.optim.SGD(params, momentum=momentum, nesterov=True)

    loss_history = []
    for _ in range(max_iter):
        optimizer.zero_grad()
        R_wc = lie.so3_to_SO3(rot_vec)
        t_wc = trans_vec / (trans_vec.norm() + 1e-10)

        F = solver._build_F_from_wc(R_wc, t_wc)
        cost_matrix = solver.compute_cost_matrix(F)
        context = torch.enable_grad() if differentiable_transport else torch.no_grad()
        with context:
            transport, _ = solver.unbalanced_sinkhorn_algorithm(
                cost_matrix,
                epsilon=epsilon,
                rho=rho,
                gate_mask=solver._last_gate_mask,
            )
            if not differentiable_transport:
                transport = transport.detach()

        transport_cost = torch.sum(transport * cost_matrix)
        T_sum = transport.sum()

        if score_type == "avg_cost":
            loss = transport_cost / (T_sum + 1e-10)
        elif score_type == "full_uot":
            a = solver.alpha1 / solver.alpha1.sum()
            b = solver.alpha2 / solver.alpha2.sum()
            row_sum = transport.sum(dim=1)
            col_sum = transport.sum(dim=0)
            eps_kl = 1e-10
            KL_row = (row_sum * torch.log((row_sum + eps_kl) / (a + eps_kl)) - row_sum + a).sum()
            KL_col = (col_sum * torch.log((col_sum + eps_kl) / (b + eps_kl)) - col_sum + b).sum()
            entropy = -(transport * torch.log(transport + eps_kl)).sum()
            eps_actual = solver._last_sinkhorn_epsilon
            rho_actual = solver._last_sinkhorn_rho
            entropic = -entropy - T_sum
            loss = transport_cost + rho_actual * (KL_row + KL_col) + eps_actual * entropic
        else:
            raise ValueError(f"Unsupported score_type: {score_type}")

        loss.backward()
        optimizer.step()
        with torch.no_grad():
            trans_vec.div_(trans_vec.norm() + 1e-10)
        loss_history.append(loss.item())

    with torch.no_grad():
        R_wc_final = lie.so3_to_SO3(rot_vec).cpu().numpy()
        t_wc_final = (trans_vec / (trans_vec.norm() + 1e-10)).cpu().numpy()

    final_rot_err = rotation_error(R_wc_final, R_wc_gt)
    final_trans_err = min(
        translation_error(t_wc_final, t_wc_gt),
        translation_error(-t_wc_final, t_wc_gt),
    )
    init_rot_err = rotation_error(R_wc_init, R_wc_gt)
    init_trans_err = min(
        translation_error(t_wc_init, t_wc_gt),
        translation_error(-t_wc_init, t_wc_gt),
    )

    return {
        "init_rot_err": init_rot_err,
        "init_trans_err": init_trans_err,
        "final_rot_err": final_rot_err,
        "final_trans_err": final_trans_err,
        "rot_improved": init_rot_err - final_rot_err,
        "trans_improved": init_trans_err - final_trans_err,
        "loss_history": loss_history,
    }


def run_step_xvi_rt_decoupled():
    """Step XVI: decoupled (R, t_dir) optimization to avoid SE3 coupling."""
    print("=" * 70)
    print("Step XVI: Decoupled (R, t_dir) Optimization")
    print("=" * 70)

    idx1, idx2 = 0, 10
    init_rot_error_deg = 30
    init_trans_error_deg = 10.0
    max_iter = int(os.getenv("STEP_XVI_MAX_ITER", "120"))
    rot_lr = 1e-3
    trans_lr = 1e-4

    data1 = load_gaussians(idx1)
    data2 = load_gaussians(idx2)
    g1 = data1["original_gaussians"]
    g2 = data2["original_gaussians"]
    ot_mass1 = data1.get("ot_mass", None)
    ot_mass2 = data2.get("ot_mass", None)

    cameras, images = load_colmap_cameras()
    cam1 = get_colmap_camera_params(cameras, images, f"{idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{idx2:04d}.png")
    K = cam1["K"]

    R_gt_wc, t_gt_wc = compute_relative_pose_wc(cam1, cam2)
    R_init_wc, t_init_wc = perturb_pose_wc(
        R_gt_wc, t_gt_wc, init_rot_error_deg, init_trans_error_deg
    )

    solver = OptimalTransportSolver(
        gaussians1=g1,
        gaussians2=g2,
        k1=K,
        k2=K,
        device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0,
        lambda_cov=0.0,
        lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        epi_clip=None,
        ot_mass1=ot_mass1,
        ot_mass2=ot_mass2,
    )

    configs = [
        ("both", R_init_wc, t_init_wc),
        ("rotation_only", R_init_wc, t_gt_wc),
        ("translation_only", R_gt_wc, t_init_wc),
    ]

    for mode, R_wc_init, t_wc_init in configs:
        print(f"\nMode: {mode}")
        res = optimize_rt_decoupled(
            solver,
            R_wc_init,
            t_wc_init,
            R_gt_wc,
            t_gt_wc,
            score_type="avg_cost",
            max_iter=max_iter,
            rot_lr=rot_lr,
            trans_lr=trans_lr,
            optimize_mode=mode,
        )
        print(
            f"  init R={res['init_rot_err']:.2f}, init t={res['init_trans_err']:.2f} | "
            f"final R={res['final_rot_err']:.2f}, final t={res['final_trans_err']:.2f}"
        )


def run_step_vii_translation_landscape():
    """Step VII: translation loss landscape with fixed GT rotation."""
    print("=" * 70)
    print("Step VII: Translation Loss Landscape")
    print("=" * 70)

    idx1, idx2 = 0, 10
    max_angle = float(os.getenv("STEP_VII_MAX_ANGLE", "60"))
    angle_step = float(os.getenv("STEP_VII_ANGLE_STEP", "10"))
    sinkhorn_max_iter = int(os.getenv("STEP_VII_SINKHORN_ITERS", "120"))

    # Load Gaussians and cameras
    data1 = load_gaussians(idx1)
    data2 = load_gaussians(idx2)
    g1 = data1['original_gaussians']
    g2 = data2['original_gaussians']
    ot_mass1 = data1.get('ot_mass', None)
    ot_mass2 = data2.get('ot_mass', None)

    cameras, images = load_colmap_cameras()
    cam1 = get_colmap_camera_params(cameras, images, f"{idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{idx2:04d}.png")
    K = cam1['K']

    # Step VIII: Use w2c convention directly
    R_wc, t_wc_gt = compute_relative_pose_wc(cam1, cam2)

    sigma_epipolar = 400.0
    epsilon = 0.05
    rho = 0.5

    solver = OptimalTransportSolver(
        gaussians1=g1,
        gaussians2=g2,
        k1=K,
        k2=K,
        device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0,
        lambda_cov=0.0,
        lambda_epipolar=1.0,
        sigma_epipolar=sigma_epipolar,
        epi_clip=None,
        ot_mass1=ot_mass1,
        ot_mass2=ot_mass2,
    )

    # Marginals for score computation
    a = solver.alpha1 / solver.alpha1.sum()
    b = solver.alpha2 / solver.alpha2.sum()

    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    angles = np.arange(-max_angle, max_angle + 1e-6, angle_step)

    results_dir = os.path.join(project_root, "results", "step_vii_landscape")
    os.makedirs(results_dir, exist_ok=True)

    scores_by_axis = {
        "avg_cost": {k: [] for k in axes},
        "full_uot": {k: [] for k in axes},
    }
    records = []

    from src.oracle_study.objective_func.test_step_f_score_functions import compute_all_scores
    import matplotlib.pyplot as plt

    with torch.no_grad():
        R_wc_t = torch.tensor(R_wc, dtype=torch.float32)
        for axis_name, axis_vec in axes.items():
            for angle in angles:
                R_delta = rodrigues_rotation(axis_vec, np.radians(angle))
                t_wc = R_delta @ t_wc_gt
                t_wc = t_wc / (np.linalg.norm(t_wc) + 1e-10)
                t_wc_t = torch.tensor(t_wc, dtype=torch.float32)

                F = solver._build_F_from_wc(R_wc_t, t_wc_t)
                cost = solver.compute_cost_matrix(F)
                transport, _ = solver.unbalanced_sinkhorn_algorithm(
                    cost_matrix=cost,
                    epsilon=epsilon,
                    rho=rho,
                    max_iter=sinkhorn_max_iter,
                    gate_mask=solver._last_gate_mask,
                )
                eps_actual = solver._last_sinkhorn_epsilon
                rho_actual = solver._last_sinkhorn_rho
                if eps_actual is None or rho_actual is None:
                    raise RuntimeError("Sinkhorn failed to set eps/rho for landscape.")

                scores = compute_all_scores(transport, cost, a, b, eps_actual, rho_actual)
                scores_by_axis["avg_cost"][axis_name].append(scores["avg_cost"])
                scores_by_axis["full_uot"][axis_name].append(scores["full_uot"])
                records.append({
                    "axis": axis_name,
                    "angle_deg": float(angle),
                    "avg_cost": scores["avg_cost"],
                    "full_uot": scores["full_uot"],
                    "T_sum": scores["T_sum"],
                })

    # Save CSV
    csv_path = os.path.join(results_dir, "translation_landscape.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["axis", "angle_deg", "avg_cost", "full_uot", "T_sum"])
        writer.writeheader()
        writer.writerows(records)

    # Plot
    fig, axes_plot = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for axis_name in axes:
        axes_plot[0].plot(angles, scores_by_axis["avg_cost"][axis_name], label=axis_name)
        axes_plot[1].plot(angles, scores_by_axis["full_uot"][axis_name], label=axis_name)
    for ax in axes_plot:
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Translation perturbation (deg)")
        ax.grid(True, alpha=0.3)
    axes_plot[0].set_title("avg_cost landscape")
    axes_plot[0].set_ylabel("avg_cost")
    axes_plot[1].set_title("full_uot landscape")
    axes_plot[1].set_ylabel("full_uot")
    axes_plot[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "translation_landscape.png"), dpi=150)
    plt.close()

    # Print summary of minima
    for score_name in ["avg_cost", "full_uot"]:
        print(f"\nMinima summary ({score_name})")
        for axis_name in axes:
            values = np.array(scores_by_axis[score_name][axis_name])
            min_idx = int(np.argmin(values))
            min_angle = angles[min_idx]
            gt_idx = int(np.where(np.isclose(angles, 0.0))[0][0])
            gt_value = values[gt_idx]
            print(
                f"  axis={axis_name}: min at {min_angle:.1f} deg "
                f"(value={values[min_idx]:.4f}), "
                f"GT=0deg value={gt_value:.4f}"
            )


def run_step_xvii_translation_landscape_with_clip():
    """Step XVII: translation landscape with epipolar gating (epi_clip)."""
    print("=" * 70)
    print("Step XVII: Translation Landscape with epi_clip")
    print("=" * 70)

    idx1, idx2 = 0, 10
    epi_clip = float(os.getenv("STEP_XVII_EPI_CLIP", "10.0"))
    max_angle = float(os.getenv("STEP_XVII_MAX_ANGLE", "60"))
    angle_step = float(os.getenv("STEP_XVII_ANGLE_STEP", "10"))
    sinkhorn_max_iter = int(os.getenv("STEP_XVII_SINKHORN_ITERS", "120"))

    data1 = load_gaussians(idx1)
    data2 = load_gaussians(idx2)
    g1 = data1["original_gaussians"]
    g2 = data2["original_gaussians"]
    ot_mass1 = data1.get("ot_mass", None)
    ot_mass2 = data2.get("ot_mass", None)

    cameras, images = load_colmap_cameras()
    cam1 = get_colmap_camera_params(cameras, images, f"{idx1:04d}.png")
    cam2 = get_colmap_camera_params(cameras, images, f"{idx2:04d}.png")
    K = cam1["K"]

    R_wc, t_wc_gt = compute_relative_pose_wc(cam1, cam2)

    solver = OptimalTransportSolver(
        gaussians1=g1,
        gaussians2=g2,
        k1=K,
        k2=K,
        device="cpu",
        epipolar_mode="sampson",
        lambda_color=0.0,
        lambda_cov=0.0,
        lambda_epipolar=1.0,
        sigma_epipolar=400.0,
        epi_clip=epi_clip,
        ot_mass1=ot_mass1,
        ot_mass2=ot_mass2,
    )

    epsilon = 0.05
    rho = 0.5
    a = solver.alpha1 / solver.alpha1.sum()
    b = solver.alpha2 / solver.alpha2.sum()

    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    angles = np.arange(-max_angle, max_angle + 1e-6, angle_step)

    results_dir = os.path.join(
        project_root, "results", f"step_xvii_landscape_clip_{_format_float_for_path(epi_clip)}"
    )
    os.makedirs(results_dir, exist_ok=True)

    scores_by_axis = {
        "avg_cost": {k: [] for k in axes},
        "full_uot": {k: [] for k in axes},
    }
    records = []

    from src.oracle_study.objective_func.test_step_f_score_functions import compute_all_scores
    import matplotlib.pyplot as plt

    with torch.no_grad():
        R_wc_t = torch.tensor(R_wc, dtype=torch.float32)
        for axis_name, axis_vec in axes.items():
            for angle in angles:
                R_delta = rodrigues_rotation(axis_vec, np.radians(angle))
                t_wc = R_delta @ t_wc_gt
                t_wc = t_wc / (np.linalg.norm(t_wc) + 1e-10)
                t_wc_t = torch.tensor(t_wc, dtype=torch.float32)

                F = solver._build_F_from_wc(R_wc_t, t_wc_t)
                cost = solver.compute_cost_matrix(F)
                transport, _ = solver.unbalanced_sinkhorn_algorithm(
                    cost_matrix=cost,
                    epsilon=epsilon,
                    rho=rho,
                    max_iter=sinkhorn_max_iter,
                    gate_mask=solver._last_gate_mask,
                )
                eps_actual = solver._last_sinkhorn_epsilon
                rho_actual = solver._last_sinkhorn_rho
                scores = compute_all_scores(transport, cost, a, b, eps_actual, rho_actual)
                scores_by_axis["avg_cost"][axis_name].append(scores["avg_cost"])
                scores_by_axis["full_uot"][axis_name].append(scores["full_uot"])
                records.append({
                    "axis": axis_name,
                    "angle_deg": float(angle),
                    "avg_cost": scores["avg_cost"],
                    "full_uot": scores["full_uot"],
                    "T_sum": scores["T_sum"],
                })

    csv_path = os.path.join(results_dir, "translation_landscape.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["axis", "angle_deg", "avg_cost", "full_uot", "T_sum"])
        writer.writeheader()
        writer.writerows(records)

    fig, axes_plot = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for axis_name in axes:
        axes_plot[0].plot(angles, scores_by_axis["avg_cost"][axis_name], label=axis_name)
        axes_plot[1].plot(angles, scores_by_axis["full_uot"][axis_name], label=axis_name)
    for ax in axes_plot:
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Translation perturbation (deg)")
        ax.grid(True, alpha=0.3)
    axes_plot[0].set_title("avg_cost landscape")
    axes_plot[0].set_ylabel("avg_cost")
    axes_plot[1].set_title("full_uot landscape")
    axes_plot[1].set_ylabel("full_uot")
    axes_plot[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "translation_landscape.png"), dpi=150)
    plt.close()

    for score_name in ["avg_cost", "full_uot"]:
        print(f"\nMinima summary ({score_name})")
        for axis_name in axes:
            values = np.array(scores_by_axis[score_name][axis_name])
            min_idx = int(np.argmin(values))
            min_angle = angles[min_idx]
            gt_idx = int(np.where(np.isclose(angles, 0.0))[0][0])
            gt_value = values[gt_idx]
            print(
                f"  axis={axis_name}: min at {min_angle:.1f} deg "
                f"(value={values[min_idx]:.4f}), "
                f"GT=0deg value={gt_value:.4f}"
            )


def run_step_xviii_differentiable_transport():
    """Step XVIII: compare differentiable_transport on full_uot."""
    print("=" * 70)
    print("Step XVIII: Differentiable Transport Comparison")
    print("=" * 70)

    idx1, idx2 = 0, 10
    init_rot_error_deg = 30
    max_iter = int(os.getenv("STEP_XVIII_MAX_ITER", "80"))

    for diff in [False, True]:
        res = test_pose_optimization(
            idx1=idx1,
            idx2=idx2,
            init_rot_error_deg=init_rot_error_deg,
            score_type="full_uot",
            epsilon_annealing=True,
            max_iter=max_iter,
            rot_lr=1e-3,
            trans_lr=1e-4,
            differentiable_transport=diff,
        )
        if res:
            print(
                f"  diff={diff}: final R={res['final_rot_err']:.2f}, "
                f"final t={res['final_trans_err']:.2f}"
            )


if __name__ == "__main__":
    if os.getenv("SKIP_QUICK_TEST", "0") != "1":
        # Quick test first
        result = test_pose_optimization(
            idx1=0,
            idx2=10,
            init_rot_error_deg=10,
            score_type="avg_cost",
            epsilon_annealing=False,
            max_iter=100,
        )

        if result:
            print("\n" + "=" * 70)
            print("Quick test completed successfully!")
            print("=" * 70)

    # Uncomment to run full test suite
    # run_comprehensive_test()

    if os.getenv("RUN_STEP_IV", "0") == "1":
        run_step_iv_cheirality_tests()
    if os.getenv("RUN_STEP_V", "0") == "1":
        run_step_v_robust_tests()
    if os.getenv("RUN_STEP_VI", "0") == "1":
        run_step_vi_two_stage_test()
    if os.getenv("RUN_STEP_VII", "0") == "1":
        run_step_vii_translation_landscape()
    if os.getenv("RUN_STEP_XVI", "0") == "1":
        run_step_xvi_rt_decoupled()
    if os.getenv("RUN_STEP_XVII", "0") == "1":
        run_step_xvii_translation_landscape_with_clip()
    if os.getenv("RUN_STEP_XVIII", "0") == "1":
        run_step_xviii_differentiable_transport()
