"""
Optimization diagnostics utilities for camera pose estimation.
Provides unified, non-redundant diagnostic reporting and visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import List, Optional
import torch


def save_optimization_diagnostics_unified(
    output_dir: str,
    loss_history: List[float],
    q_param_history: List[torch.Tensor],
    t_param_history: List[torch.Tensor],
    s_history: List[torch.Tensor],
    q_grad_history: List[Optional[torch.Tensor]],
    t_grad_history: List[Optional[torch.Tensor]],
    s_grad_history: List[torch.Tensor],
    R_history: List[torch.Tensor],
    E_history: List[torch.Tensor]
) -> None:
    """
    統合された診断情報の保存（冗長性を排除し、必要最小限の画像とテキストファイル1つに集約）
    
    Args:
        output_dir: 診断ファイルを保存するディレクトリ
        loss_history: 各イテレーションでの損失値
        q_param_history: 各イテレーションでの四元数パラメータ (S³)
        t_param_history: 各イテレーションでの並進単位ベクトル (S²)
        s_history: 各イテレーションでのスケールパラメータ
        q_grad_history: 各イテレーションでのqの勾配
        t_grad_history: 各イテレーションでのtの勾配
        s_grad_history: 各イテレーションでのsの勾配
        R_history: 各イテレーションでの回転行列
        E_history: 各イテレーションでのEssential matrix
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # データ変換
    iterations = range(len(loss_history))
    q_params_np = np.array([q.cpu().numpy() for q in q_param_history])
    t_params_np = np.array([t.cpu().numpy() for t in t_param_history])
    s_history_np = np.array([s.cpu().numpy().item() for s in s_history])
    q_grad_norms = np.array([g.norm().item() if g is not None else 0 for g in q_grad_history])
    t_grad_norms = np.array([g.norm().item() if g is not None else 0 for g in t_grad_history])
    s_grad_norms = np.array([g.norm().item() for g in s_grad_history])
    R_params_np = np.array([R.cpu().numpy() for R in R_history])
    E_params_np = np.array([E.cpu().numpy() for E in E_history])
    
    # === 1. 統合された最適化サマリープロット (重要な指標のみ) ===
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss曲線
    ax1.plot(iterations, loss_history, 'b-', linewidth=2)
    ax1.set_title('Loss Evolution')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # 勾配ノルム（対数スケール）
    ax2.semilogy(iterations, q_grad_norms, 'r-', label='‖grad q‖', linewidth=2)
    ax2.semilogy(iterations, t_grad_norms, 'b-', label='‖grad t̂‖', linewidth=2)
    ax2.semilogy(iterations, s_grad_norms, 'g-', label='‖grad log s‖', linewidth=2)
    ax2.set_title('Gradient Norms')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Norm (log scale)')
    ax2.legend()
    ax2.grid(True)
    
    # スケール推移
    ax3.plot(iterations, s_history_np, 'g-', linewidth=2)
    ax3.set_title('Scale Evolution')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Scale')
    ax3.grid(True)
    
    # t̂ノルム推移
    t_norms = np.array([np.linalg.norm(t) for t in t_params_np])
    ax4.plot(iterations, t_norms, 'm-', linewidth=2)
    ax4.axhline(y=0.2, color='r', linestyle='--', alpha=0.7, label='Target min (0.2)')
    ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Target max (1.0)')
    ax4.set_title('Translation Unit Vector Norm')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('‖t̂‖')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimization_summary.png'), dpi=150)
    plt.close()
    
    # === 2. Essential Matrix特異値分析（最終のみ） ===
    E_final = E_params_np[-1]
    U, S, Vh = np.linalg.svd(E_final)
    
    # === テキストレポート（統合された1ファイル） ===
    summary_path = os.path.join(output_dir, 'optimization_report.txt')
    with open(summary_path, 'w') as f:
        f.write("OPTIMIZATION REPORT\n")
        f.write("==================\n\n")
        
        # 基本情報
        f.write("1. BASIC INFORMATION\n")
        f.write("-------------------\n")
        f.write(f"Total iterations: {len(loss_history)}\n")
        f.write(f"Initial loss: {loss_history[0]:.8f}\n")
        f.write(f"Final loss: {loss_history[-1]:.8f}\n")
        f.write(f"Loss reduction: {((loss_history[0] - loss_history[-1]) / loss_history[0] * 100):.2f}%\n\n")
        
        # スケール分析
        f.write("2. SCALE PARAMETER ANALYSIS\n")
        f.write("---------------------------\n")
        f.write(f"Initial scale: {s_history_np[0]:.6f}\n")
        f.write(f"Final scale: {s_history_np[-1]:.6f}\n")
        f.write(f"Scale change: {s_history_np[-1] - s_history_np[0]:.6f}\n")
        f.write(f"Scale stability (std): {np.std(s_history_np):.6f}\n\n")
        
        # 勾配分析
        f.write("3. GRADIENT ANALYSIS\n")
        f.write("-------------------\n")
        f.write(f"Final q grad norm: {q_grad_norms[-1]:.8f}\n")
        f.write(f"Final t grad norm: {t_grad_norms[-1]:.8f}\n")
        f.write(f"Final s grad norm: {s_grad_norms[-1]:.8f}\n")
        f.write(f"Mean q/t grad ratio: {np.mean(q_grad_norms / (t_grad_norms + 1e-10)):.3f}\n\n")
        
        # 姿勢パラメータ分析
        f.write("4. POSE PARAMETER ANALYSIS\n")
        f.write("-------------------------\n")
        f.write(f"Final quaternion norm: {np.linalg.norm(q_params_np[-1]):.6f}\n")
        f.write(f"Final t̂ norm: {t_norms[-1]:.4f}\n")
        f.write(f"t̂ norm in target range [0.2, 1.0]: {'Yes' if 0.2 <= t_norms[-1] <= 1.0 else 'No'}\n")
        
        # 回転行列の詳細
        R_final = R_params_np[-1]
        f.write(f"Final R determinant: {np.linalg.det(R_final):.6f}\n")
        f.write(f"R orthogonality error: {np.linalg.norm(R_final @ R_final.T - np.eye(3)):.8f}\n\n")
        
        # Essential Matrix分析
        f.write("5. ESSENTIAL MATRIX ANALYSIS\n")
        f.write("---------------------------\n")
        f.write(f"Singular values: [{S[0]:.6f}, {S[1]:.6f}, {S[2]:.6f}]\n")
        f.write(f"σ1/σ2 ratio: {S[0]/S[1]:.3f} (ideal: 1.0)\n")
        f.write(f"σ3 value: {S[2]:.6f} (ideal: 0.0)\n")
        f.write(f"Essential constraint satisfaction: {'Good' if S[0]/S[1] < 1.2 and S[2] < 0.05 else 'Poor'}\n\n")
        
        # 機能別分析
        f.write("6. FEATURE ANALYSIS\n")
        f.write("------------------\n")
        f.write("Cheirality Constraint: Applied to prevent negative depth solutions\n")
        f.write("Barycentric Reprojection: Used for efficient O(K1+K2) computation\n")
        f.write("Parallax Angle Penalty: Applied to prevent degenerate parallel rays\n")
        f.write("Scale Optimization: Enabled to resolve metric ambiguity\n\n")
        
        # 収束分析
        f.write("7. CONVERGENCE ANALYSIS\n")
        f.write("----------------------\n")
        if len(loss_history) > 10:
            recent_loss_std = np.std(loss_history[-10:])
            f.write(f"Recent loss stability (last 10 iter std): {recent_loss_std:.8f}\n")
        f.write(f"Converged: {'Yes' if q_grad_norms[-1] < 1e-3 and t_grad_norms[-1] < 1e-3 else 'No'}\n")
    
    print(f"Saved unified optimization diagnostics to {output_dir}")
    print(f"Generated files: optimization_summary.png, optimization_report.txt")


def mean_parallax_angle(
    R_wc: torch.Tensor,           # (3,3)  world→cam2 回転
    transport: torch.Tensor,      # (K1,K2) Sinkhorn の輸送行列
    k1: torch.Tensor,             # (3,3) カメラ1内部パラメータ
    k2: torch.Tensor,             # (3,3) カメラ2内部パラメータ
    means1: torch.Tensor,         # (K1,2) 画像1のガウス中心
    means2: torch.Tensor,         # (K2,2) 画像2のガウス中心
    device: torch.device
) -> torch.Tensor:
    """
    画像1/2 の射線ベクトル r1, r2 の平均パララックス角 θ̄ を返す。
    射線ペアの重みには Sinkhorn の輸送量を使う（soft-inlier）。
    
    Args:
        R_wc: world→cam2 回転行列
        transport: Sinkhorn輸送行列
        k1, k2: カメラ内部パラメータ
        means1, means2: ガウス中心座標
        device: 計算デバイス
        
    Returns:
        mean_theta: 平均パララックス角 (rad)
    """
    K1_inv = torch.inverse(k1)
    K2_inv = torch.inverse(k2)

    # ------------- 射線ベクトル r1・r2 -----------------
    ones1 = torch.ones((means1.shape[0], 1), device=device)
    ones2 = torch.ones((means2.shape[0], 1), device=device)

    # 同次→正規化射線 (ワールド座標系)
    p1_cam1 = (K1_inv @ torch.cat([means1, ones1], 1).T).T          # (K1,3)
    p2_cam2 = (K2_inv @ torch.cat([means2, ones2], 1).T).T          # (K2,3)

    r1 = p1_cam1 / (p1_cam1.norm(dim=1, keepdim=True) + 1e-12)           # (K1,3)
    r2_cam1 = (R_wc.T @ p2_cam2.T).T                                     # cam2→cam1
    r2 = r2_cam1 / (r2_cam1.norm(dim=1, keepdim=True) + 1e-12)           # (K2,3)

    # ------------- cosθ の加重平均 ---------------------
    #   w_ij = transport / transport.sum()
    w = transport / (transport.sum() + 1e-12)
    cos_theta = (r1 @ r2.t()).clamp(-1+1e-6, 1-1e-6)                     # (K1,K2)

    mean_cos = (w * cos_theta).sum()
    mean_theta = torch.acos(mean_cos)                                    # scalar rad
    return mean_theta


def save_optimization_diagnostics_SE3(
    output_dir: str,
    loss_history: list,
    param_history: dict,
    grad_history: dict
) -> None:
    """
    分離したSE(3)パラメータの最適化過程に関する詳細な診断情報を保存する
    
    回転と並進を分離して最適化したSE(3)パラメータについて、以下の診断情報を生成・保存します：
    - 損失軌跡の分析
    - パラメータ進化の分析（回転と並進の各成分）
    - 勾配挙動の分析
    - 収束性分析
    - 3D軌跡可視化
    - テキスト形式のサマリーレポート
    
    Args:
        output_dir: 診断ファイルを保存するディレクトリ
        loss_history: イテレーションごとの損失値リスト
        param_history: パラメータ履歴の辞書（'rot_vec'と'trans_vec'を含む）
        grad_history: 勾配履歴の辞書（'rot_vec'と'trans_vec'を含む）
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D
    
    # 出力ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 履歴をNumPy配列に変換
    param_history_np = {}
    grad_history_np = {}
    
    for param_name, history in param_history.items():
        param_history_np[param_name] = np.array([p.detach().cpu().numpy() for p in history])
        
    for param_name, history in grad_history.items():
        grad_history_np[param_name] = np.array([g.detach().cpu().numpy() if g is not None 
                                            else np.zeros_like(param_history_np[param_name][0]) 
                                            for g in history])
    
    # イテレーション数
    iterations = range(len(loss_history))
    
    # ======================= 統合された最適化プロット =======================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 損失軌跡
    ax1.plot(iterations, loss_history, 'b-', linewidth=2)
    ax1.set_title('Loss Evolution')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # 2. SE(3)パラメータ軌跡
    rot_data = param_history_np['rot_vec']
    trans_data = param_history_np['trans_vec']
    
    ax2.plot(iterations, rot_data[:, 0], 'r-', label='wx', linewidth=2)
    ax2.plot(iterations, rot_data[:, 1], 'g-', label='wy', linewidth=2)
    ax2.plot(iterations, rot_data[:, 2], 'b-', label='wz', linewidth=2)
    ax2.set_title('Rotation Components')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Value')
    ax2.legend()
    ax2.grid(True)
    
    # 3. 並進成分
    ax3.plot(iterations, trans_data[:, 0], 'r-', label='tx', linewidth=2)
    ax3.plot(iterations, trans_data[:, 1], 'g-', label='ty', linewidth=2)
    ax3.plot(iterations, trans_data[:, 2], 'b-', label='tz', linewidth=2)
    ax3.set_title('Translation Components')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Value')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 勾配ノルム
    rot_grad_data = grad_history_np['rot_vec']
    trans_grad_data = grad_history_np['trans_vec']
    
    rot_grad_magnitude = np.linalg.norm(rot_grad_data, axis=1)
    trans_grad_magnitude = np.linalg.norm(trans_grad_data, axis=1)
    total_grad_magnitude = np.sqrt(rot_grad_magnitude**2 + trans_grad_magnitude**2)
    
    ax4.semilogy(iterations, total_grad_magnitude, 'k-', linewidth=2, label='Total')
    ax4.semilogy(iterations, rot_grad_magnitude, 'r-', linewidth=1.5, label='Rotation')
    ax4.semilogy(iterations, trans_grad_magnitude, 'b-', linewidth=1.5, label='Translation')
    ax4.set_title('Gradient Magnitude')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Gradient Norm (log scale)')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'se3_optimization_summary.png'), dpi=150)
    plt.close()
    
    # =============== テキスト形式のサマリーレポート ===============
    with open(os.path.join(output_dir, 'se3_optimization_report.txt'), 'w') as f:
        f.write("SE(3) OPTIMIZATION REPORT\n")
        f.write("========================\n\n")
        
        # 基本情報
        f.write("1. BASIC INFORMATION\n")
        f.write("-------------------\n")
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss != 0 else 0
        
        f.write(f"Total iterations: {len(loss_history)}\n")
        f.write(f"Initial loss: {initial_loss:.8f}\n")
        f.write(f"Final loss: {final_loss:.8f}\n")
        f.write(f"Loss reduction: {loss_reduction:.2f}%\n\n")
        
        # パラメータ分析
        f.write("2. SE(3) PARAMETER ANALYSIS\n")
        f.write("--------------------------\n")
        
        # 回転成分
        rot_change = np.linalg.norm(rot_data[-1] - rot_data[0])
        f.write(f"Rotation initial: {rot_data[0]}\n")
        f.write(f"Rotation final: {rot_data[-1]}\n")
        f.write(f"Total rotation change: {rot_change:.6f}\n\n")
        
        # 並進成分
        trans_change = np.linalg.norm(trans_data[-1] - trans_data[0])
        f.write(f"Translation initial: {trans_data[0]}\n")
        f.write(f"Translation final: {trans_data[-1]}\n")
        f.write(f"Total translation change: {trans_change:.6f}\n\n")
        
        # 勾配分析
        f.write("3. GRADIENT ANALYSIS\n")
        f.write("-------------------\n")
        f.write(f"Final rotation gradient norm: {rot_grad_magnitude[-1]:.8f}\n")
        f.write(f"Final translation gradient norm: {trans_grad_magnitude[-1]:.8f}\n")
        f.write(f"Final total gradient norm: {total_grad_magnitude[-1]:.8f}\n")
        
        # 勾配比率
        ratio = rot_grad_magnitude / (trans_grad_magnitude + 1e-10)
        avg_ratio = np.mean(ratio)
        f.write(f"Average rotation/translation gradient ratio: {avg_ratio:.4f}\n\n")
        
        # 収束分析
        f.write("4. CONVERGENCE ANALYSIS\n")
        f.write("----------------------\n")
        
        # 単調減少性チェック
        is_monotonic = all(loss_history[i] >= loss_history[i+1] for i in range(len(loss_history)-1))
        f.write(f"Loss decreases monotonically: {is_monotonic}\n")
        
        # 収束判定
        converged = total_grad_magnitude[-1] < 1e-3 and loss_reduction > 10
        f.write(f"Converged: {'Yes' if converged else 'No'}\n")
        
        if len(loss_history) > 10:
            recent_loss_std = np.std(loss_history[-10:])
            f.write(f"Recent loss stability (last 10 iter std): {recent_loss_std:.8f}\n")
    
    print(f"Saved SE(3) optimization diagnostics to {output_dir}")
    print(f"Generated files: se3_optimization_summary.png, se3_optimization_report.txt")


def debug_transport_matrix_issues(
    output_dir: str,
    sinkhorn_metrics: dict,
    cost_statistics: dict,
    alpha_beta_stats: dict,
    epsilon_schedule: list,
    loss_components: dict,
    transport_snapshots: dict
) -> None:
    """
    輸送行列の問題をデバッグするための詳細診断
    
    Args:
        output_dir: 診断ファイルを保存するディレクトリ
        sinkhorn_metrics: Sinkhorn収束メトリクス
        cost_statistics: コスト行列の統計情報
        alpha_beta_stats: マージナル質量の統計
        epsilon_schedule: εスケーリングの履歴
        loss_components: 損失成分の履歴
        transport_snapshots: 各εレベルでの輸送行列スナップショット
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # === 1. Sinkhorn収束解析 ===
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1-1. マージナル誤差の推移
    if 'marginal_errors' in sinkhorn_metrics:
        iterations = range(len(sinkhorn_metrics['marginal_errors']))
        ax1.semilogy(iterations, sinkhorn_metrics['marginal_errors'], 'b-', linewidth=2)
        ax1.axhline(y=1e-6, color='r', linestyle='--', alpha=0.7, label='Target tolerance')
        ax1.set_title('Sinkhorn Marginal Error')
        ax1.set_xlabel('Sinkhorn Iteration')
        ax1.set_ylabel('Error (log scale)')
        ax1.legend()
        ax1.grid(True)
    
    # 1-2. ε-スケーリング履歴
    if epsilon_schedule:
        ax2.plot(range(len(epsilon_schedule)), epsilon_schedule, 'g-o', linewidth=2, markersize=6)
        ax2.set_title('Epsilon Scaling Schedule')
        ax2.set_xlabel('Scale Level')
        ax2.set_ylabel('Epsilon Value')
        ax2.set_yscale('log')
        ax2.grid(True)
    
    # 1-3. α/β統計
    if alpha_beta_stats:
        alpha_stats = alpha_beta_stats.get('alpha', {})
        beta_stats = alpha_beta_stats.get('beta', {})
        
        categories = ['min', 'max', 'mean', 'std']
        alpha_values = [alpha_stats.get(cat, 0) for cat in categories]
        beta_values = [beta_stats.get(cat, 0) for cat in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax3.bar(x - width/2, alpha_values, width, label='Alpha', alpha=0.7)
        ax3.bar(x + width/2, beta_values, width, label='Beta', alpha=0.7)
        ax3.set_title('Alpha/Beta Marginal Statistics')
        ax3.set_ylabel('Value')
        ax3.set_yscale('log')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(True)
    
    # 1-4. コスト正規化統計
    if cost_statistics:
        cost_levels = list(cost_statistics.keys())
        p95_values = [cost_statistics[level].get('p95', 0) for level in cost_levels]
        p99_values = [cost_statistics[level].get('p99', 0) for level in cost_levels]
        mean_values = [cost_statistics[level].get('mean', 0) for level in cost_levels]
        
        ax4.plot(cost_levels, p95_values, 'r-o', label='P95', linewidth=2)
        ax4.plot(cost_levels, p99_values, 'b-o', label='P99', linewidth=2)
        ax4.plot(cost_levels, mean_values, 'g-o', label='Mean', linewidth=2)
        ax4.set_title('Cost Matrix Statistics by Level')
        ax4.set_xlabel('Cost Level')
        ax4.set_ylabel('Value')
        ax4.legend()
        ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'transport_debug_analysis.png'), dpi=150)
    plt.close()
    
    # === 2. 輸送行列"帯"パターン解析 ===
    if transport_snapshots:
        n_snapshots = len(transport_snapshots)
        fig, axes = plt.subplots(2, min(n_snapshots, 4), figsize=(16, 8))
        
        if n_snapshots == 1:
            axes = [axes]
        if n_snapshots <= 4:
            axes = axes.reshape(-1)
        
        for idx, (epsilon_val, transport_data) in enumerate(transport_snapshots.items()):
            if idx >= 4:  # 最大4つまで表示
                break
                
            T = transport_data['matrix']
            
            # 上段: 輸送行列ヒートマップ
            ax_heat = axes[idx] if n_snapshots <= 4 else axes[0, idx]
            im = ax_heat.imshow(T, cmap='hot', aspect='auto')
            ax_heat.set_title(f'Transport Matrix\nε={epsilon_val:.4f}')
            ax_heat.set_xlabel('Image 2 Gaussians')
            ax_heat.set_ylabel('Image 1 Gaussians')
            
            # 下段: 行/列和の分布
            if n_snapshots > 4:
                ax_dist = axes[1, idx]
            else:
                ax_dist = plt.subplot(2, min(n_snapshots, 4), idx + min(n_snapshots, 4) + 1)
            
            row_sums = T.sum(axis=1)
            col_sums = T.sum(axis=0)
            
            ax_dist.hist(row_sums, bins=30, alpha=0.7, label='Row sums', density=True)
            ax_dist.hist(col_sums, bins=30, alpha=0.7, label='Col sums', density=True)
            ax_dist.set_title(f'Marginal Distribution\nε={epsilon_val:.4f}')
            ax_dist.set_xlabel('Sum Value')
            ax_dist.set_ylabel('Density')
            ax_dist.legend()
            ax_dist.grid(True)
            
            # "帯"検出メトリクス
            row_max_ratio = np.max(row_sums) / (np.mean(row_sums) + 1e-10)
            col_max_ratio = np.max(col_sums) / (np.mean(col_sums) + 1e-10)
            
            # テキスト情報として保存
            with open(os.path.join(output_dir, f'transport_analysis_eps_{epsilon_val:.4f}.txt'), 'w') as f:
                f.write(f"TRANSPORT MATRIX ANALYSIS - ε={epsilon_val:.4f}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Matrix shape: {T.shape}\n")
                f.write(f"Total mass: {T.sum():.6f}\n")
                f.write(f"Max element: {T.max():.6f}\n")
                f.write(f"Min element: {T.min():.6f}\n\n")
                
                f.write("MARGINAL ANALYSIS:\n")
                f.write(f"Row sum - max/mean ratio: {row_max_ratio:.3f}\n")
                f.write(f"Col sum - max/mean ratio: {col_max_ratio:.3f}\n")
                f.write(f"Row sum variance: {np.var(row_sums):.6f}\n")
                f.write(f"Col sum variance: {np.var(col_sums):.6f}\n\n")
                
                # "帯"判定
                if row_max_ratio > 5.0 or col_max_ratio > 5.0:
                    f.write("⚠️  BANDING DETECTED\n")
                    f.write("Possible causes:\n")
                    f.write("- Epsilon too small for this scale\n")
                    f.write("- Early stopping in Sinkhorn\n")
                    f.write("- Cost matrix normalization issues\n")
                    f.write("- Alpha/Beta dynamic range too wide\n")
                else:
                    f.write("✅ Transport matrix appears well-distributed\n")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'transport_snapshots_analysis.png'), dpi=150)
        plt.close()
    
    # === 3. 総合診断レポート ===
    with open(os.path.join(output_dir, 'transport_debug_report.txt'), 'w') as f:
        f.write("TRANSPORT MATRIX DEBUG REPORT\n")
        f.write("=" * 40 + "\n\n")
        
        f.write("1. SINKHORN CONVERGENCE ANALYSIS\n")
        f.write("-" * 35 + "\n")
        if 'marginal_errors' in sinkhorn_metrics:
            final_error = sinkhorn_metrics['marginal_errors'][-1] if sinkhorn_metrics['marginal_errors'] else float('inf')
            converged = final_error < 1e-6
            f.write(f"Final marginal error: {final_error:.2e}\n")
            f.write(f"Converged (< 1e-6): {'Yes' if converged else 'No'}\n")
            
            if not converged:
                f.write("⚠️  Sinkhorn did not converge properly\n")
                f.write("Recommendations:\n")
                f.write("- Increase sinkhorn_max_iter\n")
                f.write("- Relax sinkhorn_tol\n")
                f.write("- Check for numerical instabilities\n")
        else:
            f.write("No Sinkhorn convergence data available\n")
        f.write("\n")
        
        f.write("2. EPSILON SCALING ANALYSIS\n")
        f.write("-" * 30 + "\n")
        if epsilon_schedule:
            f.write(f"Number of scales: {len(epsilon_schedule)}\n")
            f.write(f"Initial epsilon: {epsilon_schedule[0]:.6f}\n")
            f.write(f"Final epsilon: {epsilon_schedule[-1]:.6f}\n")
            f.write(f"Scaling factor: {epsilon_schedule[-1]/epsilon_schedule[0]:.6f}\n")
            
            # 急激な変化をチェック
            ratios = [epsilon_schedule[i+1]/epsilon_schedule[i] for i in range(len(epsilon_schedule)-1)]
            min_ratio = min(ratios)
            if min_ratio < 0.3:
                f.write(f"⚠️  Aggressive epsilon scaling detected (min ratio: {min_ratio:.3f})\n")
                f.write("Consider using scaling_factor=0.5 or scaling_steps=6-8\n")
        f.write("\n")
        
        f.write("3. ALPHA/BETA DYNAMIC RANGE ANALYSIS\n")
        f.write("-" * 38 + "\n")
        if alpha_beta_stats:
            alpha_stats = alpha_beta_stats.get('alpha', {})
            beta_stats = alpha_beta_stats.get('beta', {})
            
            alpha_range = alpha_stats.get('max', 1) / max(alpha_stats.get('min', 1), 1e-10)
            beta_range = beta_stats.get('max', 1) / max(beta_stats.get('min', 1), 1e-10)
            
            f.write(f"Alpha dynamic range: {alpha_range:.2f}\n")
            f.write(f"Beta dynamic range: {beta_range:.2f}\n")
            
            if alpha_range > 100 or beta_range > 100:
                f.write("⚠️  Wide dynamic range detected\n")
                f.write("Recommendations:\n")
                f.write("- Normalize alpha/beta before Sinkhorn\n")
                f.write("- Use log-domain computation\n")
                f.write("- Consider marginal renormalization\n")
        f.write("\n")
        
        f.write("4. COST MATRIX NORMALIZATION ANALYSIS\n")
        f.write("-" * 40 + "\n")
        if cost_statistics:
            f.write("Cost statistics by component:\n")
            for level, stats in cost_statistics.items():
                f.write(f"  {level}:\n")
                f.write(f"    P95: {stats.get('p95', 0):.6f}\n")
                f.write(f"    P99: {stats.get('p99', 0):.6f}\n")
                f.write(f"    Mean: {stats.get('mean', 0):.6f}\n")
                f.write(f"    Clipped ratio: {stats.get('clipped_ratio', 0):.3f}\n")
                
                if stats.get('clipped_ratio', 0) > 0.3:
                    f.write(f"    ⚠️  High clipping ratio for {level}\n")
        f.write("\n")
        
        f.write("5. IMPROVEMENT SUGGESTIONS\n")
        f.write("-" * 28 + "\n")
        f.write("Based on the analysis above:\n\n")
        
        suggestions = []
        
        if 'marginal_errors' in sinkhorn_metrics:
            final_error = sinkhorn_metrics['marginal_errors'][-1] if sinkhorn_metrics['marginal_errors'] else float('inf')
            if final_error > 1e-6:
                suggestions.append("1. Improve Sinkhorn convergence:")
                suggestions.append("   - Increase sinkhorn_max_iter to 2000")
                suggestions.append("   - Set sinkhorn_tol to 1e-7")
                suggestions.append("   - Use marginal_error_type='l1'")
        
        if epsilon_schedule and len(epsilon_schedule) < 6:
            suggestions.append("2. Make epsilon scaling more gradual:")
            suggestions.append("   - Set scaling_steps=6-8")
            suggestions.append("   - Set scaling_factor=0.5")
        
        if alpha_beta_stats:
            alpha_stats = alpha_beta_stats.get('alpha', {})
            beta_stats = alpha_beta_stats.get('beta', {})
            alpha_range = alpha_stats.get('max', 1) / max(alpha_stats.get('min', 1), 1e-10)
            beta_range = beta_stats.get('max', 1) / max(beta_stats.get('min', 1), 1e-10)
            
            if alpha_range > 100 or beta_range > 100:
                suggestions.append("3. Normalize marginals:")
                suggestions.append("   - Add alpha/beta renormalization")
                suggestions.append("   - Use robust scaling for outliers")
        
        if cost_statistics:
            high_clipping = any(stats.get('clipped_ratio', 0) > 0.3 for stats in cost_statistics.values())
            if high_clipping:
                suggestions.append("4. Improve cost normalization:")
                suggestions.append("   - Use P99 instead of P95 for clipping")
                suggestions.append("   - Consider robust z-score normalization")
                suggestions.append("   - Apply sigmoid instead of hard clipping")
        
        if not suggestions:
            suggestions.append("No major issues detected in current configuration.")
        
        for suggestion in suggestions:
            f.write(suggestion + "\n")
    
    print(f"Saved transport matrix debug analysis to {output_dir}")
    print("Generated files: transport_debug_analysis.png, transport_snapshots_analysis.png, transport_debug_report.txt")


def collect_sinkhorn_debug_data(
    cost_matrix: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    epsilon: float,
    transport_matrix: torch.Tensor,
    marginal_errors: List[float] = None
) -> dict:
    """
    Sinkhorn最適化のデバッグデータを収集
    
    Args:
        cost_matrix: コスト行列
        alpha, beta: マージナル制約
        epsilon: 現在のε値
        transport_matrix: 輸送行列
        marginal_errors: マージナル誤差の履歴
        
    Returns:
        デバッグ情報の辞書
    """
    debug_data = {}
    
    # コスト統計
    with torch.no_grad():
        C_flat = cost_matrix.flatten()
        debug_data['cost_stats'] = {
            'mean': C_flat.mean().item(),
            'std': C_flat.std().item(),
            'min': C_flat.min().item(),
            'max': C_flat.max().item(),
            'p95': torch.quantile(C_flat, 0.95).item(),
            'p99': torch.quantile(C_flat, 0.99).item(),
            'median': torch.median(C_flat).item()
        }
        
        # クリッピング比率（P95を超える値の割合）
        p95_val = debug_data['cost_stats']['p95']
        clipped_ratio = (C_flat > p95_val).float().mean().item()
        debug_data['cost_stats']['clipped_ratio'] = clipped_ratio
        
        # α/β統計
        debug_data['alpha_stats'] = {
            'min': alpha.min().item(),
            'max': alpha.max().item(),
            'mean': alpha.mean().item(),
            'std': alpha.std().item()
        }
        
        debug_data['beta_stats'] = {
            'min': beta.min().item(),
            'max': beta.max().item(),
            'mean': beta.mean().item(),
            'std': beta.std().item()
        }
        
        # 輸送行列統計
        debug_data['transport_stats'] = {
            'total_mass': transport_matrix.sum().item(),
            'max_element': transport_matrix.max().item(),
            'min_element': transport_matrix.min().item(),
            'sparsity': (transport_matrix < 1e-6).float().mean().item()
        }
        
        # 行/列和の統計
        row_sums = transport_matrix.sum(dim=1)
        col_sums = transport_matrix.sum(dim=0)
        
        debug_data['marginal_stats'] = {
            'row_sum_var': row_sums.var().item(),
            'col_sum_var': col_sums.var().item(),
            'row_max_ratio': (row_sums.max() / (row_sums.mean() + 1e-10)).item(),
            'col_max_ratio': (col_sums.max() / (col_sums.mean() + 1e-10)).item()
        }
        
        # ε情報
        debug_data['epsilon'] = epsilon
        
        # マージナル誤差
        if marginal_errors is not None:
            debug_data['marginal_errors'] = marginal_errors
            debug_data['final_marginal_error'] = marginal_errors[-1] if marginal_errors else float('inf')
        
        # "帯"検出
        banding_detected = (debug_data['marginal_stats']['row_max_ratio'] > 5.0 or 
                           debug_data['marginal_stats']['col_max_ratio'] > 5.0)
        debug_data['banding_detected'] = banding_detected
    
    return debug_data


def log_optimization_progress(
    iteration: int,
    loss_components: dict,
    gradient_norms: dict,
    parameter_stats: dict,
    epsilon_current: float = None,
    log_file: str = None
) -> None:
    """
    最適化進捗の詳細ログ出力
    
    Args:
        iteration: 現在のイテレーション
        loss_components: 損失成分の辞書
        gradient_norms: 勾配ノルムの辞書
        parameter_stats: パラメータ統計の辞書
        epsilon_current: 現在のε値
        log_file: ログファイルパス
    """
    log_entry = f"Iter {iteration:05d}: "
    
    # 損失成分
    if 'total' in loss_components:
        log_entry += f"Loss={loss_components['total']:.6f} "
    
    for component, value in loss_components.items():
        if component != 'total':
            log_entry += f"{component}={value:.6f} "
    
    # 勾配ノルム
    log_entry += "| Grads: "
    for param, norm in gradient_norms.items():
        log_entry += f"{param}={norm:.4f} "
    
    # パラメータ統計
    if parameter_stats:
        log_entry += "| Params: "
        for param, value in parameter_stats.items():
            if isinstance(value, (int, float)):
                log_entry += f"{param}={value:.4f} "
    
    # ε値
    if epsilon_current is not None:
        log_entry += f"| ε={epsilon_current:.6f}"
    
    # ログ出力
    print(log_entry)
    
    if log_file:
        with open(log_file, 'a') as f:
            f.write(log_entry + "\n")


def save_transport_snapshot(
    transport_matrix: torch.Tensor,
    epsilon: float,
    iteration: int,
    output_dir: str,
    prefix: str = "transport"
) -> str:
    """
    輸送行列のスナップショットを保存
    
    Args:
        transport_matrix: 輸送行列
        epsilon: ε値
        iteration: イテレーション
        output_dir: 出力ディレクトリ
        prefix: ファイル名プレフィックス
        
    Returns:
        保存されたファイルパス
    """
    import os
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"{prefix}_iter_{iteration:05d}_eps_{epsilon:.6f}.png"
    filepath = os.path.join(output_dir, filename)
    
    with torch.no_grad():
        T_np = transport_matrix.cpu().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.imshow(T_np, cmap='hot', aspect='auto', interpolation='nearest')
        plt.colorbar(label='Transport Mass')
        plt.title(f'Transport Matrix - Iter {iteration}, ε={epsilon:.6f}')
        plt.xlabel('Image 2 Gaussians')
        plt.ylabel('Image 1 Gaussians')
        
        # 統計情報をテキストで追加
        total_mass = T_np.sum()
        max_val = T_np.max()
        sparsity = (T_np < 1e-6).mean()
        
        plt.text(0.02, 0.98, f'Total Mass: {total_mass:.4f}\nMax Value: {max_val:.4f}\nSparsity: {sparsity:.3f}',
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150)
        plt.close()
    
    return filepath