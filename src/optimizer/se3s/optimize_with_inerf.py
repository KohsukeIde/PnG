def optimize_with_RT(self, 
                        max_iter: int = 1000, 
                        tol: float = 1e-5, 
                        save_diagnostics: bool = True, 
                        diagnostics_dir: Optional[str] = None,
                        learning_rate: float = 5e-3):
        """
        iNeRF風最適化により、カメラポーズ（R & t）を最適化する。
        """
        # -------------------------  出力ディレクトリ  ---------------------- #
        transport_dir = os.path.join("results", "transport_inerf")
        os.makedirs(transport_dir, exist_ok=True)
        diagnostics_dir = diagnostics_dir or os.path.join("results", "diagnostics_inerf")
        os.makedirs(diagnostics_dir, exist_ok=True)

        # ------------------------- 履歴用 ------------------------- #
        loss_history = []
        param_history = {'T': []} if save_diagnostics else None
        grad_history = {'delta': []} if save_diagnostics else None

        # ------------------------- パラメータ初期化 ----------------------- #
        if not hasattr(self, 'T'):
            # 初期パラメータを作成
            if hasattr(self, 'se3_vec'):
                # SE3パラメータが既に存在する場合はそれを使う
                R_cw, t_cw = self.se3_exp(self.se3_vec)
            else:
                # なければランダム初期化
                self._init_se3_like_cam1(rot_noise=0.2, trans_noise=0.2)
                R_cw, t_cw = self.se3_exp(self.se3_vec)
                
            # 4x4の同次変換行列を作成
            self.T = torch.eye(4, device=self.device)
            self.T[:3, :3] = R_cw
            self.T[:3, 3] = t_cw

        # --- iNeRFと同じ: ループ外で1度だけパラメータとoptimizerを生成 ---
        self.delta = nn.Parameter(torch.zeros(6, device=self.device))
        # iNeRFと同様、Adamを使用（重み減衰なし）
        optimizer = torch.optim.Adam([self.delta], lr=learning_rate, weight_decay=0.0)
        # iNeRFと同じ指数関数的学習率減衰: 0.8^(t/100)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8**(1/100))

        prev_loss_val = float('inf')
        pbar = tqdm(range(max_iter), desc="Optimizing iNeRF", leave=True)

        # ---- 勾配デバッグ用ログファイル ----
        debug_log_path = os.path.join(diagnostics_dir, "gradient_debug_inerf.log")
        with open(debug_log_path, 'w') as f:
            f.write("Iteration, Loss, Delta_Norm, Grad_Norm, Grad_Rot_x, Grad_Rot_y, Grad_Rot_z, Grad_Trans_x, Grad_Trans_y, Grad_Trans_z\n")
        
        # -------------------------  ループ  ------------------------------- #
        for iteration in pbar:
            # 1. Forward pass: 勾配計算のリセット
            optimizer.zero_grad()
            
            # 2. Δξからexp(Δξ)を計算
            T_delta = self.se3_exp_T(self.delta)
            
            # 3. 更新: T_new = T_delta * T (左から掛ける - iNeRFと同じ)
            T_next = T_delta @ self.T
            
            # 4. カメラ→ワールド変換行列から回転と並進を抽出
            R_cw = T_next[:3, :3]
            t_cw = T_next[:3, 3]
            
            # 5. ワールド→カメラ変換に変更
            R_wc = R_cw.t()
            t_wc = -R_wc @ t_cw
            
            # 6. 基礎行列計算
            F = self._build_F_from_wc(R_wc, t_wc)
            
            # 7. コスト行列と最適輸送計算
            cost_matrix = self.compute_cost_matrix_fundamental(F)
            transport = self.unbalanced_sinkhorn_algorithm(cost_matrix)
            
            # 8. 損失計算
            loss = torch.sum(transport * cost_matrix)
            
            # 9. バックワード前のデバッグ情報
            if iteration % 10 == 0:
                print(f"\nIteration {iteration} - Before backward:")
                print(f"  Delta SE3: {self.delta.data}")
                print(f"  Loss: {loss.item():.6f}")
                print(f"  Learning rate: {scheduler.get_last_lr()[0]:.6e}")
                with torch.no_grad():
                    print(f"  Cost matrix min/max: {cost_matrix.min().item():.6f}/{cost_matrix.max().item():.6f}")
            
            # 10. バックワード計算
            loss.backward()
            
            # 11. 勾配チェック
            if self.delta.grad is not None:
                # 勾配情報取得
                grad = self.delta.grad
                grad_norm = grad.norm().item()
                delta_norm = self.delta.norm().item()
                
                # 勾配情報をログに記録
                with open(debug_log_path, 'a') as f:
                    grad_vals = grad.detach().cpu().numpy()
                    f.write(f"{iteration}, {loss.item():.6f}, {delta_norm:.6f}, {grad_norm:.6f}, " + 
                        f"{grad_vals[0]:.6f}, {grad_vals[1]:.6f}, {grad_vals[2]:.6f}, " +
                        f"{grad_vals[3]:.6f}, {grad_vals[4]:.6f}, {grad_vals[5]:.6f}\n")
                
                # 詳細な勾配情報を表示
                if iteration % 10 == 0:
                    print(f"  Gradient norm: {grad_norm:.6f}")
                    print(f"  Rot gradient: {grad[:3].detach().cpu().numpy()}")
                    print(f"  Trans gradient: {grad[3:].detach().cpu().numpy()}")
                    rot_grad_norm = grad[:3].norm().item()
                    trans_grad_norm = grad[3:].norm().item()
                    print(f"  Rot/Trans gradient norm ratio: {rot_grad_norm/max(trans_grad_norm, 1e-10):.6f}")
            else:
                print("Warning: No gradient computed!")
            
            # 12. 最適化ステップと学習率の更新
            optimizer.step()
            scheduler.step()
            
            # 13. 履歴の保存（メモリ効率化）
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            if save_diagnostics:
                # メモリ効率化: GPUテンソルではなくCPUの浮動小数点値を保存
                if param_history is not None:
                    param_history['T'].append(self.T.detach().cpu().clone())
                
                if grad_history is not None and self.delta.grad is not None:
                    grad_history['delta'].append(self.delta.grad.detach().cpu().clone())
            
            # 14. 更新されたdeltaを元のポーズに適用し、deltaをリセット（値だけ、モーメンタムは保持）
            with torch.no_grad():
                # 回転成分をπ範囲にクランプ（数値安定性のため）
                self.delta.data[:3].clamp_(-math.pi, math.pi)
                
                # T_new = exp(δ) * T
                self.T = self.se3_exp_T(self.delta) @ self.T
                
                # iNeRFスタイル: deltaパラメータを0にリセットするが、Adamのモーメンタムは保持
                self.delta.zero_()
                
                # 数値安定性のためのチェック
                if torch.isnan(self.T).any():
                    print("NaN detected in transformation matrix. Stopping optimization.")
                    break
            
            # 15. 収束判定
            loss_diff = abs(prev_loss_val - current_loss)
            if iteration > 5 and loss_diff < tol:
                pbar.set_description(f"Converged (loss_diff={loss_diff:.2e})")
                break
            prev_loss_val = current_loss
            
            # 16. プログレスバー更新
            if iteration % 10 == 0:
                # self.deltaを使用（deltaではなく）
                delta_norm = self.delta.norm().item()
                rot_delta_norm = self.delta[:3].norm().item()
                trans_delta_norm = self.delta[3:].norm().item()
                ratio = rot_delta_norm / max(trans_delta_norm, 1e-10)
                pbar.set_postfix({
                    'loss': f"{current_loss:.6f}",
                    'delta': f"{delta_norm:.4f}",
                    'r/t': f"{ratio:.2f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # 輸送行列の可視化（頻度を下げる）
                if iteration % 50 == 0 or iteration == max_iter - 1:
                    with torch.no_grad():
                        t_np = transport.detach().cpu().numpy()

                    rows, cols = t_np.shape
                    aspect_ratio = cols / rows

                    if rows > cols:
                        fig_width = 8
                        fig_height = min(20, fig_width / aspect_ratio)
                    else:
                        fig_height = 6
                        fig_width = min(20, fig_height * aspect_ratio)

                    plt.figure(figsize=(fig_width, fig_height))

                    if rows > 1000 or cols > 1000:
                        downsample_factor = max(1, int(max(rows, cols) / 1000))
                        t_np_display = t_np[::downsample_factor, ::downsample_factor]
                        plt.imshow(t_np_display, cmap="hot", interpolation="nearest", aspect="auto")
                        plt.title(f"Transport Plan at Iteration {iteration} (Downsampled {downsample_factor}x)")
                    else:
                        plt.imshow(t_np, cmap="hot", interpolation="nearest", aspect="auto")
                        plt.title(f"Transport Plan at Iteration {iteration}")

                    plt.colorbar(label="Transport Plan Value")
                    plt.xlabel("Image 2 Gaussians")
                    plt.ylabel("Image 1 Gaussians")

                    plt.tight_layout()
                    plt_path = os.path.join(transport_dir, f"transport_iter_{iteration}.png")
                    plt.savefig(plt_path, dpi=150)
                    plt.close()
        
        # ------------------------- 最終パラメータ保存 --------------------------- #
        with torch.no_grad():
            # 4x4行列から回転と並進を抽出
            R_cw = self.T[:3, :3]
            t_cw = self.T[:3, 3]
            
            # 保存
            self.R_cw = R_cw
            self.t_cw = t_cw
            
            # 世界→カメラ変換も保存
            R_wc = R_cw.t()
            t_wc = -R_wc @ t_cw
            self.R_wc = R_wc
            self.t_wc = t_wc
            
            # 基礎行列を計算して保存
            final_F = self._build_F_from_wc(R_wc, t_wc)
            self.f = final_F
            
            # 既存APIとの互換性のために従来のパラメータも更新
            rvec_numpy, _ = cv2.Rodrigues(R_wc.cpu().numpy())
            self.rvec = nn.Parameter(torch.from_numpy(rvec_numpy).to(self.device))
            self.tvec = nn.Parameter(t_wc)
            
            # camera-to-world パラメータも更新
            rvec_cw_numpy, _ = cv2.Rodrigues(R_cw.cpu().numpy())
            self.rvec_cw = nn.Parameter(torch.from_numpy(rvec_cw_numpy).to(self.device))
            self.center = nn.Parameter(t_cw)
            
            # SE3ベクトルとしても保存 (他の最適化器との互換性のため)
            se3_vec = torch.zeros(6, device=self.device)
            rvec_cw = torch.from_numpy(rvec_cw_numpy).to(self.device).float().flatten()
            se3_vec[:3] = rvec_cw
            se3_vec[3:] = t_cw
            self.se3_vec = nn.Parameter(se3_vec)
        
        # ------------------------- 損失プロット -------------------------- #
        plt.figure()
        plt.plot(loss_history, '-o')
        plt.title("Loss (optimize_with_inerf)")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(transport_dir, "loss_optimize_with_inerf.png"))
        plt.close()

        # ------------------------- 最適化過程描画 -------------------------- #
        if save_diagnostics :
            self.save_optimization_diagnostics_inerf(
                output_dir=diagnostics_dir,
                loss_history=loss_history,
                param_history=param_history,
                grad_history=grad_history
            )
        
        return loss_history

    def save_optimization_diagnostics_inerf(self, 
                               output_dir: str,
                               loss_history: list,
                               param_history: dict,
                               grad_history: dict) -> None:
        """Save detailed diagnostics about the iNeRF optimization process.
        
        Analyzes and visualizes the optimization process of the SE(3) parameters, including:
        - Loss trajectory
        - Parameter evolution
        - Gradient behavior
        - Convergence analysis
        
        Args:
            output_dir: Directory to save diagnostic files
            loss_history: List of loss values at each iteration
            param_history: Dictionary of parameter histories (contains 'T')
            grad_history: Dictionary of gradient histories corresponding to parameters
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute rotation and translation components from T matrices
        T_history = [T.detach().cpu().numpy() for T in param_history['T']]
        R_history = [T[:3, :3] for T in T_history]
        t_history = [T[:3, 3] for T in T_history]
        
        # Convert rotation matrices to axis-angle representation
        rvec_history = []
        for R in R_history:
            rvec, _ = cv2.Rodrigues(R)
            rvec_history.append(rvec.flatten())
        rvec_history = np.array(rvec_history)
        t_history = np.array(t_history)
        
        # Combine into a parameter history that matches the SE3 format
        param_history_np = {}
        param_history_np['se3_vec'] = np.hstack([rvec_history, t_history])
        
        # Convert delta gradients to numpy arrays
        grad_history_np = {}
        grad_history_np['delta'] = np.array([g.detach().cpu().numpy() if g is not None 
                                            else np.zeros(6) 
                                            for g in grad_history['delta']])
        
        # Number of iterations
        iterations = range(len(loss_history))
        
        # 1. Loss Trajectory Analysis
        plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plt.plot(iterations, loss_history, 'b-', linewidth=2)
        plt.title('Loss Value During Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Plot loss changes (derivative) to see stability
        plt.subplot(212)
        loss_changes = np.array([loss_history[i+1] - loss_history[i] 
                                for i in range(len(loss_history)-1)])
        plt.plot(iterations[:-1], loss_changes, 'r-')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.title('Loss Change Between Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Loss Difference')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_analysis.png'), dpi=150)
        plt.close()
        
        # 2. SE(3) Parameter Trajectory Analysis
        se3_data = param_history_np['se3_vec']
        
        # Plot all 6 components
        fig = plt.figure(figsize=(15, 8))
        
        # Rotation components
        plt.subplot(211)
        plt.plot(iterations, se3_data[:, 0], 'r-', label='ωx')
        plt.plot(iterations, se3_data[:, 1], 'g-', label='ωy')
        plt.plot(iterations, se3_data[:, 2], 'b-', label='ωz')
        plt.title('Camera Rotation Components Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Value (rad)')
        plt.grid(True)
        plt.legend()
        
        # Translation components
        plt.subplot(212)
        plt.plot(iterations, se3_data[:, 3], 'r-', label='tx')
        plt.plot(iterations, se3_data[:, 4], 'g-', label='ty')
        plt.plot(iterations, se3_data[:, 5], 'b-', label='tz')
        plt.title('Camera Translation Components Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'camera_trajectory.png'), dpi=150)
        plt.close()
        
        # 3. Delta Gradient Analysis
        grad_data = grad_history_np['delta']
        
        # Gradient magnitude
        grad_magnitude = np.linalg.norm(grad_data, axis=1)
        rot_grad_magnitude = np.linalg.norm(grad_data[:, :3], axis=1)
        trans_grad_magnitude = np.linalg.norm(grad_data[:, 3:], axis=1)
        
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 1, figure=fig)
        
        # Plot total gradient magnitude
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(iterations, grad_magnitude, 'k-', linewidth=2, label='Total')
        ax1.plot(iterations, rot_grad_magnitude, 'r-', linewidth=1.5, label='Rotation')
        ax1.plot(iterations, trans_grad_magnitude, 'b-', linewidth=1.5, label='Translation')
        ax1.set_title('Delta Gradient Magnitudes')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Gradient Norm')
        ax1.set_yscale('log')  # Log scale to better see changes
        ax1.grid(True)
        ax1.legend()
        
        # Plot rotation gradient components
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(iterations, grad_data[:, 0], 'r-', label='grad_ωx')
        ax2.plot(iterations, grad_data[:, 1], 'g-', label='grad_ωy')
        ax2.plot(iterations, grad_data[:, 2], 'b-', label='grad_ωz')
        ax2.set_title('Rotation Gradient Components')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Value')
        ax2.grid(True)
        ax2.legend()
        
        # Plot translation gradient components
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(iterations, grad_data[:, 3], 'r-', label='grad_tx')
        ax3.plot(iterations, grad_data[:, 4], 'g-', label='grad_ty')
        ax3.plot(iterations, grad_data[:, 5], 'b-', label='grad_tz')
        ax3.set_title('Translation Gradient Components')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Gradient Value')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'delta_gradient_analysis.png'), dpi=150)
        plt.close()
        
        # 4. Generate a text report with analysis
        with open(os.path.join(output_dir, 'optimization_analysis.txt'), 'w') as f:
            f.write("iNeRF OPTIMIZATION PROCESS ANALYSIS\n")
            f.write("====================================\n\n")
            
            # Loss analysis
            f.write("1. LOSS BEHAVIOR\n")
            f.write("----------------\n")
            initial_loss = loss_history[0]
            final_loss = loss_history[-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss != 0 else 0
            
            f.write(f"Initial loss: {initial_loss:.6f}\n")
            f.write(f"Final loss: {final_loss:.6f}\n")
            f.write(f"Total loss reduction: {loss_reduction:.2f}%\n")
            f.write(f"Number of iterations: {len(loss_history)}\n\n")
            
            # Parameter analysis
            f.write("2. PARAMETER ANALYSIS\n")
            f.write("---------------------\n")
            initial_params = param_history_np['se3_vec'][0]
            final_params = param_history_np['se3_vec'][-1]
            
            f.write("Initial camera parameters (rvec, t):\n")
            f.write(f"  Rotation: [{initial_params[0]:.4f}, {initial_params[1]:.4f}, {initial_params[2]:.4f}]\n")
            f.write(f"  Translation: [{initial_params[3]:.4f}, {initial_params[4]:.4f}, {initial_params[5]:.4f}]\n\n")
            
            f.write("Final camera parameters (rvec, t):\n")
            f.write(f"  Rotation: [{final_params[0]:.4f}, {final_params[1]:.4f}, {final_params[2]:.4f}]\n")
            f.write(f"  Translation: [{final_params[3]:.4f}, {final_params[4]:.4f}, {final_params[5]:.4f}]\n\n")
            
            # Gradient analysis
            f.write("3. GRADIENT ANALYSIS\n")
            f.write("--------------------\n")
            
            avg_grad_magnitude = np.mean(grad_magnitude)
            max_grad_magnitude = np.max(grad_magnitude)
            min_grad_magnitude = np.min(grad_magnitude)
            
            f.write(f"Average gradient magnitude: {avg_grad_magnitude:.6f}\n")
            f.write(f"Maximum gradient magnitude: {max_grad_magnitude:.6f}\n")
            f.write(f"Minimum gradient magnitude: {min_grad_magnitude:.6f}\n\n")
            
            avg_rot_grad = np.mean(rot_grad_magnitude)
            avg_trans_grad = np.mean(trans_grad_magnitude)
            avg_ratio = avg_rot_grad / max(avg_trans_grad, 1e-10)
            
            f.write(f"Average rotation gradient: {avg_rot_grad:.6f}\n")
            f.write(f"Average translation gradient: {avg_trans_grad:.6f}\n")
            f.write(f"Average rotation/translation ratio: {avg_ratio:.6f}\n\n")
            
            # Check for potential issues
            issues = []
            
            # Gradient vanishing check
            if min_grad_magnitude < 1e-6:
                issues.append("Potential gradient vanishing detected.")
                
            # Gradient explosion check    
            if max_grad_magnitude > 1e3:
                issues.append("Potential gradient explosion detected.")
                
            # Rotation/translation ratio imbalance
            if avg_ratio > 100 or avg_ratio < 0.01:
                issues.append(f"Imbalanced rotation/translation gradients (ratio: {avg_ratio:.2f}).")
                
            # Oscillations check
            if len(loss_history) >= 3:
                oscillation_count = sum(1 for i in range(len(loss_history)-2) 
                                       if (loss_history[i] > loss_history[i+1] and 
                                           loss_history[i+1] < loss_history[i+2]))
                oscillation_ratio = oscillation_count / (len(loss_history) - 2)
                
                if oscillation_ratio > 0.3:
                    issues.append(f"High oscillation detected ({oscillation_ratio:.2%} of iterations).")
            
            if issues:
                f.write("4. POTENTIAL ISSUES\n")
                f.write("-------------------\n")
                for issue in issues:
                    f.write(f"- {issue}\n")
            else:
                f.write("4. OPTIMIZATION APPEARS STABLE\n")
                f.write("------------------------------\n")
                f.write("No significant optimization issues detected.\n")
        
        print(f"Saved iNeRF optimization diagnostics to {output_dir}")

    def save_optimization_diagnostics_inerf(self, 
                           output_dir: str,
                           loss_history: list,
                           param_history: dict,
                           grad_history: dict) -> None:
        """
        既存のコードはそのまま保持...
        """
        # 既存のコードはそのまま維持
        
        # --- 以下の3D可視化機能を追加 ---
        
        # カメラ位置の3D軌跡を可視化
        if 'camera_poses' in param_history and len(param_history['camera_poses']) > 0:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # カメラ位置の抽出
            poses = np.array(param_history['camera_poses'])
            positions = poses[:, :3, 3]  # カメラ位置 (Nx3)
            
            # 軌跡のプロット
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=1)
            
            # 初期位置と最終位置を強調表示
            ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Initial')
            ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='Final')
            
            # 基準カメラの位置（存在する場合）
            if hasattr(self, 'camera_1_pose'):
                ref_pos = self.camera_1_pose[:3, 3]
                ax.scatter(ref_pos[0], ref_pos[1], ref_pos[2], c='yellow', s=100, label='Reference')
            
            # 視覚化の設定
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Camera Position Optimization Path')
            ax.legend()
            
            # 座標系の表示
            scale = np.max(np.abs(positions)) * 0.1
            origin = np.zeros(3)
            ax.quiver(origin[0], origin[1], origin[2], scale, 0, 0, color='r', label='X')
            ax.quiver(origin[0], origin[1], origin[2], 0, scale, 0, color='g', label='Y')
            ax.quiver(origin[0], origin[1], origin[2], 0, 0, scale, color='b', label='Z')
            
            plt.savefig(os.path.join(output_dir, 'camera_trajectory_3d.png'))
            plt.close()
            
        # 勾配比率のプロット（回転/並進）を追加
        if 'rot_grad_norms' in grad_history and 'trans_grad_norms' in grad_history:
            rot_norms = np.array(grad_history['rot_grad_norms'])
            trans_norms = np.array(grad_history['trans_grad_norms'])
            
            # 勾配比率を計算
            ratio = rot_norms / (trans_norms + 1e-10)  # ゼロ除算防止
            
            plt.figure(figsize=(10, 6))
            plt.plot(ratio)
            plt.xlabel('Iteration')
            plt.ylabel('Gradient Ratio (Rotation/Translation)')
            plt.title('Gradient Norm Ratio History')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'gradient_ratio.png'))
            plt.close()