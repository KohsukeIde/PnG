def optimize_with_DSO(
        self,
        max_outer     = 200,   # 再線形化回数
        inner_steps   = 5,     # 1つの線形化点で回すGDステップ数
        lr            = 1e-2,
        momentum      = 0.9,   # innerで履歴を効かせる
        grad_clip     = 0.1,
        tol           = 1e-6,
        save_diagnostics = True,
        diagnostics_dir = None,
        seed          = None):
        """
        タイプB: outerループでposeを更新しlinearize、
                innerループで同一deltaを複数step更新するDSO-GD。
        
        Args:
            max_outer (int): 最大再線形化回数
            inner_steps (int): 1つの線形化点で実行する勾配降下ステップ数
            lr (float): 学習率
            momentum (float): モーメンタム係数
            grad_clip (float): 勾配クリッピングの閾値
            tol (float): 収束判定閾値
            save_diagnostics (bool): 診断情報を保存するかどうか
            diagnostics_dir (str): 診断情報保存先ディレクトリ
            seed (int): 乱数シード（初期化用）
        """
        # -------------------------  出力ディレクトリ  ---------------------- #
        transport_dir = os.path.join("results", "transport_DSO_B")
        os.makedirs(transport_dir, exist_ok=True)
        diagnostics_dir = diagnostics_dir or os.path.join("results", "diagnostics_dsoB")
        os.makedirs(diagnostics_dir, exist_ok=True)

        # ------------------------- 履歴用 ------------------------- #
        loss_history = []
        delta_history = []      # Δξの履歴
        delta_grad_history = [] # Δξの勾配履歴
        pose_history = []       # 姿勢の履歴

        # ------------------------- パラメータ初期化 ----------------------- #
        # optimize_with_SE3と同じ方法で初期姿勢を設定
        if not hasattr(self, "pose_cw"):
            if hasattr(self, "rot_vec") and hasattr(self, "trans_vec"):
                # 既存のrot_vecとtrans_vecから初期化
                se3_init = torch.cat([self.rot_vec.detach(), self.trans_vec.detach()])
                self.pose_cw = self.lie.se3_to_SE3(se3_init).detach()
            else:
                # _init_se3_like_cam1を使用して初期化
                self._init_se3_like_cam1(rot_noise=0.05, trans_noise=0.05, seed=seed)
                se3_init = torch.cat([self.rot_vec.detach(), self.trans_vec.detach()])
                self.pose_cw = self.lie.se3_to_SE3(se3_init).detach()
        
        pose_history.append(self.pose_cw.clone())

        # ------------------------- デバッグログ設定 ----------------------- #
        debug_log_path = os.path.join(diagnostics_dir, "gradient_debug_dsoB.log")
        with open(debug_log_path, 'w') as f:
            f.write("Outer, Inner, Loss, Delta_Norm, Grad_Norm, dw_x, dw_y, dw_z, dt_x, dt_y, dt_z\n")

        # ------------------------- 最適化ループ ----------------------- #
        prev_loss_val = float('inf')
        pbar = tqdm(range(max_outer), desc="DSO-TypeB", leave=True)

        for outer in pbar:
            # 1. Δξをゼロ初期化とオプティマイザを設定 (outer毎に新しく作成)
            delta = torch.zeros(6, device=self.device, requires_grad=True)
            optimizer = torch.optim.SGD([delta], lr=lr, momentum=momentum)
            
            # ------------ 内部ループで同一線形化点を使って複数回の勾配降下 ------------
            for inner in range(inner_steps):
                # 2. forward pass - 現在のposeにΔξを適用
                delta_SE3 = self.lie.se3_to_SE3(delta)
                R_delta = delta_SE3[:3, :3]  # 回転部分
                t_delta = delta_SE3[:3, 3:4]  # 並進部分 (3,1)の形状に

                R_pose = self.pose_cw[:3, :3]
                t_pose = self.pose_cw[:3, 3:4]  # (3,1)の形状に

                # SE(3)の合成: R' = R_delta * R_pose, t' = R_delta * t_pose + t_delta
                R_new = R_delta @ R_pose
                t_new = R_delta @ t_pose + t_delta

                # world→camera変換に変換
                R_wc = R_new.t()
                t_wc = -R_wc @ t_new

                # 3. 基礎行列計算とコスト行列計算
                F = self._build_F_from_wc(R_wc, t_wc[:,0])  # t_wcは1Dで渡す
                cost_matrix = self.compute_cost_matrix_fundamental(F)
                transport = self.unbalanced_sinkhorn_algorithm(cost_matrix)
                loss = torch.sum(transport * cost_matrix)
                
                # 4. バックワードパスとパラメータ更新
                optimizer.zero_grad()
                loss.backward()
                
                # Log inner loop details occasionally
                if outer % 10 == 0 and inner == 0:
                    print(f"\nOuter {outer}, Inner {inner} - Before step:")
                    print(f"  Delta params: {delta.data}")
                    print(f"  Loss: {loss.item():.6f}")
                    print(f"  Learning rate: {lr:.6e}")
                
                # 勾配保存（step前に！）
                if delta.grad is not None:
                    delta_grad = delta.grad.detach().clone()
                    delta_grad_norm = delta_grad.norm().item()
                    
                    # デバッグログに勾配情報を記録
                    with open(debug_log_path, 'a') as f:
                        delta_vals = delta.detach().cpu().numpy()
                        grad_vals = delta_grad.cpu().numpy()
                        f.write(f"{outer}, {inner}, {loss.item():.6f}, {delta.norm().item():.6f}, {delta_grad_norm:.6f}, " + 
                              f"{grad_vals[0]:.6f}, {grad_vals[1]:.6f}, {grad_vals[2]:.6f}, " +
                              f"{grad_vals[3]:.6f}, {grad_vals[4]:.6f}, {grad_vals[5]:.6f}\n")
                    
                    if inner == inner_steps - 1:  # 最後のinner iterationの勾配を保存
                        delta_grad_history.append(delta_grad)
                else:
                    print("Warning: No gradient computed!")
                    if inner == inner_steps - 1:
                        delta_grad_history.append(None)
                    delta_grad_norm = 0.0
                
                # 勾配クリッピングと最適化ステップ
                torch.nn.utils.clip_grad_norm_([delta], grad_clip)
                optimizer.step()
                
                # Inner loop 早期終了チェック
                if delta.norm() < tol:
                    break
            
            # 5. 最後のdeltaを保存
            delta_norm = delta.detach().norm().item()
            delta_history.append(delta.detach().clone())
            
            # 6. 現在の損失を保存
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            # 7. poseを更新（in-place、計算グラフを切断）
            with torch.no_grad():
                # left-multiplicative更新: pose ← exp(Δξ) · pose
                delta_SE3 = self.lie.se3_to_SE3(delta.detach())
                R_delta = delta_SE3[:3, :3]
                t_delta = delta_SE3[:3, 3:4]  # (3,1)形式で

                R_pose = self.pose_cw[:3, :3]
                t_pose = self.pose_cw[:3, 3:4]  # (3,1)形式で

                R_new = R_delta @ R_pose
                t_new = R_delta @ t_pose + t_delta

                # in-place更新
                self.pose_cw[:3, :3] = R_new
                self.pose_cw[:3, 3] = t_new[:, 0]  # 列ベクトルを1Dに変換
                self.pose_cw = self.pose_cw.detach()
                
                pose_history.append(self.pose_cw.clone())
                
                if outer % 10 == 0:
                    print(f"  Delta magnitude: {delta_norm:.6f}")
                    print(f"  Updated Pose: {self.pose_cw}")
            
            # 8. 収束判定（損失差分とdeltaノルム両方をチェック）
            loss_diff = abs(prev_loss_val - current_loss)
            if loss_diff < tol and delta_norm < tol:
                pbar.set_description(f"Converged (loss_diff={loss_diff:.2e}, delta={delta_norm:.2e})")
                break
            prev_loss_val = current_loss
            
            # 9. プログレスバーの更新
            pbar.set_postfix({
                'loss': f"{current_loss:.6f}",
                'delta': f"{delta_norm:.4f}",
                'inner': f"{inner+1}/{inner_steps}",
                'lr': f"{lr:.2e}"
            })
            
            # 10. トランスポートプラン可視化（定期的に）
            if outer % 10 == 0 or outer == max_outer - 1:
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
                        plt.title(f"Transport Plan at Outer {outer} (Downsampled {downsample_factor}x)")
                    else:
                        plt.imshow(t_np, cmap="hot", interpolation="nearest", aspect="auto")
                        plt.title(f"Transport Plan at Outer {outer}")
                    plt.colorbar(label="Transport Plan Value")
                    plt.xlabel("Image 2 Gaussians")
                    plt.ylabel("Image 1 Gaussians")
                    plt.tight_layout()
                    plt_path = os.path.join(transport_dir, f"transport_dsoB_outer_{outer}.png")
                    plt.savefig(plt_path, dpi=150)
                    plt.close()

        # ------------------------- 最終パラメータ保存 --------------------------- #
        with torch.no_grad():
            # 最終SE(3)パラメータから変換結果を保存
            self.R_cw = self.pose_cw[:3, :3]
            self.t_cw = self.pose_cw[:3, 3]
            self.R_wc = self.R_cw.t()
            self.t_wc = -self.R_wc @ self.t_cw
            final_F = self._build_F_from_wc(self.R_wc, self.t_wc)
            self.f = final_F
            
            # OpenCV形式のパラメータ（あれば更新）
            if cv2 is not None:
                rvec_numpy, _ = cv2.Rodrigues(self.R_wc.cpu().numpy())
                self.rvec = nn.Parameter(torch.from_numpy(rvec_numpy).to(self.device))
                self.tvec = nn.Parameter(self.t_cw)
                rvec_cw_numpy, _ = cv2.Rodrigues(self.R_cw.cpu().numpy())
                self.rvec_cw = nn.Parameter(torch.from_numpy(rvec_cw_numpy).to(self.device))
                self.center = nn.Parameter(self.t_cw)

        # ------------------------- 損失曲線保存 --------------------------- #
        plt.figure()
        plt.plot(loss_history, '-o')
        plt.title("Loss (optimize_with_DSO Type B)")
        plt.xlabel("Outer Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(transport_dir, "loss_optimize_with_DSO_B.png"))
        plt.close()

        # ------------------------- 診断情報保存 --------------------------- #
        if save_diagnostics:
            # 詳細な診断情報を保存
            self.save_optimization_diagnostics_DSO(
                output_dir=diagnostics_dir,
                loss_history=loss_history,
                delta_history=delta_history,
                delta_grad_history=delta_grad_history,
                pose_history=pose_history
            )
        
        return loss_history

    def save_optimization_diagnostics_DSO(self, 
                                   output_dir: str,
                                   loss_history: list,
                                   delta_history: list,
                                   delta_grad_history: list,
                                   pose_history: list) -> None:
        """タイプB DSO最適化（First-Estimate Jacobian + GD）の診断情報を保存
        
        タイプB DSO最適化についての詳細な診断情報を生成・保存します：
        - 損失値の軌跡分析
        - Δξパラメータの挙動と成分ごとの分析
        - 勾配動向の分析
        - 姿勢の変化と収束性
        - テキスト形式のサマリーレポート
        
        Args:
            output_dir: 診断ファイルを保存するディレクトリ
            loss_history: outerループごとの損失値リスト
            delta_history: 各outerループの最終Δξの履歴
            delta_grad_history: 各outerループの最終Δξの勾配履歴
            pose_history: 姿勢の履歴
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 履歴をNumPy配列に変換
        delta_np = np.array([d.detach().cpu().numpy() for d in delta_history])
        delta_grad_np = np.array([
            g.detach().cpu().numpy() if g is not None else np.zeros(6) 
            for g in delta_grad_history
        ])
        pose_np = np.array([p.detach().cpu().numpy() for p in pose_history])
        
        # イテレーション数（outerループ）
        iterations = range(len(loss_history))
        
        # ======================= 1. 損失軌跡の分析 =======================
        plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plt.plot(iterations, loss_history, 'b-', linewidth=2)
        plt.title('Loss Value During Optimization (DSO Type B)')
        plt.xlabel('Outer Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 損失の変化（微分）をプロット
        plt.subplot(212)
        if len(loss_history) > 1:
            loss_changes = np.array([loss_history[i+1] - loss_history[i] 
                                    for i in range(len(loss_history)-1)])
            plt.plot(iterations[:-1], loss_changes, 'r-')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.title('Loss Change Between Outer Iterations')
            plt.xlabel('Outer Iteration')
            plt.ylabel('Loss Difference')
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'loss_analysis_dsoB.png'), dpi=150)
        plt.close()
        
        # =============== 2. DSOのΔξとその勾配分析 ===============
        # 勾配の大きさ（ノルム）
        delta_norms = np.linalg.norm(delta_np, axis=1)
        delta_grad_norms = np.linalg.norm(delta_grad_np, axis=1)
        
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 1, figure=fig)
        
        # Δξのノルム推移
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(iterations, delta_norms, 'k-', linewidth=2, label='Delta Norm')
        ax1.set_title('DSO Type B Delta Magnitude')
        ax1.set_xlabel('Outer Iteration')
        ax1.set_ylabel('Norm')
        ax1.set_yscale('log')  # Log scale
        ax1.grid(True)
        ax1.legend()
        
        # Δξの回転成分
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(iterations, delta_np[:, 0], 'r-', label='delta_wx')
        ax2.plot(iterations, delta_np[:, 1], 'g-', label='delta_wy')
        ax2.plot(iterations, delta_np[:, 2], 'b-', label='delta_wz')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_title('Delta Rotation Components')
        ax2.set_xlabel('Outer Iteration')
        ax2.set_ylabel('Value')
        ax2.grid(True)
        ax2.legend()
        
        # Δξの並進成分
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(iterations, delta_np[:, 3], 'r-', label='delta_tx')
        ax3.plot(iterations, delta_np[:, 4], 'g-', label='delta_ty')
        ax3.plot(iterations, delta_np[:, 5], 'b-', label='delta_tz')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_title('Delta Translation Components')
        ax3.set_xlabel('Outer Iteration')
        ax3.set_ylabel('Value')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'delta_analysis_dsoB.png'), dpi=150)
        plt.close()
        
        # =============== 3. Δξの勾配分析 ===============
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 1, figure=fig)
        
        # 勾配ノルム推移
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(iterations, delta_grad_norms, 'k-', linewidth=2, label='Gradient Norm')
        ax1.set_title('DSO Type B Gradient Magnitude')
        ax1.set_xlabel('Outer Iteration')
        ax1.set_ylabel('Norm')
        ax1.set_yscale('log')  # Log scale
        ax1.grid(True)
        ax1.legend()
        
        # 回転勾配成分
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(iterations, delta_grad_np[:, 0], 'r-', label='grad_wx')
        ax2.plot(iterations, delta_grad_np[:, 1], 'g-', label='grad_wy')
        ax2.plot(iterations, delta_grad_np[:, 2], 'b-', label='grad_wz')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_title('Rotation Gradient Components')
        ax2.set_xlabel('Outer Iteration')
        ax2.set_ylabel('Gradient Value')
        ax2.grid(True)
        ax2.legend()
        
        # 並進勾配成分
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(iterations, delta_grad_np[:, 3], 'r-', label='grad_tx')
        ax3.plot(iterations, delta_grad_np[:, 4], 'g-', label='grad_ty')
        ax3.plot(iterations, delta_grad_np[:, 5], 'b-', label='grad_tz')
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax3.set_title('Translation Gradient Components')
        ax3.set_xlabel('Outer Iteration')
        ax3.set_ylabel('Gradient Value')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_analysis_dsoB.png'), dpi=150)
        plt.close()
        
        # =============== 4. ポーズの軌跡分析 ===============
        if len(pose_history) > 1 and pose_np.shape[1] >= 3 and pose_np.shape[2] >= 4:
            # 回転行列からオイラー角（または回転ベクトル）を計算
            from scipy.spatial.transform import Rotation as R
            
            euler_angles = []
            translations = []
            
            for pose in pose_np:
                if pose.shape[0] >= 3 and pose.shape[1] >= 4:
                    # 回転行列部分を抽出
                    rot_mat = pose[:3, :3]
                    r = R.from_matrix(rot_mat)
                    euler = r.as_euler('xyz', degrees=True)
                    euler_angles.append(euler)
                    
                    # 並進ベクトルを抽出
                    trans = pose[:3, 3]
                    translations.append(trans)
            
            euler_angles = np.array(euler_angles)
            translations = np.array(translations)
            
            # ポーズの軌跡可視化
            fig = plt.figure(figsize=(15, 10))
            
            # オイラー角の変化
            ax1 = fig.add_subplot(211)
            ax1.plot(range(len(euler_angles)), euler_angles[:, 0], 'r-', label='Roll')
            ax1.plot(range(len(euler_angles)), euler_angles[:, 1], 'g-', label='Pitch')
            ax1.plot(range(len(euler_angles)), euler_angles[:, 2], 'b-', label='Yaw')
            ax1.set_title('Camera Rotation (Euler Angles)')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Angle (degrees)')
            ax1.grid(True)
            ax1.legend()
            
            # 並進の変化
            ax2 = fig.add_subplot(212)
            ax2.plot(range(len(translations)), translations[:, 0], 'r-', label='X')
            ax2.plot(range(len(translations)), translations[:, 1], 'g-', label='Y')
            ax2.plot(range(len(translations)), translations[:, 2], 'b-', label='Z')
            ax2.set_title('Camera Translation')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Position')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'pose_trajectory_dsoB.png'), dpi=150)
            plt.close()
            
            # 3D視点での並進軌跡
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(translations[:, 0], translations[:, 1], translations[:, 2], 'b-', linewidth=2)
            ax.scatter(translations[0, 0], translations[0, 1], translations[0, 2], c='g', s=100, label='Start')
            ax.scatter(translations[-1, 0], translations[-1, 1], translations[-1, 2], c='r', s=100, label='End')
            
            ax.set_title('Camera Position Trajectory in 3D')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            
            plt.savefig(os.path.join(output_dir, 'camera_trajectory_3d_dsoB.png'), dpi=150)
            plt.close()
        
        # =============== 5. 成分ごとの勾配履歴の詳細分析 ==============
        # 回転成分（wx, wy, wz）のグラフ
        plt.figure(figsize=(15, 10))
        for i, component in enumerate(['wx', 'wy', 'wz']):
            plt.subplot(3, 1, i+1)
            plt.plot(iterations, delta_grad_np[:, i], 'b-', linewidth=1.5)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.title(f'Rotation Gradient - {component} Component')
            plt.xlabel('Outer Iteration')
            plt.ylabel(f'Gradient Value (d/d{component})')
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rot_gradient_components_detail_dsoB.png'), dpi=150)
        plt.close()

        # 並進成分（tx, ty, tz）のグラフ
        plt.figure(figsize=(15, 10))
        for i, component in enumerate(['tx', 'ty', 'tz']):
            plt.subplot(3, 1, i+1)
            plt.plot(iterations, delta_grad_np[:, i+3], 'r-', linewidth=1.5)
            plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            plt.title(f'Translation Gradient - {component} Component')
            plt.xlabel('Outer Iteration')
            plt.ylabel(f'Gradient Value (d/d{component})')
            plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trans_gradient_components_detail_dsoB.png'), dpi=150)
        plt.close()
        
        # =============== 6. Delta変化の分析 ===============
        if len(delta_np) > 1:
            plt.figure(figsize=(12, 6))
            # 各イテレーションでのDeltaの変化量
            delta_changes = np.array([np.linalg.norm(delta_np[i+1] - delta_np[i]) for i in range(len(delta_np)-1)])
            
            plt.plot(iterations[:-1], delta_changes, 'b-', linewidth=2)
            plt.title('Delta Change Magnitude Between Outer Iterations')
            plt.xlabel('Outer Iteration')
            plt.ylabel('Change Magnitude')
            plt.yscale('log')  # Log scaleで変化を見やすく
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'delta_changes_dsoB.png'), dpi=150)
            plt.close()
        
        # =============== 7. テキスト形式のサマリーレポート ===============
        with open(os.path.join(output_dir, 'optimization_analysis_dsoB.txt'), 'w') as f:
            f.write("DSO TYPE-B OPTIMIZATION PROCESS ANALYSIS\n")
            f.write("=======================================\n\n")
            
            # 損失分析
            f.write("1. LOSS BEHAVIOR\n")
            f.write("----------------\n")
            initial_loss = loss_history[0]
            final_loss = loss_history[-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss != 0 else 0
            
            f.write(f"Initial loss: {initial_loss:.6f}\n")
            f.write(f"Final loss: {final_loss:.6f}\n")
            f.write(f"Total loss reduction: {loss_reduction:.2f}%\n\n")
            
            # 単調減少性チェック
            if len(loss_history) > 1:
                is_monotonic = all(loss_history[i] >= loss_history[i+1] for i in range(len(loss_history)-1))
                f.write(f"Loss decreases monotonically: {is_monotonic}\n")
                
                # 振動とプラトー（平坦部）の検出
                oscillation_count = sum(1 for i in range(len(loss_history)-2) 
                                        if (loss_history[i] > loss_history[i+1] and 
                                            loss_history[i+1] < loss_history[i+2]))
                
                plateau_threshold = 1e-6  # プラトー判定の閾値
                plateau_count = sum(1 for i in range(len(loss_history)-1) 
                                if abs(loss_history[i] - loss_history[i+1]) < plateau_threshold)
                
                f.write(f"Number of oscillations: {oscillation_count}\n")
                f.write(f"Number of plateaus: {plateau_count}\n\n")
            
            # Δξと勾配の分析
            f.write("2. DELTA AND GRADIENT BEHAVIOR\n")
            f.write("----------------------------\n")
            
            # Δξの統計
            max_delta = np.max(delta_norms)
            min_delta = np.min(delta_norms)
            avg_delta = np.mean(delta_norms)
            f.write(f"Delta magnitude - Max: {max_delta:.6f}, " 
                    f"Min: {min_delta:.6f}, Avg: {avg_delta:.6f}\n")
            
            # 勾配の統計
            max_grad = np.max(delta_grad_norms)
            min_grad = np.min(delta_grad_norms)
            avg_grad = np.mean(delta_grad_norms)
            f.write(f"Gradient magnitude - Max: {max_grad:.6f}, "
                    f"Min: {min_grad:.6f}, Avg: {avg_grad:.6f}\n\n")
            
            # 勾配消失/爆発チェック
            vanishing_threshold = 1e-6
            exploding_threshold = 1e2
            
            vanishing_grad = any(grad < vanishing_threshold for grad in delta_grad_norms)
            exploding_grad = any(grad > exploding_threshold for grad in delta_grad_norms)
            
            f.write(f"Gradient vanishing detected: {vanishing_grad}\n")
            f.write(f"Gradient exploding detected: {exploding_grad}\n\n")
            
            # 回転/並進成分の分析
            f.write("3. ROTATION/TRANSLATION COMPONENT ANALYSIS\n")
            f.write("----------------------------------------\n")
            
            # 回転成分
            rot_delta = delta_np[:, :3]
            rot_delta_norms = np.linalg.norm(rot_delta, axis=1)
            max_rot = np.max(rot_delta_norms)
            min_rot = np.min(rot_delta_norms)
            avg_rot = np.mean(rot_delta_norms)
            f.write(f"Rotation delta - Max: {max_rot:.6f}, Min: {min_rot:.6f}, Avg: {avg_rot:.6f}\n")
            
            # 並進成分
            trans_delta = delta_np[:, 3:]
            trans_delta_norms = np.linalg.norm(trans_delta, axis=1)
            max_trans = np.max(trans_delta_norms)
            min_trans = np.min(trans_delta_norms)
            avg_trans = np.mean(trans_delta_norms)
            f.write(f"Translation delta - Max: {max_trans:.6f}, Min: {min_trans:.6f}, Avg: {avg_trans:.6f}\n\n")
            
            # 結論
            f.write("4. CONCLUSION\n")
            f.write("-------------\n")
            
            # 最適化の成功判定
            successful = loss_reduction > 50 and final_loss < initial_loss * 0.5
            
            if successful:
                f.write("Optimization appears to be SUCCESSFUL based on significant loss reduction.\n\n")
            else:
                f.write("Optimization may have ISSUES based on limited loss reduction.\n\n")
                
            # 潜在的な問題点のレポート
            issues = []
            if len(loss_history) > 2:
                if not is_monotonic and oscillation_count > len(loss_history) * 0.1:
                    issues.append("- Loss exhibits significant oscillations, suggesting unstable optimization.")
                    
                if plateau_count > len(loss_history) * 0.3:
                    issues.append("- Loss exhibits plateaus, suggesting the optimizer may be struggling to make progress.")
            
            if vanishing_grad:
                issues.append("- Gradients approach zero, suggesting vanishing gradient issues.")
                
            if exploding_grad:
                issues.append("- Gradients are very large, suggesting exploding gradient issues.")
            
            if issues:
                f.write("Potential issues detected:\n")
                for issue in issues:
                    f.write(issue + "\n")
            else:
                f.write("No significant optimization issues detected.\n")
                
            # タイプB DSO形式の利点について
            f.write("\n5. TYPE-B DSO OPTIMIZATION BENEFITS\n")
            f.write("--------------------------------\n")
            f.write("- Each outer iteration provides a fresh linearization point\n")
            f.write("- Multiple inner iterations enable momentum to be effective on the same linearization\n")
            f.write("- Left-multiplicative updates ensure numerical stability\n")
            f.write("- Multiple steps on the same linearization improve convergence efficiency\n")
            
            # 最適化改善のための提案
            f.write("\n6. SUGGESTIONS FOR IMPROVEMENT\n")
            f.write("------------------------------\n")
            
            suggestions = []
            
            if exploding_grad:
                suggestions.append("- Consider using a smaller learning rate or increasing gradient clipping threshold.")
                
            if vanishing_grad:
                suggestions.append("- Consider using a larger learning rate or different optimizer (e.g. Adam).")
                
            if len(loss_history) > 2:
                if oscillation_count > len(loss_history) * 0.2:
                    suggestions.append("- Adjust momentum parameter or learning rate to stabilize optimization.")
                    
                if plateau_count > len(loss_history) * 0.4:
                    suggestions.append("- Try increasing inner steps or using adaptive learning rate methods.")
            
            if max_rot / (max_trans + 1e-10) > 10:
                suggestions.append("- Rotation changes much larger than translation; consider balancing learning rates.")
            elif max_trans / (max_rot + 1e-10) > 10:
                suggestions.append("- Translation changes much larger than rotation; consider balancing learning rates.")
            
            if len(suggestions) > 0:
                for suggestion in suggestions:
                    f.write(suggestion + "\n")
            else:
                f.write("No specific improvements needed. The optimization appears to be well-configured.\n")
        
        print(f"Saved DSO Type-B optimization diagnostics to {output_dir}")