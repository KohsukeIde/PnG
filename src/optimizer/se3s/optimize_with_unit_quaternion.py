
def optimize_with_quat(self, max_iter=1000, tol=1e-6,
                           
                           
                           
                           
                           
                     lr=3e-3, save_diagnostics=True,
                     diagnostics_dir: Optional[str] = None):
        """OT-PnP : quaternion(4) + translation(3) を Adam で更新
        
        quaternion のみ各 step で正規化．並進のスケールは自由度として残す
        
        Args:
            max_iter (int): 最大反復回数。Default: 1000.
            tol (float): 収束判定のための閾値。Default: 1e-6.
            lr (float): 学習率。Default: 3e-3.
            save_diagnostics (bool): 診断情報を保存するかどうか。Default: True.
            diagnostics_dir (str, optional): 診断情報を保存するディレクトリ。Default: None.
        
        Returns:
            None: 最適化されたパラメータはインスタンス属性として保存されます。
        """
        # 出力用フォルダ
        log_dir = os.path.join("results", "transport_quat")
        os.makedirs(log_dir, exist_ok=True)
        diagnostics_dir = diagnostics_dir or os.path.join("results", "diagnostics_quat")
        os.makedirs(diagnostics_dir, exist_ok=True)

        # ---------- パラメータ初期化 ----------
        if not hasattr(self, "quat"):
            # 単位四元数と小さな並進で初期化
            self.quat = nn.Parameter(torch.tensor([1., 0., 0., 0.],
                                                device=self.device, dtype=torch.float32))
            self.center = nn.Parameter(torch.tensor([1., 0., 0.],
                                                    device=self.device, dtype=torch.float32))

        # ------------------------- 履歴用 ------------------------- #
        loss_history = []
        param_history = {'quat': [], 'center': []}
        grad_history = {'quat': [], 'center': []}

        # ---------- オプティマイザ ----------
        opt = torch.optim.Adam([self.quat, self.center], lr=lr)

        prev_loss = float('inf')
        pbar = tqdm(range(max_iter), desc="Optimizing quat-t", leave=True)

        # ---- 勾配デバッグ用ログファイル ----
        debug_log_path = os.path.join(diagnostics_dir, "gradient_debug_quat.log")
        with open(debug_log_path, 'w') as f:
            f.write("Iteration, Loss, Quat_Grad_Norm, Trans_Grad_Norm, Quat_w, Quat_x, Quat_y, Quat_z, Trans_x, Trans_y, Trans_z\n")

        # ---------- 最適化ループ ----------
        for iteration in pbar:
            # --- 射影: quat を正規化 ---
            with torch.no_grad():
                self.quat.data /= self.quat.data.norm() + 1e-12

            opt.zero_grad()

            # quat → R_cw,  c_w (camera center)
            R_cw = self._quat_to_matrix(self.quat)           # (3,3)
            R_wc = R_cw.t()
            t_wc = -R_wc @ self.center                       # カメラ座標系原点

            F = self._build_F_from_wc(R_wc, t_wc)

            C = self.compute_cost_matrix_fundamental(F)
            T = self.unbalanced_sinkhorn_algorithm(C)
            loss = torch.sum(T * C)
            
            # バックワードパス前のデバッグ出力
            if iteration % 10 == 0:
                print(f"\nIteration {iteration} - Before backward:")
                print(f"  Quat: {self.quat.data}")
                print(f"  Center: {self.center.data}")
                print(f"  Loss: {loss.item():.6f}")
            
            loss.backward()
            
            # 勾配デバッグ - 値とノルムを表示
            if self.quat.grad is not None and self.center.grad is not None:
                quat_grad = self.quat.grad
                center_grad = self.center.grad
                quat_grad_norm = quat_grad.norm().item()
                center_grad_norm = center_grad.norm().item()
                
                # 勾配情報をログに記録
                with open(debug_log_path, 'a') as f:
                    quat_vals = self.quat.detach().cpu().numpy()
                    center_vals = self.center.detach().cpu().numpy()
                    f.write(f"{iteration}, {loss.item():.6f}, {quat_grad_norm:.6f}, {center_grad_norm:.6f}, " + 
                          f"{quat_vals[0]:.6f}, {quat_vals[1]:.6f}, {quat_vals[2]:.6f}, {quat_vals[3]:.6f}, " +
                          f"{center_vals[0]:.6f}, {center_vals[1]:.6f}, {center_vals[2]:.6f}\n")
                
                # 定期的に詳細な勾配情報を表示
                if iteration % 10 == 0:
                    print(f"  Quat gradient norm: {quat_grad_norm:.6f}")
                    print(f"  Quat gradient: {quat_grad.detach().cpu().numpy()}")
                    print(f"  Center gradient norm: {center_grad_norm:.6f}")
                    print(f"  Center gradient: {center_grad.detach().cpu().numpy()}")
                    print(f"  Quat/Center gradient norm ratio: {quat_grad_norm/max(center_grad_norm, 1e-10):.6f}")
            else:
                print("Warning: No gradient computed!")

            # ---- ログ ----
            current_loss = loss.item()
            loss_history.append(current_loss)
            param_history['quat'].append(self.quat.clone())
            param_history['center'].append(self.center.clone())
            grad_history['quat'].append(self.quat.grad.clone() if self.quat.grad is not None else None)
            grad_history['center'].append(self.center.grad.clone() if self.center.grad is not None else None)
            
            # パラメータ更新
            opt.step()
            
            # 更新後のパラメータをデバッグ表示
            if iteration % 10 == 0:
                quat_change = torch.norm(self.quat.data - param_history['quat'][-1].data)
                center_change = torch.norm(self.center.data - param_history['center'][-1].data)
                print(f"  Quat parameter change: {quat_change.item():.6f}")
                print(f"  Center parameter change: {center_change.item():.6f}")
                print(f"  Updated quat: {self.quat.data}")
                print(f"  Updated center: {self.center.data}")

            # ---- 収束判定 ----
            loss_diff = abs(prev_loss - current_loss)
            if iteration > 5 and loss_diff < tol:
                pbar.set_description(f"Converged (loss_diff={loss_diff:.2e})")
                break
            prev_loss = current_loss

            # プログレスバー更新
            if iteration % 10 == 0 and self.quat.grad is not None and self.center.grad is not None:
                quat_grad_norm = self.quat.grad.norm().item()
                center_grad_norm = self.center.grad.norm().item()
                pbar.set_postfix({
                    'loss': f"{current_loss:.6f}",
                    'q_grad': f"{quat_grad_norm:.4f}",
                    'c_grad': f"{center_grad_norm:.4f}"
                })

            # 定期的な可視化
            if iteration % 50 == 0 or iteration == max_iter - 1:
                plt.figure()
                plt.plot(loss_history, '-o')
                plt.title("OT-loss (quat)")
                plt.xlabel("Iteration")
                plt.ylabel("Loss")
                plt.grid(True)
                plt.savefig(os.path.join(log_dir, f"loss_{iteration:04d}.png"))
                plt.close()
                
                # Transport matrix visualization
                if iteration % 50 == 0 or iteration == max_iter - 1:
                    with torch.no_grad():
                        t_np = T.detach().cpu().numpy()

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
                    plt_path = os.path.join(log_dir, f"transport_iter_{iteration}.png")
                    plt.savefig(plt_path, dpi=150)
                    plt.close()

        # ---------- 最終結果をメンバへ ----------
        with torch.no_grad():
            self.quat.data /= self.quat.data.norm() + 1e-12
            R_cw = self._quat_to_matrix(self.quat)
            R_wc = R_cw.t()
            t_wc = -R_wc @ self.center
            self.f = self._build_F_from_wc(R_wc, t_wc)

            # 互換目的で rvec, tvec も保存
            rvec_np, _ = cv2.Rodrigues(R_wc.cpu().numpy())
            self.rvec = nn.Parameter(torch.from_numpy(rvec_np).to(self.device))
            self.tvec = nn.Parameter(t_wc)
            
            # camera-to-world パラメータも更新
            rvec_cw_numpy, _ = cv2.Rodrigues(R_cw.cpu().numpy())
            self.rvec_cw = nn.Parameter(torch.from_numpy(rvec_cw_numpy).to(self.device))
            self.center = nn.Parameter(t_wc)

        # ---------- 保存 ----------
        torch.save({
            'quat': self.quat.detach().cpu(),
            'center': self.center.detach().cpu(),
            'F': self.f.detach().cpu(),
            'loss': torch.tensor(loss_history)
        }, os.path.join(log_dir, "opt_state.pt"))
        
        # 最終損失プロット
        plt.figure()
        plt.plot(loss_history, '-o')
        plt.title("Loss (optimize_with_quat)")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, "loss_optimize_with_quat.png"))
        plt.close()

        # 診断情報の保存
        if save_diagnostics:
            self.save_optimization_diagnostics_quat(
                output_dir=diagnostics_dir,
                loss_history=loss_history,
                param_history=param_history,
                grad_history=grad_history
            )
def save_optimization_diagnostics_quat(self, 
                                  output_dir: str,
                                  loss_history: list,
                                  param_history: dict,
                                  grad_history: dict) -> None:
        """Save detailed diagnostics about the quaternion optimization process.
        
        Analyzes and visualizes the optimization process of quaternion and translation parameters.
        
        Args:
            output_dir: Directory to save diagnostic files
            loss_history: List of loss values at each iteration
            param_history: Dictionary of parameter histories (contains 'quat' and 'center')
            grad_history: Dictionary of gradient histories corresponding to parameters
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert histories to numpy arrays
        param_history_np = {}
        grad_history_np = {}
        
        for param_name, history in param_history.items():
            param_history_np[param_name] = np.array([p.detach().cpu().numpy() for p in history])
            
        for param_name, history in grad_history.items():
            grad_history_np[param_name] = np.array([g.detach().cpu().numpy() if g is not None 
                                                else np.zeros_like(param_history_np[param_name][0]) 
                                                for g in history])
        
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
        
        # 2. Quaternion Parameter Trajectory Analysis
        quat_data = param_history_np['quat']
        center_data = param_history_np['center']
        
        # Plot all quaternion components
        fig = plt.figure(figsize=(15, 8))
        
        # Quaternion components
        plt.subplot(211)
        plt.plot(iterations, quat_data[:, 0], 'r-', label='w')
        plt.plot(iterations, quat_data[:, 1], 'g-', label='x')
        plt.plot(iterations, quat_data[:, 2], 'b-', label='y')
        plt.plot(iterations, quat_data[:, 3], 'c-', label='z')
        plt.title('Quaternion Components Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        # Center components
        plt.subplot(212)
        plt.plot(iterations, center_data[:, 0], 'r-', label='x')
        plt.plot(iterations, center_data[:, 1], 'g-', label='y')
        plt.plot(iterations, center_data[:, 2], 'b-', label='z')
        plt.title('Center Components Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quat_components_trajectory.png'), dpi=150)
        plt.close()
        
        # 3D Visualization of center trajectory
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(center_data[:, 0], center_data[:, 1], center_data[:, 2], 'r-', linewidth=2)
        ax.scatter(center_data[0, 0], center_data[0, 1], center_data[0, 2], c='g', s=100, label='Initial')
        ax.scatter(center_data[-1, 0], center_data[-1, 1], center_data[-1, 2], c='b', s=100, label='Final')
        ax.set_title('Camera Center Trajectory in 3D')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'center_3d_trajectory.png'), dpi=150)
        plt.close()
        
        # 3. Gradient Analysis
        quat_grad_data = grad_history_np['quat']
        center_grad_data = grad_history_np['center']
        
        # Gradient magnitude
        quat_grad_magnitude = np.linalg.norm(quat_grad_data, axis=1)
        center_grad_magnitude = np.linalg.norm(center_grad_data, axis=1)
        
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(3, 1, figure=fig)
        
        # Plot total gradient magnitude
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(iterations, quat_grad_magnitude, 'r-', linewidth=1.5, label='Quaternion')
        ax1.plot(iterations, center_grad_magnitude, 'b-', linewidth=1.5, label='Center')
        ax1.set_title('Gradient Magnitude')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Gradient Norm')
        ax1.set_yscale('log')  # Log scale to better see changes
        ax1.grid(True)
        ax1.legend()
        
        # Plot quaternion gradient components
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(iterations, quat_grad_data[:, 0], 'r-', label='grad_w')
        ax2.plot(iterations, quat_grad_data[:, 1], 'g-', label='grad_x')
        ax2.plot(iterations, quat_grad_data[:, 2], 'b-', label='grad_y')
        ax2.plot(iterations, quat_grad_data[:, 3], 'c-', label='grad_z')
        ax2.set_title('Quaternion Gradient Components')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Value')
        ax2.grid(True)
        ax2.legend()
        
        # Plot center gradient components
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(iterations, center_grad_data[:, 0], 'r-', label='grad_x')
        ax3.plot(iterations, center_grad_data[:, 1], 'g-', label='grad_y')
        ax3.plot(iterations, center_grad_data[:, 2], 'b-', label='grad_z')
        ax3.set_title('Center Gradient Components')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Gradient Value')
        ax3.grid(True)
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quat_gradient_analysis.png'), dpi=150)
        plt.close()
        
        # 4. Plot ratio of quaternion to center gradient norms
        plt.figure(figsize=(12, 6))
        # Add small epsilon to avoid division by zero
        ratio = quat_grad_magnitude / (center_grad_magnitude + 1e-10)
        plt.plot(iterations, ratio, 'b-', linewidth=2)
        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Balanced ratio (1.0)')
        plt.title('Ratio of Quaternion to Center Gradient Norms')
        plt.xlabel('Iteration')
        plt.ylabel('Ratio')
        plt.yscale('log')  # Log scale to better see changes
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gradient_ratio.png'), dpi=150)
        plt.close()
        
        # 5. Quaternion Unit Norm Analysis
        quat_norms = np.linalg.norm(quat_data, axis=1)
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, quat_norms)
        plt.axhline(y=1.0, color='r', linestyle='--')
        plt.title('Quaternion Norm Throughout Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Norm')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'quaternion_norm.png'), dpi=150)
        plt.close()
        
        # 6. Generate a text report with analysis
        with open(os.path.join(output_dir, 'optimization_analysis.txt'), 'w') as f:
            f.write("QUATERNION OPTIMIZATION PROCESS ANALYSIS\n")
            f.write("=======================================\n\n")
            
            # Loss analysis
            f.write("1. LOSS BEHAVIOR\n")
            f.write("----------------\n")
            initial_loss = loss_history[0]
            final_loss = loss_history[-1]
            loss_reduction = (initial_loss - final_loss) / initial_loss * 100 if initial_loss != 0 else 0
            
            f.write(f"Initial loss: {initial_loss:.6f}\n")
            f.write(f"Final loss: {final_loss:.6f}\n")
            f.write(f"Total loss reduction: {loss_reduction:.2f}%\n\n")
            
            # Monotonicity check
            is_monotonic = all(loss_history[i] >= loss_history[i+1] for i in range(len(loss_history)-1))
            f.write(f"Loss decreases monotonically: {is_monotonic}\n")
            
            # Find oscillations or plateaus
            oscillation_count = sum(1 for i in range(len(loss_history)-2) 
                                    if (loss_history[i] > loss_history[i+1] and 
                                        loss_history[i+1] < loss_history[i+2]))
            
            plateau_threshold = 1e-6  # Define what constitutes a plateau
            plateau_count = sum(1 for i in range(len(loss_history)-1) 
                              if abs(loss_history[i] - loss_history[i+1]) < plateau_threshold)
            
            f.write(f"Number of oscillations: {oscillation_count}\n")
            f.write(f"Number of plateaus: {plateau_count}\n\n")
            
            # Parameter analysis
            f.write("2. PARAMETER BEHAVIOR\n")
            f.write("---------------------\n")
            
            # Quaternion analysis
            f.write(f"Quaternion (initial): " + np.array2string(quat_data[0], precision=6) + "\n")
            f.write(f"Quaternion (final): " + np.array2string(quat_data[-1], precision=6) + "\n")
            quat_change = np.linalg.norm(quat_data[-1] - quat_data[0])
            f.write(f"Total quaternion change magnitude: {quat_change:.6f}\n\n")
            
            # Check quaternion norms
            quat_norm_deviation = np.abs(quat_norms - 1.0)
            max_norm_deviation = np.max(quat_norm_deviation)
            f.write(f"Maximum quaternion norm deviation from 1.0: {max_norm_deviation:.6f}\n\n")
            
            # Center analysis
            f.write(f"Center (initial): " + np.array2string(center_data[0], precision=6) + "\n")
            f.write(f"Center (final): " + np.array2string(center_data[-1], precision=6) + "\n")
            center_change = np.linalg.norm(center_data[-1] - center_data[0])
            f.write(f"Total center change magnitude: {center_change:.6f}\n\n")
            
            # Gradient analysis
            f.write("3. GRADIENT BEHAVIOR\n")
            f.write("-------------------\n")
            
            f.write(f"Quaternion gradient - Max: {np.max(quat_grad_magnitude):.6f}, " 
                    f"Min: {np.min(quat_grad_magnitude):.6f}, Avg: {np.mean(quat_grad_magnitude):.6f}\n")
            f.write(f"Center gradient - Max: {np.max(center_grad_magnitude):.6f}, "
                    f"Min: {np.min(center_grad_magnitude):.6f}, Avg: {np.mean(center_grad_magnitude):.6f}\n")
            
            # Check for vanishing/exploding gradients
            vanishing_threshold = 1e-6
            exploding_threshold = 1e2
            
            quat_vanishing = any(grad < vanishing_threshold for grad in quat_grad_magnitude)
            quat_exploding = any(grad > exploding_threshold for grad in quat_grad_magnitude)
            center_vanishing = any(grad < vanishing_threshold for grad in center_grad_magnitude)
            center_exploding = any(grad > exploding_threshold for grad in center_grad_magnitude)
            
            f.write(f"Quaternion gradient vanishing detected: {quat_vanishing}\n")
            f.write(f"Quaternion gradient exploding detected: {quat_exploding}\n")
            f.write(f"Center gradient vanishing detected: {center_vanishing}\n")
            f.write(f"Center gradient exploding detected: {center_exploding}\n\n")
            
            # Ratio of quaternion/center gradients
            avg_ratio = np.mean(ratio)
            f.write(f"Average quaternion/center gradient ratio: {avg_ratio:.4f}\n")
            f.write(f"Ideal ratio should be close to 1.0 for balanced optimization\n\n")
            
            # Conclusion
            f.write("4. CONCLUSION\n")
            f.write("-------------\n")
            
            # Determine if the optimization was successful
            successful = loss_reduction > 50 and final_loss < initial_loss * 0.5
            
            if successful:
                f.write("Optimization appears to be SUCCESSFUL based on significant loss reduction.\n\n")
            else:
                f.write("Optimization may have ISSUES based on limited loss reduction.\n\n")
                
            # Report potential issues
            issues = []
            if not is_monotonic and oscillation_count > len(loss_history) * 0.1:
                issues.append("- Loss exhibits significant oscillations, suggesting unstable optimization.")
                
            if plateau_count > len(loss_history) * 0.3:
                issues.append("- Loss exhibits plateaus, suggesting the optimizer may be struggling to make progress.")
            
            if quat_vanishing or center_vanishing:
                issues.append("- Gradients approach zero, suggesting vanishing gradient issues.")
                
            if quat_exploding or center_exploding:
                issues.append("- Gradients are very large, suggesting exploding gradient issues.")
                
            if avg_ratio > 10.0 or avg_ratio < 0.1:
                issues.append(f"- Quaternion/center gradient ratio ({avg_ratio:.2f}) is far from balanced, "
                            "which may cause biased optimization.")
            
            if issues:
                f.write("Potential issues detected:\n")
                for issue in issues:
                    f.write(issue + "\n")
            else:
                f.write("No significant optimization issues detected.\n")
                
        print(f"Saved quaternion optimization diagnostics to {output_dir}")