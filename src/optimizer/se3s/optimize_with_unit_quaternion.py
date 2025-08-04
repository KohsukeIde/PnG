
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
            
            # 勾配デバッグ - 統一されたログ関数を使用
            if self.quat.grad is not None and self.center.grad is not None:
                quat_grad_norm = self.quat.grad.norm().item()
                center_grad_norm = self.center.grad.norm().item()
                
                if iteration % 10 == 0:
                    from utils.debug.optimization_diagnostics import log_optimization_progress
                    log_optimization_progress(
                        iteration=iteration,
                        loss_components={'total': loss.item()},
                        gradient_norms={'quat': quat_grad_norm, 'center': center_grad_norm},
                        parameter_stats={'quat_norm': self.quat.norm().item(), 'center_norm': self.center.norm().item()},
                        log_file=debug_log_path
                    )
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
                
                # Transport matrix visualization using unified function
                if iteration % 50 == 0 or iteration == max_iter - 1:
                    from utils.debug.optimization_diagnostics import save_transport_snapshot
                    save_transport_snapshot(
                        transport_matrix=T,
                        epsilon=0.1,  # placeholder epsilon value
                        iteration=iteration,
                        output_dir=log_dir,
                        prefix="transport_quat"
                    )

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
            from utils.debug.optimization_diagnostics import save_optimization_diagnostics_SE3
            save_optimization_diagnostics_SE3(
                output_dir=diagnostics_dir,
                loss_history=loss_history,
                param_history=param_history,
                grad_history=grad_history
            )
