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
                
                # 勾配保存（step前に！）using unified logging
                if delta.grad is not None:
                    delta_grad = delta.grad.detach().clone()
                    delta_grad_norm = delta_grad.norm().item()
                    
                    if outer % 5 == 0 and inner == 0:  # Log less frequently
                        from utils.debug.optimization_diagnostics import log_optimization_progress
                        log_optimization_progress(
                            iteration=outer * inner_steps + inner,
                            loss_components={'total': loss.item()},
                            gradient_norms={'delta': delta_grad_norm},
                            parameter_stats={'delta_norm': delta.norm().item()},
                            log_file=debug_log_path
                        )
                    
                    if inner == inner_steps - 1:  # 最後のinner iterationの勾配を保存
                        delta_grad_history.append(delta_grad)
                else:
                    print("Warning: No gradient computed!")
                    if inner == inner_steps - 1:
                        delta_grad_history.append(None)
                
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
            
            # 10. トランスポートプラン可視化（統一関数使用）
            if outer % 10 == 0 or outer == max_outer - 1:
                from utils.debug.optimization_diagnostics import save_transport_snapshot
                save_transport_snapshot(
                    transport_matrix=transport,
                    epsilon=0.1,  # placeholder
                    iteration=outer,
                    output_dir=transport_dir,
                    prefix="transport_dsoB"
                )

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
            # 統一された診断情報を使用
            from utils.debug.optimization_diagnostics import save_optimization_diagnostics_SE3
            # Convert delta histories to SE3 format
            param_history_converted = {
                'rot_vec': [d[:3] for d in delta_history],  # rotation part
                'trans_vec': [d[3:] for d in delta_history]  # translation part
            }
            grad_history_converted = {
                'rot_vec': [g[:3] if g is not None else None for g in delta_grad_history],
                'trans_vec': [g[3:] if g is not None else None for g in delta_grad_history]
            }
            save_optimization_diagnostics_SE3(
                output_dir=diagnostics_dir,
                loss_history=loss_history,
                param_history=param_history_converted,
                grad_history=grad_history_converted
            )
        
        return loss_history

