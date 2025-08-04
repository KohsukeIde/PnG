def optimize_with_essential_pgd(self, max_iter=1000, tol=1e-6,
                                save_diagnostics=True, diagnostics_dir=None,
                                lr_E=5e-5, momentum=0.0,
                                grad_clip=0.1, seed=None):
        """
        Directly optimise the Essential matrix E (rank-2, σ1=σ2) via
        projected-gradient descent with differentiable SVD.
        """
        # ----------- I/O dirs -----------
        transport_dir  = os.path.join("results", "transport_E")
        os.makedirs(transport_dir, exist_ok=True)
        diagnostics_dir = diagnostics_dir or os.path.join("results",
                                                          "diagnostics_E")
        os.makedirs(diagnostics_dir, exist_ok=True)

        # ----------- history -----------
        loss_history, E_norm_hist = [], []
        grad_hist = []
        param_history = {'E_raw': []}

        # ----------- parameter init -----------
        torch.manual_seed(seed or 0)
        if not hasattr(self, "E_raw"):
            # random 3×3 then project once so we start on manifold
            E0 = torch.randn(3, 3, device=self.device)
            U, S, Vh = torch.linalg.svd(E0, full_matrices=False)
            V = Vh.mH  # linalg.svd returns Vh (conjugate-transpose)
            
            # 符号ロック：det(U@V^T)が負ならU[:, 2]の符号を反転
            if torch.det(U @ V.mH) < 0:
                U[:, 2] *= -1
                
            # 初期化時は完全なEssential matrix制約を適用
            E0 = U @ torch.diag(torch.tensor([1., 1., 0.], device=self.device)) @ V
            E0 /= E0.norm() + 1e-9
            self.E_raw = nn.Parameter(E0.clone())
        else:
            # ensure requires_grad on reload
            self.E_raw.requires_grad_(True)

        # -------- optimizer & scheduler --------
        optimizer = torch.optim.SGD([self.E_raw], lr=lr_E,
                                    momentum=momentum)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=0.8**(1/100)
        )

        prev_loss = float('inf')
        pbar = tqdm(range(max_iter), desc="Optimizing E", leave=True)

        debug_log = os.path.join(diagnostics_dir, "gradient_debug_E.log")
        with open(debug_log, 'w') as f:
            f.write("iter,loss,E_grad_norm,e11,e12,e13,e21,e22,e23,e31,e32,e33\n")

        # -------- main loop -----------
        for it in pbar:
            optimizer.zero_grad()

            # ----- 1. projection INSIDE graph -----
            # --- (A) グラフ内プロジェクション（安全版：σ₃=0のみ） ---
            U, S, Vh = torch.linalg.svd(self.E_raw, full_matrices=False)
            V = Vh.mH  # linalg.svd returns Vh (conjugate-transpose)
            
            # 符号ロック：det(U@V^T)が負の場合 - インプレースを避ける！
            if torch.det(U @ V.mH) < 0:
                # U[:, 2] *= -1 のかわりに新しいテンソルを作成
                new_U = U.clone()
                new_U[:, 2] = -U[:, 2]
                U = new_U
                
            S_proj = S.clone()
            S_proj[-1] = 0.0             # rank-2 だけ保証、σ1≠σ2 は触らない
            E = U @ torch.diag(S_proj) @ V
            E = E / (E.norm() + 1e-9)    # スケール正規化

            # ----- 2. build F = K2^{-T}EK1^{-1} -----
            K1_inv = torch.inverse(self.k1)
            K2_inv = torch.inverse(self.k2)
            F = K2_inv.t() @ E @ K1_inv

            # ----- 3. loss (OT) -----
            cost = self.compute_cost_matrix_fundamental(F)
            Tplan = self.unbalanced_sinkhorn_algorithm(cost)
            
            # after transport is computed
            if torch.isnan(Tplan).any():
                raise RuntimeError("transport NaN")
                
            loss = torch.sum(Tplan * cost)

            # ----- 4. back-prop -----
            # just before loss.backward()
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError("loss Nan/Inf")

            loss.backward()
            
            # 安全策②：勾配がNaNなら0に置換（保険）
            if torch.isnan(self.E_raw.grad).any():
                # NaNを検出したが、安全に続行するため0に置き換え
                torch.nan_to_num_(self.E_raw.grad, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"Warning: NaN gradients detected at iteration {it}, replaced with zeros")

            # log gradient info every 10 it using unified function
            if it % 10 == 0:
                gnorm = self.E_raw.grad.norm().item() if self.E_raw.grad is not None else 0
                
                from utils.debug.optimization_diagnostics import log_optimization_progress
                log_optimization_progress(
                    iteration=it,
                    loss_components={'total': loss.item()},
                    gradient_norms={'E_raw': gnorm},
                    parameter_stats={'E_norm': self.E_raw.norm().item()},
                    log_file=debug_log
                )
                
                # Save gradient history
                if self.E_raw.grad is not None:
                    grad_hist.append(self.E_raw.grad.detach().clone())
                
                # デバッグ情報：特異値をチェック
                with torch.no_grad():
                    _, S_debug, _ = torch.linalg.svd(self.E_raw.data, full_matrices=False)
                    print(f"  Current singular values: {S_debug.cpu().numpy()}")
                
            # Save parameter history
            param_history['E_raw'].append(self.E_raw.detach().clone())

            # grad-clip & step
            torch.nn.utils.clip_grad_norm_([self.E_raw], grad_clip)
            optimizer.step()

            # ----- 5. projection OUTSIDE graph (for next iter stability) -----
            # --- (B) no-grad 投影（ステップ後）---
            # ここでは完全なEssential matrix制約（σ₁=σ₂, σ₃=0）を適用
            with torch.no_grad():
                U, S, Vh = torch.linalg.svd(self.E_raw.data, full_matrices=False)
                V = Vh.mH
                
                # 符号ロック - ここはno_gradなのでインプレースでも問題なし
                if torch.det(U @ V.mH) < 0:
                    U[:, 2] *= -1
                    
                self.E_raw.data = U @ torch.diag(torch.tensor(
                                [1., 1., 0.], device=self.device)) @ V
                self.E_raw.data /= self.E_raw.data.norm() + 1e-9
                E_norm_hist.append(self.E_raw.data.norm().item())

            scheduler.step()
            # ----- 6. book-keeping -----
            loss_val = loss.item()
            loss_history.append(loss_val)

            pbar.set_postfix({'loss': f"{loss_val:.6f}",
                              'E|grad|': f"{self.E_raw.grad.norm().item():.2e}",
                              'lr': f"{scheduler.get_last_lr()[0]:.2e}"})

            if abs(prev_loss - loss_val) < tol and it > 5:
                pbar.set_description(f"Converged (Δloss<{tol})")
                break
            prev_loss = loss_val

            # transport-plan viz every 20 it using unified function
            if it % 20 == 0 or it == max_iter - 1:
                from utils.debug.optimization_diagnostics import save_transport_snapshot
                save_transport_snapshot(
                    transport_matrix=Tplan,
                    epsilon=0.1,  # placeholder epsilon value
                    iteration=it,
                    output_dir=transport_dir,
                    prefix="Tplan_E"
                )

        # -------- save final pose --------
        with torch.no_grad():
            # decompose E → R,t̂   (Kruppa / SVD)
            U, S, Vh = torch.linalg.svd(self.E_raw.data, full_matrices=False)
            V = Vh.mH
            
            # 符号ロック：det(U@V^T)が負ならU[:, 2]の符号を反転
            if torch.det(U @ V.mH) < 0:
                U[:, 2] *= -1
                
            W = torch.tensor([[0,-1,0],[1,0,0],[0,0,1]], device=self.device, dtype=torch.float32)
            R1 = U @ W  @ V
            R2 = U @ W.t() @ V
            t_hat = U[:, 2]

            # Ensure R is a rotation matrix (det=1)
            if torch.det(R1) < 0:
                R1 = -R1
            if torch.det(R2) < 0:
                R2 = -R2

            # choose the first R, t to store (chirality check left to user)
            self.R_wc = R1
            self.t_wc = t_hat
            self.f = K2_inv.t() @ self.E_raw.data @ K1_inv
            
            # Also compute and store the camera-to-world transformation
            self.R_cw = self.R_wc.t()
            self.t_cw = -self.R_cw @ t_hat

            # Store as OpenCV parameters if cv2 is available
            if cv2 is not None:
                rvec_numpy, _ = cv2.Rodrigues(self.R_wc.cpu().numpy())
                self.rvec = nn.Parameter(torch.from_numpy(rvec_numpy).to(self.device))
                self.tvec = nn.Parameter(self.t_cw)
                rvec_cw_numpy, _ = cv2.Rodrigues(self.R_cw.cpu().numpy())
                self.rvec_cw = nn.Parameter(torch.from_numpy(rvec_cw_numpy).to(self.device))
                self.center = nn.Parameter(self.t_cw)

        # -------- diagnostics --------
        # Simple loss plot (keep as minimal plotting)
        plt.figure()
        plt.plot(loss_history, '-o')
        plt.title("Loss (optimize_with_essential)")
        plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.grid(True)
        plt.savefig(os.path.join(transport_dir, "loss_optimize_with_E.png"))
        plt.close()

        if save_diagnostics:
            np.save(os.path.join(diagnostics_dir, "loss_E.npy"),
                    np.array(loss_history))
            np.save(os.path.join(diagnostics_dir, "E_norm.npy"),
                    np.array(E_norm_hist))
            
            # Extended diagnostics using unified optimization_diagnostics
            from utils.debug.optimization_diagnostics import save_optimization_diagnostics_SE3
            # Convert param_history to match expected format
            param_history_converted = {
                'rot_vec': [torch.zeros(3, device=self.device) for _ in param_history['E_raw']],  # placeholder
                'trans_vec': [torch.zeros(3, device=self.device) for _ in param_history['E_raw']]  # placeholder
            }
            grad_history_converted = {
                'rot_vec': [g[:3].flatten() if g is not None else None for g in grad_hist],
                'trans_vec': [g[3:6].flatten() if g is not None and g.numel() >= 6 else None for g in grad_hist]
            }
            save_optimization_diagnostics_SE3(
                output_dir=diagnostics_dir,
                loss_history=loss_history,
                param_history=param_history_converted,
                grad_history=grad_history_converted
            )
            
            print(f"Saved diagnostics to {diagnostics_dir}")

        return loss_history

