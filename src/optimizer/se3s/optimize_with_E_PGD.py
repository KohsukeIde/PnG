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

            # log gradient info every 10 it
            if it % 10 == 0:
                gnorm = self.E_raw.grad.norm().item() if self.E_raw.grad is not None else 0
                with torch.no_grad():
                    vals = self.E_raw.detach().cpu().view(-1).numpy()
                with open(debug_log, 'a') as f:
                    f.write(f"{it},{loss.item():.6f},{gnorm:.6f}," +
                            ",".join([f"{v:.6f}" for v in vals]) + "\n")
                print(f"\nIter {it}  loss {loss.item():.6f}  |E_grad| {gnorm:.3e}")
                
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

            # transport-plan viz every 20 it
            if it % 20 == 0 or it == max_iter - 1:
                with torch.no_grad():
                    T_np = Tplan.detach().cpu().numpy()
                rows, cols = T_np.shape
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
                    t_np_display = T_np[::downsample_factor, ::downsample_factor]
                    plt.imshow(t_np_display, cmap="hot", interpolation="nearest", aspect="auto")
                    plt.title(f"Transport Plan at Iteration {it} (Downsampled {downsample_factor}x)")
                else:
                    plt.imshow(T_np, cmap="hot", interpolation="nearest", aspect="auto")
                    plt.title(f"Transport Plan at Iteration {it}")
                plt.colorbar(label="Transport Plan Value")
                plt.xlabel("Image 2 Gaussians")
                plt.ylabel("Image 1 Gaussians")
                plt.tight_layout()
                plt.savefig(os.path.join(transport_dir, f"Tplan_{it:04d}.png"))
                plt.close()

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
            
            # Extended diagnostics: similar to optimize_with_SE3
            self.save_optimization_diagnostics_E(
                output_dir=diagnostics_dir,
                loss_history=loss_history,
                param_history=param_history,
                grad_history=grad_hist
            )
            
            print(f"Saved diagnostics to {diagnostics_dir}")

        return loss_history

    def save_optimization_diagnostics_E(self, 
                                       output_dir: str,
                                       loss_history: list,
                                       param_history: dict,
                                       grad_history: list) -> None:
        """Essential matrix最適化の診断情報を保存する
        
        Essential matrix直接最適化の詳細な診断情報を生成・保存します：
        - 損失軌跡の分析
        - E_rawパラメータの挙動
        - 勾配挙動の分析
        - 収束性分析
        - テキスト形式のサマリーレポート
        
        Args:
            output_dir: 診断ファイルを保存するディレクトリ
            loss_history: イテレーションごとの損失値リスト
            param_history: パラメータ履歴の辞書（'E_raw'を含む）
            grad_history: 勾配履歴のリスト
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # 出力ディレクトリ作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 履歴をNumPy配列に変換
        E_raw_history = np.array([p.detach().cpu().numpy() for p in param_history['E_raw']])
        
        # 勾配の履歴をNumPy配列に変換（Noneがある場合はゼロで置換）
        if grad_history:
            grad_history_np = np.array([g.detach().cpu().numpy() if g is not None 
                                       else np.zeros_like(E_raw_history[0]) 
                                       for g in grad_history])
        else:
            # 勾配履歴がない場合は空の配列を作成
            grad_history_np = np.array([])
        
        # イテレーション数
        iterations = range(len(loss_history))
        
        # ======================= 1. 損失軌跡の分析 =======================
        plt.figure(figsize=(12, 8))
        plt.subplot(211)
        plt.plot(iterations, loss_history, 'b-', linewidth=2)
        plt.title('Loss Value During Essential Matrix Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # 損失の変化（微分）をプロット
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
        plt.savefig(os.path.join(output_dir, 'loss_analysis_E.png'), dpi=150)
        plt.close()
        
        # =============== 2. Essential行列要素の可視化 ===============
        if E_raw_history.shape[0] > 0:
            fig = plt.figure(figsize=(15, 10))
            gs = GridSpec(3, 3, figure=fig)
            
            for i in range(3):
                for j in range(3):
                    ax = fig.add_subplot(gs[i, j])
                    ax.plot(iterations, E_raw_history[:, i, j], 'b-', linewidth=1.5)
                    ax.set_title(f'E_raw[{i},{j}]')
                    ax.set_xlabel('Iteration')
                    ax.set_ylabel('Value')
                    ax.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'essential_matrix_elements.png'), dpi=150)
            plt.close()
        
        # =============== 3. 勾配分析 ===============
        if len(grad_history_np) > 0:
            # 勾配ノルムを計算
            grad_norms = np.linalg.norm(grad_history_np.reshape(grad_history_np.shape[0], -1), axis=1)
            
            fig = plt.figure(figsize=(12, 6))
            plt.plot(range(len(grad_norms)), grad_norms, 'r-', linewidth=2)
            plt.title('Essential Matrix Gradient Norm')
            plt.xlabel('Iteration')
            plt.ylabel('Gradient Norm')
            plt.yscale('log')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'gradient_norm_analysis_E.png'), dpi=150)
            plt.close()
            
            # 勾配要素のヒートマップ
            if grad_history_np.shape[0] > 0:
                plt.figure(figsize=(10, 8))
                grad_avg = np.mean(np.abs(grad_history_np), axis=0)  # 各要素の絶対値の平均
                plt.imshow(grad_avg, cmap='hot', interpolation='nearest')
                plt.colorbar(label='Avg Absolute Gradient')
                plt.title('Average Absolute Gradient Magnitude per Matrix Element')
                
                # 行列要素のラベル
                element_labels = [f'e{i+1}{j+1}' for i in range(3) for j in range(3)]
                element_labels = np.array(element_labels).reshape(3, 3)
                
                # 各セルに値を表示
                for i in range(3):
                    for j in range(3):
                        plt.text(j, i, f'{element_labels[i,j]}\n{grad_avg[i,j]:.2e}', 
                                 ha='center', va='center', color='w' if grad_avg[i,j] > np.mean(grad_avg) else 'k')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'gradient_heatmap_E.png'), dpi=150)
                plt.close()
        
        # =============== 4. Essential行列のノルムとRank分析 ===============
        if E_raw_history.shape[0] > 0:
            # Frobenius ノルムを計算
            E_norms = np.linalg.norm(E_raw_history.reshape(E_raw_history.shape[0], -1), axis=1)
            
            # 各イテレーションでのSVD特異値を計算
            svd_values = []
            for E_mat in E_raw_history:
                U, S, V = np.linalg.svd(E_mat)
                svd_values.append(S)
            svd_values = np.array(svd_values)
            
            fig = plt.figure(figsize=(15, 10))
            
            # Frobeniusノルム
            ax1 = fig.add_subplot(211)
            ax1.plot(iterations, E_norms, 'b-', linewidth=2)
            ax1.set_title('Essential Matrix Frobenius Norm')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Norm')
            ax1.grid(True)
            
            # 特異値
            ax2 = fig.add_subplot(212)
            for i in range(3):
                ax2.plot(iterations, svd_values[:, i], 
                         label=f'σ{i+1}', linewidth=2)
            ax2.set_title('Singular Values of Essential Matrix')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Value')
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'essential_norm_rank_analysis.png'), dpi=150)
            plt.close()
            
            # σ1/σ2比率の分析（理想的には1）
            plt.figure(figsize=(10, 6))
            sigma_ratio = svd_values[:, 0] / (svd_values[:, 1] + 1e-10)
            plt.plot(iterations, sigma_ratio, 'g-', linewidth=2)
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Ideal ratio (σ1=σ2)')
            plt.title('Ratio of First to Second Singular Values (σ1/σ2)')
            plt.xlabel('Iteration')
            plt.ylabel('Ratio')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'singular_value_ratio.png'), dpi=150)
            plt.close()
            
            # σ3の分析（理想的には0）
            plt.figure(figsize=(10, 6))
            plt.plot(iterations, svd_values[:, 2], 'r-', linewidth=2)
            plt.axhline(y=0.0, color='g', linestyle='--', alpha=0.7, label='Ideal value (σ3=0)')
            plt.title('Third Singular Value (σ3)')
            plt.xlabel('Iteration')
            plt.ylabel('Value')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'third_singular_value.png'), dpi=150)
            plt.close()
        
        # =============== 5. テキスト形式のサマリーレポート ===============
        with open(os.path.join(output_dir, 'optimization_analysis_E.txt'), 'w') as f:
            f.write("ESSENTIAL MATRIX OPTIMIZATION PROCESS ANALYSIS\n")
            f.write("===========================================\n\n")
            
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
            
            # Essential行列の特性分析
            if E_raw_history.shape[0] > 0:
                f.write("2. ESSENTIAL MATRIX PROPERTIES\n")
                f.write("----------------------------\n")
                
                # 初期値と最終値
                f.write("Initial E_raw matrix:\n")
                f.write(str(E_raw_history[0]) + "\n\n")
                
                f.write("Final E_raw matrix:\n")
                f.write(str(E_raw_history[-1]) + "\n\n")
                
                # ノルム分析
                initial_norm = np.linalg.norm(E_raw_history[0])
                final_norm = np.linalg.norm(E_raw_history[-1])
                f.write(f"Initial Frobenius norm: {initial_norm:.6f}\n")
                f.write(f"Final Frobenius norm: {final_norm:.6f}\n\n")
                
                # 特異値分析
                U, S, V = np.linalg.svd(E_raw_history[-1])
                f.write(f"Final singular values: {S[0]:.6f}, {S[1]:.6f}, {S[2]:.6f}\n")
                f.write(f"σ1/σ2 ratio: {S[0]/S[1]:.6f} (ideal is 1.0)\n")
                f.write(f"σ3 value: {S[2]:.6e} (ideal is 0.0)\n\n")
            
            # 勾配分析
            if len(grad_history_np) > 0:
                f.write("3. GRADIENT BEHAVIOR\n")
                f.write("-------------------\n")
                
                # 勾配の統計
                max_grad = np.max(grad_norms)
                min_grad = np.min(grad_norms)
                avg_grad = np.mean(grad_norms)
                f.write(f"Gradient norm - Max: {max_grad:.6f}, " 
                        f"Min: {min_grad:.6f}, Avg: {avg_grad:.6f}\n")
                
                # 勾配消失/爆発チェック
                vanishing_threshold = 1e-6
                exploding_threshold = 1e2
                
                vanishing_grad = any(grad < vanishing_threshold for grad in grad_norms)
                exploding_grad = any(grad > exploding_threshold for grad in grad_norms)
                
                f.write(f"Gradient vanishing detected: {vanishing_grad}\n")
                f.write(f"Gradient exploding detected: {exploding_grad}\n\n")
            
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
            if not is_monotonic and oscillation_count > len(loss_history) * 0.1:
                issues.append("- Loss exhibits significant oscillations, suggesting unstable optimization.")
                
            if plateau_count > len(loss_history) * 0.3:
                issues.append("- Loss exhibits plateaus, suggesting the optimizer may be struggling to make progress.")
            
            if E_raw_history.shape[0] > 0:
                U, S, V = np.linalg.svd(E_raw_history[-1])
                if S[0]/S[1] > 1.1:
                    issues.append(f"- Final σ1/σ2 ratio ({S[0]/S[1]:.2f}) is not close to 1.0, " 
                                  "which violates Essential matrix constraints.")
                
                if S[2] > 0.01:
                    issues.append(f"- Final σ3 value ({S[2]:.2e}) is not close to 0.0, "
                                  "which violates Essential matrix rank-2 constraint.")
            
            if len(grad_history_np) > 0:
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
                
            # 最適化改善のための提案
            f.write("\n5. SUGGESTIONS FOR IMPROVEMENT\n")
            f.write("------------------------------\n")
            
            suggestions = []
            
            if len(grad_history_np) > 0 and exploding_grad:
                suggestions.append("- Consider using a smaller learning rate or increasing gradient clipping threshold.")
                
            if len(grad_history_np) > 0 and vanishing_grad:
                suggestions.append("- Consider using a larger learning rate or different optimizer (e.g. Adam).")
                
            if oscillation_count > len(loss_history) * 0.2:
                suggestions.append("- Increase momentum or add decay to learning rate to stabilize optimization.")
                
            if plateau_count > len(loss_history) * 0.4:
                suggestions.append("- Try different learning rate schedule or optimizer to escape plateaus.")
                
            if E_raw_history.shape[0] > 0:
                U, S, V = np.linalg.svd(E_raw_history[-1])
                if S[0]/S[1] > 1.1 or S[2] > 0.01:
                    suggestions.append("- Consider more frequent or more accurate projections onto the Essential matrix manifold.")
                    suggestions.append("- Try different initialization or parameterization of the Essential matrix.")
            
            if len(suggestions) > 0:
                for suggestion in suggestions:
                    f.write(suggestion + "\n")
            else:
                f.write("No specific improvements needed. The optimization appears to be well-configured.\n")
        
        print(f"Saved Essential matrix optimization diagnostics to {output_dir}")