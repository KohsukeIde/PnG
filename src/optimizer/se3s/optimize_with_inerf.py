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
                # 勾配情報取得 - 統一ログ関数使用
                grad = self.delta.grad
                grad_norm = grad.norm().item()
                delta_norm = self.delta.norm().item()
                
                if iteration % 10 == 0:
                    rot_grad_norm = grad[:3].norm().item()
                    trans_grad_norm = grad[3:].norm().item()
                    
                    from utils.debug.optimization_diagnostics import log_optimization_progress
                    log_optimization_progress(
                        iteration=iteration,
                        loss_components={'total': loss.item()},
                        gradient_norms={'total': grad_norm, 'rot': rot_grad_norm, 'trans': trans_grad_norm},
                        parameter_stats={'delta_norm': delta_norm},
                        log_file=debug_log_path
                    )
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
        if save_diagnostics:
            # Use unified diagnostics from optimization_diagnostics
            from utils.debug.optimization_diagnostics import save_optimization_diagnostics_SE3
            # Convert param_history to match expected format
            param_history_converted = {
                'rot_vec': [T[:3, :3].detach().cpu() for T in param_history['T']],  # Extract rotation matrices
                'trans_vec': [T[:3, 3].detach().cpu() for T in param_history['T']]  # Extract translation vectors
            }
            grad_history_converted = {
                'rot_vec': [g[:3].detach().cpu() if g is not None else None for g in grad_history['delta']],
                'trans_vec': [g[3:].detach().cpu() if g is not None else None for g in grad_history['delta']]
            }
            save_optimization_diagnostics_SE3(
                output_dir=diagnostics_dir,
                loss_history=loss_history,
                param_history=param_history_converted,
                grad_history=grad_history_converted
            )
        
        return loss_history

