/* stylelint-disable */
/* stylelint-disable */
# Real-Data Loss Eval Memo

Quick log of Loss(F_ref) vs F_bad experiments on DTU scan63 with fitted GS (apple_200_5k_pkls).

## Setup / F_ref
- Data: DTU `scan63` (`data/DTU/scan63`), Gaussians: `data/fitted_gs/apple_200_5k_pkls/fitted_gaussians_XXXX.pkl`.
- Intrinsics/poses (current/GT): cameras.npz with scale_mat applied (`c2w = scale_mat @ world_mat_inv`, then `w2c = inv(c2w)`) → K,R,t.  
  *COLMAP poses/K are **not** used for F_ref anymore because they differ in scale/K from GT; F_ref now uses GT.*
- F_ref construction: relative pose (cam1 at origin) `R_rel = R2 @ R1^T`, `t_rel = t2 - R_rel @ t1`, then `F_ref = K2^{-T} [t_rel]_x R_rel K1^{-1}` via `_build_F_from_wc`.
- Loss_eval script: `src/oracle_study/camera_pose/real_data/loss_eval.py` (default epipolar_mode=sampson; can switch to sed/robust noise).
- GT option: `--use-gt-cameras` uses `cameras.npz` (with `scale_mat @ world_mat_inv` → `c2w` → `w2c`) to build K/R/t; filenames assumed `0000.png` style.

## Commands
Base script:
```
python src/oracle_study/camera_pose/real_data/loss_eval.py --image1 <img1> --image2 <img2> [extra args]
```

## Pair selection note
- SIFT/RANSACインライアが閾値未満、または Sampson(F_ref) が F_bad より明確に低くならないペアは除外（幾何情報不足）。
- 問題ペア（除外）: 0000–0001, 0000–0038（各種強化でもGT非優位）。

## Current key results (GT cameras, auto ε/ρ)
- 0000–0019: GT優位（デフォルト／epi-onlyとも優位）
- 0004–0048: GT非優位（Color/Covが悪化要因、epi-onlyでも非優位）
- 0005–0018: GT非優位（epi-onlyでも非優位）
- 0028–0048: GT非優位（epi-onlyでも非優位）
- Color/Covは幾何識別に寄与せずノイズ源になりがち → Pose実験は epipolar-only 推奨。

## Historical (pre-GT fix / exploratory sweeps)
- 各種 λ/σ スイープ、SED向き修正、ロバストノイズ、固定NN Sampson などを実施。問題ペア（0000–0001, 0000–0038）は一貫してGT非優位。
- SIFT/RANSAC: 0000–0001 で 118マッチ/74インライアだが Sampson_ref > Sampson_bad(30°)。0000–0038 はマッチ不足。
- OT ε/ρ固定でも問題ペアは非優位継続。

## Current key results (after GT fix)
- GT優位（epi-onlyも優位）: 0000–0019
- GT非優位（epi-onlyでも非優位）: 0004–0048, 0005–0018, 0028–0048
- 問題ペア（除外）: 0000–0001, 0000–0038
- Color/Cov は現状の Real GS では幾何識別に寄与せず、ノイズ源になりがち → Pose実験は epipolar-only 推奨。

## fixed_nn_sampson (epi-only, GT cameras)
- 0000–0019: ref 26452; F_bad(±,30°,60°) 11979–48094 → GT優位
- 0004–0048: ref 92619; F_bad min 58811 → GT優位（NNでは優位だが OT-lossでは非優位だった）
- 0005–0018: ref 105937; F_bad min 53525 → GT優位（NNでは優位だが OT-lossでは非優位だった）
- 0028–0048: ref 328295; F_bad min 180277 → GT優位（NNでは優位だが OT-lossでは非優位だった）

## Epi-only + fixed ε/ρ (epsilon=0.05, rho=0.5), GT cameras
- 0004–0048: Loss_ref 8.23; F_bad 60° 1.95 < ref → 非優位継続
- 0005–0018: Loss_ref 9.35; F_bad 60° 3.10 < ref → 非優位継続
- 0028–0048: Loss_ref 7.52; F_bad 60° 3.23 < ref → 非優位継続

## Epi-only + ε/ρ sweep (epsilon ∈ {0.02, 0.05, 0.1}, rho ∈ {0.5, 1, 5}), GT cameras
- 0004–0048: 最良ケース (ε=0.02, ρ=0.5) Loss_ref 3.28; F_bad min 0.79 → 非優位継続（全設定で F_bad < ref）
- 0005–0018: 最良ケース (ε=0.02, ρ=0.5) Loss_ref 2.80; F_bad min 1.10 → 非優位継続（全設定で F_bad < ref）
- 0028–0048: 最良ケース (ε=0.02, ρ=0.5) Loss_ref 1.84; F_bad min 1.26 → 非優位継続（全設定で F_bad < ref）

## Spatial term (pixel distance) in OT cost
- λ_spatial ∈ {0, 0.5, 1.0} で再評価したが Loss_ref/Loss_bad に変化なし → 現実的な効果なし（窓制約などより強い制限が必要）
- λ_spatial を 10, 50, 100 まで上げても Loss_ref/Loss_bad 変化なし → 現状の epi-only cost では spatial 項を足すだけでは逆転しない

### Spatial window gating (hard cutoff by pixel distance)
- 目的: epi-only + entropic OT が遠距離マッチを許すのを防ぐため、|p1-p2|>τ を大コスト化（τ=15px, 25px）。
- 実装: compute_cost_matrix(..., spatial_window_px=τ) で mask_far * 1e6 を加算（view間距離としては厳密でないが探索窓として使用）。
- 結果 (ε=0.05, ρ=0.5, F_bad=40°回転):  
  - 0004–0048: ref 8.23, bad 5.22 → 非優位継続 (τ=15,25 でも同値)  
  - 0005–0018: ref 9.35, bad 6.27 → 非優位継続 (τ=15,25 でも同値)  
  - 0028–0048: ref 7.52, bad 8.06 → GT優位維持 (τ=15,25 でも同値)  
- Insight: 単純なピクセル窓だけでは OT が好む「全体的に少し安い誤対応」を抑制できず、Loss_ref/Loss_bad の逆転は解消しない。

## SE(3) 最適化テスト（epi-only, GTカメラ, ペア0000–0019）
- 初期 GT Loss ≈ 0.413
- optimize_with_SE3(200iter, rot_lr=5e-3, trans_lr=5e-4, cpu, differentiable_transport=False)  
  → 収束 Loss ≈ 0.222, rot err ≈ 180°, trans dir err ≈ 102°（対称解に落下）
- 示唆: loss landscape がフラット/対称で勾配最適化が GT 近傍に収束しない可能性（最適化設計の見直しも必要）
- optimize_with_SE3(400iter, diff_transport=True, cheirality=0.1)  
  → 最終 Loss ≈ 1.526, rot err ≈ 0°, trans dir err ≈ 72.7°（回転は正解、並進は方向誤り）
- OT を「固定E-step」にし、Sinkhornを毎回 detach して R/t のみ最適化 (150iter, ε=0.05, ρ=0.5)  
  → rot err ≈ 0°, trans dir err ≈ 84.5°, Loss ≈ 1.60（回転は合うが並進は依然曖昧）

### Local EM: OT=E-step 専用 (0000–0019)
- F0: GT を 8° 回転摂動して初期化。各イテで OT (epi-only, ε=0.05, ρ=0.5, detach) → top-K=2000 を weighted 8点法で F 更新。
- 5iter で Sampson(F_est) は常に Sampson(F_ref) より小さい方向へ（例: iter0 err_ref≈3.4e4, err_est≈5.5e3 → iter4 err_ref≈3.1e4, err_est≈2.2e3）。GT から乖離した F に収束しがち。
- F_ref 固定で得た top-K OT 対応に対し、weighted 8点法で再推定すると Sampson(F_ref)≈8.7e2, Sampson(F_est)≈1.3e3（依然大きく、対応がGTと整合していない）。
- Insight: OTを局所E-step専用にしても、対応がGTに寄らず「都合のよいF」へ drift する。局所でも誤対応を抑える追加制約（エピポーラ帯域ゲートなど）が必要。

### Local EM + エピポーラ帯域ゲート (band=1.5px, topK=2000, 0000–0019)
- OT対応から Sampson 残差 < 1.5px だけ残して weighted 8点法 → recoverPose で R,t 更新を試行。
- 結果: 1iter目で帯域後の対応が6本しか残らず中断（insufficient_after_band）。
- Insight: OTが吐く top-K の大半が帯域外で、帯域ゲートをかけると対応が枯渇。GT近傍で有効に使うには、OT自体を帯域制約下で解くか、もっと緩い/多段のフィルタが必要。

### Local EM + OTコスト内でエピポーラ残差クリップ (epi_clip_px=1.5, topK=2000, 0000–0019)
- OTのサンプソン残差に閾値を入れ、帯域外に大コスト(1e6)を付与してから Sinkhorn → detach → weighted 8点法 → recoverPose。
- 5iter で Sampson(F_est) が継続的に Sampson(F_ref) より小さく推移（iter0 err_ref≈1.8e4, err_est≈8.2e3 → iter4 err_ref≈1.6e4, err_est≈2.0e3）。GTに寄らず、依然「都合のよいF」に drift。
- Insight: コスト内部での残差クリップを入れても、局所EMはGT保持に失敗。帯域内でなお誤対応が支配的で、追加の幾何制約（multi-view正則化/構造付きOT）が必要。

## OTなし（Gauss中心のMutual NN + RANSAC/8点法）でのF推定テスト
- 0004–0048: マッチ数1（RANSAC不可） → データ不足
- 0005–0018: マッチ数3（RANSAC不可） → データ不足
- 0028–0048: マッチ数20でRANSAC通過、Sampson_gt ≈ 1.08e6, Sampson_est ≈ 0.038, frob|F_est-F_gt| ≈ 1.41 → GTより推定FのほうがSampsonが桁違いに小さい（完全にGTを上回る別解）

### OTなし・NN対応を近傍ウィンドウ制限（threshold 5px/8px）
- 0000–0019: NN+cross-check で 8〜12本しか残らず、RANSAC後のインライアは7本で 8点法不可（too_few_inliers）。
- 0004–0048: 1〜4本でRANSAC不可（データ不足）。
- 0005–0018: 3〜5本でRANSAC不可（データ不足）。
- 0028–0048: 5px→20本(20 inliers)、8px→23本(20 inliers)。Sampson_gt ≈1.08e6→1.02e6、Sampson_F_est ≈0.038→0.08 と、依然 GT より推定Fが極端に良く見える（誤対応由来の“都合のいいF”が作られる）。
- Insight: 近傍ウィンドウで対応を絞っても、問題ペアは対応が枯渇するか、0028–0048のように誤対応に特化したFを生む。OTなしでも「対応が怪しいとFが暴れる」構造は同じ。

### 簡易 3view サロゲート（0000-0019-0048, OT-lossを2本足し合わせ）
- view0=0000 と view1=0019, view2=0048 の OT-Loss を合計（epi-only, ε=0.05, ρ=0.5）。view1/view2 の相対回転だけを摂動（20°, 40°）。
- rot 0°(GT): total 15.65 (l01 7.33, l02 8.32)
- rot 20°: total 16.89 (l01 10.27, l02 6.62)
- rot 40°: total 16.61 (l01 10.37, l02 6.24)
- Insight: 2view和という素朴な multi-view でも GT が最小だが差は~1.0–1.2 と小さい。真にmulti-viewらしくするには、ビュー間整合の正則化や構造付きOTが必要。

### 3view + 行方向マス整合正則化（0000-0019-0048, λ_cons ∈ {0.5,2,5}, epi_clip=1.0）
- OT(view0-view1) と OT(view0-view2) の行和をMSEで縛る簡易正則化を total = L01+L02+λ_cons*‖row(T01)-row(T02)‖² で追加。
- 結果: λ_cons=1.0 と似た挙動で、むしろ摂動20°/40°のほうが total が小さくなる（例: λ_cons=2.0 で rot0 total≈0.162, rot20≈0.138, rot40≈0.091）。GTマージンは広がらず、むしろ逆転。
- Insight: 行和一致だけでは mass を「平均化」してしまい、誤対応を抑制できず逆効果。multi-viewでGTを押し戻すには、エピポーラ整合を跨いだ構造付きOT（共通3D点仮説）など、より強い整合が必要。

## Pair selection rule
- SIFT/RANSACインライア不足、または Sampson(F_ref) が F_bad より明確に低くならないペアは初期ペアから除外。

## Next moves (with hypothesis)
1) 幾何的に曖昧なペアは除外し、GT優位なペアのみで pipeline を回す（epipolar-onlyで挙動確認）。
2) それでも不足する場合は multi-view / 3D 制約（構造付きOT、3DGS-in-the-loop 等）を検討。
3) OT の柔らかさ調整や構造付き制約を入れる場合は、GT優位ペアで少量実験してから広げる。

## Current summary & insights
- 小基線（0005–0018, 0028–0048）と 0000–0019 は現行 or Epi-onlyでGT優位。Color+Covのみでは角度に依存せず定数化。
- 0004–0048 は Color/Cov が悪化要因。Epi-onlyならGT優位。弱めても完全解消はしないが悪化は緩和。
- 0000–0001, 0000–0038 は Epi-onlyでもGT非優位。λ_epi↑, σ_epi↓, λ_color↓, σ_color↑ でも解決せず。目的関数の形そのもの（SED修正や別正則化）を検討した方がよい領域。
- Color/Cov項は少なくとも「識別力の足し」にはなっておらず、場合によってはノイズ源。

### 2DGS 実データの軽量統計（apple_200_5k_pkls, imgs: 0000/0019/0048/0004/0018/0028）
- N: 180–200/枚。mean座標は大きくはみ出す例あり（例: 0019 min≈[-1393,-1154], max≈[2830,1837]）。
- scale: 画像によって極端。0000 の median(scale_x,y) ≈ (0.02, 0.14) に対し、0019/0048/0018/0028 は median が ~1e-37（実質0）、一方 p95 は 30–98、max は 100–160 と超ブロード。スケール推定が壊れている個体が多数含まれる可能性。
- alpha: 多くが高透明度（median ≈0.76〜0.9999）、p05 は ~1e-6 オーダの極小も混在。
- Insight: スケール推定のスプレッドが非常に大きく、座標も画枠外が多い。Sim2Realを設計する際は「実データのこの壊れ方」をノイズ分布として再現する必要がある。

### 画像外 + 異常スケールのGaussを除外して再評価（W=1600,H=1200, pad=20px, scale<=20）
- フィルタ後の残存数: 0000:109/200, 0019:25/196, 0048:23/180, 0004:49/200, 0018:26/196, 0028:26/188（kept ratio 0.13〜0.55）。
- OT-Loss (epi-only, ε=0.05, ρ=0.5) の GT/bad (30°rot) 比較:  
  - 0000–0019: ref 2.02 < bad 3.24 → GT優位  
  - 0004–0048: ref 1.57 > bad 1.32 → 非優位継続  
  - 0005–0018: ref 1.70 > bad 1.60 → 非優位（僅差）  
  - 0028–0048: ref 1.84 > bad 1.44 → 非優位  
- Insight: 画像外/巨大スケール除去でマスは大幅減少したが、問題ペアのGT非優位はほぼ改善せず。データクリーニングだけでは不十分で、目的関数/対応生成側の構造的問題が残る。

### 軽量 Sim2Real 試行（暫定・要修正）
- GTカメラ(0000,0019)でランダム3D点を投影し、ピクセルノイズσ∈{0.5,1.5,3,6}、スケールにlognormalノイズσ∈{0.2,0.5,1.0}＋10% heavy-tail×50を付与。DummyGSで OT-Loss(F_gt) vs F_bad(30°回転) を評価。
- 結果 (修正版, sigma_epipolar=1e6):  
  - noise_xy=0.5, scale_sigma=0.2: Loss_ref ≈ 5.5e-05, Loss_bad ≈ 41.91 (GT 大幅優位)  
  - noise_xy=1.5, scale_sigma=0.5: Loss_ref ≈ 5.6e-05, Loss_bad ≈ 41.91 (GT 大幅優位)  
  - noise_xy=3.0, scale_sigma=0.5: Loss_ref ≈ 5.3e-05, Loss_bad ≈ 41.91 (GT 大幅優位)  
  - noise_xy=6.0, scale_sigma=1.0: Loss_ref ≈ 5.2e-05, Loss_bad ≈ 41.91 (GT 大幅優位)  
- Insight: シンプルなSyntheticでは、sigma_epipolarを大きくすると GT が強く優位（Loss_bad が高い）。Real で GT 非優位になるのは、実データの壊れたスケール/座標分布や OT の誤対応許容が主要因と見られる。今後は実データ統計に近いノイズ（座標はみ出し・巨大スケール）を明示的に再現する必要がある。

### Sim2Real（実データ寄せノイズ: 座標はみ出し + スケールheavy-tail強化 + αばらつき）
- 3D点500、座標ノイズσ_xy、30–40%を±1000〜1500pxシフト（画像外模倣）。スケール: base 0.02px × lognormal(σ_scale)、50%を×200（p95/ max を100–200超に寄せるイメージ）。αはBeta(0.5,0.5)で広くばらつかせる。sigma_epipolar=1e6。
- Loss_ref / Loss_bad (30°回転):  
  - (σ_xy=2.0, σ_scale=0.7): 1.53e-4 / 33.01  
  - (σ_xy=4.0, σ_scale=1.0): 1.59e-4 / 30.85  
  - (σ_xy=8.0, σ_scale=1.2): 1.53e-4 / 32.11  
  - (σ_xy=12.0, σ_scale=1.5): 1.73e-4 / 31.98  
- Insight: ここまでノイズを盛っても GT は依然大幅優位（Loss_ref ≪ Loss_bad）。RealでのGT非優位は、より異常なスケール/座標分布や質量分布の歪み、もしくはOTの自由度による誤対応が主因と推測される。

## Next moves (with hypothesis)
1) 目的関数側の再設計を少量試す（WHY: 0000–0001/0000–0038でEpi-onlyすら非優位）  
   - 例: λ_epiさらに↑＋σ_epi 80–120 で1–2本だけ確認。  
   - 見込み: これで改善しなければ、スケール調整では解けず、別の形（正則化/距離定義）の検討が必要と判断できる。
2) K・姿勢の簡易健全性チェック（WHY: F_ref側のズレがあれば目的関数をいくら強くしても勝てない）  
   - 近接ペアでSfM由来K/poseを再確認、あるいは基線小ペアでF_ref再計算。  
   - 見込み: ズレがあればF_ref改善が先。ズレなければ目的関数強化に絞れる。
3) 軽量Sim2Realを1–2条件だけ実施（WHY: どのノイズ量からLoss分離が崩れるかを把握し、Realがその範囲内か外かを判断するため）  
   - 見込み: 低ノイズでもGT非優位なら目的関数が弱い。低ノイズでGT優位なら、Realは「目的関数の情報量を超えるノイズ領域」にいると判断できる。
