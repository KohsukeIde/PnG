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

---

## New EM 2DGS (scan63_em_200) 再評価タスク
- 生成元: `data/fitted_gs/scan63_em_200/`（mask対応EM, K=200, pkl出力あり）。
- これを用いて再度 Real Loss eval / pose 推定を実施（0000–0019, 0004–0048, 0005–0018, 0028–0048）。
- 実施項目:
  - fixed_nn_sampson(F_ref/F_bad) 再計測
  - OT-Loss (epi-only, ε=0.05, ρ=0.5) 再計測
  - 必要に応じ multi-view/構造付きOT 簡易検証
- 目的: GT優位マージンが改善するかを確認し、残る問題が OT/構造側かを切り分ける。

### scan63_em_200 再計測結果（epi-only, ε=0.05, ρ=0.5, F_bad=±60°）
- OT-Loss（epiのみ, λ_color=λ_cov=0, σ_epi=400, ε=0.05, ρ=0.5）※results/*.json 出力  
  - 0000–0019: ref 0.165 / bad 0.036/0.289 → GT優位  
  - 0004–0048: ref 0.186 / bad 0.101/0.070 → GT非優位継続  (そもそもこのペアは視野角を共有しないので，ペアの選び直しが必要→後述)
  - 0005–0018: ref 0.124 / bad 0.203/0.214 → GT優位  
  - 0028–0048: ref 0.247 / bad 0.230/0.291 → 僅差で非優位  
- fixed_nn_sampson（片側NN, 2DGS中心）  
  - 0000–0019: ref 2.73e4 > bad 4.24e5/1.25e5 → NNではbad優位  
  - 0004–0048: ref 2.08e5 < bad 3.56e5/3.54e5 → NNではGT優位  
  - 0005–0018: ref 2.72e5 > bad 4.00e4/1.83e5 → NNではbad優位  
  - 0028–0048: ref 1.38e5 < bad 4.82e4/8.83e4 → NNではGT優位  
- Insight: 旧形式pklで正式に再計測しても、0004–0048と0028–0048のOT-Loss非優位は残存。NNでは優劣が逆転するペアがあり、OTの自由度（誤対応）側の問題が引き続き濃厚。

### 視野共有フィルタ後の再評価（SIFTマッチ＋基線）
- ペア選定: `results/pairs.json`（min_matches=60, top_k=12, min_index_gap=3, min_baseline=0.05m, GTカメラ中心距離）。0004–0048, 0028–0048 など視野非共有ペアは含まれない。
- スイープ: (ε,ρ) ∈ {(0.02,0.5), (0.05,0.5), (0.1,1.0)}, epi-only, λ_color=λ_cov=0, σ_epi=400, F_bad=±60°, GTカメラ。
- 代表結果（eps=0.02, ρ=0.5, min bad loss / ref loss）  
  - 0014–0024: bad 0.0126 / ref 0.0799  
  - 0013–0025: bad 0.0195 / ref 0.0489  
  - 0015–0023: bad 0.0750 / ref 0.0871  
  - 0014–0025: bad 0.0133 / ref 0.0797  
  - 0025–0031: bad 0.0393 / ref 0.0903  
  - 0012–0027: bad 0.0786 / ref 0.0333 (唯一 ref<bad)  
  - 0031–0045: bad 0.0929 / ref 0.0688 (bad優位)  
  - 0024–0032: bad 0.0778 / ref 0.0818  
  - 0014–0023: bad 0.0446 / ref 0.0800  
  - 0030–0046: bad 0.0924 / ref 0.0696 (bad優位)  
  - 0013–0026: bad 0.0590 / ref 0.0487 (bad優位)  
  - 0032–0044: bad 0.0938 / ref 0.0575 (bad優位)  
- 傾向: ε/ρを変えても多くのペアで bad≲ref（GT非優位）。fixed_nn_sampson も符号が割れ、OT優位性と一致しない。
- Insight: 視野共有＋基線フィルタを入れても GT優位は安定せず、目的関数/構造側（ウィンドウ制限やmulti-view等）を検討する必要がある。

### 視野共有ペアでの σ_epi 強弱＋ε/ρ 再スイープ（新2DGS, epi-only, λ_color=λ_cov=0, F_bad=±60°）
- 設定: σ_epi ∈ {200, 600}, (ε,ρ) ∈ {(0.02,0.5), (0.05,0.5), (0.1,1.0)}, GTカメラ、ペアは上記12本。
- 集計（GT優位本数 / bad優位本数 / tie）  
  - σ=200, ε=0.02: 7 / 5 / 0  
  - σ=200, ε=0.05: 7 / 5 / 0  
  - σ=200, ε=0.1: 5 / 7 / 0  
  - σ=600, ε=0.02: 2 / 10 / 0  
  - σ=600, ε=0.05: 2 / 10 / 0  
  - σ=600, ε=0.1: 2 / 10 / 0  
- 例（σ=200, ε=0.02, bad/ref）: 0014–0024 0.014/0.051, 0013–0025 0.020/0.050, 0012–0027 0.047/0.052 (ref優位), 0032–0044 0.024/0.0011 (ref優位) ほか。  
- 傾向: σを下げてもGT優位が僅かに増える程度で、多数ペアは bad≲ref。σを上げるとさらに bad優位が増える。ロバストノイズ（cauchy/huber, ε=0.02,ρ=0.5）でも逆転は解消せず。
- Insight: サンプソン残差の強弱やロバスト化を変えても、OTが誤対応を許す構造はそのまま。single-view内では窓制約/構造付き対応など追加の制約が必要。

### OTなし（Gauss中心のMutual NN + 8点法RANSAC, window 8px） @視野共有12ペア
- 手順: mutual NN (8px) → RANSAC-F (cv2.FM_RANSAC, thresh=1px, conf=0.99) → インライア上の Sampson(F_gt) / Sampson(F_est) を計算（GTカメラ）。全ペア RANSAC 成功（inliers: 37–81）。
- 数値例（sam_ref / sam_est）  
  - 0014–0024: 2.02e4 / 0.16  
  - 0013–0025: 7.66e3 / 0.145  
  - 0015–0023: 5.35e4 / 0.145  
  - 0014–0025: 1.45e4 / 0.127  
  - 0025–0031: 3.63e4 / 0.135  
  - 0012–0027: 1.45e3 / 0.126  
  - 0031–0045: 7.50e4 / 0.117  
  - 0024–0032: 5.92e4 / 0.140  
  - 0014–0023: 2.67e4 / 0.169  
  - 0030–0046: 7.11e4 / 0.126  
  - 0013–0026: 4.71e3 / 0.124  
  - 0032–0044: 8.63e4 / 0.149  

### 旧GS（apple_200_5k_pkls）での比較（視野共有12ペア, epi-only, ε=0.05, ρ=0.5, σ_epi=400）
- 全ペアで Loss_ref ≫ Loss_bad（例: 0014–0024 ref 9.90 / bad 3.44/3.50, 0013–0025 ref 9.14 / bad 3.31/3.30, 0032–0044 ref 6.88 / bad 2.61/2.73）。
- 傾向: 旧データは新GSよりさらに bad 優位が強く、GTマージンは大幅悪化。新GSで多少改善はあるが依然 GT優位は安定しない → データ改善だけでは不十分で目的関数/対応選択の構造的問題が残る。
- Insight: OTなし（NN+RANSAC）でも推定Fが桁違いに低いサンプソンとなり、“都合の良いF”が容易に生成される。OT/NNいずれも誤対応が支配的で、帯域・構造・多視点など強い幾何制約が必要。

### 座標・解像度の整合修正（原因と処置）
- 原因: 2DGS生成時は `images_resized`(384x288) で (y,x) 保存していた一方、評価は `images`(1554x1162) の K/F_gt を用いていたため、解像度不一致と軸不整合で Sampson が桁爆発していた。
- 処置:
  - `image_to_gmm_em.py`: 保存時に (y,x)→(x,y) へ並べ替え、cov を軸交換してから `projected_gaussians` を書き出すよう変更。既存 `scan63_em_200` も一括変換。
  - `loss_eval.py`: Gauss を評価画像サイズにリスケール（reconstructed_image.pngから推定 or means最大値 fallback）し、(x,y) 前提で処理。
  - `optimal_transport_solver_torch.py`: (x,y) 前提に戻し、同次座標をそのまま構築。
- 効果: Sampson(F_gt) の桁ズレは解消（依然GT>F_estだが1e3〜1e5程度に低減）。OT-Lossも解像度整合後は0.03〜0.27程度に整合。

### エピポーラゲート（epi_clip）導入 (新GS, 視野共有12ペア, epi-only, ε=0.05, ρ=0.5, σ_epi=400)
- 実装: Sampson残差が τ^2 を超える (i,j) に 1e6 を加算して輸送候補を実質マスク（tau=3px, 5px）。
- 代表結果: ノークリップでは bad 優位多数（例: 0014–0024 ref 0.139 / bad 0.079/0.077）。  
  - τ=3: 0014–0024 ref 5.9e-7 / bad 8.4e-6 → ref優位に反転（桁落ち）  
  - τ=5: 同ペアで ref 5.9e-7 / bad 2.5e-5 → ref優位  
  - 他ペアも多くで Loss が桁落ちし GT優位化するが、いくつかは bad≲ref のまま残る（詳細は `results/*_clip{tau3|tau5}.json`）。
- Insight: 距離ゲートを入れると誤対応が大きく抑制され、GT優位ペアが増える。ただし全ペアで安定GTとはならず、さらなる帯域/構造・多視点制約が依然必要。

### rhoバグ修正後の再評価（balanced比較含む, 新GS, 視野共有12ペア）
- 修正: Sinkhornで外部指定rhoをそのまま使うように変更（これまで ρ が内部で 10*ε に上書きされていた）。
- 設定: epi-only, ε=0.05, ρ∈{0.5, 1e6}, σ_epi=400, λ_color=λ_cov=0, clip∈{なし, τ=3, τ=5}, GTカメラ。
- 集計（GT優位 / bad優位 / tie, 12ペア）  
  - ρ=0.5: noclip 0/12/0, τ=3 12/0/0, τ=5 12/0/0  
  - ρ=1e6: noclip 0/12/0, τ=3 0/12/0, τ=5 0/12/0  
- Insight: ρ を効かせると、ρ=1e6（balanced風）では clip を入れても bad優位が残る一方、ρ=0.5 では clip で全ペア ref 優位に反転。質量保存よりも「帯域ゲート」が効いており、unbalanced の柔らかさ＋広い候補集合が主因だった可能性が高い。

### Sampson候補数の比較（GT vs bad, τ=3/5, 新GS）
- τ=3 では多くのペアで GT 側に数本〜十数本の候補、bad は 0 本が多数（例: 0014–0024 cnt_ref=8, cnt_bad=0）。τ=5 でも同様に bad の候補がほぼ 0、GTのみ少数候補。  
- 一部ペアは GT も 0（0025–0031, 0031–0045, 0024–0032, 0030–0046, 0032–0044 など）→ overlap/帯域で候補枯渇。  
- Insight: clip で GT優位になるのは「badの帯域内候補がほぼ無い／GTも僅少」ため。候補自体が乏しいペアでは Loss が極小化しやすく、スケール設計に注意。

### σ_epi スイープ（400→10→3→1, τ=3 clip, ε=0.05, ρ=0.5, λ_color=0）
- σ_epi を下げると全ペアで Loss_ref/bad がほぼ0に収束（勾配が極小化）。識別力を見たい場合は σ_epi を小さくしすぎない方がよい（400程度が妥当域）。

### λ_color スイープ（0.0→0.1→1.0, τ=3 clip, σ_epi=400, ε=0.05, ρ=0.5）
- clip強めの状態では λ_color を入れても全ペアで Loss_ref/bad ≈ 0 となり、識別への寄与は観測できず。帯域が厳しいとカラー項までゼロ化するため、カラーを効かせるには帯域設計と合わせて再考が必要。

### 追加: Unbalanced vs Balanced風（rho→∞）＋質量ログ
- 設定: epi-only, ε=0.05, ρ=0.5 vs rho=1e9（質量保存近似）, σ_epi=400, 視野共有12ペア。
- 傾向: loss_bal はごく僅かに増加、mass_sum も unbalanced とほぼ同程度（例: 0014–0024 mass_unb 1.588 / mass_bal 1.601）。→ 質量捨て（unbalanced）だけがbad優位の主因ではない。

### 追加: epi_clipなし/ありのLoss比較（まとめ）
- ノークリップ: 多数ペアで bad 優位。
- τ=3,5: 多くのペアで ref≪bad に反転し桁落ち。ただし一部は bad≲ref が残存。

### epi_clip+topk ログ拡充テスト（新GS, 0000–0019, epi-only, ε=0.05, ρ=0.5, σ_epi=400, τ=3, topk=20, λ_color=λ_cov=0）
- 結果（loss_ref=0.00938, mass_ref=0.204, avg_cost_ref=0.0459, KL_row=1.42, KL_col=2.09, inlier_mass_ref=0.204）  
  - bad (±10/30/60°): loss_bad ∈ [5.1e-5, 1.63e-2], mass_bad ∈ [0.09, 0.68], avg_cost_bad ∈ [7.6e-5, 0.18]。最小は +30°（loss 7.4e-5, mass 0.678）。  
  - KL は bad でも ~0.29–3.42、row/col 質量の min–max は 1e-4〜8e-2 程度で NaN/Inf なし。
- Insight: clip+topk を入れても 0000–0019 では依然 bad 優位（loss_ref > 全ての bad）。帯域＋候補縮小で数値は安定し質量も確保できているが、F の識別は改善しない → 次は topk/clip を固定したまま他ペア・パラメータ（σ_epi, ρ）を再スイープし、どこで GT 優位に転じるかを確認する。

### epi_clip+topk 他ペア追試（新GS, epi-only, ε=0.05, ρ=0.5, σ_epi=400, τ=3, topk=20, λ_color=λ_cov=0）
- 0004–0048: loss_ref 0.0141 (mass 0.054), bad 最小 +60° loss 0.00622 (mass 0.868) → bad 優位。KL_row/col ≈2.49/2.61。質量は ref で枯渇気味。
- 0005–0018: loss_ref 0.0264 (mass 0.087), bad 最小 −60° loss 1.1e-4 (mass 1.23) → bad 優位かつ大差。KL_row/col ref ≈1.59/2.47。
- 0028–0048: loss_ref 0.0278 (mass 0.113), bad 最小 +60° loss 0.00181 (mass 1.01) → bad 優位。KL_row/col ref ≈1.57/2.10。
- Insight: clip+topk=20 でも他ペアはすべて bad 優位のまま。ref 側の質量が小さく、bad 側で大きく輸送され平均コストが下がるケースが多い。次は σ_epi/ρ/ topk を振りつつ、質量分布（row/col sum, KL）を比較し GT 優位化する条件を探索する。

### σ_epi 200 スイープ（0000–0019, τ=3, topk=20, ε=0.05, λ_color=λ_cov=0）
- ρ=0.5: loss_ref 0.00850 (mass 0.166, avg_cost 0.0513), bad 最小 +60° loss 0.000201 (mass 0.665) → bad 優位。KL_row/col ref ≈1.90/1.85。
- ρ=1e6: loss/avg_cost が桁跳ね（loss_ref 1.92e5, avg_cost 3.41e5）、bad も高コストだが相対比較では依然 bad ≪ ref。質量は保存に近づき inlier_mass~0.37。→ balanced 方向に振ると数値爆発しつつ bad 優位。
- Insight: σ_epi を 200 にしても GT 優位化せず。ρ を大きくするとスケールが吹き上がり、比較困難。次は σ_epi=600 や topk 縮小を試すか、質量正規化/コストスケールの扱いを再検討する必要あり。

### dustbin + massペナルティ付きスコア導入テスト（0000–0019, τ=3, topk=20, ε=0.05, ρ=0.5, σ_epi=400, λ_mass=0.1, λ_inlier_avg=0.1, dustbin_cost=1.0）
- 追加実装: dustbin 行列をコストに付与（ノーマッチ定数コスト）、score_mass_pen = avg_cost + λ_mass*(KL_row+KL_col)、score_inlier = -inlier_mass + λ_inlier_avg*inlier_avg_cost を記録。
- 結果: loss_ref 0.311 (mass 0.490, avg_cost 0.634, score_mass_ref 0.721, score_inlier_ref -0.180)。bad 最小は +30°（loss 0.205, mass 0.876, score_mass 0.360, score_inlier -0.672）で依然 bad 優位（score 指標でも bad < ref）。
- 観察: dustbin で質量が潰れにくくなり、KL も小さく安定。ただし score_mass / score_inlier いずれも bad が小さい → スコア再設計だけでは逆転せず。候補設計や c_unmatch のスイープが必要。

### dustbin_cost スイープ開始（0000–0019, τ=3, topk=20, ε=0.05, ρ=0.5, σ_epi=400, λ_mass=0.1, λ_inlier_avg=0.1）
- c_unmatch=0.1: loss_ref 0.207 (mass 2.139, avg_cost 0.0966, score_mass_ref 0.0972, score_inlier_ref -0.0740)。bad 最小 +30°（score_mass 0.0866, score_inlier -0.350）で依然 bad 優位。KL_row/col 極小（~0.002–0.014）で質量ほぼ均等に分配。
- c_unmatch=1.0: (上記 dustbin テスト) でも bad 優位。
- Insight: c_unmatch を小さくしても、質量は dustbin を含め広く分配され、score_mass/score_inlier とも bad 側が有利。さらなる c_unmatch スイープ（大きめ 5.0 など）と τ/σ_epi の再設計が必要。

### 追加: c_unmatch=5.0（同条件）
- loss_ref 0.00988 (mass 0.204, score_mass_ref 0.538, score_inlier_ref -0.199)。bad 最小 +30° score_mass 0.274 / score_inlier -0.678 → 依然 bad 優位。KL_row/col ref 2.11/2.78 と質量は再び偏り気味。
- Insight: c_unmatch を上げても順位は逆転せず。質量偏りと候補設計の両面を調整する必要がある。

### 追加: adaptive clip pct=1%（閾値不足で τ=3 fallback）※0000–0019
- 実装は min_j Sampson 分布から zero-row 1% 以下を目標に τ を決めるが、dists 次元不足で fallback τ=3 に。結果は dustbin=1.0 と同等オーダーで bad 優位のまま（score_mass_ref 0.486, score_inlier_ref -0.517; bad 最小 score_mass 0.278, score_inlier -0.962）。
- Insight: Sampson 距離の行列計算（pairwise）が得られず適応計算が動かなかった。pairwise Sampson を正しく計算するか、最小候補分布を別途取得する必要あり。

### 追加: σ_epi=600（τ=3, topk=20, dustbin=1.0）
- loss_ref 0.307 (mass 0.492, score_mass_ref 0.714, score_inlier_ref -0.186)。bad 最小 +30° score_mass 0.360 / score_inlier -0.672 → 依然 bad 優位。σ_epi を広げても順位は変わらず。
- Insight: コスト帯域を広げても bad 優位は解消せず。候補設計・スコア設計の根本見直し（pairwise Sampsonでの適応τや window制約）が必要。

### 追加: adaptive clip (zero-row 1%) + dustbin=1.0
- pairwise Sampson から τ を自動決定（zero-row率≒1% → τ≈458px）。結果は依然 bad 優位（score_mass_ref 0.486 / score_inlier_ref -0.517；bad 最小 +10° score_mass 0.278 / score_inlier -0.962）。τ が極端に広がり、実質ノークリップに近い挙動。
- Insight: ゼロ行率を抑えると帯域が広がり過ぎて誤対応自由度が復活 → 依然 bad 優位。帯域を広げすぎないよう window 制約や topk 縮小と組み合わせる必要がある。

### 追加: window 制約 + tighter band（τ=10, window=20px, topk=10, dustbin=1.0）
- 0000–0019: score_mass_ref 0.747 / score_inlier_ref -0.159、bad 最小 +30° score_mass 0.346 / score_inlier -0.724 → 依然 bad 優位。ただし τ/窓を絞っても順位は変わらず。
- Insight: 空間窓と帯域を締めても bad 優位が続く。候補自由度を下げても、依然コスト設計（epi-only）と質量配分が bad に有利に働いている。

### 追加: color 試行（wide band, window=50, topk=50, τなし, σ_color=0.2, λ_color=0.5）
- dustbin無し: 0000–0019 で transport が全ゼロ（loss/avg_cost/mass すべて0）→実質崩壊。
- dustbinあり(c_unmatch=1): loss_ref=0.365, avg_cost=1.0, mass=0.365、inlier_mass=0（すべて mask 外に流れた扱い）。bad も全角度同値で ref/bad 差なし。
- Insight: color を足しても gating/topk との相性が悪く、輸送が崩壊または均一化して識別不能。帯域設計と質量制約を見直すか、color を当面切り離したほうがよい。

### 追加: dustbin_cost をさらに増大（c_unmatch=10, 20）
- c=10: score_mass_ref 0.536, score_inlier_ref -0.199、bad 最小 +30° score_mass 0.274 / score_inlier -0.677 → 依然 bad 優位。KL_row/col ref ≈2.12/2.79。
- c=20: score_mass_ref 0.536, score_inlier_ref -0.199、bad 最小 +30° score_mass 0.274 / score_inlier -0.675 → 依然 bad 優位。質量・スコアとも大きな変化なし。
- Insight: c_unmatch をさらに上げても順位は逆転せず。質量偏りは残り、score_inlier も bad 側が優位のまま。

### (x,y) 統一の rescale 修正後の再実行（0000–0019, τ=3, topk=20, ε=0.05, ρ=0.5, λ_color=λ_cov=0）
- loss_ref 0.00935 (mass 0.199, score_mass_ref 0.0469, score_inlier_ref -0.199)、bad 最小 +60° loss 4.65e-5 (mass 0.658) → 依然 bad 優位。KL_row/col ref ≈1.45/2.12。SIFT inlier 9本（Sampson_ref ~8.55e4 と大きく、Sampson_bad_min ~9.7e3〜7.1e4 で悪い F の方が小さい）。
- Insight: 座標系を (x,y) に揃えても bad 優位は継続。SIFT サンプソンでも bad 側が低く、そもそも GT F を支持する対応が得られていない可能性が高い（座標/解像度/K の整合か、SIFT が別解を指しているか要確認）。

### 追加: pairwise Sampson の最小分布（(x,y) 統一＆rescale修正後, 0000–0019, GTカメラ）
- F_ref: min_row mean/med/10% = 6.44e+2 / 1.62 / 1.53e-2, min_col mean/med/10% = 5.18 / 0.98 / 1.53e-2
- F_bad (+10/-10/+30/-30/+60/-60°):
  - +10°: min_row 7.81e+3 / 3.15 / 6.90e-2, min_col 5.53e+2 / 2.12 / 6.90e-2
  - -10°: min_row 5.98e+2 / 1.69 / 1.74e-2, min_col 1.11e+1 / 1.15 / 1.49e-2
  - +30°: min_row 5.87e+4 / 1.90e+4 / 4.67e-1, min_col 1.03e+4 / 1.25e+3 / 6.37e-1
  - -30°: min_row 5.40e+4 / 3.19e+4 / 1.45e+0, min_col 3.06e+4 / 1.06e+4 / 1.40e+0
  - +60°: min_row 1.08e+5 / 7.96e+4 / 6.64e-1, min_col 1.16e+3 / 1.08e+2 / 3.92e-1
  - -60°: min_row 7.66e+5 / 7.40e+5 / 4.53e+5, min_col 7.88e+5 / 7.68e+5 / 4.47e+5
- Insight: pairwise Sampson では F_ref が圧倒的に優位（特に 10%分位で桁違いに小さい）。点群の幾何情報は GT を強く支持しており、OT 実験で bad 優位となるのはコスト設計/輸送の扱いが信号を壊していることを示唆。

### 整合性チェック & 切り分け状況（Step0〜Step1）
- 座標系: rescale を (x,y) 前提に修正済み。pairwise Sampson でも F_ref が圧倒的に優位 → 点群幾何は GT を支持。
- **GTローダにバグがあった**: `world_mat_inv` を c2w と誤解しており、R がランク1に近かった。`world_mat` を投影行列として RQ 分解（cv2.decomposeProjectionMatrix）し直すことで解消。  
  - 修正後: 離れたペア（0000–0019）で GT が圧倒的優位（min_row_median ≈2.43、bad+30° ≈3.66e4 など）。
  - 隣接ペア（0000–0001）はベースライン小で差が小さいが想定内。
- SIFT: 旧GTローダ時の結果は無効。修正後ローダで再取得が必要。
- OT スコア: 旧GTローダでの実験は無効。修正後ローダ＋COLMAPで再評価が必要。


### 修正後GTローダ再評価（scan63_em_200, epi-only, ε=0.05, ρ=0.5, σ_epi=400, angles=±10/30/60）
- 0000–0019: loss_ref 0.0333、min loss_bad 0.00526 (-60°, mass 2.78e-3)、min avg_cost bad -10° 0.0199 < ref 0.0228。pairwise min_row_med ref 2.43、bad(-10) 1.27、bad(-60) 6.46e5。
- 0000–0001: loss_ref 0.0281、min loss_bad 0.0267 (+10°)。pairwise min_row_med ref 0.997、bad(+10) 0.686。
- 0004–0048: loss_ref 0.0205、min loss_bad 0.0212 (-10°) → GTわずかに優位。pairwise min_row_med ref 1.01、bad(-10) 1.07。
- 0005–0018: loss_ref 0.0181、min loss_bad 2.17e-26 (+60°, mass 7.37e-28) → mass collapse。pairwise min_row_med ref 0.414、bad(+10) 5.78e3。
- 0028–0048: loss_ref 0.0181、min loss_bad 2.77e-24 (-60°, mass 1.04e-25) → mass collapse。pairwise min_row_med ref 1.09、bad(+10) 1.74e3。
- Insight: loss最小が「輸送質量ほぼ0」のbadで出るケースがあり、loss単体比較は危険。avg_cost/score_mass 併記が必要。

### COLMAPカメラ再評価（scan63_em_200, 同条件）
- 0000–0019: loss_ref 0.0270、min loss_bad 0.00297 (-60°, mass 1.36e-3)。
- 0000–0001: loss_ref 0.0269、min loss_bad 0.0261 (+10°)。
- 0004–0048: loss_ref 0.0197、min loss_bad 0.0219 (-10°) → GT同様にrefが僅差で優位。
- 0005–0018: loss_ref 0.0181、min loss_bad 7.21e-26 (+60°, mass 2.50e-27) → mass collapse。
- 0028–0048: loss_ref 0.0180、min loss_bad 1.84e-23 (-60°, mass 7.18e-25) → mass collapse。
- Insight: GT/Colmapでオーダは近く、順位も概ね同様。mass collapseが主要課題。
- Note: COLMAP はGTではないので、以降の評価はGTのみを使用（この節は参考扱い）。

### mutual top-k / mask+dustbin / primal 再評価（GTローダ修正後, 0000–0019）
- mutual top-k（τ=3, topk=20, mutual, dustbinなし）: loss_ref 8.09e-4、min loss_bad 2.22e-4 (-10°)。bad優位は継続（primal_score_min_bad -0.238）。
- mask+dustbin（τ=3, topk=20, dustbin=1.0, λ_mass=λ_inlier_avg=0.1）: loss_ref 0.1569、min loss_bad 0.1493 (-10°)。score_mass_ref 0.2186 vs min_bad 0.207、score_inlier_ref -1.107 vs min_bad -1.149 → 依然 bad 優位。

### SIFT/track 再評価（GTローダ修正後）
- 0000–0001: SIFT matches 118 / inliers 74。Sampson_ref 89.2、bad(+10) 30.8（bad優位）。
- 0000–0019: SIFT matches 19 / inliers 9。Sampson_ref 5.37e4、bad(-10) 5.33e4（ほぼ同等だがbad僅差）。
- shared tracks: `images.txt` が無いため取得不可（`images_correct.txt` しか無い）→ track評価は別途対応が必要。

### GT-only 追加チェック（scan63_em_200, epi-only, ε=0.05, ρ=0.5, σ_epi=400）
- 0005–0018: pairwise min_row_med ref 0.414、bad(-10) 5.78e3。SIFT matches 65 / inliers 38、Sampson_ref 6.63、min bad 1.09e5 → GT強優位。primal_ref -0.739、min primal_bad -0.052 → primalでもGT優位。
- 0028–0048: pairwise min_row_med ref 1.09、bad(+10) 1.74e3。SIFT matches 208 / inliers 157、Sampson_ref 5.71、min bad 1.18e5 → GT強優位。primal_ref -0.720、min primal_bad -0.0606 → primalでもGT優位。
- 0004–0048: pairwise min_row_med ref 1.01、bad(-10) 1.07（僅差）。SIFTはマッチ不足で取得不可 → 依然不確定。
- 0000–0001 / 0000–0019: pairwise/SIFTとも bad が僅差で優位（ベースライン小 or 低重なりの疑い）。識別ペアとしては弱い。
- Insight: 0005–0018/0028–0048 は「幾何（pairwise/SIFT）ではGTが勝つが、OT lossはmass collapseでbadが勝つ」構図。評価指標の見直しが最優先。

### 強ペアの候補制約テスト（GT-only, 0005–0018 / 0028–0048）
- mutual top-k（τ=3, topk=20, mutual, dustbinなし）: loss最小は依然 mass collapse の bad が出るが、**primal は ref が最小**（0005–0018: ref -0.393 < bad_min -2.35e-7、0028–0048: ref -0.351 < bad_min -1.20e-8）。
- mask+dustbin（τ=3, topk=20, dustbin=1.0, λ_mass=λ_inlier_avg=0.1）: ref が loss/score で優位。  
  - 0005–0018: loss_ref 0.149 < bad_min 0.271、score_mass_ref 0.182 < bad_min 0.462  
  - 0028–0048: loss_ref 0.150 < bad_min 0.256、score_mass_ref 0.188 < bad_min 0.420  
- Insight: 候補制約＋dustbin で mass collapse を抑えると、GT優位が回復する。評価指標は primal/score_mass を主に使うべき。

### TODO（やり直し＆再評価）
- [x] 修正後GTローダで主要ペア（例: 0000–0019, 0000–0001 ほか）を再評価（pairwise Sampson, OTスコア）し、旧結果を置き換える。
- [x] SIFT/track を修正後ローダで再取得し、Sampson(F_ref) < Sampson(F_bad) となるか検証（trackは images.txt 不在で未取得）。
- [x] mutual top-k / mask+dustbin / primal 比較を、修正後ローダで再実行し、候補設計を見直す。
- [x] SIFT を 0005–0018, 0028–0048 に拡張（GT優位を確認）。
- [x] OTスコアを raw loss ではなく primal / avg_cost / mass-aware 指標で比較し、mass collapse の影響を排除できるか確認（primal/score_massでGT優位が回復）。
- [x] 強ペア（0005–0018, 0028–0048）に限定し、mutual top-k + dustbin + mass制約でGT優位化できるか確認。
- [ ] 0004–0048 は SIFTマッチ不足のため、別の弱い外観特徴（小パッチ/勾配）で候補集合を絞れるか試す。

---

## 2024-12-25: DTU cameras.npz フォーマット詳細調査

### 調査の動機
- Codex の実験結果で「bad優位」（摂動したFがGT Fより良いスコア）が多発
- DTU の cameras.npz からの F 行列が RANSAC F と大きく異なる（Sampson ~50000 vs ~0.01）
- カメラデータの解釈に問題がある可能性を調査

### NeuS/IDR フォーマットの正しい解釈

参考: [IDR DATA_CONVENTION.md](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md), [NeuS Issue #27](https://github.com/Totoro97/NeuS/issues/27)

```python
# NeuS の正しいアプローチ
P = world_mat @ scale_mat  # 4x4 @ 4x4
P = P[:3, :4]              # 3x4 投影行列
K, R, t = cv2.decomposeProjectionMatrix(P)
```

- `world_mat`: 4x4 (上3x4が投影行列 P = K[R|t])
- `scale_mat`: 4x4 (3D座標の正規化用、スケール ~225)
- `P = world_mat @ scale_mat` で 1600x1200 画像のピクセル座標に投影

### 検証結果: カメラ姿勢 (R, t) は正確

```
=== DTU vs COLMAP 比較 ===
R (回転行列):     差分 = 0.000000  ✓ 完全一致
t (並進ベクトル): 差分 = 0.000000  ✓ 完全一致
K (内部パラメータ): 差分 = 126.88   × 不一致
```

カメラ中心も完全一致（誤差 0.0000）:
```
COLMAP 0000.png: center=[ 1.23254404  0.17633213 -3.0419736 ]
DTU    idx 0:    center=[ 1.23254407  0.17633213 -3.04197364]
```

### 内部パラメータ K の不一致原因

| パラメータ | DTU (1600x1200) | DTU (スケール後) | COLMAP | 差分 |
|-----------|-----------------|------------------|--------|------|
| fx        | 2892.33         | 2809.18          | 2892.33| ~0   |
| fy        | 2883.18         | 2791.87          | 2883.18| ~0   |
| cx        | 823.21          | 799.54           | 777.00 | -23px|
| cy        | 619.07          | 599.47           | 581.00 | -19px|

**根本原因: 非対称クロップ**

```
元画像:   1600 x 1200
実画像:   1554 x 1162  (46px x 38px 小さい)

クロップ方向（左上から切り取り）:
  ┌─────────────────────┐
  │←46px→              │
  │  ↑                  │
  │ 38px                │
  │  ↓                  │
  │    実際の画像領域    │
  │                     │
  └─────────────────────┘
```

- DTU は 1600x1200 の中心付近 (823, 619) を主点として想定
- 画像が**左上から非対称にクロップ**された
- COLMAP は bundle adjustment で正しい主点 (777, 581) を再推定
- 単純なスケーリングではクロップオフセットを補正できない

### SIFT マッチングの問題

| 画像ペア | ベースライン | SIFTマッチ | インライア率 | F(カメラ) Sampson | F(RANSAC) Sampson |
|---------|-------------|-----------|-------------|-------------------|-------------------|
| 0000-0001 | 0.57 | 118 | 78% | **0.86** ✓ | 0.49 |
| 0000-0002 | 1.11 | 53 | 68% | **0.48** ✓ | 0.31 |
| 0000-0010 | 0.61 | 242 | 85% | **0.60** ✓ | 0.27 |
| 0000-0005 | 2.31 | 18 | 50% | **144716** ✗ | 0.08 |
| 0000-0019 | 3.08 | 19 | 47% | **48016** ✗ | 0.01 |

**結論:**
- 近いペア（ベースライン < 1.5）: カメラからの F が正しく機能（Sampson ~0.5-1）
- 遠いペア（ベースライン > 2）: SIFT が**誤マッチ**を生成、RANSAC は誤マッチにフィットする**退化した F** を生成

### 「bad優位」の真の原因

以前の実験で「摂動した F が GT F より良いスコア」となった理由:

1. **ベースラインが大きいペアで SIFT が誤マッチ**
2. **RANSAC は誤マッチにフィットする F を生成**（真のカメラジオメトリと無関係）
3. **摂動した F が偶然誤マッチにフィット**する場合、GT F より良いスコアになる
4. **mass collapse**: 極端な摂動（±60°）で輸送質量 → 0、見かけ上の loss が低下

### load_gt_npz_mapping の修正

```python
def load_gt_npz_mapping(npz_path: str, actual_image_size: tuple[int, int] | None = None):
    """NeuS スタイルで DTU cameras.npz を読み込む"""
    # P = world_mat @ scale_mat で分解
    world_mat = data[f"world_mat_{idx}"]
    scale_mat = data[f"scale_mat_{idx}"]
    P = (world_mat @ scale_mat)[:3, :4]

    K, R, t_homog = cv2.decomposeProjectionMatrix(P)

    # 実画像サイズにスケール（ただし主点オフセットは補正できない）
    if actual_image_size:
        sx = W_actual / 1600
        sy = H_actual / 1200
        K[0, :] *= sx
        K[1, :] *= sy
```

### 推奨事項

1. **評価には COLMAP カメラを使用**（`--use-gt-cameras` なし）
   - K が bundle adjustment で最適化済み
   - 主点のオフセット問題を回避

2. **画像ペアの選択**
   - ベースライン < 1.5 のペアを使用（SIFT が正しくマッチ）
   - 推奨: 0000-0001, 0000-0002, 0000-0010, 0009, 0011-0014

3. **評価指標**
   - loss 単体ではなく primal / score_mass を使用（mass collapse 回避）
   - COLMAP の検証済みトラック（images_correct.txt）を参照可能


