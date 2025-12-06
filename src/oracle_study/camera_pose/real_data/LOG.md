# Real-Data Loss Eval Memo

Quick log of Loss(F_ref) vs F_bad experiments on DTU scan63 with fitted GS (`apple_200_5k_pkls`).  
All runs use `epipolar_mode=sampson`, `noise_model=gaussian`, λ_color=1.0, λ_cov=0.3, λ_epi=1.0, σ_epi=400, σ_color=0.5, σ_cov=8.0 unless noted.

## Commands
Base script:
```
python src/oracle_study/camera_pose/real_data/loss_eval.py --image1 <img1> --image2 <img2> [extra args]
```

## Results (defaults)
- 0000–0001: Loss_ref 24.37; some F_bad (10/30°) < ref → GT not preferred.
- 0000–0038 (large baseline/angle): Loss_ref 74.72; F_bad 30° < ref → GT not preferred.
- 0000–0019: Loss_ref 65.77 is minimum → GT preferred.
- 0004–0048: Loss_ref 60.54; several F_bad < ref → GT not preferred.
- 0005–0018 (very small baseline): Loss_ref 49.99 is minimum → GT preferred.
- 0028–0048 (very small baseline): Loss_ref 70.86 is minimum → GT preferred.

## Epi-only ablation (λ_color=λ_cov=0)
- 0000–0001: Loss_ref 0.367; F_bad 30° 0.305 < ref → still not preferred.
- 0000–0038: Loss_ref 1.514; F_bad 30° 1.160 < ref → still not preferred.
- 0004–0048: Loss_ref 1.169 is minimum → turns GT preferred once color/cov are off.

## Extra ablations / tweaks (0000–0038, problematic pair)
- Epi+Color only (λ_color=1, λ_cov=0): Loss_ref 53.53; F_bad 30° 49.17 < ref → まだ非優位
- Epi+Cov only (λ_color=0, λ_cov=0.3): Loss_ref 10.98; F_bad 30° 8.39 < ref → 非優位
- σ_epi=150: Loss_ref 126.23; F_bad 30° 110.54 < ref → 非優位（悪化）
- σ_color=1.5: Loss_ref 21.07; F_bad 30° 17.52 < ref → 非優位（改善せず）

## Extra ablations / tweaks (0004–0048, color/covが悪さするケース)
- Epi+Cov only (λ_color=0, λ_cov=0.3): Loss_ref 8.33; F_bad 10° 9.80 / -10° 7.92 → ほぼ拮抗だが一部で非優位
- Epi+Color only (λ_color=1, λ_cov=0, σ_color=1.5): Loss_ref 7.97; F_bad 10° 8.63 / -10° 7.55 → わずかに非優位になる角度がある
- Epi-onlyではGT優位。色/共分散を弱めても完全には解消しきれないが悪化は緩和。

## Other pairs (Epi-only / Color+Cov only)
- 0000–0019:
  - Epi-only: Loss_ref 1.48 (min) → GT優位
  - Color+Cov only (λ_epi=0): Loss_ref=Loss_bad=34.76（角度影響なし）
- 0005–0018:
  - Epi-only: Loss_ref 1.26 (min) → GT優位
  - Color+Cov only: Loss_ref=Loss_bad=26.20（角度影響なし）
- 0028–0048:
  - Epi-only: Loss_ref 2.64 (min) → GT優位
  - Color+Cov only: Loss_ref=Loss_bad=35.31（角度影響なし）

## Extra tweak (0000–0001, stronger epi)
- λ_epi=3, σ_epi=200: Loss_ref 48.46; F_bad 30° 46.11 < ref → 依然非優位

## Strong-epi sweeps (λ_epi=5, σ_epi=100, λ_color弱め or off)
- 0000–0001:
  - λ_epi=5, σ_epi=100, λ_color=0.2, σ_color=2.0: Loss_ref 33.19; F_bad 30° 28.32 < ref → 非優位継続
  - λ_epi=5, σ_epi=100, λ_color=0, λ_cov=0: Loss_ref 29.36; F_bad 30° 24.41 < ref → 非優位継続
- 0000–0038:
  - λ_epi=5, σ_epi=100, λ_color=0.2, σ_color=2.0: Loss_ref 251.51; F_bad 30° 191.03 < ref → 非優位継続

## Current summary & insights
- 小基線（0005–0018, 0028–0048）と 0000–0019 は現行 or Epi-onlyでGT優位。Color+Covのみでは角度に依存せず定数化。
- 0004–0048 は Color/Cov が悪化要因。Epi-onlyならGT優位。弱めても完全解消はしないが悪化は緩和。
- 0000–0001, 0000–0038 は Epi-onlyでもGT非優位。λ_epi↑, σ_epi↓, λ_color↓, σ_color↑ でも解決せず。目的関数の形そのもの（SED修正や別正則化）を検討した方がよい領域。
- Color/Cov項は少なくとも「識別力の足し」にはなっておらず、場合によってはノイズ源。

## Next moves (candidate)
1) 根本調整（目的関数側）を検討: SED実装の向き修正を試す or エピポーラ項の別スケール設計。
2) Step3へ進み、Syntheticでノイズ増加に対する Loss(F_gt) vs F_bad の分離度を測り、現目的関数の限界を把握。
3) 追加スイープするなら、λ_epiさらに↑とσ_epi 80–120の組み合わせを少数だけ試し、改善しなければ切り上げ。
## Takeaways so far
- Color/Cov terms hurt at least for 0004–0048; Epi-only fixes it.
- Even Epi-only cannot make GT preferred for 0000–0001, 0000–0038 → need epi scale/σ tuning or different weighting.
- Small-baseline pairs (0005–0018, 0028–0048) are fine with current settings.

## Next tweaks to try
- For problematic pairs (0000–0001, 0000–0038):
  - Increase epi weight / reduce σ_epi (e.g., σ_epi=100–200 or larger λ_epipolar).
  - Weaken color: increase σ_color (1.0–2.0) or reduce λ_color.
- For 0004–0048: lower color/cov weights or increase σ_color/σ_cov; Epi-only already works.
