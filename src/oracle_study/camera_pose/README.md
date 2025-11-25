# Camera Pose Oracle Study

Utilities and experiments for evaluating camera pose optimisation with synthetic
correspondences and known ground-truth poses.

## Getting Started

Run the baseline oracle experiment (SE(3) + S³×S²) for a single scenario:

```bash
python src/oracle_study/run_experiment.py pose_oracle_basic --scenario baseline_yaw --epipolar-mode hybrid
```

Key options:
- `--scenario`: scenario name or `all` to sweep every predefined case
- `--init-mode {random,gt}`: optimiser initialisation (`random` noise or near ground truth)
- `--skip-se3` / `--skip-geoopt`: disable a solver branch
- `--lambda-*`, `--sigma-*`, `--noise-model`: cost weighting/robustness controls
- `--results-root`: override output directory (default `camera_pose/results/`)

Each run writes metrics and artefacts under
`src/oracle_study/camera_pose/results/<scenario>/<method>/`, including:
- `metrics.json` (rotation/translation errors, Sampson residuals, diagonal concentration)
- `F_est.npy`, `R_est.npy`, `t_est.npy`, `sampson_residuals.npy`
- optional loss/transport plots (if enabled in the solver)

## Future Work

- Scenario sweeps & aggregate benchmarking (translation, large baseline, noise, occlusion)
- Hyperparameter tuning / Optuna-based optimisation
- Integration with downstream reconstruction pipeline once pose oracle is stable

Contributions and experiment additions belong under `camera_pose/experiments/`.
