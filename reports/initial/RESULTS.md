# Initial method experiments

All classification/anomaly results below use the same 6,081-image validation split. No supplied test images were scored.
CNN: 3 epochs; symmetry/plain/denoising AE epochs: 2/2/2. These are exploratory CPU prototypes and are not a claim of convergence or final generalization.

| Method | Average precision | ROC-AUC | Recall at <=1% FPR |
|---|---:|---:|---:|
| hough | 0.2655 | 0.7963 | 0.1272 |
| cnn | 0.7806 | 0.9772 | 0.6763 |
| symmetry_ae | 0.0318 | 0.1611 | 0.0000 |
| plain_ae | 0.0418 | 0.3567 | 0.0000 |
| denoising_ae | 0.0474 | 0.4204 | 0.0000 |

The original image-statistics baseline had validation AP 0.5138 and ROC-AUC 0.9482.
Higher AE reconstruction error was fixed as the anomaly direction; scores were not inverted to improve results.
Architecture, fitting objective and inference cost differ, so matched-budget and component ablations are still required.

## Symmetry checks

```json
{
  "max_abs_D4_equivariance_error": 1.7881393432617188e-07,
  "max_abs_positive_brightness_equivariance_error": 3.5762786865234375e-07,
  "max_abs_shape_descriptor_invariance_error": 9.5367431640625e-07,
  "scope": "Algebraic architectural properties on three examples; not proof of physical luminosity invariance."
}
```

## Synthetic PINN — separate task

```json
{
  "model": "SIS Poisson PINN (synthetic boundary-value benchmark)",
  "seed": 429,
  "steps": 1000,
  "equation": "laplacian(psi)=2*kappa=b/r; alpha=gradient(psi); beta=theta-alpha",
  "einstein_radius_dimensionless": 0.3,
  "domain_radius": [
    0.15,
    1.0
  ],
  "holdout_potential_mae": 0.009486114606261253,
  "holdout_deflection_rmse": 0.0248741265386343,
  "holdout_poisson_residual_rmse": 0.03859424963593483,
  "holdout_points": 4096,
  "observational_data_used": false,
  "known_inputs": [
    "synthetic SIS mass/convergence field",
    "inner/outer potential boundary values"
  ],
  "limitations": "Known synthetic mass, no PSF/noise, no mass inference. This is a separate PDE task; AP/AUC comparisons to lens classifiers are not appropriate."
}
```

The PINN uses known synthetic mass and boundary values, not DeepLense images. Its PDE errors cannot be ranked against detection AP/AUC.