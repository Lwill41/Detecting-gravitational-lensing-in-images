# Hough, CNN, autoencoders, then a physics benchmark

These are first CPU prototypes, not final model-selection experiments. All
observational methods use the existing seeded manifest. The supplied test set
is not scored. Repeated validation comparisons are exploratory and must not be
presented as independent test performance.

## Run in PowerShell

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements_experiments.txt
.\.venv\Scripts\python.exe -B -m unittest discover -v
.\.venv\Scripts\python.exe -u experiment_suite.py
```

Each run creates a fresh `runs/methods_...` directory with a terminal log,
per-method metrics, predictions, plots, weights and a comparison JSON. The
default run uses 3 CNN epochs, 2 epochs for each AE and 1000 PINN updates.
Subsequent suite runs create a new timestamped directory, preserving earlier work.
Individual methods can be run with `--output` pointing to a new directory:

```powershell
python hough.py --data prepared_v1 --output runs/hough_new
python neural_train.py --kind cnn --epochs 3 --data prepared_v1 --output runs/cnn_new
python neural_train.py --kind symmetry_ae --epochs 2 --data prepared_v1 --output runs/symmetry_new
python neural_train.py --kind plain_ae --epochs 2 --data prepared_v1 --output runs/plain_new
python neural_train.py --kind denoising_ae --epochs 2 --data prepared_v1 --output runs/denoising_new
python pinn.py --steps 1000 --output runs/pinn_new
```

## Hough approach

Estimate a central foreground galaxy within a 12x12 central window. Compute
radial mean intensity per channel and smooth that profile with sigma=3 pixels
before subtracting it. Smoothing prevents a narrow complete ring from being
entirely absorbed by the foreground profile. Scale residuals by an outer-region
MAD estimate, combine positive residuals across channels and apply Canny edges.
Vote for circles of radii 4,6,...,20 pixels with candidate centers within 6 pixels
of the estimated galaxy center. Keep radius-wise perimeter-normalized votes,
angular coverage, ring support, center offset and radial gradient alignment.

Report the raw maximum Hough vote and a class-weighted logistic classifier on
the geometric features. Feature standardization and classifier fitting use
training data only. Parameters are fixed for the initial run; no search over
validation labels is performed. The gallery deliberately includes high-scoring
negatives. Circular geometry misses noncircular arcs and can detect isophotes,
stars, spiral arms and padding artifacts. It is a lens-candidate score, not a
physical lens fit. Elliptical/generalized Hough is a later extension.

API reference: https://scikit-image.org/docs/stable/auto_examples/edges/plot_circular_elliptical_hough_transform.html

## CNN

Three convolution/pooling blocks (16,32,64 channels), a spatial fully connected
head and dropout. Train on standardized images with orientation augmentation
and balanced sampling, without additionally weighting BCE. AdamW, learning
rate=0.001, weight decay=0.0001. Save the epoch with highest validation AP.
The initial CPU run is deliberately short; convergence is not assumed.
`gradcam_examples.png` shows high/low-scoring validation examples per class and
coarse positive-logit Grad-CAM sensitivity maps. These maps are not arc masks
or causal explanations. The inspection script checks the saved checkpoint
reproduces the stored scores for the selected examples.

## Autoencoders

Fit only non-lens training images and select checkpoints by the mean relative
reconstruction error on non-lens validation images. Lens labels are used for
reporting anomaly-detection performance, not AE fitting. Higher reconstruction
error is the predefined anomaly direction; do not silently invert scores if
results are poor. Report both raw MSE and energy-normalized MSE. AEs may learn
brightness/background differences rather than lens geometry, or reconstruct
lenses well, so good reconstruction does not imply good detection.

The three models share a convolutional bottleneck with shape (8,8,8):

1. **Symmetry AE (first):** decompose the image into RMS amplitude s and shape
   x/s. Model shape, then multiply the reconstruction by s. At inference,
   average inverse-transformed predictions across all 8 square symmetries D4.
   This makes reconstruction D4-equivariant and equivariant to positive global
   intensity scaling (above the numerical amplitude floor). Averaging pooled
   encoded descriptors across D4 yields an invariant shape descriptor. Training
   uses reconstruction plus 0.1 times a stochastic equivariance penalty.
2. **Plain convolutional AE:** raw-image reconstruction baseline.
3. **Denoising AE:** add Gaussian intensity noise (sigma=0.03), clip the noisy
   input to [0,1], and reconstruct the unchanged clean processed image.

Global intensity scaling is a mathematical nuisance experiment, not a claim
that intrinsic luminosity is irrelevant to lensing. No per-band scaling is used,
because it would change color. There is no scale, translation, shear or arbitrary
rotation equivariance guarantee. Extra controlled ablations (amplitude only,
D4 only, matched capacity/compute) are needed to attribute any improvement.

`symmetry_probe.json` records controlled score changes under a 90-degree
rotation/reflection, common brightness scaling by 1.5, and both. It uses 32
validation examples per class, with no clipping during scaling. This is a
transformation check, not a population accuracy or physical luminosity test.
Score units differ between classifiers and reconstruction models, so their
absolute changes should not be ranked as if they measured the same quantity.

## PINN: a separate synthetic problem

The current images have unknown upstream intensity transforms and lack PSF,
noise and mass metadata. A quantitative physical inversion cannot be justified
from them alone. The initial PINN instead solves the known SIS potential
boundary-value problem on 0.15 <= r <= 1:

`psi = b*r`, `laplacian(psi) = 2*kappa = b/r`, `b = 0.3`.

A coordinate MLP learns potential from Poisson residuals at random collocation
points plus known inner/outer potential boundary values. Autodiff gives the
deflection `alpha = gradient(psi)`; source coordinates are
`beta = theta - alpha`. A Gaussian source then illustrates surface-brightness
remapping through the learned potential. Singular central pixels are excluded.
This model does not learn unknown lens mass from observed data. It has no
observational lens/non-lens AP or AUC; report potential, deflection and PDE errors
on independent synthetic collocation points instead. Real-data physics modeling
requires a calibrated forward model including foreground light, PSF and noise.

Reference for source-to-image mapping:
https://lenstronomy.readthedocs.io/en/latest/_modules/lenstronomy/ImSim/image2source_mapping.html
