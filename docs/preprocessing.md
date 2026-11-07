# COS 429: DeepLense observational lens finding

This is the active pipeline for the downloaded observational DeepLense dataset.
The older `lensing_pipeline_20261107` directory targets Bologna Challenge data
and is a separate, unused implementation.

## Reproduce (PowerShell, from this directory)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -p test_data.py
.\.venv\Scripts\python.exe prepare.py --archive ..\deeplense_observational_20261107\deeplense_observational.zip --output prepared_v1
.\.venv\Scripts\python.exe quality.py --data prepared_v1
.\.venv\Scripts\python.exe baseline.py --data prepared_v1 --output baseline_v1
```

Python 3.14 was used for the initial run. `environment.txt` records the exact
installed environment used. Outputs must not already exist; choose new names
for reruns. The original ZIP is preserved and no archive paths are extracted.
The raw image cache is about 2.46 GB and is memory-mapped, avoiding a full-RAM load.

## Preprocessing contract

- Validate every real NPY member: shape (3,64,64), float32, finite values in [0,1].
- Ignore macOS metadata and directories. Save raw pixels unchanged in images.npy.
- Labels: lens=1, non-lens=0. Paths identify examples; filenames alone are not
  universal object identifiers and repeat between folders.
- Retain the supplied test set. Deduplicate exact pixel arrays within each
  original split, and exclude training copies matching the test set. Contradictory
  labels for identical pixels fail explicitly.
- Make a seeded, stratified 80/20 train/validation split of eligible original
  training images. No exact image hash may span the resulting splits.
- Compute per-channel pixel mean and population standard deviation on training
  rows only; apply the same z-score to validation and test. No additional clipping,
  per-image scaling, resizing, ImageNet statistics, or galaxy subtraction.
- Channel order stays as supplied. The official instructions say three filters
  but do not identify their order, so figures label channels 1/2/3.
- Augmentation is optional and training-only: 90-degree rotations and reflection.
  These are candidate invariances; their effect on instrumental patterns should
  be tested in later experiments. No image interpolation is needed.
- Optional balanced sampling draws with replacement using inverse class counts.
  Validation/test preserve their natural prevalence. Prefer either balanced
  sampling or a weighted loss initially; combining both overweights positives.

Without sky coordinates, exact-image checks cannot guarantee that different
cutouts or near-duplicates of the same sky object are split independently.
The images appear to have upstream preprocessing; [0,1] values are not calibrated
flux measurements. Pixel-level arc masks are not provided.

## Load batches

```python
from data import LensData
dataset = LensData('prepared_v1')
for images, labels, row_ids in dataset.batches(
        'train', batch_size=64, epoch=0, balanced=True, augment=True):
    # images: float32 NCHW; labels: int64; row_ids locate manifest entries
    pass
```

Shuffle/sampling and augmentation are deterministic for a given seed and epoch.
Set `normalize=False` to access preserved original pixels. A future PyTorch
training loop can convert each contiguous NumPy batch with `torch.from_numpy`.

## Outputs

`prepared_v1/manifest.csv` records original paths, labels, source split, assigned
split, row indices and pixel hashes. Excluded duplicates remain in the raw cache
but are never returned by train/validation/test loaders. `normalization.json`
stores train-only statistics. `validation.json` records full-data checks and
counts. `positive_negative.png` and `preview_rows.json` show traceable training
examples using a fixed [0,1] display scale.

The preview revealed a cutout with a large flat region. `quality.py` flags images
where at least 10% of pixels in any channel exactly match its top-left corner
value, as a heuristic for padding or constant regions. The flags are for inspection;
they do not prove corruption and no images are removed automatically. Quality
flags and counts are saved beside the prepared data. A later masking experiment
must define its policy using training/validation data, not test performance.

`baseline_v1` contains validation metrics, a precision-recall plot, predictions,
and model coefficients in NPZ format. The baseline uses logistic regression on
channel means, standard deviations and annular means. It is a pipeline sanity
check, not the proposed CNN/Hough comparison. It fits preprocessing and balanced
class weights using training data only. Test labels are not used for model tuning
or baseline evaluation. A reported threshold is selected on validation data and
is not an independent performance estimate.

## Next experiments

1. Train a small CNN on the same split; select checkpoints using validation
   average precision. Compare rotation augmentation on/off.
2. Implement a foreground subtraction + edge/Hough baseline with validation-only
   parameter selection. At 64×64, inspect whether arcs are resolved well enough.
3. Compare geometry-assisted learning and rotation consistency with the CNN.
4. Freeze configurations, then evaluate once on the supplied test set, reporting
   average precision, ROC-AUC, recall at fixed false-positive rate and uncertainty.
   Noise/blur experiments here act on processed images, not raw detector data.

## Source

ML4SCI DeepLense Specific Test V (observational lens/non-lens images):
https://docs.google.com/document/d/10APh49fvayGoSftzO4fGXs2HP3uvYSzG-fSrq4xHL1w/edit

Archive: https://drive.google.com/file/d/1doUhVoq1-c9pamZVLpvjW1YRDMkKO1Q5/view

Use the dataset authors' citation and usage requirements when publishing results;
the evaluation instructions do not establish a separate dataset license.
