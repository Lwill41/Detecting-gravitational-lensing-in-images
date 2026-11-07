"""Validated, memory-mapped DeepLense images and reproducible NumPy batches."""
import csv
import json
from pathlib import Path
import numpy as np

SHAPE = (3, 64, 64)


def stratified_split(records, seed=429, fraction=.2):
    """Group exact duplicates; retain official test and exclude matching training copies."""
    if not 0 < fraction < 1:
        raise ValueError('Validation fraction must lie between 0 and 1')
    labels_by_hash = {}
    for r in records:
        labels_by_hash.setdefault(r['sha256'], set()).add(int(r['label']))
    conflicts = [h for h, labels in labels_by_hash.items() if len(labels) > 1]
    if conflicts:
        raise ValueError(f'{len(conflicts)} identical pixel arrays have contradictory labels')
    test_hashes = {r['sha256'] for r in records if r['source_split'] == 'test'}
    seen = set()
    for r in records:
        key = (r['source_split'], r['sha256'])
        if r['source_split'] == 'train' and r['sha256'] in test_hashes:
            r['split'] = 'excluded_test_duplicate'
        elif key in seen:
            r['split'] = 'excluded_duplicate'
        else:
            r['split'] = r['source_split']
            seen.add(key)
    rng = np.random.default_rng(seed)
    for label in (0, 1):
        candidates = sorted((r for r in records if r['split'] == 'train' and r['label'] == label), key=lambda r: r['path'])
        if len(candidates) < 2:
            raise ValueError('Need at least two distinct training examples per class')
        n = min(len(candidates)-1, max(1, round(len(candidates) * fraction)))
        for index in rng.permutation(len(candidates))[:n]:
            candidates[int(index)]['split'] = 'validation'
    return records


class LensData:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.images = np.load(self.directory / 'images.npy', mmap_mode='r', allow_pickle=False)
        with (self.directory / 'manifest.csv').open(newline='', encoding='utf-8') as stream:
            self.records = list(csv.DictReader(stream))
        self.labels = np.array([int(r['label']) for r in self.records], dtype=np.int64)
        stats = json.loads((self.directory / 'normalization.json').read_text())
        self.mean = np.array(stats['mean'], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(stats['std'], dtype=np.float32).reshape(1, 3, 1, 1)
        if len(self.images) != len(self.records) or self.images.shape[1:] != SHAPE:
            raise ValueError('Cached images and manifest do not match')
        if not np.all(self.std > 0):
            raise ValueError('Normalization standard deviation must be positive')

    def indices(self, split):
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Expected train, validation or test')
        return np.array([i for i, r in enumerate(self.records) if r['split'] == split], dtype=np.int64)

    def batches(self, split='train', batch_size=64, epoch=0, seed=429,
                balanced=False, augment=False, normalize=True):
        if batch_size < 1:
            raise ValueError('Batch size must be positive')
        if split != 'train' and (balanced or augment):
            raise ValueError('Resampling and augmentation are training-only')
        indices = self.indices(split)
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        if split == 'train':
            if balanced:
                counts = np.bincount(self.labels[indices], minlength=2)
                if min(counts) == 0:
                    raise ValueError('Balanced sampling requires both classes')
                probabilities = 1 / counts[self.labels[indices]]
                probabilities /= probabilities.sum()
                indices = rng.choice(indices, len(indices), replace=True, p=probabilities)
            else:
                indices = rng.permutation(indices)
        for start in range(0, len(indices), batch_size):
            ids = indices[start:start+batch_size]
            x = np.array(self.images[ids], dtype=np.float32, copy=True)
            if augment:
                for i in range(len(x)):
                    x[i] = np.rot90(x[i], int(rng.integers(4)), axes=(-2, -1))
                    if rng.random() < .5:
                        x[i] = x[i, :, :, ::-1]
            if normalize:
                x = (x - self.mean) / self.std
            yield np.ascontiguousarray(x), self.labels[ids].copy(), ids


def channel_statistics(images, indices, batch_size=128):
    """Merge batch moments in float64; fit on explicitly supplied training rows only."""
    count, mean, m2 = 0, np.zeros(3), np.zeros(3)
    for start in range(0, len(indices), batch_size):
        x = np.asarray(images[indices[start:start+batch_size]], dtype=np.float64)
        n = x.shape[0] * x.shape[2] * x.shape[3]
        batch_mean = x.mean(axis=(0, 2, 3))
        batch_m2 = ((x - batch_mean[None, :, None, None]) ** 2).sum(axis=(0, 2, 3))
        delta = batch_mean - mean
        m2 += batch_m2 + delta**2 * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    if count == 0:
        raise ValueError('No training pixels')
    std = np.sqrt(m2 / count)
    if np.any(std < 1e-8):
        raise ValueError('Constant or near-constant training channel')
    return {'mean': mean.tolist(), 'std': std.tolist(), 'pixels_per_channel': count,
            'fit_split': 'train', 'transform': '(raw - channel_mean) / channel_std',
            'channel_order': 'Original archive order; filter names not independently verified'}
