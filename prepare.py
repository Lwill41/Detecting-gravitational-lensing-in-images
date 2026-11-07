"""Validate all archive images, cache raw pixels, split, normalize and visualize."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import time
import zipfile
import numpy as np
from data import SHAPE, LensData, channel_statistics, stratified_split

FOLDERS = {'train_lenses': ('train', 1), 'train_nonlenses': ('train', 0),
           'test_lenses': ('test', 1), 'test_nonlenses': ('test', 0)}


def preview(dataset, directory, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    chosen = []
    for label in (1, 0):
        ids = [i for i in dataset.indices('train') if dataset.labels[i] == label]
        chosen.extend(rng.choice(ids, size=min(4, len(ids)), replace=False).tolist())
    fig, axes = plt.subplots(len(chosen), 3, figsize=(8, len(chosen)*2.2), squeeze=False)
    for row, i in enumerate(chosen):
        for channel in range(3):
            axes[row, channel].imshow(dataset.images[i, channel], cmap='gray', vmin=0, vmax=1)
            axes[row, channel].set_title(f"{'Lens' if dataset.labels[i] else 'Non-lens'} · row {i} · channel {channel+1}", fontsize=9)
            axes[row, channel].axis('off')
    fig.suptitle('DeepLense training examples · raw archive values · fixed [0,1] display')
    fig.tight_layout(rect=(0,0,1,.98))
    fig.savefig(directory / 'positive_negative.png', dpi=140)
    plt.close(fig)
    (directory / 'preview_rows.json').write_text(json.dumps([dataset.records[i] for i in chosen], indent=2))


def prepare(archive_path, output, seed=429, validation_fraction=.2):
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'started', 'seed': seed, 'archive': str(archive_path.resolve())}
    started = time.monotonic()
    try:
        with archive_path.open('rb') as stream:
            report['archive_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
        with zipfile.ZipFile(archive_path) as archive:
            entries, ignored = [], 0
            for info in archive.infolist():
                path = PurePosixPath(info.filename)
                if info.is_dir() or '__MACOSX' in path.parts or path.name.startswith('._'):
                    ignored += 1
                    continue
                if len(path.parts) == 2 and path.parts[0] in FOLDERS and path.suffix == '.npy':
                    entries.append(info)
                else:
                    raise ValueError(f'Unexpected archive entry: {info.filename}')
            entries.sort(key=lambda info: info.filename)
            if len({i.filename for i in entries}) != len(entries):
                raise ValueError('Duplicate archive member paths')
            if not entries:
                raise ValueError('No recognized dataset arrays')
            images = np.lib.format.open_memmap(output / 'images.npy', mode='w+', dtype=np.float32,
                                              shape=(len(entries), *SHAPE))
            records, minimum, maximum = [], np.full(3, np.inf), np.full(3, -np.inf)
            for index, info in enumerate(entries):
                x = np.load(io.BytesIO(archive.read(info)), allow_pickle=False)
                if x.shape != SHAPE or x.dtype != np.float32 or not np.isfinite(x).all():
                    raise ValueError(f'Bad array {info.filename}: {x.shape}, {x.dtype}')
                if x.min() < -1e-6 or x.max() > 1+1e-6:
                    raise ValueError(f'Unexpected values outside [0,1]: {info.filename}')
                images[index] = x
                minimum = np.minimum(minimum, x.min(axis=(1, 2)))
                maximum = np.maximum(maximum, x.max(axis=(1, 2)))
                folder = PurePosixPath(info.filename).parts[0]
                source_split, label = FOLDERS[folder]
                records.append({'row': index, 'path': info.filename, 'source_split': source_split,
                                'label': label, 'sha256': hashlib.sha256(x.tobytes(order='C')).hexdigest()})
                if (index+1) % 5000 == 0:
                    print(f'Validated {index+1}/{len(entries)} images', flush=True)
            images.flush()
        stratified_split(records, seed, validation_fraction)
        train = np.array([r['row'] for r in records if r['split'] == 'train'])
        stats = channel_statistics(images, train)
        (output / 'normalization.json').write_text(json.dumps(stats, indent=2))
        with (output / 'manifest.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        del images
        dataset = LensData(output)
        split_counts = {split: dict(Counter(str(r['label']) for r in records if r['split'] == split))
                        for split in ('train', 'validation', 'test')}
        hashes = {split: {r['sha256'] for r in records if r['split'] == split} for split in split_counts}
        if any(hashes[a] & hashes[b] for a, b in [('train','validation'), ('train','test'), ('validation','test')]):
            raise ValueError('Exact pixel duplicate leakage across splits')
        x, y, ids = next(dataset.batches(batch_size=32, balanced=True, augment=True))
        if x.shape != (32, *SHAPE) or not np.isfinite(x).all():
            raise ValueError('Batch smoke check failed')
        preview(dataset, output, seed)
        report.update(status='passed', total_images=len(records), all_shapes=[3,64,64],
                      all_dtype='float32', all_finite=True, channel_min=minimum.tolist(),
                      channel_max=maximum.tolist(), split_counts=split_counts,
                      original_counts=dict(Counter(r['path'].split('/')[0] for r in records)),
                      excluded_counts=dict(Counter(r['split'] for r in records if r['split'].startswith('excluded'))),
                      ignored_metadata_or_directories=ignored,
                      exact_pixel_leakage=False, batch_shape=list(x.shape),
                      normalization=stats, elapsed_seconds=round(time.monotonic()-started, 1),
                      limitation='No coordinates or object catalog supplied: same-sky-object and near-duplicate leakage cannot be ruled out by exact pixel hashes.')
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (output / 'validation.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New cache directory; cannot exist')
    parser.add_argument('--seed', type=int, default=429)
    parser.add_argument('--validation-fraction', type=float, default=.2)
    a = parser.parse_args()
    prepare(a.archive, a.output, a.seed, a.validation_fraction)
