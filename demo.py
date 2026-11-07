"""Inspect a prepared dataset and load batches without changing saved data."""
import argparse
import numpy as np
from data import LensData


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='prepared_v1')
    p.add_argument('--split',choices=['train','validation','test'],default='train')
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--batches',type=int,default=2)
    p.add_argument('--epoch',type=int,default=0)
    p.add_argument('--seed',type=int,default=429)
    p.add_argument('--balanced',action='store_true')
    p.add_argument('--augment',action='store_true')
    p.add_argument('--raw',action='store_true',help='Disable channel standardization')
    a=p.parse_args()
    if a.batches<1:
        p.error('--batches must be positive')
    if a.split!='train' and (a.balanced or a.augment):
        p.error('--balanced and --augment are training-only')
    d=LensData(a.data)
    print('DATASET: original cached pixels are unchanged')
    for split in ['train','validation','test']:
        labels=d.labels[d.indices(split)]
        print(f'  {split}: {len(labels):,} images | lenses={int(labels.sum()):,} | non-lenses={int((labels==0).sum()):,}')
    print('\nNORMALIZATION: statistics fitted on training pixels only')
    print('  mean:',d.mean.ravel())
    print('  std: ',d.std.ravel())
    print('  formula: (raw - mean) / std')
    print(f'\nBATCH SETTINGS: split={a.split}, balanced={a.balanced}, augment={a.augment}, normalize={not a.raw}, seed={a.seed}, epoch={a.epoch}')
    iterator=d.batches(a.split,batch_size=a.batch_size,epoch=a.epoch,seed=a.seed,
                       balanced=a.balanced,augment=a.augment,normalize=not a.raw)
    for number in range(a.batches):
        try:
            x,y,ids=next(iterator)
        except StopIteration:
            break
        print(f'\nBATCH {number+1}')
        print(f'  images: shape={x.shape}, dtype={x.dtype}, finite={bool(np.isfinite(x).all())}')
        print(f'  labels: shape={y.shape}, dtype={y.dtype}, lenses={int(y.sum())}, non-lenses={int((y==0).sum())}')
        print(f'  output range: [{x.min():.4f}, {x.max():.4f}]')
        print('  channel means:',x.mean(axis=(0,2,3)))
        print('  channel stds: ',x.std(axis=(0,2,3)))
        print('  first three examples:')
        for i in ids[:3]:
            r=d.records[int(i)]
            print(f"    row={i}, label={r['label']}, path={r['path']}")
        i=int(ids[0])
        raw=float(d.images[i,0,32,32])
        normalized=(raw-float(d.mean[0,0,0,0]))/float(d.std[0,0,0,0])
        print(f'  normalization example BEFORE augmentation: raw channel-1 pixel [32,32]={raw:.6f} -> {normalized:.6f}')
    print('\nDone. A small batch need not have zero mean, unit variance, or exactly equal class counts.')


if __name__=='__main__':
    main()
