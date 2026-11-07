"""Flag large regions exactly equal to a corner value; do not auto-remove images."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
from data import LensData


def audit(data, threshold=.1):
    dataset=LensData(data)
    flags=[]
    for start in range(0,len(dataset.images),128):
        x=np.asarray(dataset.images[start:start+128])
        fractions=(x==x[:,:,0:1,0:1]).mean(axis=(2,3))
        for offset in np.flatnonzero(fractions.max(axis=1)>=threshold):
            row=start+int(offset)
            record=dataset.records[row]
            flags.append({'row':row,'path':record['path'],'label':record['label'],'split':record['split'],
                          'max_corner_value_fraction':float(fractions[offset].max())})
    report={'flag_rule':f'At least {threshold:.0%} of pixels in any channel exactly equal its top-left corner',
            'flagged_images':len(flags),'total_images':len(dataset.images),
            'by_split_and_label':dict(Counter(r['split']+' label='+r['label'] for r in flags)),
            'interpretation':'Heuristic for padding/constant regions, not proof of corrupt data. No images removed. Inspect before designing masks; do not tune exclusions using test outcomes.'}
    with (Path(data)/'quality_flags.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['row','path','label','split','max_corner_value_fraction'])
        writer.writeheader()
        writer.writerows(flags)
    (Path(data)/'quality_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    args=p.parse_args()
    audit(args.data)
