"""Controlled transformation checks on 64 validation examples, not accuracy tests."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from data import LensData
from neural_models import SmallCNN,ConvAE,SymmetryAE,transform


def run(data,root):
    torch.set_num_threads(4)
    dataset=LensData(data)
    validation=dataset.indices('validation')
    rng=np.random.default_rng(429)
    rows=np.concatenate([rng.choice(validation[dataset.labels[validation]==label],32,replace=False) for label in [1,0]])
    raw=torch.tensor(np.array(dataset.images[rows]),dtype=torch.float32)
    mean=torch.tensor(dataset.mean);std=torch.tensor(dataset.std)
    result={'rows':rows.tolist(),'sample_size':len(rows),'split':'validation',
            'scope':'Controlled pixel transformations; global scaling preserves channel ratios and is not intrinsic luminosity or realistic noise variation.',
            'models':{}}
    for name in ['cnn','symmetry_ae','plain_ae','denoising_ae']:
        model=SmallCNN() if name=='cnn' else SymmetryAE() if name=='symmetry_ae' else ConvAE()
        checkpoint=torch.load(root/name/'model.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        def score(x):
            if name=='cnn':return torch.sigmoid(model((x-mean)/std))
            return (model(x)-x).square().mean((1,2,3))/(x.square().mean((1,2,3))+1e-8)
        with torch.no_grad():
            reference=score(raw)
            changes={}
            for label,image in [('rotate90_reflect',transform(raw,1,True)),('brightness_times_1.5',1.5*raw),
                                ('both',transform(1.5*raw,1,True))]:
                delta=(score(image)-reference).abs()
                changes[label]={'mean_absolute_score_change':float(delta.mean()),'max_absolute_score_change':float(delta.max())}
        result['models'][name]={'score_units':'sigmoid score' if name=='cnn' else 'relative reconstruction error',**changes}
    (root/'symmetry_probe.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['models'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('prepared_v1'))
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();run(a.data,a.run)
