"""Train small reproducible CNN/AE prototypes; validation only, never score test."""
import argparse
import copy
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from data import LensData
from experiment_utils import metrics,save_results,provenance
from neural_models import SmallCNN,ConvAE,SymmetryAE,transform


def configure(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(4)
    # Use CPU for consistent behavior on this machine's older GPU and installed wheel.
    torch.use_deterministic_algorithms(True)
    return torch.device('cpu')


def evaluate(model,dataset,kind,batch_size=128):
    labels=[]; rows=[]; scores=[]; raw_scores=[]
    model.eval()
    with torch.no_grad():
        for x,y,ids in dataset.batches('validation',batch_size=batch_size,normalize=(kind=='cnn')):
            z=torch.from_numpy(x)
            if kind=='cnn':
                score=torch.sigmoid(model(z)).numpy()
                raw=score
            else:
                reconstruction=model(z)
                error=(reconstruction-z).square().mean((1,2,3))
                score=(error/(z.square().mean((1,2,3))+1e-8)).numpy()
                raw=error.numpy()
            labels.append(y); rows.append(ids); scores.append(score); raw_scores.append(raw)
    return tuple(np.concatenate(a) for a in [labels,scores,rows,raw_scores])


def reconstructions(model,dataset,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids=np.concatenate([dataset.indices('validation')[dataset.labels[dataset.indices('validation')]==label][:2] for label in [1,0]])
    x=torch.tensor(np.array(dataset.images[ids]),dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predicted=model(x).numpy()
    fig,axes=plt.subplots(4,3,figsize=(8,10))
    for j,row in enumerate(ids):
        axes[j,0].imshow(x[j,0],cmap='gray',vmin=0,vmax=1)
        axes[j,1].imshow(predicted[j,0],cmap='gray',vmin=0,vmax=1)
        axes[j,2].imshow(np.abs(predicted[j,0]-x[j,0].numpy()),cmap='magma')
        for ax in axes[j]: ax.axis('off')
        axes[j,0].set_title(f"{'Lens' if dataset.labels[row] else 'Non-lens'} row {row}")
        axes[j,1].set_title('Reconstruction (channel 1)')
        axes[j,2].set_title('Absolute residual')
    fig.tight_layout();fig.savefig(output/'reconstructions.png',dpi=140);plt.close(fig)


def run(data,output,kind,epochs=3,batch_size=64,seed=429):
    if epochs<1: raise ValueError('epochs must be positive')
    output.mkdir(parents=True,exist_ok=False)
    configure(seed)
    dataset=LensData(data)
    model=SmallCNN() if kind=='cnn' else SymmetryAE() if kind=='symmetry_ae' else ConvAE()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    history=[]; best=None; best_value=-np.inf if kind=='cnn' else np.inf
    started=time.monotonic()
    print(f'{kind}: CPU, {sum(p.numel() for p in model.parameters()):,} parameters; {epochs} epochs',flush=True)
    for epoch in range(epochs):
        model.train(); loss_sum=0.; seen=0
        for step,(x,y,ids) in enumerate(dataset.batches('train',batch_size=batch_size,epoch=epoch,seed=seed,
                    balanced=(kind=='cnn'),augment=True,normalize=(kind=='cnn'))):
            if kind!='cnn':
                x=x[y==0]  # Fit the normal (non-lens) population only.
                if not len(x): continue
            z=torch.from_numpy(x)
            optimizer.zero_grad(set_to_none=True)
            if kind=='cnn':
                loss=F.binary_cross_entropy_with_logits(model(z),torch.tensor(y,dtype=torch.float32))
            elif kind=='symmetry_ae':
                loss=model.training_loss(z,int(torch.randint(4,()).item()),bool(torch.rand(())<.5))
            else:
                input_image=(z+.03*torch.randn_like(z)).clamp(0,1) if kind=='denoising_ae' else z
                loss=F.mse_loss(model(input_image),z)
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            loss.backward();optimizer.step()
            loss_sum+=float(loss.detach())*len(z);seen+=len(z)
            if (step+1)%100==0: print(f'  epoch {epoch+1}, batch {step+1}, loss={loss_sum/seen:.6f}',flush=True)
        labels,score,rows,raw=evaluate(model,dataset,kind)
        result=metrics(labels,score)
        criterion=result['average_precision'] if kind=='cnn' else float(score[labels==0].mean())
        improvement=criterion>best_value if kind=='cnn' else criterion<best_value
        if improvement:
            best_value=criterion;best=copy.deepcopy(model.state_dict())
        event={'epoch':epoch+1,'train_loss':loss_sum/seen,'validation':result,
               'selection_value':criterion,'elapsed_seconds':round(time.monotonic()-started,1)}
        history.append(event)
        print(json.dumps(event),flush=True)
        (output/'history.json').write_text(json.dumps(history,indent=2))
    model.load_state_dict(best)
    torch.save({'kind':kind,'state_dict':best,'seed':seed,'epochs':epochs},output/'model.pt')
    labels,score,rows,raw=evaluate(model,dataset,kind)
    result=save_results(output,labels,score,rows,
       {'model':kind,'epochs':epochs,'seed':seed,'device':'cpu','prototype_run':True,
        'parameters':sum(p.numel() for p in model.parameters()),'train_count':len(dataset.indices('train')),
        'training_labels':'both classes, balanced sampling' if kind=='cnn' else 'non-lenses only',
        'checkpoint_selection':'maximum validation AP' if kind=='cnn' else 'minimum validation non-lens relative reconstruction error',
        'anomaly_score':None if kind=='cnn' else 'MSE / (mean squared input intensity + 1e-8); larger means more lens-like anomaly',
        'raw_reconstruction_mse_metrics':None if kind=='cnn' else metrics(labels,raw),
        'elapsed_seconds':round(time.monotonic()-started,1),**provenance(data),
        'limitation':'Short CPU prototype; validation used for selection; no independent test claim.'})
    if kind!='cnn': reconstructions(model,dataset,output)
    else:
        from inspect_cnn import run as inspect_cnn
        inspect_cnn(data,output)
    if kind=='symmetry_ae':
        x=torch.tensor(np.array(dataset.images[dataset.indices('validation')[:3]]),dtype=torch.float32)
        with torch.no_grad():
            original=model(x)
            geometry_error=float((model(transform(x,1,True))-transform(original,1,True)).abs().max())
            brightness_error=float((model(1.5*x)-1.5*original).abs().max())
            descriptor_error=float((model.shape_descriptor(transform(1.5*x,1,True))-model.shape_descriptor(x)).abs().max())
        (output/'symmetry_checks.json').write_text(json.dumps({'max_abs_D4_equivariance_error':geometry_error,
            'max_abs_positive_brightness_equivariance_error':brightness_error,
            'max_abs_shape_descriptor_invariance_error':descriptor_error,
            'scope':'Algebraic architectural properties on three examples; not proof of physical luminosity invariance.'},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('prepared_v1'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=['cnn','symmetry_ae','plain_ae','denoising_ae'],required=True)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--seed',type=int,default=429)
    a=p.parse_args();run(a.data,a.output,a.kind,a.epochs,a.batch_size,a.seed)
