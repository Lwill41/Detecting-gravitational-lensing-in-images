"""Qualitative Grad-CAM examples; coarse sensitivity maps are not arc masks."""
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from data import LensData
from neural_models import SmallCNN


def run(data,run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    torch.set_num_threads(2)
    dataset=LensData(data)
    checkpoint=torch.load(run/'model.pt',map_location='cpu',weights_only=True)
    model=SmallCNN();model.load_state_dict(checkpoint['state_dict']);model.eval()
    with np.load(run/'validation_predictions.npz',allow_pickle=False) as pred:
        labels,score,rows=pred['labels'],pred['score'],pred['rows']
    chosen=[]
    for label in [1,0]:
        ids=np.flatnonzero(labels==label)
        chosen.extend([int(ids[np.argmax(score[ids])]),int(ids[np.argmin(score[ids])])])
    fig,axes=plt.subplots(4,2,figsize=(7,11))
    for j,i in enumerate(chosen):
        raw=np.array(dataset.images[rows[i]])
        z=torch.tensor((raw[None]-dataset.mean)/dataset.std)
        activation=model.features(z)
        activation.retain_grad()
        logit=model.head(activation).squeeze()
        model.zero_grad(set_to_none=True);logit.backward()
        cam=torch.relu((activation.grad.mean((-2,-1),keepdim=True)*activation).sum(1,keepdim=True))
        cam=F.interpolate(cam,size=(64,64),mode='bilinear',align_corners=False)[0,0].detach().numpy()
        cam=cam/max(float(cam.max()),1e-8)
        probability=float(torch.sigmoid(logit.detach()))
        if not np.isclose(probability,score[i],atol=1e-5):raise ValueError('Checkpoint and saved predictions do not agree')
        for ax in axes[j]:
            ax.imshow(raw.mean(0),cmap='gray',vmin=0,vmax=1);ax.axis('off')
        axes[j,0].set_title(f"{'Lens' if labels[i] else 'Non-lens'} row {rows[i]}\nscore={score[i]:.3f}",fontsize=10)
        axes[j,1].imshow(cam,cmap='inferno',alpha=.45,vmin=0,vmax=1)
        axes[j,1].set_title('Positive-logit Grad-CAM\n(coarse 8x8 map)',fontsize=10)
    fig.suptitle('Validation extremes per class: qualitative model sensitivity, not segmentation')
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(run/'gradcam_examples.png',dpi=140);plt.close(fig)
    print('Saved Grad-CAM examples; checkpoint predictions agree with saved scores.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('prepared_v1'))
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();run(a.data,a.run)
