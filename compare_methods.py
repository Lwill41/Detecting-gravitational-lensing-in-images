"""Plot completed validation experiments; keep synthetic PINN metrics separate."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score,precision_recall_curve


def run(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['hough','cnn','symmetry_ae','plain_ae','denoising_ae']
    reports={name:json.loads((root/name/'metrics.json').read_text()) for name in names}
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    ap=[]; labels_reference=None
    for name in names:
        with np.load(root/name/'validation_predictions.npz',allow_pickle=False) as d:
            labels,score,rows=d['labels'],d['score'],d['rows']
        if labels_reference is None:
            labels_reference,rows_reference=labels,rows
        if not np.array_equal(labels,labels_reference) or not np.array_equal(rows,rows_reference):
            raise ValueError('Comparison requires identical validation rows and labels')
        p,r,_=precision_recall_curve(labels,score)
        value=average_precision_score(labels,score);ap.append(value)
        axes[0].plot(r,p,label=f'{name}: {value:.3f}')
    prevalence=labels_reference.mean()
    axes[0].axhline(prevalence,c='gray',ls='--',label=f'Prevalence: {prevalence:.3f}')
    axes[0].set(xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1),title='Precision-recall curves')
    axes[0].legend(fontsize=8)
    bars=axes[1].bar(names,ap)
    for bar,value in zip(bars,ap):axes[1].text(bar.get_x()+bar.get_width()/2,value+.015,f'{value:.3f}',ha='center')
    axes[1].axhline(prevalence,c='gray',ls='--')
    axes[1].set(ylabel='Average precision',ylim=(0,1),title='Same validation split; short initial runs')
    axes[1].tick_params(axis='x',rotation=25)
    fig.suptitle('Exploratory validation comparison — not independent test performance')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(root/'method_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name in names[1:]:
        history=json.loads((root/name/'history.json').read_text())
        axes[0].plot([h['epoch'] for h in history],[h['train_loss'] for h in history],marker='o',label=name)
        axes[1].plot([h['epoch'] for h in history],[h['validation']['average_precision'] for h in history],marker='o',label=name)
    axes[0].set(xlabel='Epoch',ylabel='Training objective (different scales)',title='Training progress',yscale='log')
    axes[1].set(xlabel='Epoch',ylabel='Validation average precision',title='Validation progress',ylim=(0,1))
    for ax in axes:ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=150);plt.close(fig)
    # Inspect whether AE scores track observed image brightness, a possible confound.
    from data import LensData
    dataset=LensData(Path(__file__).resolve().parent/'prepared_v1')
    rms=np.array([np.sqrt(np.mean(np.square(dataset.images[i]),dtype=np.float64)) for i in rows_reference])
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,name in zip(axes,['symmetry_ae','plain_ae','denoising_ae']):
        with np.load(root/name/'validation_predictions.npz',allow_pickle=False) as d:
            score=d['score'];labels=d['labels']
        for label,color in [(0,'gray'),(1,'tab:orange')]:
            mask=labels==label
            ax.scatter(rms[mask],score[mask],s=5,alpha=.3,color=color,label='Lens' if label else 'Non-lens')
        ax.set(xlabel='Observed raw intensity RMS',ylabel='Relative reconstruction error',title=name,yscale='log')
        ax.legend(fontsize=8)
    fig.suptitle('Brightness diagnostic: processed intensities, not physical luminosities')
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(root/'ae_brightness_diagnostic.png',dpi=150);plt.close(fig)
    lines=['# Initial method experiments','',
        'All classification/anomaly results below use the same 6,081-image validation split. No supplied test images were scored.',
        f"CNN: {reports['cnn']['epochs']} epochs; symmetry/plain/denoising AE epochs: "+'/'.join(str(reports[n]['epochs']) for n in names[2:])+'. These are exploratory CPU prototypes and are not a claim of convergence or final generalization.','',
        '| Method | Average precision | ROC-AUC | Recall at <=1% FPR |',
        '|---|---:|---:|---:|']
    for name in names:
        m=reports[name]['validation']
        lines.append(f"| {name} | {m['average_precision']:.4f} | {m['roc_auc']:.4f} | {m['recall_at_fpr_le_0.01']:.4f} |")
    lines+=['','The original image-statistics baseline had validation AP 0.5138 and ROC-AUC 0.9482.',
        'Higher AE reconstruction error was fixed as the anomaly direction; scores were not inverted to improve results.',
        'Architecture, fitting objective and inference cost differ, so matched-budget and component ablations are still required.','',
        '## Symmetry checks','','```json',json.dumps(json.loads((root/'symmetry_ae'/'symmetry_checks.json').read_text()),indent=2),'```','',
        '## Synthetic PINN — separate task','','```json',json.dumps(json.loads((root/'pinn'/'metrics.json').read_text()),indent=2),'```','',
        'The PINN uses known synthetic mass and boundary values, not DeepLense images. Its PDE errors cannot be ranked against detection AP/AUC.']
    (root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Saved method comparison, learning curves and RESULTS.md',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args();run(a.run)
