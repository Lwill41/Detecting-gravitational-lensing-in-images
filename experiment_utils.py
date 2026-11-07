"""Shared validation metrics. The supplied test set is never scored here."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve


def metrics(labels, scores):
    fpr,tpr,thresholds=roc_curve(labels,scores)
    result={'average_precision':float(average_precision_score(labels,scores)),
            'roc_auc':float(roc_auc_score(labels,scores)), 'count':len(labels),
            'positives':int(np.sum(labels)), 'prevalence':float(np.mean(labels))}
    for target in [.01,.001]:
        good=np.flatnonzero(fpr<=target)
        index=good[np.argmax(tpr[good])]
        result[f'recall_at_fpr_le_{target}']=float(tpr[index])
    return result


def save_results(output, labels, scores, rows, details, extra_scores=None):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    result={**details,'validation':metrics(labels,scores),'test_evaluated':False}
    (output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    np.savez(output/'validation_predictions.npz',labels=labels,score=scores,rows=rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5))
    for name, values in [(details['model'],scores),*(extra_scores or {}).items()]:
        p,r,_=precision_recall_curve(labels,values)
        ax.plot(r,p,label=f'{name}: AP={average_precision_score(labels,values):.3f}')
    ax.axhline(np.mean(labels),color='gray',ls='--',label='Validation prevalence')
    ax.set(xlabel='Recall',ylabel='Precision',title='Validation only',xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'precision_recall.png',dpi=140)
    plt.close(fig)
    print(json.dumps(result,indent=2),flush=True)
    return result


def provenance(data):
    p=Path(data)
    return {name+'_sha256':hashlib.sha256((p/name).read_bytes()).hexdigest()
            for name in ['manifest.csv','normalization.json']}
