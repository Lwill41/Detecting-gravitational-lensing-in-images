"""Small image-statistics classifier; fit train, report validation, leave test unused."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data import LensData


def features(x):
    yy, xx = np.mgrid[:64, :64]
    radius = np.sqrt((xx-31.5)**2 + (yy-31.5)**2)
    columns = [x.mean(axis=(2,3)), x.std(axis=(2,3))]
    for lo, hi in [(0, 6), (6, 12), (12, 20), (20, 46)]:
        columns.append(x[:, :, (radius >= lo) & (radius < hi)].mean(axis=2))
    return np.concatenate(columns, axis=1)


def run(data, output):
    output.mkdir(parents=True, exist_ok=False)
    dataset = LensData(data)
    def collect(split):
        fs, ys = [], []
        for x, y, _ in dataset.batches(split, batch_size=128, normalize=False):
            fs.append(features(x))
            ys.append(y)
        return np.concatenate(fs), np.concatenate(ys)
    train_x, train_y = collect('train')
    val_x, val_y = collect('validation')
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000, random_state=429))
    model.fit(train_x, train_y)
    p = model.predict_proba(val_x)[:,1]
    report = {'model': 'Logistic regression on 18 image-statistics features',
              'features': '3 channel means + 3 standard deviations + 4 annular means per channel',
              'train_count':len(train_y), 'validation_count':len(val_y),
              'validation_positive_count':int(val_y.sum()),
              'validation_prevalence':float(val_y.mean()),
              'validation_average_precision':float(average_precision_score(val_y,p)),
              'validation_roc_auc':float(roc_auc_score(val_y,p)),
              'test_evaluated':False,
              'note':'Sanity baseline; neither CNN nor Hough method. Balanced loss weights apply to training only.'}
    fpr, tpr, thresholds = roc_curve(val_y,p)
    eligible = np.flatnonzero(fpr <= .01)
    best = eligible[np.argmax(tpr[eligible])]
    report.update(validation_recall_at_fpr_le_1pct=float(tpr[best]), validation_fpr=float(fpr[best]),
                  validation_threshold=float(thresholds[best]) if np.isfinite(thresholds[best]) else None)
    scaler, classifier = model.steps[0][1], model.steps[1][1]
    np.savez(output/'model.npz', mean=scaler.mean_, scale=scaler.scale_,
             coefficient=classifier.coef_, intercept=classifier.intercept_)
    np.savez(output/'validation_predictions.npz', labels=val_y, probability=p)
    (output/'metrics.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    precision, recall, _ = precision_recall_curve(val_y,p)
    fig, ax = plt.subplots(figsize=(6,5))
    ax.plot(recall,precision,label=f"Baseline AP={report['validation_average_precision']:.3f}")
    ax.axhline(val_y.mean(), color='gray', linestyle='--',label='Positive prevalence')
    ax.set(xlabel='Recall',ylabel='Precision',title='Validation only: image-statistics baseline',xlim=(0,1),ylim=(0,1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/'precision_recall.png',dpi=150)
    plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    run(args.data,args.output)
