"""Circular Hough arc baseline: smooth foreground subtraction, Canny, circle votes.

Geometry is an approximation: arcs need not be circular; stars, spirals and galaxy
isophotes can trigger detections. A detected circle is not lens confirmation.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, sobel
from skimage.feature import canny
from skimage.transform import hough_circle
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from data import LensData
from experiment_utils import save_results, metrics, provenance

RADII=np.arange(4,22,2)
YY,XX=np.mgrid[:64,:64]
FEATURE_NAMES=[f'normalized_vote_radius_{r}' for r in RADII]+['best_vote','radius','center_offset','angular_coverage','ring_support','radial_gradient_alignment','edge_density']


def extract(image, return_debug=False):
    if image.shape!=(3,64,64) or not np.isfinite(image).all():
        raise ValueError('Expected finite three-channel 64x64 image')
    gray=image.mean(axis=0)
    smooth=gaussian_filter(gray,2)
    cy,cx=np.unravel_index(np.argmax(smooth[26:38,26:38]),(12,12))
    cy,cx=cy+26,cx+26
    distance=np.hypot(XX-cx,YY-cy)
    bins=np.floor(distance).astype(int)
    residuals=[]
    for plane in image:
        radial=np.bincount(bins.ravel(),weights=plane.ravel())/np.maximum(np.bincount(bins.ravel()),1)
        foreground=gaussian_filter1d(radial,3)[bins]
        residual=plane-foreground
        sky=residual[distance>24]
        scale=max(float(np.median(np.abs(sky-np.median(sky)))*1.4826),1e-4)
        residuals.append(residual/scale)
    residual=np.maximum(np.max(residuals,axis=0),0)
    hi=max(float(np.percentile(residual,99.5)),1e-6)
    residual=np.clip(residual/hi,0,1)
    mask=(distance>3)&(distance<25)&(XX>3)&(XX<60)&(YY>3)&(YY<60)
    edges=canny(residual,sigma=1,low_threshold=.70,high_threshold=.90,use_quantiles=True,mask=mask)
    space=hough_circle(edges,RADII,normalize=True)
    centers=np.hypot(XX-cx,YY-cy)<=6
    restricted=np.where(centers[None],space,-1)
    ri,by,bx=np.unravel_index(np.argmax(restricted),restricted.shape)
    votes=restricted[:,centers].max(axis=1)
    radius=int(RADII[ri])
    radius_map=np.hypot(XX-bx,YY-by)
    ring=np.abs(radius_map-radius)<=1.5
    selected=edges&ring
    angle=(np.arctan2(YY-by,XX-bx)+np.pi)/(2*np.pi)
    sectors=np.minimum((angle*24).astype(int),23)
    coverage=len(np.unique(sectors[selected]))/24 if selected.any() else 0.
    gy,gx=sobel(residual,axis=0),sobel(residual,axis=1)
    alignment=np.abs(gx*(XX-bx)+gy*(YY-by))/(np.hypot(gx,gy)*np.maximum(radius_map,1)+1e-6)
    features=np.array([*votes,float(votes.max()),radius,np.hypot(bx-cx,by-cy),coverage,
                       selected.sum()/max(int(ring.sum()),1),alignment[selected].mean() if selected.any() else 0,
                       edges.sum()/max(int(mask.sum()),1)],dtype=np.float32)
    if return_debug:
        return features,{'gray':gray,'residual':residual,'edges':edges,'cx':bx,'cy':by,'radius':radius,'foreground_center':(cx,cy)}
    return features


def gallery(dataset,rows,labels,score,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=[int(i) for label in [1,0] for i in np.flatnonzero(labels==label)[np.argsort(score[labels==label])[-2:]]]
    fig,axes=plt.subplots(4,4,figsize=(11,11))
    for r,i in enumerate(chosen):
        _,debug=extract(np.array(dataset.images[rows[i]]),True)
        for c,key in enumerate(['gray','residual','edges','gray']):
            axes[r,c].imshow(debug[key],cmap='gray')
            axes[r,c].axis('off')
        axes[r,0].set_title(f"{'Lens' if labels[i] else 'Non-lens'} row {rows[i]}")
        axes[r,1].set_title('Foreground residual')
        axes[r,2].set_title('Canny edges')
        axes[r,3].add_patch(plt.Circle((debug['cx'],debug['cy']),debug['radius'],fill=False,color='cyan'))
        axes[r,3].set_title(f'Circle candidate; score={score[i]:.2f}')
    fig.suptitle('Highest-scoring validation examples: two per label (includes false positives)')
    fig.tight_layout(rect=(0,0,1,.96))
    fig.savefig(output/'hough_gallery.png',dpi=140)
    plt.close(fig)


def run(data,output):
    output.mkdir(parents=True,exist_ok=False)
    dataset=LensData(data)
    train=dataset.indices('train'); validation=dataset.indices('validation')
    rows=np.concatenate([train,validation])
    started=time.monotonic()
    features=[]
    for i,row in enumerate(rows):
        features.append(extract(dataset.images[row]))
        if (i+1)%1000==0:
            print(f'Hough features: {i+1}/{len(rows)} ({time.monotonic()-started:.0f}s)',flush=True)
    features=np.stack(features)
    np.savez(output/'features.npz',features=features,rows=rows,names=np.array(FEATURE_NAMES))
    model=make_pipeline(StandardScaler(),LogisticRegression(class_weight='balanced',max_iter=1500,random_state=429))
    model.fit(features[:len(train)],dataset.labels[train])
    probability=model.predict_proba(features[len(train):])[:,1]
    raw_score=features[len(train):,len(RADII)]
    result=save_results(output,dataset.labels[validation],probability,validation,
       {'model':'Hough-feature logistic regression','feature_names':FEATURE_NAMES,'train_count':len(train),
        'radii_pixels':RADII.tolist(),'center_search_radius_pixels':6,'canny_sigma':1,
        'canny_quantile_thresholds':[.7,.9],'foreground_radial_smoothing_sigma':3,
        'raw_hough_validation':metrics(dataset.labels[validation],raw_score),
        'elapsed_seconds':round(time.monotonic()-started,1),**provenance(data),
        'limitations':'Fixed circular approximation, approximate central galaxy, processed intensities, padding can produce false circles.'},
        {'Raw maximum Hough vote':raw_score})
    scaler,clf=model.steps[0][1],model.steps[1][1]
    np.savez(output/'classifier.npz',mean=scaler.mean_,scale=scaler.scale_,coefficient=clf.coef_,intercept=clf.intercept_)
    gallery(dataset,validation,dataset.labels[validation],probability,output)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('prepared_v1'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); run(a.data,a.output)
