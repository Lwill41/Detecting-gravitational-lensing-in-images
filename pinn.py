"""Synthetic PINN for the SIS lens-potential Poisson equation on an annulus.

This solves a known physical boundary-value problem, not observational lens
classification or unknown-mass inference. psi=b*r, laplacian(psi)=2*kappa=b/r.
The central singularity is excluded. Boundary potentials fix the gauge.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from neural_train import configure

B=.3
RMIN=.15


class Potential(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2,48),nn.Tanh(),nn.Linear(48,48),nn.Tanh(),
                               nn.Linear(48,48),nn.Tanh(),nn.Linear(48,1))
    def forward(self,x):return self.net(x)


def derivatives(function,x):
    psi=function(x)
    gradient=torch.autograd.grad(psi.sum(),x,create_graph=True)[0]
    laplacian=sum(torch.autograd.grad(gradient[:,j].sum(),x,create_graph=True,retain_graph=True)[0][:,j:j+1] for j in [0,1])
    return psi,gradient,laplacian


def points(n,generator):
    theta=2*torch.pi*torch.rand(n,generator=generator)
    r=(RMIN**2+(1-RMIN**2)*torch.rand(n,generator=generator)).sqrt()
    return torch.stack([r*theta.cos(),r*theta.sin()],1)


def run(output,steps=1000,seed=429):
    output.mkdir(parents=True,exist_ok=False)
    configure(seed)
    model=Potential()
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    generator=torch.Generator().manual_seed(seed)
    history=[]
    for step in range(steps):
        x=points(256,generator).requires_grad_(True)
        radius=x.square().sum(1,keepdim=True).sqrt()
        _,_,laplacian=derivatives(model,x)
        pde=(laplacian-B/radius).square().mean()
        angles=2*torch.pi*torch.rand(128,generator=generator)
        boundary_radius=torch.cat([torch.full((64,),RMIN),torch.ones(64)])
        boundary=torch.stack([boundary_radius*angles.cos(),boundary_radius*angles.sin()],1)
        bc=(model(boundary).squeeze(1)-B*boundary_radius).square().mean()
        loss=pde+10*bc
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if (step+1)%100==0:
            event={'step':step+1,'loss':float(loss.detach()),'pde_mse':float(pde.detach()),'boundary_mse':float(bc.detach())}
            history.append(event); print('PINN '+json.dumps(event),flush=True)
    holdout=points(4096,torch.Generator().manual_seed(seed+1)).requires_grad_(True)
    radius=holdout.square().sum(1,keepdim=True).sqrt()
    psi,gradient,laplacian=derivatives(model,holdout)
    report={'model':'SIS Poisson PINN (synthetic boundary-value benchmark)', 'seed':seed,'steps':steps,
        'equation':'laplacian(psi)=2*kappa=b/r; alpha=gradient(psi); beta=theta-alpha',
        'einstein_radius_dimensionless':B,'domain_radius':[RMIN,1.],
        'holdout_potential_mae':float((psi-B*radius).abs().mean().detach()),
        'holdout_deflection_rmse':float((gradient-B*holdout/radius).square().mean().sqrt().detach()),
        'holdout_poisson_residual_rmse':float((laplacian-B/radius).square().mean().sqrt().detach()),
        'holdout_points':4096,'observational_data_used':False,
        'known_inputs':['synthetic SIS mass/convergence field','inner/outer potential boundary values'],
        'limitations':'Known synthetic mass, no PSF/noise, no mass inference. This is a separate PDE task; AP/AUC comparisons to lens classifiers are not appropriate.'}
    (output/'metrics.json').write_text(json.dumps(report,indent=2))
    (output/'history.json').write_text(json.dumps(history,indent=2))
    torch.save(model.state_dict(),output/'potential.pt')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    coordinates=torch.linspace(-1,1,96)
    yy,xx=torch.meshgrid(coordinates,coordinates,indexing='ij')
    grid=torch.stack([xx.ravel(),yy.ravel()],1).requires_grad_(True)
    prediction=model(grid)
    alpha=torch.autograd.grad(prediction.sum(),grid)[0]
    r=grid.square().sum(1,keepdim=True).sqrt().clamp_min(1e-6)
    beta_true=grid-B*grid/r; beta_pred=grid-alpha
    def source(beta):return torch.exp(-((beta[:,0]-.08)**2+(beta[:,1]-.05)**2)/(2*.1**2))
    mask=((r[:,0]>=RMIN)&(r[:,0]<=1)).numpy().reshape(96,96)
    def plane(values):
        arr=values.detach().numpy().reshape(96,96)
        return np.ma.masked_where(~mask,arr)
    fig,axes=plt.subplots(2,3,figsize=(11,7))
    panels=[(B*r,'True potential'),(prediction,'PINN potential'),(prediction-B*r,'Potential error'),
            (source(beta_true),'True lensed source'),(source(beta_pred),'PINN-rendered lensed source'),
            (source(beta_pred)-source(beta_true),'Rendered intensity error')]
    for ax,(values,title) in zip(axes.ravel(),panels):
        artist=ax.imshow(plane(values),origin='lower',extent=[-1,1,-1,1],cmap='viridis')
        ax.set_title(title);fig.colorbar(artist,ax=ax,fraction=.046)
    fig.suptitle('Synthetic SIS physics benchmark: known mass and boundary values; singular core excluded')
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(output/'pinn_benchmark.png',dpi=140);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000)
    a=p.parse_args();run(a.output,a.steps)
