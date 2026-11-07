"""Small CNN and autoencoders with explicit rotation/brightness contracts."""
import torch
from torch import nn
from torch.nn import functional as F


def transform(x,k,flip=False):
    x=torch.rot90(x,k,(-2,-1))
    return x.flip(-1) if flip else x


def inverse(x,k,flip=False):
    if flip:
        x=x.flip(-1)
    return torch.rot90(x,-k,(-2,-1))


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        # Retain coarse spatial structure instead of globally averaging faint arcs.
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(64*8*8,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,1))
    def forward(self,x):
        return self.head(self.features(x)).squeeze(1)


class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.ReLU(),
             nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.Conv2d(32,8,3,2,1),nn.ReLU())
        self.decoder=nn.Sequential(nn.ConvTranspose2d(8,32,4,2,1),nn.ReLU(),
             nn.ConvTranspose2d(32,16,4,2,1),nn.ReLU(),nn.ConvTranspose2d(16,3,4,2,1),nn.Softplus())
    def forward(self,x):
        return self.decoder(self.encoder(x))


class SymmetryAE(nn.Module):
    """D4-equivariant reconstruction and global positive-brightness equivariance.

    RMS is an observed nuisance amplitude, not intrinsic luminosity. Inference
    averages all eight transformed reconstructions. Training uses a cheaper
    stochastic equivariance penalty. The model is not scale/translation invariant.
    """
    def __init__(self):
        super().__init__()
        self.core=ConvAE()
    @staticmethod
    def separate(x):
        amplitude=x.square().mean((1,2,3),keepdim=True).sqrt().clamp_min(1e-6)
        return x/amplitude,amplitude
    def forward(self,x,symmetrize=True):
        shape,amplitude=self.separate(x)
        if not symmetrize:
            return self.core(shape)*amplitude
        reconstruction=sum(inverse(self.core(transform(shape,k,f)),k,f)
                           for k in range(4) for f in [False,True])/8
        return reconstruction*amplitude
    def shape_descriptor(self,x):
        shape,_=self.separate(x)
        return sum(self.core.encoder(transform(shape,k,f)).mean((-2,-1))
                   for k in range(4) for f in [False,True])/8
    def training_loss(self,x,k,flip):
        shape,_=self.separate(x)
        reconstructed=self.core(shape)
        transformed=self.core(transform(shape,k,flip))
        reconstruction=F.mse_loss(reconstructed,shape)
        equivariance=F.mse_loss(transformed,transform(reconstructed,k,flip))
        return reconstruction+.1*equivariance
