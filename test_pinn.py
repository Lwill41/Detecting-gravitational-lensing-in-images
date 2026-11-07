import unittest
import torch
from pinn import derivatives,B

class PinnTests(unittest.TestCase):
    def test_analytic_poisson_and_deflection(self):
        x=torch.tensor([[.3,.4],[.7,-.2],[-.5,.1]],dtype=torch.float64,requires_grad=True)
        def analytic(z):return B*z.square().sum(1,keepdim=True).sqrt()
        psi,alpha,lap=derivatives(analytic,x)
        r=x.square().sum(1,keepdim=True).sqrt()
        torch.testing.assert_close(alpha,B*x/r)
        torch.testing.assert_close(lap,B/r)

if __name__=='__main__':unittest.main()
