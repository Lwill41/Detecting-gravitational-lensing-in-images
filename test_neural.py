import unittest
import torch
from neural_models import SmallCNN,ConvAE,SymmetryAE,transform,inverse

class NeuralTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1);torch.set_num_threads(2)
    def test_transforms_and_shapes(self):
        x=torch.rand(2,3,64,64)
        for k in range(4):
            for f in [False,True]:
                self.assertTrue(torch.equal(x,inverse(transform(x,k,f),k,f)))
        self.assertEqual(SmallCNN()(x).shape,(2,))
        self.assertEqual(ConvAE()(x).shape,x.shape)
    def test_symmetry_contracts_and_gradient(self):
        x=torch.rand(2,3,64,64)
        model=SymmetryAE()
        with torch.no_grad():
            y=model(x)
            torch.testing.assert_close(model(transform(x,1,True)),transform(y,1,True),atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(model(1.7*x),1.7*y,atol=1e-6,rtol=1e-5)
            torch.testing.assert_close(model.shape_descriptor(x),model.shape_descriptor(transform(1.7*x,2,True)),atol=1e-6,rtol=1e-5)
        loss=model.training_loss(x,1,True);loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

if __name__=='__main__':unittest.main()
