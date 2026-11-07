import unittest
import numpy as np
from hough import extract,XX,YY,RADII,FEATURE_NAMES

class HoughTests(unittest.TestCase):
    def test_ring_and_blank(self):
        r=np.hypot(XX-32,YY-32)
        ring=np.exp(-.5*((r-12)/.65)**2)+2*np.exp(-r**2/8)
        f,debug=extract(np.repeat(ring[None],3,axis=0).astype(np.float32),True)
        self.assertEqual(len(f),len(FEATURE_NAMES))
        self.assertLessEqual(abs(debug['radius']-12),2)
        self.assertLessEqual(np.hypot(debug['cx']-32,debug['cy']-32),2)
        blank=extract(np.zeros((3,64,64),dtype=np.float32))
        self.assertTrue(np.isfinite(blank).all())
        self.assertGreater(f[len(RADII)],blank[len(RADII)])

if __name__=='__main__': unittest.main()
