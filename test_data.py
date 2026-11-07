import unittest
import csv
import json
from pathlib import Path
import tempfile
import numpy as np
from data import LensData, stratified_split, channel_statistics
from baseline import features


class DataTests(unittest.TestCase):
    def test_duplicate_groups_and_official_test(self):
        records = [{'row':i,'path':str(i),'source_split':'train','label':i%2,'sha256':str(i)} for i in range(20)]
        records += [dict(records[0],row=20,path='copy'), dict(records[1],row=21,path='heldout',source_split='test')]
        stratified_split(records)
        self.assertIn(records[0]['split'], ('train','validation'))
        self.assertEqual(records[20]['split'],'excluded_duplicate')
        self.assertEqual(records[1]['split'],'excluded_test_duplicate')
        self.assertEqual(records[21]['split'],'test')
        sets=[{r['sha256'] for r in records if r['split']==s} for s in ['train','validation','test']]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])

    def test_conflicting_duplicate_labels_fail(self):
        with self.assertRaises(ValueError):
            stratified_split([{'sha256':'same','label':0}, {'sha256':'same','label':1}])

    def test_training_only_normalization(self):
        x=np.random.default_rng(0).random((8,3,64,64),dtype=np.float32)
        stats=channel_statistics(x,np.array([0,1,2]),batch_size=2)
        np.testing.assert_allclose(stats['mean'],x[:3].mean(axis=(0,2,3),dtype=np.float64))
        np.testing.assert_allclose(stats['std'],x[:3].std(axis=(0,2,3),dtype=np.float64))
        x[3:]=1000
        self.assertEqual(stats,channel_statistics(x,np.array([0,1,2]),batch_size=2))

    def test_features(self):
        x=np.ones((2,3,64,64),dtype=np.float32)
        f=features(x)
        self.assertEqual(f.shape,(2,18))
        np.testing.assert_array_equal(f[:,:3],1)
        np.testing.assert_array_equal(f[:,3:6],0)

    def test_loader_augmentation_and_holdout(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            raw=np.random.default_rng(0).random((20,3,64,64),dtype=np.float32)
            np.save(root/'images.npy',raw,allow_pickle=False)
            with (root/'manifest.csv').open('w',newline='') as stream:
                writer=csv.writer(stream)
                writer.writerow(['row','label','split'])
                writer.writerows((i,i%2,'train' if i<10 else 'validation' if i<15 else 'test') for i in range(20))
            (root/'normalization.json').write_text(json.dumps({'mean':[.5]*3,'std':[.2]*3}))
            dataset=LensData(root)
            a=next(dataset.batches(batch_size=6,balanced=True,augment=True,epoch=3))
            b=next(dataset.batches(batch_size=6,balanced=True,augment=True,epoch=3))
            for left,right in zip(a,b):
                np.testing.assert_array_equal(left,right)
            np.testing.assert_array_equal(dataset.images,raw)
            x,y,ids=next(dataset.batches('validation',batch_size=5))
            np.testing.assert_array_equal(ids,np.arange(10,15))
            np.testing.assert_allclose(x,(raw[10:15]-.5)/.2)
            with self.assertRaises(ValueError):
                next(dataset.batches('test',augment=True))
            with self.assertRaises(ValueError):
                next(dataset.batches('validation',balanced=True))
            # Release the read-only map before Windows temporary-directory cleanup.
            dataset.images._mmap.close()


if __name__=='__main__':
    unittest.main()
