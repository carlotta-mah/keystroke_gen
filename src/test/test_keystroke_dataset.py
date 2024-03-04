import torch

import src.ks_dataset as KeystrDataset

def test_keystroke_dataset():
    dataset = KeystrDataset.KeystrokeDataset(data=None, labels=None)

    assert len(dataset) == 0

    # Test append method
    dataset.append_item(data_point=[[1, 2, 3]], label=[1])
    assert len(dataset) == 1

    dataset = KeystrDataset.KeystrokeDataset(data=[[1, 2, 3], [1, 2, 3]], labels=[1, 2])
    assert len(dataset) == 2

    # Test get item
    assert dataset[0] == ([1, 2, 3], 1)

