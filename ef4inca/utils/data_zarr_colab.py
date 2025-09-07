import os
import zarr
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class ZarrINCADataset(Dataset):
    def __init__(self, zarr_path):
        self.store = zarr.open(zarr_path, mode='r')
        self.past = self.store['past']
        self.future = self.store['future']
        self.names = self.store.get('names', None)
        self.N = self.past.shape[0]
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        x = self.past[idx]   # (in_len, H, W, C)
        y = self.future[idx] # (out_len, H, W, 1)
        sample = {
            "sample_past": torch.from_numpy(x.astype('float32')),
            "sample_future": torch.from_numpy(y.astype('float32')),
            "name": "" if self.names is None else str(self.names[idx])
        }
        return sample

class ZarrINCADataModule(pl.LightningDataModule):
    def __init__(self, params):
        super().__init__()
        # expects keys: train_path, val_path, test_path, BATCH_SIZE, NUM_WORKERS
        self.params = params
        self.train_path = params.get('train_path')
        self.val_path   = params.get('val_path')
        self.test_path  = params.get('test_path')
        self.bs = params.get('BATCH_SIZE', 1)
        self.nw = params.get('NUM_WORKERS', 2)

    def setup(self, stage=None):
        if stage in (None, "fit", "train"):
            self.train_dataset = ZarrINCADataset(self.train_path)
            self.val_dataset   = ZarrINCADataset(self.val_path)
        if stage in (None, "test"):
            self.test_dataset  = ZarrINCADataset(self.test_path)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.bs, shuffle=True,  num_workers=self.nw)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,   batch_size=self.bs, shuffle=False, num_workers=self.nw)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,  batch_size=1,       shuffle=False, num_workers=self.nw)

    @property
    def num_train_samples(self):
        return len(self.train_dataset)
    @property
    def num_val_samples(self):
        return len(self.val_dataset)
    @property
    def num_test_samples(self):
        return len(self.test_dataset)