import os
import zarr
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F

def _to_float32(x):
    return torch.from_numpy(x.astype('float32'))

def _pad_to_multiple_lastHW(x_torch: torch.Tensor, mult_h=12, mult_w=12):
    """
    x_torch: (T, H, W, C)
    Pads H and W (right/bottom) so they become multiples of mult_h/mult_w.
    """
    T, H, W, C = x_torch.shape
    pad_h = (mult_h - (H % mult_h)) % mult_h
    pad_w = (mult_w - (W % mult_w)) % mult_w
    if pad_h == 0 and pad_w == 0:
        return x_torch
    # pad tuple is (C_left, C_right, W_left, W_right, H_left, H_right) for last 3 dims
    return F.pad(x_torch, (0, 0, 0, pad_w, 0, pad_h))
    
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
        x = _to_float32(self.past[idx])     # (T_in, H, W, C)
        y = _to_float32(self.future[idx])   # (T_out, H, W, 1)

        # >>> Critical: pad BOTH H and W to multiples of 12
        x = _pad_to_multiple_lastHW(x, mult_h=12, mult_w=12)
        y = _pad_to_multiple_lastHW(y, mult_h=12, mult_w=12)

        name = "" if self.names is None else str(self.names[idx])
        return {"sample_past": x, "sample_future": y, "name": name}

        # sample = {
        #     "sample_past": x,
        #     "sample_future": y,
        #     "name": "" if self.names is None else str(self.names[idx])
        # }
        # return sample

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