"""Paired HF/LF data from OpenNeuro ds006557 with the paper's 5-fold subject splits.

Expected layout (see README, "Data"):

    <data_root>/ds006557_HFC_T1/sub-HYPE00/{HF_synthstrip,LF_synthstrip}.nii.gz
    <data_root>/ds006557_HFC_T2/sub-HYPE00/{HF_synthstrip,LF_synthstrip}.nii.gz
    ...
HF and LF of a subject are co-registered on the same grid.
"""
from pathlib import Path
from typing import Dict, List

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

from h2lo.model import make_coord_grid, normalize_volume

CONTRAST_DIR = {"T1w": "ds006557_HFC_T1", "T2w": "ds006557_HFC_T2"}
N_SUBJECTS = 23
EXCLUDED = {"T1w": {6}, "T2w": set()}  # sub-HYPE06 has no low-field T1w
FOLDS = {
    0: {"val": [14, 15], "test": [16, 17, 18, 19, 20, 21, 22]},
    1: {"val": [19, 22], "test": [4, 7, 8, 12, 13, 18, 21]},
    2: {"val": [1, 8], "test": [2, 4, 5, 6, 11, 19, 21]},
    3: {"val": [1, 13], "test": [6, 7, 8, 11, 15, 19, 22]},
    4: {"val": [15, 18], "test": [1, 2, 6, 7, 10, 13, 14]},
}


def split_subjects(contrast: str, fold: int) -> Dict[str, List[str]]:
    val, test = set(FOLDS[fold]["val"]), set(FOLDS[fold]["test"])
    ids = {"train": set(range(N_SUBJECTS)) - val - test, "val": val, "test": test}
    return {k: [f"sub-HYPE{i:02d}" for i in sorted(v - EXCLUDED[contrast])] for k, v in ids.items()}


class PairedVolumes(Dataset):
    """Max-normalized (HF, LF) volumes of a list of subjects."""

    def __init__(self, data_root: str, contrast: str, subjects: List[str]):
        root = Path(data_root) / CONTRAST_DIR[contrast]
        self.pairs = []
        for s in subjects:
            hf, lf = root / s / "HF_synthstrip.nii.gz", root / s / "LF_synthstrip.nii.gz"
            if not (hf.exists() and lf.exists()):
                raise FileNotFoundError(f"missing {hf} or {lf}")
            self.pairs.append((s, hf, lf))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        s, hf, lf = self.pairs[i]
        load = lambda p: normalize_volume(nib.load(str(p)).get_fdata(dtype=np.float32))
        return {"subject": s, "hf": load(hf), "lf": load(lf)}


def make_splits(data_root: str, contrast: str = "T1w", fold: int = 0) -> Dict[str, PairedVolumes]:
    return {k: PairedVolumes(data_root, contrast, v) for k, v in split_subjects(contrast, fold).items()}


class PointSampleDataset(Dataset):
    """Training samples: the HF volume, K random LF voxels (coordinates + targets) and the full LF
    volume (for the gradient loss on a random sub-volume)."""

    def __init__(self, volumes: PairedVolumes, K: int = 8000):
        self.volumes, self.K = volumes, K

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, i):
        v = self.volumes[i]
        hf, lf = torch.from_numpy(v["hf"]), torch.from_numpy(v["lf"])
        coord_all = make_coord_grid(hf.shape)
        n = coord_all.shape[0]
        idx = torch.randperm(n)[:self.K] if n > self.K else torch.arange(n)
        return {"hf": hf.unsqueeze(0), "coord": coord_all[idx], "gt": lf.reshape(-1, 1)[idx], "lf": lf}
