"""Manifest-backed MRI slices and patient-level splitting."""
from pathlib import Path
from typing import Callable
import pandas as pd
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from sklearn.model_selection import StratifiedGroupKFold

LABELS = {"CN": 0, "MCI": 1, "AD": 2}

def patient_split(frame: pd.DataFrame, seed: int = 42, fractions=(0.70, 0.15, 0.15)):
    """Return train/val/test frames without patient leakage; stratify by patient label."""
    required={"filepath","label","patient_id"}
    if not required.issubset(frame.columns): raise ValueError(f"Manifest needs {sorted(required)}")
    if abs(sum(fractions)-1)>1e-6 or min(fractions)<=0: raise ValueError("fractions must be positive and sum to one")
    # A patient must have one diagnosis for meaningful group stratification.
    labels=frame.groupby("patient_id")["label"].agg(lambda x: x.mode().iloc[0])
    groups=labels.index.to_numpy(); y=labels.to_numpy()
    n_splits=max(2, round(1/fractions[2]))
    splitter=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    trainval_idx,test_idx=next(splitter.split(frame,frame.label,groups=frame.patient_id))
    trainval=frame.iloc[trainval_idx].copy(); test=frame.iloc[test_idx].copy()
    # Split remaining patients into train and validation with groups intact.
    val_ratio=fractions[1]/(fractions[0]+fractions[1])
    n_val_splits=max(2,round(1/val_ratio))
    split2=StratifiedGroupKFold(n_splits=n_val_splits,shuffle=True,random_state=seed+1)
    a,b=next(split2.split(trainval,trainval.label,groups=trainval.patient_id))
    return trainval.iloc[a].reset_index(drop=True),trainval.iloc[b].reset_index(drop=True),test.reset_index(drop=True)

class GaussianBlur:
    """PIL Gaussian blur with randomized radius."""
    def __call__(self, image): return image.filter(ImageFilter.GaussianBlur(radius=float(torch.empty(1).uniform_(0.1,2.0))))

def simclr_transform(size=224):
    """Build the two independently augmented views expected by SimCLR."""
    aug=T.Compose([T.RandomResizedCrop(size,scale=(0.6,1.0)),T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(.4,.4,.4,.1)],p=.8),T.RandomGrayscale(p=.15),
        T.RandomApply([GaussianBlur()],p=.5),T.ToTensor(),T.Lambda(lambda x:x.repeat(3,1,1)),
        T.Normalize((.5,)*3,(.5,)*3)])
    return aug

class TwoCropsTransform:
    def __init__(self, transform: Callable): self.transform=transform
    def __call__(self, image): return self.transform(image),self.transform(image)

class MRISliceDataset(Dataset):
    """Read manifest columns filepath,label,patient_id; labels may be CN/MCI/AD or 0/1/2."""
    def __init__(self, manifest, transform=None, simclr=False):
        self.frame=pd.read_csv(manifest) if isinstance(manifest,(str,Path)) else manifest.reset_index(drop=True)
        needed={"filepath","label","patient_id"}
        if not needed.issubset(self.frame.columns): raise ValueError(f"Missing columns: {needed-set(self.frame.columns)}")
        self.transform=TwoCropsTransform(simclr_transform()) if simclr else (transform or T.Compose([T.Resize((224,224)),T.ToTensor(),T.Lambda(lambda x:x.repeat(3,1,1)),T.Normalize((.5,)*3,(.5,)*3)]))
    def __len__(self): return len(self.frame)
    def __getitem__(self,index):
        row=self.frame.iloc[index]
        with Image.open(row.filepath) as im: image=im.convert("L")
        label=LABELS.get(str(row.label),int(row.label) if str(row.label).isdigit() else -1)
        if label not in (0,1,2): raise ValueError(f"Unknown label: {row.label}")
        value=self.transform(image)
        return (value[0],value[1]) if isinstance(self.transform,TwoCropsTransform) else (value,torch.tensor(label,dtype=torch.long))
