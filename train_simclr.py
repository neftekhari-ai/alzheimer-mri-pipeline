"""Stage 1: self-supervised SimCLR pretraining."""
import argparse, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import Config
from src.dataset import MRISliceDataset
from src.model import SimCLRModel
from src.simclr import train_simclr

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=None); p.add_argument('--epochs',type=int); p.add_argument('--batch-size',type=int)
    p.add_argument('--lr',type=float); p.add_argument('--output',default='checkpoints/simclr.pt'); p.add_argument('--workers',type=int)
    a=p.parse_args(); c=Config()
    manifest=a.manifest or c.data_dir; epochs=a.epochs or c.epochs; batch=a.batch_size or c.batch_size
    lr=a.lr or c.lr; workers=c.num_workers if a.workers is None else a.workers
    random.seed(c.seed); np.random.seed(c.seed); torch.manual_seed(c.seed)
    ds=MRISliceDataset(manifest,simclr=True)
    loader=DataLoader(ds,batch_size=batch,shuffle=True,num_workers=workers,pin_memory=c.device.startswith('cuda'),drop_last=True)
    if len(loader)==0: raise ValueError('Dataset must contain at least one full batch (batch size >= 2).')
    model=SimCLRModel().to(c.device); optimizer=torch.optim.AdamW(model.parameters(),lr=lr)
    losses=train_simclr(model,loader,optimizer,c.device,c.temperature,epochs)
    output=__import__('pathlib').Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'epochs':epochs,'losses':losses},output)
    print(f'Saved {output}; final loss={losses[-1]:.4f}')
if __name__=='__main__': main()
