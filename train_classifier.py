"""Stage 2: patient-level supervised classifier finetuning."""
import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from config import Config
from src.dataset import MRISliceDataset, patient_split, LABELS
from src.model import AlzheimerClassifier

def label_ids(series): return series.map(lambda x: LABELS.get(str(x),int(x) if str(x).isdigit() else -1)).to_numpy()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',default=None); p.add_argument('--encoder',default=None); p.add_argument('--epochs',type=int)
    p.add_argument('--batch-size',type=int); p.add_argument('--lr',type=float); p.add_argument('--output',default='checkpoints/classifier.pt')
    p.add_argument('--workers',type=int); a=p.parse_args(); c=Config()
    random.seed(c.seed); np.random.seed(c.seed); torch.manual_seed(c.seed)
    frame=pd.read_csv(a.manifest or c.data_dir); tr,va,te=patient_split(frame,c.seed)
    train=DataLoader(MRISliceDataset(tr),batch_size=a.batch_size or c.batch_size,shuffle=True,num_workers=c.num_workers if a.workers is None else a.workers,pin_memory=c.device.startswith('cuda'))
    val=DataLoader(MRISliceDataset(va),batch_size=a.batch_size or c.batch_size,shuffle=False,num_workers=c.num_workers if a.workers is None else a.workers)
    test=DataLoader(MRISliceDataset(te),batch_size=a.batch_size or c.batch_size,shuffle=False,num_workers=c.num_workers if a.workers is None else a.workers)
    model=AlzheimerClassifier(c.num_classes,encoder_checkpoint=a.encoder).to(c.device)
    classes=np.arange(c.num_classes); weights=compute_class_weight('balanced',classes=classes,y=label_ids(tr.label))
    criterion=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=c.device))
    optimizer=torch.optim.Adam(model.parameters(),lr=a.lr or c.lr); epochs=a.epochs or c.epochs
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs); best=-1.; out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(epochs):
        model.train()
        for x,y in train:
            x,y=x.to(c.device),y.to(c.device); optimizer.zero_grad(set_to_none=True); loss=criterion(model(x),y); loss.backward(); optimizer.step()
        model.eval(); truth=[]; pred=[]
        with torch.no_grad():
            for x,y in val:
                pred.extend(model(x.to(c.device)).argmax(1).cpu().tolist()); truth.extend(y.tolist())
        score=f1_score(truth,pred,average='macro',zero_division=0); scheduler.step()
        print(f'Epoch {epoch+1}/{epochs}: val_macro_f1={score:.4f}')
        if score>best:
            best=score; torch.save({'model':model.state_dict(),'num_classes':c.num_classes,'best_val_macro_f1':best},out)
    ckpt=torch.load(out,map_location=c.device,weights_only=False); model.load_state_dict(ckpt['model']); model.eval(); truth=[]; pred=[]
    with torch.no_grad():
        for x,y in test:
            pred.extend(model(x.to(c.device)).argmax(1).cpu().tolist()); truth.extend(y.tolist())
    print('Test report (CN=0, MCI=1, AD=2):')
    print(classification_report(truth,pred,labels=classes,target_names=['CN','MCI','AD'],zero_division=0))
    print(f'Best checkpoint: {out}; validation macro-F1={best:.4f}')
if __name__=='__main__': main()
