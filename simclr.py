"""NT-Xent objective and SimCLR training loop."""
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

class NTXentLoss(nn.Module):
    def __init__(self,temperature=0.5):
        super().__init__()
        if temperature<=0: raise ValueError("temperature must be positive")
        self.temperature=temperature
    def forward(self,z1,z2):
        n=z1.shape[0]
        if n<2: raise ValueError("NT-Xent requires batch size >= 2")
        z=F.normalize(torch.cat([z1,z2],dim=0),dim=1)
        sim=z@z.T/self.temperature
        sim.fill_diagonal_(float("-inf"))
        positives=torch.cat([torch.arange(n,2*n),torch.arange(n)]).to(z.device)
        return F.cross_entropy(sim,positives)

def train_simclr(model,loader,optimizer,device,temperature=.5,epochs=1):
    """Train model on batches of augmented (view1, view2); return epoch loss history."""
    criterion=NTXentLoss(temperature).to(device); history=[]
    for epoch in range(epochs):
        model.train(); total=0.0; count=0
        for views in tqdm(loader,desc=f"SimCLR {epoch+1}/{epochs}"):
            x1,x2=(v.to(device,non_blocking=True) for v in views)
            optimizer.zero_grad(set_to_none=True); loss=criterion(model(x1),model(x2)); loss.backward(); optimizer.step()
            total+=loss.item()*x1.size(0); count+=x1.size(0)
        history.append(total/max(count,1))
    return history
