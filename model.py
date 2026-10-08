"""ResNet-50 SimCLR and downstream classifier models."""
from pathlib import Path
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

class SimCLRModel(nn.Module):
    def __init__(self, pretrained=False, projection_dim=128):
        super().__init__()
        weights=ResNet50_Weights.DEFAULT if pretrained else None
        self.encoder=resnet50(weights=weights); dim=self.encoder.fc.in_features
        self.encoder.fc=nn.Identity()
        self.projector=nn.Sequential(nn.Linear(dim,512),nn.ReLU(inplace=True),nn.Linear(512,projection_dim))
    def forward(self,x): return self.projector(self.encoder(x))

class AlzheimerClassifier(nn.Module):
    """ResNet-50 classifier. Optionally initialize its encoder from SimCLR weights."""
    def __init__(self,num_classes=3,encoder_checkpoint=None,pretrained=False):
        super().__init__()
        self.model=resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.model.fc=nn.Linear(self.model.fc.in_features,num_classes)
        if encoder_checkpoint:
            state=torch.load(encoder_checkpoint,map_location="cpu",weights_only=False)
            state=state.get("model",state.get("state_dict",state))
            # Accept SimCLRModel checkpoints, stripping encoder prefix and excluding its projection head.
            enc={k.removeprefix("encoder."):v for k,v in state.items() if k.startswith("encoder.")}
            if not enc: enc={k:v for k,v in state.items() if k.startswith(("conv1.","bn1.","layer"))}
            missing,unexpected=self.model.load_state_dict(enc,strict=False)
            if len(enc)==0: raise ValueError(f"No ResNet encoder parameters found in {encoder_checkpoint}")
    def forward(self,x): return self.model(x)
