"""Grad-CAM for ResNet layer4 with image overlay export."""
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

class GradCAM:
    def __init__(self,model,target_layer=None):
        self.model=model; self.layer=target_layer or model.model.layer4; self.activations=None; self.gradients=None
        self.layer.register_forward_hook(self._forward_hook); self.layer.register_full_backward_hook(self._backward_hook)
    def _forward_hook(self,module,inputs,output): self.activations=output
    def _backward_hook(self,module,grad_input,grad_output): self.gradients=grad_output[0]
    def __call__(self,x,class_idx=None):
        self.model.zero_grad(set_to_none=True); logits=self.model(x)
        idx=int(logits.argmax(1).item()) if class_idx is None else int(class_idx)
        logits[:,idx].sum().backward(retain_graph=False)
        weights=self.gradients.mean(dim=(2,3),keepdim=True)
        cam=torch.relu((weights*self.activations).sum(dim=1,keepdim=True))
        cam=torch.nn.functional.interpolate(cam,size=x.shape[-2:],mode="bilinear",align_corners=False)[0,0]
        cam=cam.detach().cpu().numpy(); return (cam-cam.min())/(cam.max()-cam.min()+1e-8),logits.detach()

def preprocess_image(path,image_size=224):
    image=Image.open(path).convert("L"); tensor=T.Compose([T.Resize((image_size,image_size)),T.ToTensor(),T.Lambda(lambda x:x.repeat(3,1,1)),T.Normalize((.5,)*3,(.5,)*3)])(image)
    return image,tensor.unsqueeze(0)

def save_overlay(image,heatmap,path,alpha=.4):
    """Save color Grad-CAM overlay at path."""
    base=np.asarray(image.convert("RGB").resize((heatmap.shape[1],heatmap.shape[0])))
    color=cv2.applyColorMap(np.uint8(255*heatmap),cv2.COLORMAP_JET)[:,:,::-1]
    overlay=cv2.addWeighted(base,1-alpha,color,alpha,0); Path(path).parent.mkdir(parents=True,exist_ok=True)
    if not cv2.imwrite(str(path),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR)): raise OSError(f"Could not write {path}")
