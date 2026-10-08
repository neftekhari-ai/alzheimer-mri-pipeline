"""Single-slice prediction with Grad-CAM heatmap export."""
import argparse
import torch
from src.model import AlzheimerClassifier
from src.gradcam import GradCAM,preprocess_image,save_overlay

LABELS=['CN','MCI','AD']
def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('image'); p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',default='gradcam_overlay.jpg'); p.add_argument('--device',default=None); p.add_argument('--image-size',type=int,default=224)
    a=p.parse_args(); device=a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    state=torch.load(a.checkpoint,map_location=device,weights_only=False); model=AlzheimerClassifier(num_classes=int(state.get('num_classes',3))).to(device)
    model.load_state_dict(state['model'] if 'model' in state else state); model.eval()
    image,x=preprocess_image(a.image,a.image_size); x=x.to(device); cam=GradCAM(model)
    heat,logits=cam(x); probs=torch.softmax(logits,dim=1)[0]; idx=int(probs.argmax())
    save_overlay(image,heat,a.output)
    print(f'Prediction: {LABELS[idx]} (class {idx}), confidence={probs[idx].item():.4f}')
    print(f'Grad-CAM overlay saved to: {a.output}')
if __name__=='__main__': main()
