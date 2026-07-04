from ultralytics import YOLO
import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForImageTextToText
import torchvision.transforms.functional as TF
import json

def load_yolo_model(model_path: str) -> YOLO:
    """
    Load a YOLO model from the specified path.
    Args:
        model_path (str): Path to the YOLO model file.
    Returns:
        YOLO: Loaded YOLO model.
    """
    model = YOLO(model_path)
    pytorch_model = model.model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pytorch_model.to(device)

    return pytorch_model

class YOLO_RL_Adapter(nn.Module):
    def __init__(self, raw_yolo_model):
        super().__init__()
        
        # 1. Steal the backbone (Layers 0-9) from your loaded YOLO model
        # This bypasses the massive anchor head but keeps the powerful vision layers
        self.backbone = raw_yolo_model.model[:10] 
        
        # 2. Freeze the YOLO backbone so the RL loop doesn't destroy its pretrained knowledge
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 3. Add the RL-specific statistical heads
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # YOLOv8 nano features output 256 channels
        print("Automatically detected YOLO backbone output channels: 256")
        self.box_mean_head = nn.Linear(256, 4) 
        self.box_std_head = nn.Linear(256, 4)

    def forward(self, x):
        features = self.backbone(x)
        flat_features = self.flatten(self.pool(features))
        
        # Output exactly [Batch, 4] for both means and stds
        means = torch.sigmoid(self.box_mean_head(flat_features)) 
        raw_stds = self.box_std_head(flat_features)
        stds = torch.clamp(torch.nn.functional.softplus(raw_stds), min=0.01, max=0.2)
        return means, stds


def load_medgemma_critic(device="cuda" if torch.cuda.is_available() else "cpu"):
    print("Loading Multimodal MedGemma 1.5 4B-IT...")
    
    model_id = "google/medgemma-1.5-4b-it"
    
    # Load processor (handles text tokenization and medical image scaling)
    processor = AutoProcessor.from_pretrained(model_id)
    # Load the vision-to-language model
    model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False  # Freeze entirely
        
    print("MedGemma reward loaded.")
    return model, processor
