import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from transformers import AutoModel
from datasets import load_dataset

from src.data.geometry import CoordsToS2
from src.data.transform import CropTransform
from src.models.head import CombineModel

def main():
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # download and load models to device
    print("Loading Foundation Models...")
    siglip2 = AutoModel.from_pretrained("google/siglip2-base-patch16-224").to(device).eval()
    #dinov3 = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    dinov3 = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m").to(device).eval()
    
    # save vram, only train head
    for param in siglip2.parameters(): param.requires_grad = False
    for param in dinov3.parameters(): param.requires_grad = False

    # used AdamW to decouple weight decay from learning rate
    # cross-entropy loss to see how bad model guessed
    NUM_CLASSES = 50000 
    fusion_head = CombineModel(siglip2_dim=768, dino_dim=384, num_s2_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.AdamW(fusion_head.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # streaming to load one by one, immediately discarding when done, saving vRAM and storage
    print("Connecting to OSV5M Stream...")
    dataset = load_dataset(
        'osv5m/osv5m',             # 1. Point directly to the official dataset repo
        split="train", 
        streaming=True,
        trust_remote_code=True
    )
    cropper = CropTransform()

    print("Starting Training Loop...")
    fusion_head.train()
    accumulation_steps = 4 
    optimizer.zero_grad()

    for step, item in enumerate(dataset):
        lat, lon = item['lat'], item['lon']
        s2_label = CoordsToS2(lat, lon, level=12)
        
        raw_image = item['image'].convert("RGB")
        image_tensors = cropper(raw_image).to(device) 

        with torch.no_grad():
            siglip2_out = siglip2(pixel_values=image_tensors).pooler_output 
            siglip2_vector = siglip2_out.mean(dim=0, keepdim=True) 

            dino_out = dinov3(pixel_values=image_tensors).pooler_output
            dino_vector = dino_out.mean(dim=0, keepdim=True)

        target_label = torch.tensor([s2_label % NUM_CLASSES]).to(device) 
        
        logits = fusion_head(siglip2_vector, dino_vector)
        loss = loss_fn(logits, target_label)
        
        loss = loss / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"Step {step+1} | Loss: {loss.item() * accumulation_steps:.4f}")

        #if step > 10: break # Small test break

if __name__ == "__main__":
    main()