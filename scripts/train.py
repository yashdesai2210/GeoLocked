import sys
import os

# Force Python to ignore broken Windows SSL paths
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
if "CURL_CA_BUNDLE" in os.environ:
    del os.environ["CURL_CA_BUNDLE"]
# ----------------------

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset
import pytorch_lightning as pl

# Import your custom modules
from src.data.geometry import CoordsToS2
from src.data.transform import CropTransform
from src.models.head import CombineModel

NUM_CLASSES = 50000

# ==========================================
# 1. Custom Data Collator (Handles Text & Images)
# ==========================================
class OSV5MCollator:
    def __init__(self, processor, cropper):
        self.processor = processor
        self.cropper = cropper

    def __call__(self, batch):
        images = []
        captions = []
        labels = []
        
        for item in batch:
            # 1. Target S2 Label
            lat, lon = item['latitude'], item['longitude']
            s2_label = CoordsToS2(lat, lon, level=12)
            labels.append(s2_label % NUM_CLASSES)
            
            # 2. Synthetic Caption Generation
            city = item.get('city', 'Unknown City')
            region = item.get('region', 'Unknown Region')
            country = item.get('country', 'Unknown Country')
            climate = item.get('climate', 'Unknown')
            drive_side = 'left' if item.get('drive_side') == 1 else 'right'
            
            caption = f"A street view photo taken in {city}, {region}, {country}. The climate classification is {climate} and cars drive on the {drive_side} side of the road."
            captions.append(caption)
            
            # 3. Image Transforms (yields 5 crops per image)
            raw_image = item['image'].convert("RGB")
            img_tensors = self.cropper(raw_image)
            images.append(img_tensors)
        
        # Stack images into shape: [Batch_Size, 5, 3, 224, 224]
        batched_images = torch.stack(images) 
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Tokenize text captions
        text_inputs = self.processor(
            text=captions, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": batched_images,
            "input_ids": text_inputs["input_ids"],
            "attention_mask": text_inputs.get("attention_mask", None),
            "labels": labels
        }

# ==========================================
# 2. PyTorch Lightning Model
# ==========================================
class GeoLightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # Load Foundation Models
        self.siglip = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        self.dinov3 = AutoModel.from_pretrained("facebook/dinov3-vits16-pretrain-lvd1689m")
        
        # Freeze Backbones, but Unfreeze the SigLIP Vision Pooler for Text Alignment
        for name, param in self.siglip.named_parameters():
            if "vision_model.head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for param in self.dinov3.parameters():
            param.requires_grad = False
            
        # Fusion Head (768 from SigLIP Base, 384 from DINOv3 Small)
        self.fusion_head = CombineModel(siglip2_dim=768, dino_dim=384, num_s2_classes=NUM_CLASSES)
        
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def forward(self, pixel_values):
        # Flatten the 5 crops into the batch dimension: [B*5, C, H, W]
        B, N, C, H, W = pixel_values.shape
        flat_images = pixel_values.view(B * N, C, H, W)
        
        # --- SigLIP Vision ---
        # Explicitly extract the tensor using .pooler_output
        siglip_out = self.siglip.vision_model(pixel_values=flat_images)
        siglip_vecs = siglip_out.pooler_output 
        siglip_vecs = siglip_vecs.view(B, N, -1).mean(dim=1) # Average the 5 crops -> [B, 768]
        
        # --- DINOv3 Vision ---
        dino_out = self.dinov3(flat_images).pooler_output
        dino_vecs = dino_out.view(B, N, -1).mean(dim=1) # Average the 5 crops -> [B, 384]
        
        # --- S2 Prediction ---
        s2_logits = self.fusion_head(siglip_vecs, dino_vecs)
        
        return s2_logits, siglip_vecs

    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        
        # 1. Forward pass for images
        s2_logits, siglip_image_vecs = self(pixel_values)
        
        # 2. Forward pass for text (Built-in extractor)
        text_out = self.siglip.text_model(
            input_ids=batch['input_ids'], 
            attention_mask=batch.get('attention_mask')
        )
        siglip_text_vecs = text_out.pooler_output
        
        # 3. Calculate S2 Geographic Classification Loss
        s2_loss = F.cross_entropy(s2_logits, labels)
        
        # 4. Calculate Text-Image Alignment Loss
        # We target '1' because we want the cosine similarity to be perfectly matched
        alignment_target = torch.ones(siglip_image_vecs.size(0), device=self.device)
        text_loss = self.cosine_loss(siglip_image_vecs, siglip_text_vecs, alignment_target)
        
        # Combine Losses (0.5 weight on text to prioritize S2 coordinate accuracy)
        total_loss = s2_loss + (0.5 * text_loss)
        
        # Logging for Weights & Biases / TensorBoard
        self.log('s2_loss', s2_loss, prog_bar=True)
        self.log('text_loss', text_loss, prog_bar=True)
        self.log('loss', total_loss, prog_bar=True)
        
        return total_loss

    def configure_optimizers(self):
        # Only optimize parameters that require gradients (Fusion Head + Projections)
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
        return optimizer

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    print("Initializing Training Pipeline...")
    
    # Setup Data Pipeline
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    cropper = CropTransform()
    collator = OSV5MCollator(processor, cropper)
    
    print("Connecting to OSV5M Stream...")
    dataset = load_dataset(
        "osv5m/osv5m", 
        split="train", 
        streaming=True,
        trust_remote_code=True
    ) # For testing, we take a subset of the stream. Remove .take() for full training.
    
    # DataLoader
    # batch_size=4 is safe for A100 when dealing with 5 crops per image (effectively batch size 20 to the backbones)
    train_loader = DataLoader(
        dataset, 
        batch_size=32, 
        collate_fn=collator, 
        num_workers=1,           # TODO: make 0 if working on personal desktop
        pin_memory=True,         # Speeds up the transfer from RAM to GPU VRAM
        prefetch_factor=4,        # Always keep 2 batches ready and waiting for the GPU
        persistent_workers=True
        ).shuffle(seed=42, buffer_size=1000)
    
    # Initialize Model
    model = GeoLightningModel()
    
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="geoguessr-{epoch:02d}-{step:05d}",
        every_n_train_steps=500, # Save every 500 batches
        save_top_k=3, # Keep all checkpoints (or set to 3 to save space)
        monitor="loss",   # <--- Tell it to watch the 'loss' you logged
        mode="min",       # <--- We want the SMALLEST loss
        save_last=True    # <--- Also always keeps the very latest one
        )
    
    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_steps=50000,             # Use max_steps instead of epochs for streaming data
        accelerator="gpu",           # Auto-detects your A100
        devices=1,
        accumulate_grad_batches=4,   # Simulates a larger batch size of 16
        precision="16-mixed",        # MASSIVE speedup on A100 GPUs
        log_every_n_steps=10,
        callbacks=[checkpoint]
    )
    
    print("Starting Training Loop...")
    trainer.fit(model, train_loader)

if __name__ == "__main__":
    main()