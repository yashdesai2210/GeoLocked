import sys
import os

# Tell Python to use your home drive for temporary worker files instead of /tmp
HOME_DIR = os.path.expanduser("~")
os.environ["TMPDIR"] = os.path.join(HOME_DIR, "tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

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
from datasets import IterableDataset
import pytorch_lightning as pl

# Import your custom modules
from src.data.geometry import CoordsToS2
from src.data.transform import CropTransform
from src.models.head import CombineModel

NUM_CLASSES = 50000

def distance(lat1, lon1, lat2, lon2):
    # Radius of Earth in kilometers
    R = 6371.0 
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    
    return R * c

class OSV5MCollator:
    def __init__(self, processor, cropper, s2_to_class):
        self.processor = processor
        self.cropper = cropper
        self.s2_to_class = s2_to_class

    def __call__(self, batch):
        images = []
        captions = []
        labels = []
        actual_lat = []
        actual_lon = []
        
        for item in batch:
            # 1. Target S2 Label
            lat, lon = item['latitude'], item['longitude']
            actual_lat.append(lat)
            actual_lon.append(lon)
            s2_label = CoordsToS2(lat, lon, level=12)
            class_id = self.s2_to_class.get(s2_label, 0)
            labels.append(class_id)
            
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
            "labels": labels,
            "actual_lat": torch.tensor(actual_lat, dtype=torch.float),
            "actual_lon": torch.tensor(actual_lon, dtype=torch.float)
        }

# ==========================================
# 2. PyTorch Lightning Model
# ==========================================
class GeoLightningModel(pl.LightningModule):
    def __init__(self, class_centroids):
        super().__init__()
        
        # Load Foundation Models
        self.register_buffer("class_centroids", class_centroids, persistent=False)
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
        with torch.no_grad(): # Do not track gradients for this!
            # Get the model's guess (the class with the highest probability)
            preds = torch.argmax(s2_logits, dim=1)
            
            # Lookup the coordinates for those guesses
            pred_lat = self.class_centroids[preds, 0]
            pred_lon = self.class_centroids[preds, 1]
            
            # Get the true coordinates from the batch
            actual_lat = batch['actual_lat'].to(self.device)
            actual_lon = batch['actual_lon'].to(self.device)
            
            # Calculate Haversine and get the average distance for the batch
            distances = distance(actual_lat, actual_lon, pred_lat, pred_lon)
            mean_dist_km = torch.mean(distances)
        
        # Logging for Weights & Biases / TensorBoard
        self.log('s2_loss', s2_loss, prog_bar=True)
        self.log('text_loss', text_loss, prog_bar=True)
        self.log('loss', total_loss, prog_bar=True)
        self.log('dist_km', mean_dist_km, prog_bar=True)
        
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
    import json
    vocab_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'vocab.json')
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
        
    s2_to_class = {int(k): v for k, v in vocab["s2_to_class"].items()}
    class_to_s2 = {int(k): int(v) for k, v in vocab["class_to_s2"].items()}

    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    cropper = CropTransform()
    collator = OSV5MCollator(processor, cropper, s2_to_class)
    
    print("Connecting to OSV5M Stream...")
    dataset = load_dataset(
        "osv5m/osv5m", 
        split="train", 
        streaming=True,
        trust_remote_code=True
    )
    dataset = dataset.shuffle(seed=42, buffer_size=5000) # For testing, we take a subset of the stream.

    # DataLoader
    # batch_size=4 is safe for A100 when dealing with 5 crops per image (effectively batch size 20 to the backbones)
    train_loader = DataLoader(
        dataset, 
        batch_size=32, 
        collate_fn=collator, 
        num_workers=1,           # TODO: make 0 if working on personal desktop
        pin_memory=True,         # Speeds up the transfer from RAM to GPU VRAM
        prefetch_factor=8,        # Always keep 2 batches ready and waiting for the GPU
        persistent_workers=True,
        drop_last=True,           # Prevents weird small batches at the very end
        )
    
    s2_to_class = vocab["s2_to_class"]
    class_to_s2 = vocab["class_to_s2"] # Get the reverse mapping too!
    
    import s2sphere
    
    centroids = torch.zeros((NUM_CLASSES, 2))
    
    for class_id, s2_str in class_to_s2.items():
        class_id = int(class_id)
        cell = s2sphere.CellId(int(s2_str))
        lat_lng = cell.to_lat_lng()
        centroids[class_id, 0] = lat_lng.lat().degrees
        centroids[class_id, 1] = lat_lng.lng().degrees
        
    model = GeoLightningModel(centroids)
    
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="geoguessr-{epoch:02d}-{step:05d}",
        every_n_train_steps=50, # Save every 100 batches
        save_top_k=1, # Keep all checkpoints (or set to 1 to save space)
        save_last=True,    # <--- Also always keeps the very latest one
        monitor=None   # <--- Tell it to watch the 'loss' you logged
        #mode="min",       # <--- We want the SMALLEST loss
        )
    
    # Initialize PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_steps=50000,             # Use max_steps instead of epochs for streaming data
        accelerator="gpu",           # Auto-detects your A100
        devices=1,
        accumulate_grad_batches=1,   # Simulates a larger batch size of 16
        precision="16-mixed",        # MASSIVE speedup on A100 GPUs
        log_every_n_steps=10,
        callbacks=[checkpoint]
    )
    
    print("Starting Training Loop...")
    ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'last.ckpt')
    
    # Pass it into trainer.fit()
    trainer.fit(
        model, 
        train_loader,
        ckpt_path=ckpt_path
    )

if __name__ == "__main__":
    main()