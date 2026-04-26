import sys
import os

# Tell Python to use your home drive for temporary worker files instead of /tmp
HOME_DIR = os.path.expanduser("~")
os.environ["TMPDIR"] = os.path.join(HOME_DIR, "tmp")
os.makedirs(os.environ["TMPDIR"], exist_ok=True)
# Tell Hugging Face to wait 5 minutes instead of 10 seconds before giving up
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

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
import torchvision.transforms as T
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor
from datasets import load_dataset
import pytorch_lightning as pl

# Import your custom modules
from src.data.geometry import CoordsToS2
from src.data.transform import Transform
from src.models.head import CombineModel

NUM_CLASSES = 50000

def distance(lat1, lon1, lat2, lon2):
    # Radius of Earth in kilometers
    R = 6378.1
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    
    return R * c

class OSV5MCollator:
    def __init__(self, processor, fast_transform, s2_to_class):
        self.processor = processor
        self.fast_transform = fast_transform
        self.s2_to_class = s2_to_class

    def __call__(self, batch):
        images = []
        captions = []
        labels = []
        actual_lat = []
        actual_lon = []
        cities, countries = [], []
        
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
            cities.append(city)
            countries.append(country)
            climate = item.get('climate', 'Unknown')
            drive_side = 'left' if item.get('drive_side') == 1 else 'right'
            
            caption = f"A street view photo taken in {city}, {region}, {country}. The climate classification is {climate} and cars drive on the {drive_side} side of the road."
            captions.append(caption)
            
            # 3. Fast CPU Image Transform (Just resize and normalize)
            raw_image = item['image'].convert("RGB")
            img_tensor = self.fast_transform(raw_image)
            images.append(img_tensor)
        
        # Stack images into shape: [Batch_Size, 3, 224, 224]
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
            "actual_lon": torch.tensor(actual_lon, dtype=torch.float),
            "city": cities,
            "country": countries
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

        # =========================================================
        # THE GPU MATH TOOLS (Runs on CUDA instantly)
        # =========================================================
        self.gpu_blur = T.GaussianBlur(kernel_size=9, sigma=(2.0, 5.0))
        self.gpu_crop = T.RandomResizedCrop(size=(224, 224), scale=(0.1, 0.4))

    def forward(self, pixel_values):
        # Flatten the 3 crops into the batch dimension: [B*3, C, H, W]
        B, N, C, H, W = pixel_values.shape
        flat_images = pixel_values.view(B * N, C, H, W)
        
        # --- SigLIP Vision ---
        siglip_out = self.siglip.vision_model(pixel_values=flat_images)
        siglip_vecs = siglip_out.pooler_output 
        siglip_vecs = siglip_vecs.view(B, N, -1).mean(dim=1) # Average the 3 crops -> [B, 768]
        
        # --- DINOv3 Vision ---
        dino_out = self.dinov3(flat_images).pooler_output
        dino_vecs = dino_out.view(B, N, -1).mean(dim=1) # Average the 3 crops -> [B, 384]
        
        # --- S2 Prediction ---
        s2_logits = self.fusion_head(siglip_vecs, dino_vecs)
        
        return s2_logits, siglip_vecs

    def training_step(self, batch, batch_idx):
        # Shape from CPU: [Batch, 3, 224, 224]
        base_images = batch['pixel_values'] 
        labels = batch['labels']
        
        # =======================================================
        # BORED GPU HACK: Generate crops and blur inside VRAM
        # =======================================================
        global_view = self.gpu_blur(base_images)
        crop1 = self.gpu_crop(base_images)
        crop2 = self.gpu_crop(base_images)
        
        # Stack them on the GPU: Shape becomes [Batch, 3, 3, 224, 224]
        gpu_pixel_values = torch.stack([global_view, crop1, crop2], dim=1)
        
        # 1. Forward pass for images (using the GPU-augmented images)
        s2_logits, siglip_image_vecs = self(gpu_pixel_values)
        
        # 2. Forward pass for text (Built-in extractor)
        text_out = self.siglip.text_model(
            input_ids=batch['input_ids'], 
            attention_mask=batch.get('attention_mask')
        )
        siglip_text_vecs = text_out.pooler_output
        
        # 3. Calculate S2 Geographic Classification Loss
        s2_loss = F.cross_entropy(s2_logits, labels, label_smoothing=0.15) 
        
        # 4. Calculate Text-Image Alignment Loss
        alignment_target = torch.ones(siglip_image_vecs.size(0), device=self.device)
        text_loss = self.cosine_loss(siglip_image_vecs, siglip_text_vecs, alignment_target)
        
        # Combine Losses (0.5 weight on text to prioritize S2 coordinate accuracy)
        total_loss = s2_loss + (0.5 * text_loss)
        
        with torch.no_grad(): # Do not track gradients for this!
            # Get the model's guess
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

        if batch_idx % 50 == 0:
            true_city = batch.get('city', ['Unknown'])[0]
            true_country = batch.get('country', ['Unknown'])[0]
            
            # 1. Get Top 3 probabilities
            probs = torch.softmax(s2_logits[0], dim=0)
            top_probs, top_classes = torch.topk(probs, 3)
            
            # 2. THE SPEED HACK: Pull EVERYTHING to the CPU in one single trip
            top_probs_cpu = top_probs.detach().cpu()
            g_coords_cpu = self.class_centroids[top_classes].detach().cpu()
            t_lat = actual_lat[0].detach().cpu()
            t_lon = actual_lon[0].detach().cpu()
            
            # 3. Calculate distance for all 3 guesses at once mathematically 
            dists_cpu = distance(
                t_lat.expand(3), t_lon.expand(3), 
                g_coords_cpu[:, 0], g_coords_cpu[:, 1]
            )
            
            # 4. Format the text in RAM
            log_dir = os.path.join(os.path.dirname(__file__), '..', 'lightning_logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "guess_log.txt")
            
            log_str = f"\n--- Step {self.global_step:05d} | Actual: {true_city}, {true_country} ---\n"
            print(f"\n[Step {self.global_step}] Sanity Check: ACTUAL -> {true_city}, {true_country}")
            
            for i in range(3):
                p_val = top_probs_cpu[i].item() * 100
                g_lat = g_coords_cpu[i, 0].item()
                g_lon = g_coords_cpu[i, 1].item()
                dist_val = dists_cpu[i].item()
                
                line = f"  {i+1}. {p_val:5.1f}% | {dist_val:5.0f} km | Guess: ({g_lat:6.2f}, {g_lon:6.2f})"
                print(line)
                log_str += line + "\n"
                
            # 5. Write to the hard drive ONCE
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(log_str)
        
        # Logging for Weights & Biases / TensorBoard
        self.log('s2_loss', s2_loss, prog_bar=True)
        self.log('text_loss', text_loss, prog_bar=True)
        self.log('loss', total_loss, prog_bar=True)
        self.log('dist_km', mean_dist_km, prog_bar=True)
        
        return total_loss

    def configure_optimizers(self):
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=3e-4, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=50000,
            T_mult=1,
            eta_min=1e-6 
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step" 
            }
        }

# ==========================================
# 3. Main Execution
# ==========================================
def main():
    print("Initializing Training Pipeline...")
    
    import json
    vocab_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'vocab.json')
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
        
    s2_to_class = {int(k): v for k, v in vocab["s2_to_class"].items()}
    class_to_s2 = {int(k): int(v) for k, v in vocab["class_to_s2"].items()}

    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    fast_transform = Transform()
    collator = OSV5MCollator(processor, fast_transform, s2_to_class)
    
    print("Connecting to OSV5M Stream...")
    dataset = load_dataset(
        "osv5m/osv5m", 
        split="train", 
        streaming=True,
        trust_remote_code=True
    )
    dataset = dataset.shuffle(seed=42, buffer_size=5000) 

    train_loader = DataLoader(
        dataset, 
        batch_size=48, 
        collate_fn=collator, 
        num_workers=1,           
        pin_memory=True,         
        prefetch_factor=8,        
        persistent_workers=True,
        drop_last=True,           
    )
    
    import s2sphere
    
    centroids = torch.zeros((NUM_CLASSES, 2))
    
    for class_id, s2_str in class_to_s2.items():
        class_id = int(class_id)
        cell = s2sphere.CellId(int(s2_str))
        lat_lng = cell.to_lat_lng()
        centroids[class_id, 0] = lat_lng.lat().degrees
        centroids[class_id, 1] = lat_lng.lng().degrees
        
    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint = ModelCheckpoint(
        dirpath="checkpoints_stage2/", 
        filename="geoguessr-stage2-{epoch:02d}-{step:05d}",
        every_n_train_steps=50, 
        save_top_k=1, 
        save_last=True,    
        monitor=None   
    )
    
    trainer = pl.Trainer(
        max_steps=50000,             
        accelerator="gpu",           
        devices=1,
        accumulate_grad_batches=5,
        gradient_clip_val=1.0, 
        precision="16-mixed",        
        log_every_n_steps=10,
        callbacks=[checkpoint]
    )
    
    model = GeoLightningModel(centroids)

    # =========================================================
    # THE RESUME LOGIC (Bulletproof Stage 2)
    # =========================================================
    stage2_ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints_stage2', 'last.ckpt')
    stage1_ckpt_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'last.ckpt')

    if os.path.exists(stage2_ckpt_path):
        print("Found existing Stage 2 checkpoint! Resuming where we left off...")
        trainer.fit(
            model, 
            train_loader,
            ckpt_path=stage2_ckpt_path
        )
        
    else:
        print("Loading Stage 1 Brain and executing Surgical Strike on Peru Bias...")
        
        checkpoint_data = torch.load(stage1_ckpt_path, map_location="cpu")
        model.load_state_dict(checkpoint_data['state_dict'])
        
        peru_lat, peru_lon = -12.12, -77.02
        distances = (centroids[:, 0] - peru_lat)**2 + (centroids[:, 1] - peru_lon)**2
        peru_idx = torch.argmin(distances).item()

        for name, param in model.fusion_head.named_parameters():
            if param.shape[0] == NUM_CLASSES: 
                if "bias" in name:
                    param.data[peru_idx] = -1000.0  
                else:
                    param.data[peru_idx] = 0.0
        
        trainer.fit(
            model, 
            train_loader
        )

if __name__ == "__main__":
    main()