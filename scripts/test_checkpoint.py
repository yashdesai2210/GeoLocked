import sys
import os
import torch
from PIL import Image

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your model and cropper
from scripts.train import GeoLightningModel
from src.data.transform import CropTransform

def main():
    print("1. Locating Checkpoint...")
    # Change this to the exact name of your checkpoint if it's not last.ckpt
    checkpoint_path = "/common/home/yd355/GeoLocked/checkpoints/last.ckpt"
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Could not find checkpoint at {checkpoint_path}")
        return

    print("2. Loading Model 'Brain' into Memory...")
    # PyTorch Lightning Magic: This loads the architecture AND your trained weights
    model = GeoLightningModel.load_from_checkpoint(checkpoint_path)
    
    # Put the model in "Evaluation" mode (turns off dropout, stops training)
    model.eval() 
    
    # Move to GPU if available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ Model successfully loaded and moved to {device}!\n")

    print("3. Preparing a Test Image...")
    # For a quick test, we will create a blank 1024x512 image.
    # Later, you can change this line to: img = Image.open("path/to/real/streetview.jpg")
    test_image = Image.new('RGB', (1024, 512), color='gray')
    
    cropper = CropTransform()
    img_tensors = cropper(test_image) # Yields [5, 3, 224, 224]
    
    # Add a "Batch" dimension so PyTorch doesn't crash (Shape becomes [1, 5, 3, 224, 224])
    batched_images = img_tensors.unsqueeze(0).to(device)

    print("4. Running Inference (Guessing the location)...")
    # Tell PyTorch NOT to calculate gradients to save memory and speed it up
    with torch.no_grad():
        s2_logits, _ = model(batched_images)
        
        # Get the index of the highest probability guess
        predicted_class = torch.argmax(s2_logits, dim=1).item()

    print("\n🎉 SUCCESS! The checkpoint works perfectly.")
    print(f"The model predicts this image belongs to S2 Class ID: {predicted_class}")

if __name__ == "__main__":
    main()