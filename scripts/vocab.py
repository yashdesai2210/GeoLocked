import sys
import os
import json
from collections import Counter
from datasets import load_dataset

# Add the root 'GeoLocked' directory to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your existing function!
from src.data.geometry import CoordsToS2

def main():
    dataset = load_dataset(
        "osv5m/osv5m", 
        split="train", 
        streaming=True,
        trust_remote_code=True
    )
    dataset = dataset.select_columns(["latitude", "longitude"])

    # We will count the S2 cells we see
    cell_counter = Counter()
    
    # Looking at 500,000 images gives us a massive, highly accurate sample of the Earth
    SAMPLE_SIZE = 500000 
    
    print(f"Scanning {SAMPLE_SIZE} locations...")
    
    # Stream the data and count the cells
    count = 0
    for item in dataset:
        print(count)
        lat, lon = item['latitude'], item['longitude']
        s2_id = CoordsToS2(lat, lon, level=12)
        cell_counter[s2_id] += 1
        
        count += 1
        if count % 10000 == 0:
            print(f"Scanned {count} / {SAMPLE_SIZE}...")
        if count >= SAMPLE_SIZE:
            break

    print("\nScanning complete! Building the dictionary...")
    
    # Get the 50,000 most common S2 Cell IDs
    top_50k = cell_counter.most_common(50000)
    
    # Create our mappings
    s2_to_class = {}
    class_to_s2 = {}
    
    for class_id, (s2_id, frequency) in enumerate(top_50k):
        s2_to_class[str(s2_id)] = class_id   # JSON keys must be strings
        class_to_s2[class_id] = str(s2_id)
        
    # Save it to your data folder
    vocab_data = {
        "s2_to_class": s2_to_class,
        "class_to_s2": class_to_s2
    }
    
    save_path = os.path.join(PROJECT_ROOT, "src", "data", "vocab.json")
    
    with open(save_path, "w") as f:
        json.dump(vocab_data, f)
        
    print(f"\nEarth Vocab saved to {save_path}")

if __name__ == "__main__":
    main()