import torch
import torchvision.transforms as T

class Transform:
    def __init__(self):
        # The CPU now only does the absolute lightweight minimum: resize, convert, normalize.
        self.base = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        # Returns a single [3, 224, 224] tensor instead of 3 stacked crops
        return self.base(image)