import torch
import torchvision.transforms as T

class CropTransform:
    def __init__(self):
        #squish entire image into 224x224
        self.global_transform = T.Compose([
            T.Resize((224, 224)),
            T.GaussianBlur(kernel_size=9, sigma=(2.0, 5.0)), # Strong Blur
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        #create random crops of image, between 10-40% of original size
        self.local_transform = T.Compose([
            T.RandomResizedCrop(size=(224, 224), scale=(0.1, 0.4)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        global_view = self.global_transform(image)
        cropped1 = self.local_transform(image)
        cropped2 = self.local_transform(image)
        cropped3 = self.local_transform(image)
        cropped4 = self.local_transform(image)
        # 1 blurry image (provides macro-architecture) + 4 cropped images (micro-features)
        return torch.stack([global_view, cropped1, cropped2, cropped3, cropped4])