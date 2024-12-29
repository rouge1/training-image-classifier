# inference.py
import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np
from enum import Enum

class ModelType(Enum):
    MNIST = "mnist"         # For 28x28   models
    STANDARD = "standard"   # For 224x224 models
    INCEPTION = "inception" # For 299x299 models

class ImagePredictor:
    def __init__(self, model, device, class_mapping, model_type: ModelType):
        self.model = model
        self.device = device
        self.class_mapping = class_mapping
        self.model_type = model_type
        self.transform = self._get_transform()

    def _get_transform(self):
        """Get the appropriate transform for the model type"""
        if self.model_type == ModelType.MNIST:
            return transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        elif self.model_type == ModelType.STANDARD:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
        elif self.model_type == ModelType.INCEPTION:
            return transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])

    def load_and_preprocess_image(self, image_path):
        try:
            # Handle both file paths and uploaded file objects
            if isinstance(image_path, (str, Path)):
                image = Image.open(image_path)
            else:
                image = Image.open(image_path)

            # Convert to RGB if not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            # Apply transformations
            tensor = self.transform(image)
            
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            
            return tensor, None
            
        except Exception as e:
            return None, f"Error processing image: {str(e)}"

    def predict(self, image_path):
        # Load and preprocess
        tensor, error = self.load_and_preprocess_image(image_path)
        if error:
            return None, error

        # Move to device and make prediction
        try:
            self.model.eval()
            with torch.no_grad():
                tensor = tensor.to(self.device)
                output = self.model(tensor)
                
                # Apply softmax to get probabilities
                probabilities = torch.nn.functional.softmax(output, dim=1)
                
                # Get prediction and confidence
                pred_prob, pred_idx = torch.max(probabilities, dim=1)
                pred = pred_idx.item()
                confidence = pred_prob.item() * 100  # Convert to percentage
                
                # Get class name
                pred_class = [k for k, v in self.class_mapping.items() if v == pred][0]
                
                return {
                    'prediction': pred, 
                    'class_name': pred_class,
                    'confidence': confidence
                }, None
                
        except Exception as e:
            return None, f"Error making prediction: {str(e)}"