import torch
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# LOAD MODEL & PROCESSOR
MODEL_PATH = "./model_weights" # Path where config.json and model.safetensors are
processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b2")
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_PATH).to("cuda")
model.eval()

# ID Mappings
reverse_map = {0: 100, 1: 200, 2: 300, 3: 500, 4: 550, 5: 600, 6: 700, 7: 800, 8: 7100, 9: 10000}
class_names = ['Trees', 'Bushes', 'DryGrass', 'DryBush', 'Clutter', 'Flowers', 'Logs', 'Rocks', 'Ground', 'Sky']

# Inference Logic
def run_test(img_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    test_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg'))]
    
    for file_name in tqdm(test_files):
        img = Image.open(os.path.join(img_dir, file_name)).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        upsampled = torch.nn.functional.interpolate(outputs.logits, size=img.size[::-1], mode="bilinear")
        pred = upsampled.argmax(dim=1)[0].cpu().numpy()

        # Save as grayscale competition ID mask
        final_mask = np.zeros_like(pred, dtype=np.uint16)
        for model_id, original_id in reverse_map.items():
            final_mask[pred == model_id] = original_id
        
        Image.fromarray(final_mask).save(os.path.join(output_dir, file_name.replace('.jpg', '.png')))

print("Inference Ready. Run run_test(data_dir, output_dir) to generate masks.")
