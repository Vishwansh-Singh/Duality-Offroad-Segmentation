import os
import numpy as np
import torch
from datasets import Dataset, Image
from transformers import (
    SegformerImageProcessor, 
    SegformerForSemanticSegmentation, 
    TrainingArguments, 
    Trainer
)

# 1. PATHS & CONFIG
TRAIN_IMG_DIR = "/content/dataset/Color_Images"
TRAIN_MASK_DIR = "/content/dataset/Segmentation"

# Competition Class Mapping (The secret to your 0.6788 score)
label_map = {100:0, 200:1, 300:2, 500:3, 550:4, 600:5, 700:6, 800:7, 7100:8, 10000:9}

# 2. DATASET PREPARATION
def get_data():
    images = sorted([os.path.join(TRAIN_IMG_DIR, f) for f in os.listdir(TRAIN_IMG_DIR) if f.endswith('.jpg')])
    masks = sorted([os.path.join(TRAIN_MASK_DIR, f) for f in os.listdir(TRAIN_MASK_DIR) if f.endswith('.png')])
    return Dataset.from_dict({"image": images, "label": masks}).cast_column("image", Image()).cast_column("label", Image())

dataset = get_data()
processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b2")

def train_transforms(example_batch):
    images = [x for x in example_batch["image"]]
    labels = [np.array(x) for x in example_batch["label"]]
    # Map high-value IDs to 0-9
    labels = [np.vectorize(lambda x: label_map.get(x, 0))(l) for l in labels]
    
    inputs = processor(images, labels, return_tensors="pt")
    return inputs

dataset.set_transform(train_transforms)

# 3. MODEL INITIALIZATION (SegFormer-B2)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b2",
    num_labels=10,
    id2label={i: str(i) for i in range(10)},
    label2id={str(i): i for i in range(10)},
    ignore_mismatched_sizes=True
)

# 4. TRAINING ARGUMENTS (Your optimized settings)
training_args = TrainingArguments(
    output_dir="./segformer_b2_offroad",
    learning_rate=6e-5,
    num_train_epochs=21,
    per_device_train_batch_size=4,
    evaluation_strategy="no",
    save_strategy="epoch",
    save_total_limit=5,
    fp16=True,
    push_to_hub=False,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# 5. START TRAINING
print("Starting SegFormer-B2 Training...")
trainer.train()
