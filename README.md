# Byte Force: Precision Perception in Unstructured Terrains
### Offroad Autonomy Segmentation Challenge | GHR 2.0 Hackathon

## # 🚀 Overview
Byte Force provides a robust semantic segmentation solution for Unmanned Ground Vehicles (UGVs) operating in complex desert environments. By transitioning from a standard baseline to a Transformer-based **SegFormer-B2** architecture, we achieved a significant performance leap, reaching a final **Mean IoU of 0.6788**.

## # 📊 Performance Summary
* **Final Mean IoU:** 0.6788 (Peak at Epoch 21)
* **Mean Average Precision (mAP):** 0.314
* **Baseline IoU:** ~0.29
* **Improvement:** +130% over benchmark

## # 🏗️ Architectural Evolution
Our project reflects a multi-iterative approach to scene understanding:
* **SegFormer-B2 (Primary Model):** Leveraged for its global receptive field, which is essential for distinguishing broad "Landscape" context from granular "Ground Clutter."
* **DeepLabv3+ (Comparative Model):** Implemented with Atrous Spatial Pyramid Pooling (ASPP) to capture multi-scale features, specifically helping with depth ambiguity and overlapping features (e.g., Logs behind Bushes).



## # 🛠️ Challenges & Technical Solutions
1. **Severe Class Imbalance ("Sky-Dominance"):** We implemented weighted logic to mathematically penalize minority-class misclassifications, ensuring the model captured critical features like "Logs" and "Rocks."
2. **Domain Shift & Prediction Fragmentation:** To stop "pixel scattering" in noisy desert textures, we utilized heavy hue/saturation randomization during training and morphological smoothing during inference.
3. **Occlusion & Depth Ambiguity:** We utilized the DinoV2 backbone for superior spatial feature extraction, maintaining sharp boundary detection for overlapping off-road features.



## # 📈 Visual Analysis
Our final submission includes a comprehensive performance dashboard:
* **Confusion Matrix:** Validates the model's success in minimizing false positives in the Landscape/DryGrass boundary.
* **Loss Curves:** Demonstrate steady exponential decay, reaching stability at Epoch 18.

`![Performance Dashboard](final_performance_dashboard.png)`

## # 📁 Repository Structure
* `train_segmentation.py`: Full training pipeline for the SegFormer-B2 model.
* `test_segmentation.py`: Master inference script used for mask generation and metrics visualization.
* `config.json`: Configuration for the SegFormer-B2 architecture.
* `requirements.txt`: Environment dependencies.
* `Hackathon_Report.pdf`: In-depth 8-page analysis of methodology and failure cases.
* **Releases Section:** Contains `model.safetensors` (104MB weights) and `vishwansh_ghr_submission.zip` (1,000+ predictions).

## # 🧑‍💻 Team: Byte Force
* **Vishwansh Singh** (24BCE10900) — Lead
* **Abhishek Choudhary** (24BCE11056)
* **Aditya Talreja** (24BCE10891)
* **Akshay Saxena** (24BCE10443)

## # 🔧 Installation & Usage
To reproduce our results, follow these steps:

### ## 1. Install Dependencies
```bash
pip install -r requirements.txt
