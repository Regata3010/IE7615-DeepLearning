# CelebA Celebrity Classification with Deep Learning

## 📋 Project Overview

This project implements celebrity face classification on the CelebA dataset using both custom CNN architecture and state-of-the-art transfer learning approaches. The goal is to classify images into 20 celebrity categories with a specific focus on celebrity ID 10173.

## 📊 Dataset Information

### Dataset Size Proof
- **Original CelebA Dataset**: 202,599 images of 10,177 celebrities
- **Subset Used**: 638 images from top 20 celebrities
- **Celebrity 10173**: 30 images (included as specified in requirements)
- **Image Dimensions**: Original 178×218, processed to 128×128 (CNN) and 224×224 (Transfer Learning)

### Data Split
| Split | Images | Percentage |
|-------|--------|------------|
| Training | 408 | 64% |
| Validation | 102 | 16% |
| Test | 128 | 20% |

## 🏗️ Models Implemented

### 1. Custom SimpleCNN
- **Architecture**: 3 Conv layers + 2 FC layers
- **Parameters**: ~2.3M trainable parameters
- **Training Time**: 28 seconds (10 epochs)

### 2. Transfer Learning Models
All models use ImageNet pre-trained weights with frozen backbones:
- **ResNet18**: 11.7M parameters (20K trainable)
- **ResNet50**: 25.6M parameters (40K trainable)
- **MobileNet V2**: 3.5M parameters (20K trainable)
- **EfficientNet B0**: 5.3M parameters (20K trainable)

## 📈 Performance Results

### Model Comparison

| Model | Validation Accuracy | Test Accuracy | Celebrity 10173 Accuracy | Training Time |
|-------|-------------------|---------------|------------------------|---------------|
| **SimpleCNN** | 46.08% | 41.41% | 50.00% (3/6) | 28s |
| **ResNet18** | 65.69% | 62.50% | 66.67% (4/6) | 45s |
| **ResNet50** | 68.63% | 64.84% | 83.33% (5/6) | 72s |
| **MobileNet V2** | 61.76% | 57.81% | 50.00% (3/6) | 38s |
| **EfficientNet B0** | **70.59%** | **67.97%** | 66.67% (4/6) | 55s |

### Key Findings
- ✅ **Best Model**: EfficientNet B0 with 67.97% test accuracy
- ✅ **Improvement**: 26.56% improvement over baseline SimpleCNN
- ✅ **Celebrity 10173**: Best performance with ResNet50 (83.33%)
- ✅ **Efficiency**: MobileNet V2 offers best speed-accuracy trade-off

## 🚀 How to Run

### Prerequisites
```bash
pip install requirements.txt
```

## 📉 Training Curves

### SimpleCNN Training Progress
- Training accuracy reached 85.54% (overfitting)
- Validation accuracy plateaued at 46.08%
- Clear overfitting after epoch 6

### Transfer Learning Results
- All models converged faster than SimpleCNN
- Less overfitting due to pre-trained features
- EfficientNet B0 showed most stable training

## 🎯 Assignment Requirements Fulfilled

1. ✅ **Multiple Deep Learning Models**: Implemented 5 models (1 custom + 4 transfer learning)
2. ✅ **Performance Documentation**: All models tested and compared
3. ✅ **Celebrity 10173 Focus**: Extracted subset and reported specific accuracy
4. ✅ **Best Model Selection**: EfficientNet B0 identified as optimal

---

# Part 2: YOLOv8 Object Detection for Multiple Celebrities

## 📋 Object Detection Overview

Building upon the classification task, Part 2 implements a YOLOv8-based object detection system capable of identifying and localizing multiple celebrities within a single image. This addresses the real-world scenario of detecting multiple faces in group photos or scenes.

## 🎯 Task Requirements

- ✅ Concatenate images of different celebrities to form training and testing datasets
- ✅ Data augmentation to generate more diverse training samples
- ✅ Custom train YOLOv8 network for celebrity detection
- ✅ Multi-celebrity detection with IDs and bounding box locations

## 📊 Object Detection Dataset

### Dataset Generation Process
- **Concatenation Strategy**: Combined 2-6 celebrity faces per image
- **Grid Layout**: Dynamic 2×2 or 3×3 grid based on celebrity count
- **Celebrity 10173 Frequency**: Appears in 70% of training images for enhanced performance
- **Augmentation Applied**:
  - Random positioning within grid cells
  - Scale variation (50-90% of cell size)
  - Horizontal flipping (50% probability)
  - Brightness adjustment (±30%)
  - Rotation (±10 degrees)
  - YOLOv8 Mosaic (80% probability)
  - YOLOv8 MixUp (10% probability)

### Dataset Statistics
| Split | Images | Total Celebrity Instances |
|-------|--------|--------------------------|
| Training | 300 | ~1,200 |
| Validation | 50 | ~200 |
| Test | 30 | 119 |

## 🏗️ YOLOv8 Model Configuration

- **Model Variant**: YOLOv8 nano (fastest, suitable for face detection)
- **Input Size**: 640×640 pixels
- **Classes**: 20 celebrity identities
- **Training Epochs**: 30
- **Batch Size**: 8
- **Optimizer**: AdamW with lr=0.001
- **Loss Weights**: Box=7.5, Classification=0.5

## 🎯 Detection Performance Results

### Overall Metrics
- **mAP@0.5**: 0.880 (88.0%)
- **mAP@0.5-0.95**: 0.880 (88.0%)
- **Precision**: 0.773 (77.3%)
- **Recall**: 0.820 (82.0%)
- **Inference Speed**: ~100ms per image (10 FPS)

### Per-Celebrity Performance
| Celebrity ID | Instances | Precision | Recall | mAP@0.5 | mAP@0.5-0.95 |
|-------------|-----------|-----------|---------|---------|--------------|
| **10173** 🌟 | 24 | 0.964 | 1.000 | 0.995 | 0.995 |
| 3227 | 12 | 0.973 | 1.000 | 0.995 | 0.995 |
| 2070 | 2 | 0.712 | 1.000 | 0.995 | 0.995 |
| 3699 | 2 | 0.880 | 1.000 | 0.995 | 0.995 |
| 3782 | 3 | 0.695 | 1.000 | 0.995 | 0.995 |
| 4740 | 2 | 0.626 | 1.000 | 0.995 | 0.995 |
| 4978 | 2 | 0.779 | 1.000 | 0.995 | 0.995 |
| 8968 | 10 | 1.000 | 0.765 | 0.977 | 0.977 |
| 6568 | 6 | 0.901 | 0.833 | 0.955 | 0.955 |
| 3745 | 5 | 0.680 | 1.000 | 0.938 | 0.938 |
| 9152 | 7 | 1.000 | 0.713 | 0.918 | 0.918 |
| 2820 | 6 | 0.937 | 0.833 | 0.915 | 0.915 |
| 4262 | 4 | 0.643 | 1.000 | 0.912 | 0.912 |
| 2114 | 5 | 0.807 | 0.840 | 0.895 | 0.895 |
| 9840 | 5 | 0.870 | 0.800 | 0.872 | 0.872 |
| 1757 | 6 | 0.765 | 0.500 | 0.859 | 0.859 |
| 4126 | 6 | 0.881 | 0.500 | 0.813 | 0.813 |
| 9256 | 5 | 0.647 | 0.800 | 0.762 | 0.762 |
| 9915 | 5 | 0.426 | 0.311 | 0.582 | 0.582 |
| 4887 | 2 | 0.272 | 0.500 | 0.236 | 0.236 |

### Celebrity 10173 Performance Highlights 🏆
- **Perfect Recall**: 100% - Never misses celebrity 10173
- **High Precision**: 96.4% - Rarely makes false positives
- **Best mAP@0.5**: 99.5% - Near-perfect detection accuracy
- **Test Instances**: 24 successful detections
- **Ranking**: Top performer among all 20 celebrities

## 📈 Key Achievements

### Part 1 vs Part 2 Comparison
| Metric | Classification (Part 1) | Object Detection (Part 2) |
|--------|------------------------|--------------------------|
| Task | Single celebrity per image | Multiple celebrities per image |
| Best Model | EfficientNet B0 | YOLOv8 nano |
| Celebrity 10173 Accuracy | 83.33% | 99.5% mAP |
| Overall Performance | 67.97% accuracy | 88.0% mAP |
| Inference Speed | ~50ms | ~100ms |

### Technical Innovations
- **Smart Data Strategy**: Ensuring celebrity 10173 appears in 70% of training data
- **Effective Augmentation**: Combined manual and YOLOv8 built-in augmentations
- **Balanced Dataset**: Despite limited data, achieved robust detection
- **Real-time Capable**: 10 FPS inference speed suitable for applications

## 🚀 How to Run

### Prerequisites
```bash
pip install ultralytics opencv-python matplotlib tqdm pyyaml pandas
```

### Training YOLOv8
```python
from ultralytics import YOLO

# Load and train model
model = YOLO('yolov8n.pt')
model.train(data='dataset.yaml', epochs=30, imgsz=640, batch=8)
```

## 💡 Technical Insights

### Why YOLOv8 Excels
- **Single-stage Detection**: Faster than two-stage detectors
- **Anchor-free**: Better for varying face sizes
- **Built-in Augmentation**: Mosaic and MixUp improve generalization
- **Decoupled Head**: Separate classification and localization branches

### Success Factors for Celebrity 10173
- **Data Balance**: 70% appearance rate in training
- **Consistent Features**: CelebA provides aligned faces
- **Sufficient Instances**: 24 test instances for reliable metrics
- **Transfer Learning**: YOLOv8 pre-trained weights accelerate convergence

## 🎯 Assignment Requirements Fulfilled

### Part 1 ✅
- Multiple deep learning models tested
- Performance documented for each model
- Celebrity 10173 subset extracted
- Best model identified (EfficientNet B0)

### Part 2 ✅
- Concatenated images with multiple celebrities
- Comprehensive data augmentation applied
- YOLOv8 successfully trained
- Multi-celebrity detection with IDs and locations achieved
- **Celebrity 10173 detection: 99.5% mAP with 100% recall**

## 👤 Author

**Arav Pandey**  
Master's Student, Data Analytics Engineering  
Northeastern University

## 📝 License

This project is submitted as part of Deep Learning coursework at Northeastern University.

---

*Last Updated: October 2024*