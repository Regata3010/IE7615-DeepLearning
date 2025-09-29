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
pip install torch torchvision pandas numpy matplotlib scikit-learn tqdm Pillow
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

## 💡 Technical Insights

### Why Transfer Learning Works Better
1. **Pre-trained Features**: ImageNet features transfer well to face recognition
2. **Less Overfitting**: Frozen layers act as regularization
3. **Faster Convergence**: Only training classifier layer
4. **Better Generalization**: Learned features are more robust

### SimpleCNN Limitations
- Limited depth for complex feature extraction
- Prone to overfitting with small dataset
- Requires more data for better performance

## 🔮 Future Improvements

1. **Data Augmentation**: Add rotation, color jitter, and cutout
2. **Fine-tuning**: Unfreeze last few layers of pre-trained models
3. **Ensemble Methods**: Combine predictions from multiple models
4. **Learning Rate Scheduling**: Implement cosine annealing or ReduceLROnPlateau
5. **Larger Dataset**: Use full CelebA dataset for better generalization

## 👤 Author

**[Arav Pandey]**  
Master's Student, Data Analytics Engineering  
Northeastern University

## 📝 License

This project is submitted as part of Deep Learning coursework at Northeastern University.

---

*Last Updated: September 2024*