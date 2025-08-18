# Emotion Detection with CNN  

This project uses a **Convolutional Neural Network (CNN)** to detect facial emotions in real-time from a webcam feed.  
The model is trained on images of faces categorized into seven emotions:  

- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

The system detects faces using OpenCV’s Haar Cascades and predicts emotions using a Keras CNN model.  

---

## 📌 Features
- Real-time facial emotion recognition via webcam  
- CNN architecture with Conv2D, BatchNormalization, MaxPooling, Dropout layers  
- Handles class imbalance with weighted loss  
- Model can be saved and reloaded for inference (`Emotional_detection.h5`)  

---

## 📥 Dataset

The model is trained on the **[Kaggle Facial Expression Recognition Dataset (FER2013)](https://www.kaggle.com/datasets/msambare/fer2013)**.  
- Dataset contains grayscale 48x48 images  
- Organize training images into `train/<emotion>` folders and validation images into `test/<emotion>` folders  
