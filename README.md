# 🌱 Plant Disease Detection Using Deep Learning

A deep learning-based image classification system for identifying plant diseases from leaf images using transfer learning and convolutional neural networks (CNNs).

## 📌 Project Overview

Plant diseases can significantly reduce crop yield and quality. Early detection is essential for effective disease management. This project uses deep learning models trained on the PlantVillage dataset to automatically classify plant leaf diseases from images.

The project compares multiple state-of-the-art CNN architectures and evaluates their performance on the same dataset.

---

## 📂 Dataset

**Dataset:** PlantVillage Dataset (RGB Color Images)

- 38 plant disease and healthy classes
- Color images only (RGB)
- Grayscale and segmented versions were not used
- Images split into:
  - Training Set
  - Validation Set
  - Test Set

---

## 🧠 Models Implemented

The following transfer learning models were trained and evaluated:

1. ResNet50
2. MobileNetV2
3. DenseNet121
4. InceptionV3

---

## 📊 Model Performance

| Model | Test Accuracy |
|---------|---------:|
| DenseNet121 | 95.44% |
| MobileNetV2 | 94.13% |
| InceptionV3 | 89.51% |
| ResNet50 | 44.38% |

### 🏆 Best Model

**DenseNet121**

- Test Accuracy: **95.44%**
- Weighted F1-Score: **95%**
- Excellent generalization on unseen images

---

## 📈 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Classification Report
- Confusion Matrix

Training performance was monitored using:

- Accuracy Curves
- Loss Curves

---

## 📁 Project Structure

```text
Plant-Disease-Detection/
│
├── notebooks/
│   ├── plant_disease_detection.ipynb
│   └── checking.ipynb
│
├── graphs/
│   ├── densenet121_accuracy.png
│   ├── densenet121_loss.png
│   ├── densenet121_confusion_matrix.png
│   ├── mobilenetv2_accuracy.png
│   ├── mobilenetv2_loss.png
│   ├── mobilenetv2_confusion_matrix.png
│   ├── inceptionv3_accuracy.png
│   ├── inceptionv3_loss.png
│   ├── inceptionv3_confusion_matrix.png
│   ├── resnet50_accuracy.png
│   ├── resnet50_loss.png
│   ├── resnet50_confusion_matrix.png
│   └── final_model_comparison.png
│
├── reports/
│   ├── classification reports
│   ├── training history
│   ├── model comparison
│   └── performance summary
│
├── README.md
└── .gitignore
```

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 🚀 How to Run

### Clone Repository

```bash
git clone https://github.com/aah194/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Notebook

Open:

```text
notebooks/plant_disease_detection.ipynb
```

and execute all cells.

---

## 🔍 Sample Prediction

The final DenseNet121 model can predict plant diseases from uploaded leaf images and return:

- Predicted Disease
- Confidence Score
- Top 5 Predictions

Example:

```text
Disease: Apple___Black_rot
Confidence: 96.34%
```

---

## 📊 Key Findings

- DenseNet121 achieved the highest performance among all tested models.
- MobileNetV2 provided excellent accuracy with lower computational cost.
- InceptionV3 achieved strong results but underperformed compared to DenseNet121.
- ResNet50 struggled to converge effectively on this dataset configuration.

---

## 🔮 Future Improvements

- Deploy using Streamlit or Flask
- Mobile application integration
- Real-time camera-based disease detection
- Support for additional crop species
- Fine-tuning of pretrained models

---

## Live Demo

Deployed Application:
https://plant-disease-detection-1-k1rl.onrender.com

## 👨‍💻 Author

**Sumit Bombale**

MCA (Artificial Intelligence & Data Science)

Deep Learning | Computer Vision | AI Applications in Agriculture