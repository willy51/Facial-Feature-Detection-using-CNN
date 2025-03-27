# 😁🔍 Facial Feature Detection using CNN  
**School Project — Deep Learning on the CelebA Dataset**


## 🎯 Project Overview

This school project focuses on building a deep learning pipeline to **analyze facial features** (e.g., **smiling**, **young**) using the **CelebA dataset**.  
It combines **data preprocessing**, **CNN model training**, **evaluation**, and **visual analysis** — all done with Python, TensorFlow/Keras, and matplotlib.


## 📦 Main Features

- 📁 **Automatic folder creation** for organized training and testing data per facial feature
- 🧠 **CNN models** to detect:
  - **Smiling**
  - **Young**
- 🔍 **Visualization tools** for predictions and model performance
- 📊 **GridSearchCV** to optimize hyperparameters (number of filters)
- 📸 **Image classification** and accuracy reports using confusion matrix

## 🧠 Technologies Used

- Python 🐍
- TensorFlow / Keras
- Pandas, NumPy, Matplotlib
- scikit-learn (for evaluation and GridSearch)

## 🗃 Dataset

- **CelebA Dataset**: [https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- Attributes: 40 facial features per image
- Used attributes:
  - `Smiling`
  - `Young`

## 🛠 How It Works

1. **Data Preprocessing**
   - Load CelebA attribute CSV
   - Create folder structure for 40 attributes
   - Split dataset into 80% training / 20% testing
   - Copy images into presence/absence folders

2. **CNN Model Training**
   - Image size: `64x64`
   - Binary classification for "Smiling" and "Young"
   - Trained for 5 epochs

3. **Model Evaluation**
   - Accuracy plots
   - Confusion matrix
   - Predictions on unseen test images

4. **Hyperparameter Tuning**
   - Uses `GridSearchCV` to find best filter number (5-10)
   - Best for Smiling: `filters = 10` ✅  
   - Best for Young: `filters = 8` ✅

5. **Visual Comparisons**
   - Side-by-side charts comparing AI predictions vs. dataset truths
   - Insight into correlations between smiling and age

## 🖼️ Output Examples

📈 Training & Validation Accuracy (Smiling & Young)  
🧪 Predicting real faces from the test set  
🧾 Classification reports & confusion matrices  
📊 Bar plots comparing model vs. dataset on facial expressions

## 🚀 Getting Started

### 1. Install dependencies:
```bash
pip install tensorflow keras scikit-learn matplotlib pandas numpy
```

### 2. Prepare the dataset:
- Place:
  - `list_attr_celeba.csv` in `/archive/`
  - All images inside `/archive/img_align_celeba/img_align_celeba/`

### 3. Run the script:
```bash
python Projet.py
```

> ⚠️ Make sure you have enough disk space — many images are copied into categorized folders.

## 📁 Folder Structure

```
archive/
├── list_attr_celeba.csv
└── img_align_celeba/
    └── img_align_celeba/
Training/
├── Smiling/
│   ├── Presence_of_feature/
│   └── Absence_of_feature/
├── Young/
Testing/
...
```

## 📚 Authors

- Tristan DESROUSSEAUX  
- Gauthier HORVILLE  
- Charles PRETET  
- Willy ZHENG

Project completed as part of our academic curriculum.  
Special thanks to the instructors and peers for feedback and support 🙏

