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

- **Dataset Used**: [CelebA Dataset on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) 📥  
- 200,000+ celebrity face images annotated with 40 facial attributes.
- Used attributes in this project:
  - `Smiling`
  - `Young`

## 🛠 How It Works

1. **Data Preprocessing**
2. **CNN Training for Smiling & Young**
3. **Evaluation & Visualization**
4. **Hyperparameter Optimization with GridSearchCV**
5. **Distribution Comparison (AI vs. Real Labels)**

## 🚀 Getting Started

### 1. Download the dataset:
👉 [Kaggle - CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)

Unzip and place:
- `list_attr_celeba.csv` in `/archive/`
- All images inside `/archive/img_align_celeba/img_align_celeba/`

### 2. Install dependencies:
```bash
pip install tensorflow keras scikit-learn matplotlib pandas numpy
```

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
