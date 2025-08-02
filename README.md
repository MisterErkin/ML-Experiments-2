
# Income Classification Pipeline (Adult Dataset)

This project is a full-featured machine learning pipeline built for classifying income levels based on the UCI Adult dataset. The pipeline includes preprocessing, model training, sampling techniques to handle class imbalance, LDA dimensionality reduction, and visualization of evaluation metrics.

## 📦 Requirements

Install required libraries using:

```
pip install -r requirements.txt
```

## 🧠 Models Used

The following classifiers were implemented and compared:

- Logistic Regression (`logreg`)
- Random Forest (`random_forest`)
- K-Nearest Neighbors (`knn`)
- Gradient Boosting (`gradient_boost`)

(Note: SVM was removed due to high computation time.)

## 🔁 Sampling Techniques

To address class imbalance, the following sampling methods were evaluated:

- Random Oversampling
- SMOTE (Synthetic Minority Oversampling Technique)
- Random Undersampling
- Tomek Links

## 📉 Dimensionality Reduction

Linear Discriminant Analysis (LDA) was optionally applied for 1D transformation and model evaluation.

## 🧪 Test Configuration

- **Test size**: 0.3  
- **Random state**: 66  
- **Scaler**: MinMaxScaler  
- **Apply LDA**: ✅

## 📊 Results Summary

Below is a summary of model performance under different conditions:

### 🔹 Logistic Regression

- Accuracy: 0.854  
- F1-Score: 0.667  
- ROC-AUC: 0.905  

### 🔹 Random Forest

- Accuracy: 0.855  
- F1-Score: 0.673  
- ROC-AUC: 0.904  

### 🔹 KNN

- Accuracy: 0.825  
- F1-Score: 0.615  
- ROC-AUC: 0.843  

### 🔹 Gradient Boost

- Accuracy: 0.868  
- F1-Score: 0.691  
- ROC-AUC: 0.921  

(Refer to the `/results` folder for confusion matrix visualizations.)

## 📁 Outputs

All evaluation figures are saved to the `results/` directory, including:

- Confusion Matrices (Baseline, Sampling, LDA)
- Income Class Distribution
- Feature Distribution Histograms

## 📷 Sample Visualizations

- `Confusion_Matrix_-_logreg.png`
- `Confusion_Matrix_-_SMOTE_logreg.png`
- `Distribution_of_Numerical_Features.png`

---

✅ This project demonstrates a modular and extensible ML pipeline.  
Feel free to fork, extend or adapt it for your own classification tasks.

