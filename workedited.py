import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, 
    confusion_matrix, roc_curve, ConfusionMatrixDisplay
)
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Load the dataset
file_path = r"C:\Users\M_Erk\OneDrive\Masaüstü\Denemeler\ML-Experiments-2/adult.data.csv"
df = pd.read_csv(file_path, header=None, delimiter=",", skipinitialspace=True)

# Assign column names
column_names = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]

# Check if 'income' column exists and is non-empty before plotting
if 'income' in df.columns and not df['income'].empty:
    plt.figure(figsize=(6, 4))
    df['income'].value_counts().plot(kind='bar', colormap='viridis')  # Use colormap to avoid color mismatch
    plt.title("Income Class Distribution")
    plt.xlabel("Income")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.show()
else:
    print("Warning: 'income' column is missing or empty.")

# Check if df has numeric columns before plotting histogram
numeric_cols = df.select_dtypes(include=['number']).columns

if not numeric_cols.empty:
    df[numeric_cols].hist(figsize=(6, 4), bins=20, edgecolor='black')
    plt.suptitle("Distribution of Numerical Features", fontsize=14)
    plt.show()
else:
    print("Warning: No numerical columns found in the DataFrame.")

# Trim spaces and map income to binary values
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
df["income"] = df["income"].map({'<=50K': 0, '>50K': 1})

# Identify categorical and numerical columns
categorical_cols = ["workclass", "education", "marital_status", "occupation", 
                    "relationship", "race", "sex", "native_country"]
numerical_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

# One-hot encoding categorical variables
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Normalize numerical features
scaler = MinMaxScaler(feature_range=(-1, 1))
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Split dataset into features (X) and target (y)
X = df.drop(columns=["income"])
y = df["income"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train logistic regression model
def train_model(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred_proba)
    }
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50K", ">50K"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    
    return metrics

# Baseline Model Results
baseline_results = train_model(X_train, y_train, X_test, y_test)
print("Baseline Model Results:")
print(baseline_results)

# Apply Sampling Methods and Compare Results
sampling_methods = {
    "Random Oversampling": RandomOverSampler(random_state=42),
    "SMOTE": SMOTE(random_state=42),
    "Random Undersampling": RandomUnderSampler(random_state=42),
    "Tomek Links": TomekLinks()
}

sampling_results = {}

for method, sampler in sampling_methods.items():
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    sampling_results[method] = train_model(X_resampled, y_resampled, X_test, y_test)

# Display all resampling method results
sampling_results_df = pd.DataFrame(sampling_results)
print("Sampling Method Results:")
print(sampling_results_df)

# Apply LDA for Dimensionality Reduction
lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

lda_results = train_model(X_train_lda, y_train, X_test_lda, y_test)
print("LDA Model Results:")
print(lda_results)
