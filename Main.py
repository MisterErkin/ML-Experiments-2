import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import Tk, filedialog
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import time
import os

# -------------------------- CONFIG CLASS --------------------------
class PipelineConfig:
    def __init__(self,
                 file_path=None,
                 test_size=0.2,
                 random_state=42,
                 scaler_type='minmax',
                 model_type='logreg',
                 sampling_choice='all',
                 apply_lda=True):

        self.file_path = file_path or self.ask_file_path()
        self.test_size = self.get_user_input("Test size (0.0 - 1.0)", test_size, float)
        self.random_state = self.get_user_input("Random state (int)", random_state, int)
        self.scaler_type = self.get_user_input("Scaler type ('minmax' or 'standard')", scaler_type, str).lower()
        self.model_type = self.get_user_input("Model type ('logreg', 'random_forest', 'knn', 'gradient_boost', or 'all')", model_type, str).lower()
        self.sampling_choice = self.get_user_input("Sampling method ('random', 'smote', 'undersample', 'tomek', or 'all')", sampling_choice, str).lower()
        self.apply_lda = self.get_user_input("Apply LDA? (True/False)", apply_lda, lambda x: x.lower() == 'true' if isinstance(x, str) else bool(x))

    def ask_file_path(self):
        Tk().withdraw()
        file_path = filedialog.askopenfilename(
            title="Select your dataset (CSV file)",
            filetypes=[("CSV files", "*.csv")]
        )
        if not file_path:
            raise FileNotFoundError("No file selected. Exiting.")
        return file_path

    def get_user_input(self, prompt, default, cast_type):
        try:
            value = input(f"{prompt} [Default: {default}]: ").strip()
            return cast_type(value) if value else default
        except:
            print(f"Invalid input. Using default: {default}")
            return default


# -------------------------- MODEL MAPPING --------------------------
def get_model(model_name, random_state=42):
    models = {
        "logreg": LogisticRegression(max_iter=1000, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "gradient_boost": GradientBoostingClassifier(random_state=random_state)
    }
    return models.get(model_name, LogisticRegression(max_iter=1000, random_state=random_state))

# -------------------------- DATA LOADER --------------------------
def load_and_preprocess_data(config: PipelineConfig):
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
        "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
        "hours_per_week", "native_country", "income"
    ]

    try:
        with open(config.file_path, 'r', encoding='utf-8') as f:
            sample = f.read(2048)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample)
            delimiter = dialect.delimiter

        df = pd.read_csv(config.file_path, header=None, delimiter=delimiter, skipinitialspace=True)
        if df.shape[1] != len(column_names):
            raise ValueError(f"Expected 15 columns, but got {df.shape[1]}. Detected delimiter: '{delimiter}'")
        df.columns = column_names

    except Exception as e:
        print(f"❌ Error reading dataset: {e}")
        raise

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df["income"] = df["income"].map({'<=50K': 0, '>50K': 1})

    categorical_cols = ["workclass", "education", "marital_status", "occupation",
                        "relationship", "race", "sex", "native_country"]
    numerical_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    scaler = MinMaxScaler(feature_range=(-1, 1)) if config.scaler_type == 'minmax' else StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop(columns=["income"])
    y = df["income"]

    return train_test_split(X, y, test_size=config.test_size, stratify=y, random_state=config.random_state), df

# -------------------------- VISUALIZATION --------------------------
def visualize_distribution(df):
    os.makedirs("results", exist_ok=True)
    if 'income' in df.columns and not df['income'].empty:
        plt.figure(figsize=(6, 4))
        df['income'].value_counts().plot(kind='bar', colormap='viridis')
        title = "Income Class Distribution"
        plt.title(title)
        plt.xlabel("Income")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        filename = title.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "") + ".png"
        plt.savefig(os.path.join("results", filename))
        plt.show()

    numeric_cols = df.select_dtypes(include=['number']).columns
    if not numeric_cols.empty:
        df[numeric_cols].hist(figsize=(10, 6), bins=20, edgecolor='black')
        plt.suptitle("Distribution of Numerical Features", fontsize=14)
        plt.savefig(os.path.join("results", "Distribution_of_Numerical_Features.png"))
        plt.show()

# -------------------------- MODEL TRAINING --------------------------
def train_model(X_train, y_train, X_test, y_test, model, title="Confusion Matrix"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_pred_proba)
    }

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["<=50K", ">50K"])
    disp.plot(cmap=plt.cm.Blues)
    os.makedirs("results", exist_ok=True)
    filename = title.replace(" ", "_").replace("(", "").replace(")", "").replace(":", "") + ".png"
    plt.title(title)
    plt.savefig(os.path.join("results", filename))
    plt.show()

    return metrics

# -------------------------- SAMPLING COMPARISON --------------------------
def compare_sampling_methods(X_train, y_train, X_test, y_test, model_type, method_filter='all'):
    sampling_methods = {
        "Random Oversampling": RandomOverSampler(random_state=42),
        "SMOTE": SMOTE(random_state=42),
        "Random Undersampling": RandomUnderSampler(random_state=42),
        "Tomek Links": TomekLinks()
    }
    results = {}
    for name, sampler in sampling_methods.items():
        if method_filter != 'all' and method_filter.lower() not in name.lower():
            continue
        X_res, y_res = sampler.fit_resample(X_train, y_train)
        model = get_model(model_type)
        metrics = train_model(X_res, y_res, X_test, y_test, model, title=f"Confusion Matrix - {name} ({model_type})")
        results[name] = metrics
    return pd.DataFrame(results)

# -------------------------- MAIN PIPELINE --------------------------
def run_pipeline(config: PipelineConfig):
    print("📦 Loading and Preprocessing Data...")
    (X_train, X_test, y_train, y_test), raw_df = load_and_preprocess_data(config)

    print("📊 Visualizing Dataset Distribution...")
    visualize_distribution(raw_df)

    models_to_run = ['logreg', 'random_forest', 'knn', 'gradient_boost'] if config.model_type == 'all' else [config.model_type]

    for m in models_to_run:
        model_start = time.time()
        print(f"\n🚀 Training Baseline Model ({m})...")
        model = get_model(m, random_state=config.random_state)
        baseline_metrics = train_model(X_train, y_train, X_test, y_test, model, title=f"Confusion Matrix - {m}")
        print(f"Baseline Results ({m}):", baseline_metrics)
        print(f"⏱️ Baseline Model ({m}) took {time.time() - model_start:.2f} seconds")

        print("\n🔁 Comparing Sampling Techniques...")
        sampling_start = time.time()
        sampling_df = compare_sampling_methods(X_train, y_train, X_test, y_test, m, config.sampling_choice)
        print(f"Sampling Results for {m}:")
        print(sampling_df)
        print(f"⏱️ Sampling Techniques ({m}) took {time.time() - sampling_start:.2f} seconds")

        if config.apply_lda:
            print("\n📉 Applying LDA for Dimensionality Reduction...")
            lda_start = time.time()
            lda = LDA(n_components=1)
            X_train_lda = lda.fit_transform(X_train, y_train)
            X_test_lda = lda.transform(X_test)
            lda_model = get_model(m)
            lda_metrics = train_model(X_train_lda, y_train, X_test_lda, y_test, lda_model, title=f"Confusion Matrix - LDA ({m})")
            print(f"LDA Model Results ({m}):", lda_metrics)
            print(f"⏱️ LDA ({m}) took {time.time() - lda_start:.2f} seconds")

# -------------------------- ENTRY POINT --------------------------
if __name__ == "__main__":
    config = PipelineConfig()
    run_pipeline(config)