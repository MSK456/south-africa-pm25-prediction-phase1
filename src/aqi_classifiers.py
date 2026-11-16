import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from config import (
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    SEED,
    TABLES_DIR,
    save_latex_tabular,
)

def assign_aqi_class(pm25):
    if pm25 < 51.0:
        return 0  # Good
    elif pm25 < 101.0:
        return 1  # Moderate
    elif pm25 < 151.0:
        return 2  # Severe
    elif pm25 < 201.0:
        return 3  # Unhealthy
    elif pm25 < 301.0:
        return 4  # Very Unhealthy
    else:
        return 5  # Hazardous

def calculate_macro_sensitivity_specificity(y_true, y_pred, labels=None):
    if labels is None:
        labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total_samples = cm.sum()
    
    sensitivities = []
    specificities = []
    
    for i in range(len(labels)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total_samples - (tp + fn + fp)
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        sensitivities.append(sens)
        specificities.append(spec)
        
    return float(np.mean(sensitivities)), float(np.mean(specificities))

def run_aqi_classification():
    print("\n=== AQI Status Classification Experiment ===")
    
    clean_path = PROCESSED_DATA_DIR / "pm25_clean.csv"
    df = pd.read_csv(clean_path)
    
    # Assign target labels from Median_PM25
    df["AQI_Class"] = df["Median_PM25"].apply(assign_aqi_class)
    
    X = df[["Median_PM25"]]
    y = df["AQI_Class"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, shuffle=True
    )
    
    print(f"Dataset size: {len(df)} rows. Train: {len(X_train)}, Test: {len(X_test)}")
    print("Test set class distribution:\n", y_test.value_counts().sort_index())
    
    models = {
        "Random Forest": RandomForestClassifier(random_state=SEED),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "SVM": SVC(random_state=SEED),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
    }
    
    metrics_list = []

    
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
    
    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred) * 100.0
        sens, spec = calculate_macro_sensitivity_specificity(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        

        
        metrics_list.append({
            "Model": name,
            "Accuracy": f"{acc:.3f}%",
            "Sensitivity": f"{sens:.4f}",
            "Specificity": f"{spec:.4f}",
            "MAE": f"{mae:.6f}",
            "MSE": f"{mse:.6f}",
            "RMSE": f"{rmse:.6f}",
            "raw_acc": acc,
        })
        
        print(f"\nModel: {name}")
        print(f"  Accuracy: {acc:.3f}%, Sensitivity: {sens:.4f}, Specificity: {spec:.4f}")
        print(f"  MAE: {mae:.6f}, MSE: {mse:.6f}, RMSE: {rmse:.6f}")

        print("  Classification Report:")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx], cbar=False)
        axes[idx].set_title(f"{name} Confusion Matrix", fontsize=11, fontweight="bold")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
        
    plt.tight_layout()
    cm_path = FIGURES_DIR / "figure14_classification_confusion_matrices.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"\nSaved confusion matrices to {cm_path}")
    
    # Save metrics table
    res_df = pd.DataFrame(metrics_list)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(TABLES_DIR / "aqi_classifier_metrics.csv", index=False)
    
    tex_df = res_df[["Model", "Accuracy", "Sensitivity", "Specificity", "MAE", "MSE", "RMSE"]].copy()
    save_latex_tabular(tex_df, TABLES_DIR / "aqi_classifier_metrics.tex")
    
    return res_df

if __name__ == "__main__":
    run_aqi_classification()
