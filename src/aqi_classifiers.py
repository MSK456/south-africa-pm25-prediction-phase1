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
    demo_list = []
    demo_input = pd.DataFrame([[125.55]], columns=["Median_PM25"])
    
    # Two-column layout (3 rows, 2 columns) for high readability
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 9.6), dpi=300)
    axes = axes.flatten()
    
    for idx, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred) * 100.0
        sens, spec = calculate_macro_sensitivity_specificity(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        demo_pred = int(model.predict(demo_input)[0])
        demo_list.append({
            "Model": name,
            "Input Median_PM25": 125.55,
            "Predicted AQI Class": demo_pred,
            "AQI Category": "Severe" if demo_pred == 2 else str(demo_pred)
        })
        
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
        print(f"  Demo prediction for 125.55: Class {demo_pred}")
        print("  Classification Report:")
        print(classification_report(y_test, y_pred, digits=4, zero_division=0))
        
        # Plot confusion matrix with explicit 0-5 class labels and readable numbers
        cm = confusion_matrix(y_test, y_pred, labels=list(range(6)))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            cbar=False,
            annot_kws={"size": 8.0, "weight": "bold"},
            xticklabels=[f"Class {i}" for i in range(6)],
            yticklabels=[f"Class {i}" for i in range(6)],
        )
        axes[idx].set_title(f"{name} Confusion Matrix", fontsize=10, fontweight="bold")
        axes[idx].set_xlabel("Predicted AQI Class (0 to 5)", fontsize=8.5)
        axes[idx].set_ylabel("Actual AQI Class (0 to 5)", fontsize=8.5)
        axes[idx].tick_params(axis="both", labelsize=7.5)

    # Sixth subplot: Class guide legend
    axes[5].axis("off")
    axes[5].text(
        0.15, 0.50,
        "AQI Class Definitions:\n\n"
        "• Class 0: Good (0 – 50)\n"
        "• Class 1: Moderate (51 – 100)\n"
        "• Class 2: Severe (101 – 150)\n"
        "• Class 3: Unhealthy (151 – 200)\n"
        "• Class 4: Very Unhealthy (201 – 300)\n"
        "• Class 5: Hazardous (301+)",
        fontsize=9,
        va="center",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#ced4da", linewidth=1.2)
    )
        
    plt.tight_layout()
    cm_path = FIGURES_DIR / "figure14_classification_confusion_matrices.png"
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nSaved confusion matrices to {cm_path}")
    
    # Save metrics table
    res_df = pd.DataFrame(metrics_list)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(TABLES_DIR / "aqi_classifier_metrics.csv", index=False)
    
    tex_df = res_df[["Model", "Accuracy", "Sensitivity", "Specificity", "MAE", "MSE", "RMSE"]].copy()
    save_latex_tabular(tex_df, TABLES_DIR / "aqi_classifier_metrics.tex")
    
    # Save demo predictions table
    demo_df = pd.DataFrame(demo_list)
    demo_df.to_csv(TABLES_DIR / "aqi_demo_predictions.csv", index=False)
    save_latex_tabular(demo_df, TABLES_DIR / "aqi_demo_predictions.tex")
    
    return res_df, demo_df

if __name__ == "__main__":
    run_aqi_classification()
