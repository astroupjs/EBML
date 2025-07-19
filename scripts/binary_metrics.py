import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.calibration import calibration_curve
from IPython.display import display

def map_binary_labels(series):
    """
    Map binary class labels to 0/1: 0 = det, 1 = over.
    Accepts: 'det', 'DET', 0 -> 0; 'over', 'OVER', 1 -> 1
    """
    return series.map({'det': 0, 'DET': 0, 0: 0, 'over': 1, 'OVER': 1, 1: 1})

__all__ = [
    'map_binary_labels',
    'print_metrics_table',
    'print_auc_table',
    'plot_roc_curves',
    'plot_reliability_diagram',
    'plot_probability_histograms',
]

def print_metrics_table(df, label_col, pred_cols, class_name):
    print(f"\nMetrics for {class_name} systems:")
    metrics = []
    for pred_col in pred_cols:
        valid = df[pred_col].notnull() & df[label_col].notnull()
        if valid.sum() == 0:
            continue
        y_true = df.loc[valid, label_col]
        y_pred = df.loc[valid, pred_col]
        y_true_bin = map_binary_labels(y_true)
        y_pred_bin = map_binary_labels(y_pred)
        acc = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
        cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0,1])
        metrics.append({
            'Model/Passband': pred_col,
            'Accuracy': round(acc, 2),
            'Precision': round(prec, 2),
            'Recall': round(rec, 2),
            'F1-score': round(f1, 2),
            'TN': int(cm[0,0]),
            'FP': int(cm[0,1]),
            'FN': int(cm[1,0]),
            'TP': int(cm[1,1])
        })
    metrics_df = pd.DataFrame(metrics)
    display(metrics_df)

def print_auc_table(df, label_col, prob_cols, class_name):
    print(f'\nAUC for {class_name} systems:')
    aucs = []
    y_true_bin = map_binary_labels(df[label_col])
    for prob_col in prob_cols:
        valid = df[prob_col].notnull() & df[label_col].notnull()
        if valid.sum() == 0:
            continue
        y_score = df.loc[valid, prob_col]
        try:
            auc_val = roc_auc_score(y_true_bin[valid], y_score)
        except Exception:
            auc_val = None
        aucs.append({'Model/Passband': prob_col, 'AUC': round(auc_val, 3) if auc_val is not None else None})
    auc_df = pd.DataFrame(aucs)
    display(auc_df)

def plot_roc_curves(df, label_col, prob_cols, class_name):
    plt.figure(figsize=(8, 6))
    y_true_bin = map_binary_labels(df[label_col])
    for prob_col in prob_cols:
        valid = df[prob_col].notnull() & df[label_col].notnull()
        if valid.sum() == 0:
            continue
        y_score = df.loc[valid, prob_col]
        fpr, tpr, _ = roc_curve(y_true_bin[valid], y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{prob_col} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves for {class_name} Systems')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

def plot_reliability_diagram(df, label_col, prob_cols, class_name, n_bins=10):
    plt.figure(figsize=(8, 6))
    y_true_bin = map_binary_labels(df[label_col])
    for prob_col in prob_cols:
        valid = df[prob_col].notnull() & df[label_col].notnull()
        if valid.sum() == 0:
            continue
        y_prob = df.loc[valid, prob_col]
        prob_true, prob_pred = calibration_curve(y_true_bin[valid], y_prob, n_bins=n_bins, strategy='uniform')
        plt.plot(prob_pred, prob_true, marker='o', label=prob_col)
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Reliability Diagram for {class_name} Systems')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def plot_probability_histograms(df, label_col, prob_cols, class_name, bins=20):
    y_true_bin = map_binary_labels(df[label_col])
    for prob_col in prob_cols:
        valid = df[prob_col].notnull() & df[label_col].notnull()
        if valid.sum() == 0:
            continue
        y_prob = df.loc[valid, prob_col]
        plt.figure(figsize=(7, 4))
        plt.hist(y_prob[y_true_bin[valid] == 0], bins=bins, alpha=0.6, label='True: Class 0', color='tab:blue', density=True)
        plt.hist(y_prob[y_true_bin[valid] == 1], bins=bins, alpha=0.6, label='True: Class 1', color='tab:orange', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title(f'Predicted Probability Histogram for {prob_col} ({class_name})')
        plt.legend()
        plt.grid(True)
        plt.show()
