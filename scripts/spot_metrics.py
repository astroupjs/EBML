import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, auc
)
from sklearn.calibration import calibration_curve

def print_metrics_table(df, label_col, pred_cols, class_name):
    print(f"\nMetrics for {class_name} systems:")
    metrics = []
    for pred_col in pred_cols:
        valid = df[pred_col].notnull() & df[label_col].notnull()
        if valid.sum() == 0:
            continue
        y_true = df.loc[valid, label_col].map({'n': 0, 's': 1, 'N': 0, 'S': 1, 0: 0, 1: 1})
        y_pred = df.loc[valid, pred_col].map({'n': 0, 's': 1, 'N': 0, 'S': 1, 0: 0, 1: 1})
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
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
    for prob_col in prob_cols:
        valid = df[prob_col].notnull() & df[label_col].notnull()
        if valid.sum() == 0:
            continue
        y_true = df.loc[valid, label_col].map({'n': 0, 's': 1, 'N': 0, 'S': 1, 0: 0, 1: 1})
        y_score = df.loc[valid, prob_col]
        try:
            auc_val = roc_auc_score(y_true, y_score)
        except Exception:
            auc_val = None
        aucs.append({'Model/Passband': prob_col, 'AUC': round(auc_val, 3) if auc_val is not None else None})
    auc_df = pd.DataFrame(aucs)
    display(auc_df)

def plot_roc_curves(df, label_col, prob_cols, class_name):
    plt.figure(figsize=(8, 6))
    for prob_col in prob_cols:
        valid = df[prob_col].notnull() & df[label_col].notnull()
        if valid.sum() == 0:
            continue
        y_true = df.loc[valid, label_col].map({'n': 0, 's': 1, 'N': 0, 'S': 1, 0: 0, 1: 1})
        y_score = df.loc[valid, prob_col]
        fpr, tpr, _ = roc_curve(y_true, y_score)
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
    for prob_col in prob_cols:
        valid = df[prob_col].notnull() & df[label_col].notnull()
        if valid.sum() == 0:
            continue
        y_true = df.loc[valid, label_col].map({'n': 0, 's': 1, 'N': 0, 'S': 1, 0: 0, 1: 1})
        y_prob = df.loc[valid, prob_col]
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
        plt.plot(prob_pred, prob_true, marker='o', label=prob_col)
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Reliability Diagram for {class_name} Systems')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

def plot_probability_histograms(df, label_col, prob_cols, class_name, bins=20):
    for prob_col in prob_cols:
        valid = df[prob_col].notnull() & df[label_col].notnull()
        if valid.sum() == 0:
            continue
        y_true = df.loc[valid, label_col].map({'n': 0, 's': 1, 'N': 0, 'S': 1, 0: 0, 1: 1})
        y_prob = df.loc[valid, prob_col]
        plt.figure(figsize=(7, 4))
        plt.hist(y_prob[y_true == 0], bins=bins, alpha=0.6, label='True: No Spot', color='tab:blue', density=True)
        plt.hist(y_prob[y_true == 1], bins=bins, alpha=0.6, label='True: Spot', color='tab:orange', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title(f'Predicted Probability Histogram for {prob_col} ({class_name})')
        plt.legend()
        plt.grid(True)
        plt.show()
