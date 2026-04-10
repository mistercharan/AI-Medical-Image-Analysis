import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

def evaluate_model(model, test_gen, threshold=0.5):
    print("\n" + "="*50)
    print("EVALUATING MODEL ON TEST SET")
    print("="*50)
    test_gen.reset()
    y_prob = model.predict(test_gen, verbose=1)
    y_pred = (y_prob > threshold).astype(int).flatten()
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    metrics = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob.flatten(),
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'class_names': class_names
    }
    return metrics

def plot_confusion_matrix(metrics):
    os.makedirs("outputs", exist_ok=True)
    cm = confusion_matrix(metrics['y_true'], metrics['y_pred'])
    class_names = metrics['class_names']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0]
    )
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm, annot=True, fmt='.2%', cmap='Greens',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1]
    )
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.close()
    print("Confusion matrix saved to outputs/confusion_matrix.png")

def plot_roc_curve(metrics):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(
        metrics['fpr'], metrics['tpr'],
        color='darkorange', lw=2,
        label=f"ROC Curve (AUC = {metrics['roc_auc']:.4f})"
    )
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Pneumonia Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('outputs/roc_curve.png', dpi=150)
    plt.close()
    print("ROC curve saved to outputs/roc_curve.png")

def print_evaluation_summary(metrics):
    from sklearn.metrics import (
        accuracy_score, precision_score,
        recall_score, f1_score
    )
    y_true = metrics['y_true']
    y_pred = metrics['y_pred']
    print("\n" + "="*50)
    print("FINAL EVALUATION SUMMARY")
    print("="*50)
    print(f"Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision : {precision_score(y_true, y_pred):.4f}")
    print(f"Recall    : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC   : {metrics['roc_auc']:.4f}")
    print("="*50)