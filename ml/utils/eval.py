from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import pandas as pd
import numpy as np

# def auc_ci_bootstrap(y_true, y_scores, n_bootstraps=1000, alpha=0.95, random_seed=42):
#     rng = np.random.RandomState(random_seed)
#     bootstrapped_scores = []
    
#     y_true = np.array(y_true)
#     y_scores = np.array(y_scores)
    
#     n_samples = len(y_true)

#     for _ in range(n_bootstraps):
#         # 有放回抽样的索引
#         indices = rng.randint(0, n_samples, n_samples)
#         if len(np.unique(y_true[indices])) < 2:
#             # 跳过样本标签单一的情况，无法计算AUC
#             continue
#         score = roc_auc_score(y_true[indices], y_scores[indices])
#         bootstrapped_scores.append(score)

#     sorted_scores = np.array(bootstrapped_scores)
#     sorted_scores.sort()

#     lower_bound = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
#     upper_bound = np.percentile(sorted_scores, (alpha + (1 - alpha) / 2) * 100)

#     auc = roc_auc_score(y_true, y_scores)
#     return auc, lower_bound, upper_bound

# def eval_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     y_pred_prob = model.predict_proba(X_test)[:, 1]
    
#     # 计算基本指标
#     accuracy = accuracy_score(y_test, y_pred)
#     auc = roc_auc_score(y_test, y_pred_prob)
#     precision = precision_score(y_test, y_pred)
#     recall = recall_score(y_test, y_pred)
#     f1 = f1_score(y_test, y_pred)
    
#     auc_ci, lower_bound, upper_bound = auc_ci_bootstrap(y_test, y_pred_prob)
    
#     cm = confusion_matrix(y_test, y_pred)
#     tn, fp, fn, tp = cm.ravel()
#     specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0  # 防止除以0
    
#     return {
#         "AUC": f"{auc:.4f}({lower_bound:.4f}-{upper_bound:.4f})",
#         "Accuracy": f"{accuracy:.4f}",
#         "Precision": f"{precision:.4f}",
#         "Recall": f"{recall:.4f}",
#         "Specificity": f"{specificity:.4f}",
#         "F1": f"{f1:.4f}"
#     }
import numpy as np
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    precision_recall_curve, auc
)

def metric_ci_bootstrap(y_true, y_scores, y_preds, metric_fn, n_bootstraps=1000, alpha=0.95, random_seed=42):
    rng = np.random.RandomState(random_seed)
    bootstrapped_scores = []
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_preds = np.array(y_preds)
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        y_t = y_true[indices]
        y_s = y_scores[indices]
        y_p = y_preds[indices]
        
        if metric_fn.__name__ == 'roc_auc_score':
            if len(np.unique(y_t)) < 2:
                continue
            score = metric_fn(y_t, y_s)
        else:
            score = metric_fn(y_t, y_p)

        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    lower = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper = np.percentile(sorted_scores, (alpha + (1 - alpha) / 2) * 100)
    mean = np.mean(bootstrapped_scores)

    if metric_fn.__name__ == 'roc_auc_score':
        return metric_fn(y_true, y_scores), lower, upper
    return metric_fn(y_true, y_preds), lower, upper


def auprc_ci_bootstrap(y_true, y_scores, n_bootstraps=1000, alpha=0.95, random_seed=42):
    rng = np.random.RandomState(random_seed)
    bootstrapped_scores = []
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    n_samples = len(y_true)

    for _ in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        y_t = y_true[indices]
        y_s = y_scores[indices]

        if len(np.unique(y_t)) < 2:
            continue

        precision, recall, _ = precision_recall_curve(y_t, y_s)
        score = auc(recall, precision)
        bootstrapped_scores.append(score)

    sorted_scores = np.sort(bootstrapped_scores)
    lower = np.percentile(sorted_scores, (1 - alpha) / 2 * 100)
    upper = np.percentile(sorted_scores, (alpha + (1 - alpha) / 2) * 100)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    score = auc(recall, precision)

    return score, lower, upper


def eval_model(model, X_test, y_test, n_bootstraps=1000):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    auc_mean, auc_lower, auc_upper = metric_ci_bootstrap(
        y_test, y_pred_prob, y_pred, roc_auc_score, n_bootstraps
    )

    auprc_mean, auprc_lower, auprc_upper = auprc_ci_bootstrap(
        y_test, y_pred_prob, n_bootstraps
    )

    acc_mean, acc_lower, acc_upper = metric_ci_bootstrap(
        y_test, y_pred_prob, y_pred, accuracy_score, n_bootstraps
    )

    prec_mean, prec_lower, prec_upper = metric_ci_bootstrap(
        y_test, y_pred_prob, y_pred, precision_score, n_bootstraps
    )

    rec_mean, rec_lower, rec_upper = metric_ci_bootstrap(
        y_test, y_pred_prob, y_pred, recall_score, n_bootstraps
    )

    f1_mean, f1_lower, f1_upper = metric_ci_bootstrap(
        y_test, y_pred_prob, y_pred, f1_score, n_bootstraps
    )

    # specificity
    def specificity_fn(y_t, y_p):
        cm = confusion_matrix(y_t, y_p)
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    spec_mean, spec_lower, spec_upper = metric_ci_bootstrap(
        y_test, y_pred_prob, y_pred, specificity_fn, n_bootstraps
    )

    return {
        "AUC": f"{auc_mean:.3f} ({auc_lower:.3f}-{auc_upper:.3f})",
        "AUPRC": f"{auprc_mean:.3f} ({auprc_lower:.3f}-{auprc_upper:.3f})",
        "Accuracy": f"{acc_mean:.3f} ({acc_lower:.3f}-{acc_upper:.3f})",
        "Precision": f"{prec_mean:.3f} ({prec_lower:.3f}-{prec_upper:.3f})",
        "Recall": f"{rec_mean:.3f} ({rec_lower:.3f}-{rec_upper:.3f})",
        "Specificity": f"{spec_mean:.3f} ({spec_lower:.3f}-{spec_upper:.3f})",
        "F1": f"{f1_mean:.3f} ({f1_lower:.3f}-{f1_upper:.3f})"
    }

