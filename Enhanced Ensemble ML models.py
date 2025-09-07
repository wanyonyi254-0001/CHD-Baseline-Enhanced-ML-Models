# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 22:07:50 2025

@author: iamwa
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 22 17:05:58 2025
Updated to include scalability metrics (training time, inference latency, model size)
and plots for ensemble & base models.

@author: iamwa
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time, joblib, os, psutil
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, learning_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from scipy.stats import mode
import shap
import lime.lime_tabular

# ----------------------------------
# 1) Load & prepare data
# ----------------------------------
df = pd.read_csv('C:/Users/iamwa/Desktop/CHD data.csv')
y = df['HeartDiseaseorAttack']
X = df.drop(columns=['HeartDiseaseorAttack'])

print("Class Distribution:\n", y.value_counts())

# SMOTE to balance
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Downsample to manageable subset (keeps stratification)
sss = StratifiedShuffleSplit(n_splits=1, train_size=9787, random_state=42)
for train_idx, _ in sss.split(X_resampled, y_resampled):
    X_sample = X_resampled.iloc[train_idx]
    y_sample = y_resampled.iloc[train_idx]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample, test_size=0.2, stratify=y_sample, random_state=42
)

# ----------------------------------
# 2) Define “base” models (as per your naming)
# ----------------------------------
anrdt = RandomForestClassifier(n_estimators=100, random_state=42)
hirf = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
pgbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
esvm = SVC(probability=True, kernel='rbf', random_state=42)

models = {'ANRDT': anrdt, 'HIRF': hirf, 'PGBM': pgbm, 'ESVM': esvm}

# Train base models
for name, model in models.items():
    model.fit(X_train, y_train)

# Predictions & probabilities for base models
predictions = {name: model.predict(X_test) for name, model in models.items()}
probs = {name: model.predict_proba(X_test)[:, 1] for name, model in models.items()}

# ----------------------------------
# 3) Additional Ensembles
# ----------------------------------
# BMA over base models' probabilities
bma_prob = np.mean(np.column_stack(list(probs.values())), axis=1)
bma_pred = (bma_prob >= 0.5).astype(int)

# Majority Voting over base models' hard predictions
mv_pred = np.squeeze(mode(np.array(list(predictions.values())), axis=0, keepdims=True).mode)

# Separate bagging/boosting/stacking (kept to mirror your original script)
bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
bagging.fit(X_train, y_train)
bagging_pred = bagging.predict(X_test)
bagging_prob = bagging.predict_proba(X_test)[:, 1]

boosting = GradientBoostingClassifier(n_estimators=100, random_state=42)
boosting.fit(X_train, y_train)
boosting_pred = boosting.predict(X_test)
boosting_prob = boosting.predict_proba(X_test)[:, 1]

meta_learner = LogisticRegression(max_iter=1000, random_state=42)
stacked_model = StackingClassifier(
    estimators=[('ANRDT', anrdt), ('HIRF', hirf), ('PGBM', pgbm), ('ESVM', esvm)],
    final_estimator=meta_learner
)
stacked_model.fit(X_train, y_train)
stacked_pred = stacked_model.predict(X_test)
stacked_prob = stacked_model.predict_proba(X_test)[:, 1]

# ----------------------------------
# 4) Evaluation (performance)
# ----------------------------------
def evaluate_models(y_true, predictions_dict, probs_dict):
    rows = []
    for name, y_pred in predictions_dict.items():
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred)
        spec = recall_score(y_true, y_pred, pos_label=0)
        if name in probs_dict and probs_dict[name] is not None:
            fpr, tpr, _ = roc_curve(y_true, probs_dict[name])
            auc_roc = auc(fpr, tpr)
        else:
            auc_roc = np.nan
        rows.append([name, acc, prec, sens, spec, auc_roc])
    return pd.DataFrame(rows, columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'AUC-ROC'])

# Collect all predictions
all_preds = {
    **predictions,               # base models
    'BMA': bma_pred,
    'Majority Voting': mv_pred,
    'Bagging': bagging_pred,
    'Boosting': boosting_pred,
    'Stacking': stacked_pred
}

# Collect probabilities where available (MV has no natural prob)
all_probs = {
    **probs,                     # base models
    'BMA': bma_prob,
    'Majority Voting': None,
    'Bagging': bagging_prob,
    'Boosting': boosting_prob,
    'Stacking': stacked_prob
}

performance_df = evaluate_models(y_test, all_preds, all_probs)
print("\nPerformance Metrics:")
print(performance_df.round(4))

# ----------------------------------
# 5) Evaluation (scalability/feasibility)
# ----------------------------------
def model_file_name(name):
    return f"{name.replace(' ', '_').lower()}_model.pkl"

def measure_model(model, name, X_train, y_train, X_test):
    # Train time
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Inference latency per sample (predict)
    t0 = time.time()
    _ = model.predict(X_test)
    pred_latency = (time.time() - t0) / len(X_test)

    # Save and measure size
    path = model_file_name(name)
    joblib.dump(model, path)
    size_kb = os.path.getsize(path) / 1024.0
    return train_time, pred_latency, size_kb

def measure_bma_latency_and_size(base_models, X_test):
    # latency: time the required predict_proba calls + averaging
    t0 = time.time()
    _ = np.mean([m.predict_proba(X_test)[:, 1] for m in base_models.values()], axis=0)
    latency = (time.time() - t0) / len(X_test)
    # size: sum of component model sizes
    total_kb = 0.0
    for name, m in base_models.items():
        path = model_file_name(name)  # already saved below, or save now
        if not os.path.exists(path):
            joblib.dump(m, path)
        total_kb += os.path.getsize(path) / 1024.0
    return latency, total_kb

def measure_mv_latency_and_size(base_models, X_test):
    t0 = time.time()
    _ = np.squeeze(mode(np.array([m.predict(X_test) for m in base_models.values()]), axis=0, keepdims=True).mode)
    latency = (time.time() - t0) / len(X_test)
    total_kb = 0.0
    for name, m in base_models.items():
        path = model_file_name(name)
        if not os.path.exists(path):
            joblib.dump(m, path)
        total_kb += os.path.getsize(path) / 1024.0
    return latency, total_kb

# We will (re)measure scalability for:
#   - Base models: ANRDT, HIRF, PGBM, ESVM
#   - Additional ensembles: Bagging, Boosting, Stacking
#   - Composites: BMA, Majority Voting
scal_rows = []

# Base models
for name, model in models.items():
    tt, lat, size_kb = measure_model(model, name, X_train, y_train, X_test)
    scal_rows.append([name, tt, lat, size_kb])

# Additional ensemble objects
for name, model in [('Bagging', BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)),
                    ('Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
                    ('Stacking', StackingClassifier(
                        estimators=[('ANRDT', anrdt), ('HIRF', hirf), ('PGBM', pgbm), ('ESVM', esvm)],
                        final_estimator=LogisticRegression(max_iter=1000, random_state=42)
                    ))]:
    tt, lat, size_kb = measure_model(model, name, X_train, y_train, X_test)
    scal_rows.append([name, tt, lat, size_kb])

# BMA & MV (no extra training; latency + effective size = sum of bases)
bma_latency, bma_size_kb = measure_bma_latency_and_size(models, X_test)
scal_rows.append(['BMA', 0.0, bma_latency, bma_size_kb])

mv_latency, mv_size_kb = measure_mv_latency_and_size(models, X_test)
scal_rows.append(['Majority Voting', 0.0, mv_latency, mv_size_kb])

scalability_df = pd.DataFrame(scal_rows, columns=['Model', 'Train Time (s)', 'Inference Latency (s/sample)', 'Model Size (KB)'])
print("\nScalability Metrics:")
print(scalability_df.round(6))

# Save tables
performance_df.to_csv("ensemble_performance_metrics.csv", index=False)
scalability_df.to_csv("ensemble_scalability_metrics.csv", index=False)

# ----------------------------------
# 6) Plots
# ----------------------------------
# Accuracy bar plot
plt.figure(figsize=(9, 5))
sns.barplot(x='Model', y='Accuracy', data=performance_df, order=performance_df['Model'])
plt.title('Accuracy: Base & Ensemble Models')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Training time
plt.figure(figsize=(9, 5))
sns.barplot(x='Model', y='Train Time (s)', data=scalability_df, order=scalability_df['Model'])
plt.title('Training Time by Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Inference latency
plt.figure(figsize=(9, 5))
sns.barplot(x='Model', y='Inference Latency (s/sample)', data=scalability_df, order=scalability_df['Model'])
plt.title('Inference Latency by Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Model size
plt.figure(figsize=(9, 5))
sns.barplot(x='Model', y='Model Size (KB)', data=scalability_df, order=scalability_df['Model'])
plt.title('Model Size by Model')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Confusion matrices for ensemble outputs
for name, pred in {'BMA': bma_pred, 'Majority Voting': mv_pred, 'Bagging': bagging_pred, 'Boosting': boosting_pred, 'Stacking': stacked_pred}.items():
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# ROC curves (for models with probabilities)
plt.figure(figsize=(9, 6))
for name, prob in all_probs.items():
    if prob is None:  # skip MV
        continue
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves (Base & Ensembles)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()

# SHAP (feature importance) for ANRDT (RandomForest)
explainer = shap.TreeExplainer(anrdt)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# Precision-Recall curves (for models with probabilities)
plt.figure(figsize=(9, 6))
for name, prob in all_probs.items():
    if prob is None:
        continue
    precision, recall, _ = precision_recall_curve(y_test, prob)
    avg_prec = average_precision_score(y_test, prob)
    plt.plot(recall, precision, label=f'{name} (AP = {avg_prec:.2f})')
plt.title('Precision-Recall Curves (Base & Ensembles)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.tight_layout()
plt.show()

# Calibration curves (for models with probabilities)
plt.figure(figsize=(9, 6))
for name, prob in all_probs.items():
    if prob is None:
        continue
    prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=name)
plt.plot([0, 1], [0, 1], 'k--')
plt.title('Calibration Curves (Base & Ensembles)')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.legend()
plt.tight_layout()
plt.show()

# Learning curves
def plot_learning_curve(model, X, y, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', label='CV score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    plt.title(f'Learning Curve: {model_name}')
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_learning_curve(anrdt, X_train, y_train, "ANRDT")
plot_learning_curve(hirf, X_train, y_train, "HIRF")
plot_learning_curve(pgbm, X_train, y_train, "PGBM")
plot_learning_curve(esvm, X_train, y_train, "ESVM")


