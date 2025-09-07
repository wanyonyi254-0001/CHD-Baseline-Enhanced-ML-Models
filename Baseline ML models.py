# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 21:58:26 2025

@author: iamwa
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 22 17:08:43 2025
Updated to include scalability metrics: training time, inference latency, model size.

@author: iamwa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time, joblib, os
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from scipy.stats import mode
import shap

# ======================================================
# Load dataset
# ======================================================
df = pd.read_csv('C:/Users/iamwa/Desktop/CHD data.csv')
y = df['HeartDiseaseorAttack']
X = df.drop(columns=['HeartDiseaseorAttack'])

# Apply SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Stratified Shuffle Split for sampling 10,000 observations
sss = StratifiedShuffleSplit(n_splits=1, train_size=10000, random_state=42)
for train_index, _ in sss.split(X_resampled, y_resampled):
    X_sample = X_resampled.iloc[train_index]
    y_sample = y_resampled.iloc[train_index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_sample, y_sample,
    test_size=0.2,
    stratify=y_sample,
    random_state=42
)

# ======================================================
# Define models
# ======================================================
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
svm = SVC(kernel='rbf', probability=True, random_state=42)

models = {
    'Decision Tree': dt,
    'Random Forest': rf,
    'Gradient Boosting': gb,
    'SVM': svm
}

# ======================================================
# Train base models
# ======================================================
for name, model in models.items():
    model.fit(X_train, y_train)

# Predictions and probabilities
predictions = {name: model.predict(X_test) for name, model in models.items()}
probs = {name: model.predict_proba(X_test)[:, 1] for name, model in models.items()}

# ======================================================
# Ensembles
# ======================================================
def bma_prediction(probs):
    avg_prob = np.mean(list(probs.values()), axis=0)
    return (avg_prob >= 0.5).astype(int)

bma_pred = bma_prediction(probs)

def majority_voting(predictions):
    pred_array = np.array(list(predictions.values()))
    return np.squeeze(mode(pred_array, axis=0, keepdims=True).mode)

mv_pred = majority_voting(predictions)

bagging = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, random_state=42)
bagging.fit(X_train, y_train)
bagging_pred = bagging.predict(X_test)

boosting = GradientBoostingClassifier(n_estimators=100, random_state=42)
boosting.fit(X_train, y_train)
boosting_pred = boosting.predict(X_test)

meta_learner = LogisticRegression(max_iter=1000, random_state=42)
stacking = StackingClassifier(
    estimators=[('dt', dt), ('rf', rf), ('gb', gb), ('svm', svm)],
    final_estimator=meta_learner
)
stacking.fit(X_train, y_train)
stacking_pred = stacking.predict(X_test)

# ======================================================
# Performance Evaluation
# ======================================================
def evaluate_models(y_true, predictions, probs):
    metrics = []
    for name, y_pred in predictions.items():
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred)
        spec = recall_score(y_true, y_pred, pos_label=0)
        auc_roc = auc(*roc_curve(y_true, probs[name])[:2]) if name in probs else np.nan
        metrics.append([name, acc, prec, sens, spec, auc_roc])
    return pd.DataFrame(metrics, columns=['Model', 'Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'AUC-ROC'])

all_preds = {
    **predictions,
    'BMA': bma_pred,
    'Majority Voting': mv_pred,
    'Bagging': bagging_pred,
    'Boosting': boosting_pred,
    'Stacking': stacking_pred
}

performance_metrics = evaluate_models(y_test, all_preds, probs)
print(performance_metrics)

# ======================================================
# Scalability Evaluation
# ======================================================
def evaluate_scalability(models, X_train, y_train, X_test):
    scalability_metrics = []
    
    for name, model in models.items():
        print(f"Evaluating scalability for {name}...")
        
        # Training time
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        # Inference latency
        start = time.time()
        _ = model.predict(X_test)
        total_inference = time.time() - start
        inference_latency = total_inference / len(X_test)
        
        # Model size (KB)
        temp_path = f"{name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(model, temp_path)
        model_size = os.path.getsize(temp_path) / 1024
        
        scalability_metrics.append([name, train_time, inference_latency, model_size])
    
    return pd.DataFrame(
        scalability_metrics,
        columns=["Model", "Train Time (s)", "Inference Latency (s/sample)", "Model Size (KB)"]
    )

scalability_df = evaluate_scalability(models, X_train, y_train, X_test)
print("\nScalability Metrics:")
print(scalability_df.round(4))
scalability_df.to_csv("scalability_metrics.csv", index=False)

# ======================================================
# Visualizations
# ======================================================
# Accuracy Bar Plot
plt.figure(figsize=(8, 5))
sns.barplot(x=performance_metrics['Model'], y=performance_metrics['Accuracy'])
plt.title('Model Accuracy Scores (Base + Ensembles)')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Scalability Plots
plt.figure(figsize=(8,6))
sns.barplot(x="Model", y="Train Time (s)", data=scalability_df)
plt.title("Training Time by Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.barplot(x="Model", y="Inference Latency (s/sample)", data=scalability_df)
plt.title("Inference Latency by Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,6))
sns.barplot(x="Model", y="Model Size (KB)", data=scalability_df)
plt.title("Model Size by Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Confusion Matrices
def plot_conf_matrix(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

for name, pred in all_preds.items():
    plot_conf_matrix(name, y_test, pred)

# ROC Curves
plt.figure(figsize=(8, 6))
for name, prob in probs.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()

# SHAP (Random Forest)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# Precision-Recall Curve
plt.figure(figsize=(8, 6))
for name, prob in probs.items():
    precision, recall, _ = precision_recall_curve(y_test, prob)
    avg_prec = average_precision_score(y_test, prob)
    plt.plot(recall, precision, label=f'{name} (AP = {avg_prec:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves')
plt.legend()
plt.show()

# Calibration Curve
plt.figure(figsize=(8, 6))
for name, prob in probs.items():
    prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=name)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Predicted Probability')
plt.ylabel('True Probability')
plt.title('Calibration Curves')
plt.legend()
plt.show()

# Learning Curve Function
def plot_learning_curve(model, X, y, model_name):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', label='Training score')
    plt.plot(train_sizes, test_mean, 'o-', label='Cross-validation score')
    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.show()

plot_learning_curve(dt, X_train, y_train, "Decision Tree")
plot_learning_curve(rf, X_train, y_train, "Random Forest")
plot_learning_curve(gb, X_train, y_train, "Gradient Boosting")
plot_learning_curve(svm, X_train, y_train, "Support Vector Machine")
