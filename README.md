# Baseline and Enhanced Machine Learning Models for Coronary Heart Disease Prediction

This repository contains the Python code used in the study:

**“Baseline and Enhanced Machine Learning Models for Coronary Heart Disease Prediction”**  
by **Maurice Wanyonyi**

The scripts implement baseline and enhanced machine learning (ML) models to predict Coronary Heart Disease (CHD).  
They include performance evaluation, ensemble approaches, scalability assessment (training time, inference latency, model size), and explainability (SHAP and LIME).

---

## 📂 Repository Contents
- baseline_model.py – Baseline ML models (Decision Tree, Random Forest, Gradient Boosting, SVM, ensembles).  
- enhanced_model.py – Enhanced ML models (ANRDT, HIRF, PGBM, ESVM, plus ensembles with SHAP/LIME).  
- requirements.txt – List of Python dependencies.  
- README.md – Documentation and usage instructions.  
- LICENSE – MIT License (open source).

---

## ⚙️ Requirements
The code requires **Python 3.8+** and the following libraries (already listed in requirements.txt):

- numpy  
- pandas  
- matplotlib  
- seaborn  
- scikit-learn  
- imbalanced-learn  
- scipy  
- shap  
- lime  
- joblib  
- psutil  

Install all dependencies with:
```bash
pip install -r requirements.txt
```
---

## ▶️ Usage
1. Clone this repository:
   
```bash
   git clone https://github.com/<your-username>/CHD-Baseline-Enhanced-ML-Models.git
   cd CHD-Baseline-Enhanced-ML-Models
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the baseline model:
```bash
python baseline_model.py
```

4. Run the enhanced model:
```bash
python enhanced_model.py
```

## 📊 Data

The models were trained and evaluated on the Coronary Heart Disease dataset used in the manuscript.
Please refer to the paper for details on preprocessing and data availability.

## 📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

## 📌 Citation
If you use this code, please cite our study:

> Wanyonyi et al. (2025). *Baseline and Enhanced Machine Learning Models for Coronary Heart Disease Prediction*. PLOS ONE. DOI: [Zenodo DOI will go here].

## 🙌 Acknowledgements

We thank the PLOS ONE editorial team and reviewers for emphasizing best practices in reproducible research.
