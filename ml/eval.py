#!/usr/bin/env python
# coding: utf-8


import sys
import os
sys.path.append(os.path.abspath('..'))

from sklearn.metrics import roc_auc_score
from utils.data_loading import load_dataset
import joblib
import pandas as pd

from utils.compare_auc_delong_xu import delong_roc_test
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
from utils.eval import eval_model



feature_name_dict = {
    'gender (1/0)': 'Gender',
    'family_history (1/0)': 'Family History',
    'preterm_birth (1/0)': 'Preterm Birth',
    'prenatal_phenotype (1/0)': 'Prenatal Phenotype',
    'cakut_subphenotype': 'Cakut Subphenotype',
    'behavioral_cognitive_abnormalities (1/0)': 'Behavioral Cognitive Abnormalities',
    'motor_retardation (1/0)': 'Motor Retardation',
    'congenital_heart_disease (1/0)': 'Congenital Heart Disease',
    'skeletal_anormalies (1/0)': 'Skeletal Anormalies',
    'genitoreproductive (1/0)': 'Genitoreproductive',
    'central_nervous_system (1/0)': 'Central Nervous System',
    'face (1/0)': 'Face',
    'hearing (1/0)': 'Hearing',
    'ocular (1/0)': 'Ocular',
    'external_ear (1/0)': 'External Ear',
    'gastrointestinal_tract (1/0)': 'Gastrointestinal Tract',
    'age_first_diagnose': 'Age At First Diagnosis',
    'ckd_stage_first_diagnose': 'Ckd Stage At Baseline',
    'short_stature (1/0)': 'Short Stature',
    'hyperuricemia(1/0)': 'Hyperuricemia',
    'CNV': 'CNV',
    'Chromosomal_abnormality': 'Chromosomal Abnormality',
    'PAX2': 'PAX2',
    'TNXB': 'TNXB',
    'EYA1': 'EYA1',
    'HNF1β': 'HNF1β',
    'GATA3': 'GATA3',
    'SALL1': 'SALL1',
    'COL4A1': 'COL4A1',
    'Other_gene': 'OtherGene'
}
cols = ['gender (1/0)', 'family_history (1/0)', 'preterm_birth (1/0)',
       'prenatal_phenotype (1/0)', 'cakut_subphenotype',
       'behavioral_cognitive_abnormalities (1/0)', 'motor_retardation (1/0)',
       'congenital_heart_disease (1/0)', 'skeletal_anormalies (1/0)',
       'genitoreproductive (1/0)', 'central_nervous_system (1/0)',
       'face (1/0)', 'hearing (1/0)', 'ocular (1/0)',
       'external_ear (1/0)', 'gastrointestinal_tract (1/0)',
       'age_first_diagnose',
       'ckd_stage_first_diagnose', 'short_stature (1/0)', 'hyperuricemia(1/0)',
       'CNV', 'Chromosomal_abnormality', 'PAX2', 'TNXB', 'EYA1', 'HNF1β', 'GATA3', 'SALL1', 'COL4A1', 'Other_gene',
       'esrd_1y', 'esrd_3y', 'esrd_5y']
selected_cols = ['PAX2', # 基因变量
 'age_first_diagnose',
 'behavioral_cognitive_abnormalities (1/0)',
 'cakut_subphenotype',
 'ckd_stage_first_diagnose',
 'congenital_heart_disease (1/0)',
 'family_history (1/0)',
 'gender (1/0)',
 'ocular (1/0)',
 'prenatal_phenotype (1/0)',
 'preterm_birth (1/0)',
 'short_stature (1/0)']

outcome_cols = ['esrd_1y', 'esrd_3y', 'esrd_5y']

model_list = ['xgb', 'rf', 'svm', 'knn', 'ann', 'catboost', 'gbm', 'adaboost', 
            #  'gbdt'
              ]

professional_model_name = {
    'xgb': 'XGBoost',
    'adaboost': 'AdaBoost',
    'ann': 'ANN',
    'catboost': 'CatBoost',
    'gbm': 'GBM',
    'knn': 'KNN',
    'rf': 'Random Forest',
    'svm': 'SVM'
}


if __name__ == "__main__":
    internal_dataset = '../dataset/复旦儿科_135年_特征1.csv'
    external_dataset = '../dataset/外院_135年_特征1.csv'

    internal_results = []
    external_results = []
    for year in [1,3,5]:
        selected_internal_X, selected_internal_y = load_dataset(internal_dataset, year, selected_cols+outcome_cols)
        selected_external_X, selected_external_y = load_dataset(external_dataset, year, selected_cols+outcome_cols)
        selected_X_train, selected_X_test, selected_y_train, selected_y_test = train_test_split(selected_internal_X, selected_internal_y, test_size=0.3, random_state=42, stratify=selected_internal_y)
        
        for model_name in model_list:
            selected_model = joblib.load(f'../output/no_gene/{model_name}_{year}yr.pkl')
            
            internal_metrics = eval_model(selected_model, selected_X_test, selected_y_test)
            external_metrics = eval_model(selected_model, selected_external_X, selected_external_y)
            
            internal_metrics['Model'] = model_name
            internal_metrics['Year'] = year
            external_metrics['Model'] = model_name
            external_metrics['Year'] = year
            internal_results.append(internal_metrics)
            external_results.append(external_metrics)

    internal_results_df = pd.DataFrame(internal_results)
    external_results_df = pd.DataFrame(external_results)
    
    internal_results_df.to_excel("../selected_internal_results_df.xlsx", index=False)
    external_results_df.to_excel("../selected_external_results_df.xlsx", index=False)

