#!/usr/bin/env python
# coding: utf-8

import sys
import os
sys.path.append(os.path.abspath('..'))

from sksurv.metrics import concordance_index_censored
from sklearn.metrics import roc_auc_score
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import compute_esrd_status
from utils import feature_importance_selector
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'


cols1 = ['gender (1/0)', 'family_history (1/0)', 'preterm_birth (1/0)',
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

if __name__ == "__main__":
    # 1. ESRD status at fixed time
    internal_dataset = '../dataset/复旦儿科570例总数据-更新基因.xlsx'
    internal_df = pd.read_excel(internal_dataset)

    external_dataset = '../dataset/外院142例总数据-更新基因.xlsx'
    external_df = pd.read_excel(external_dataset)

    internal_df_processed = compute_esrd_status(internal_df)
    external_processed = compute_esrd_status(external_df)
    internal_df_processed[cols1].to_csv("../dataset/复旦儿科_135年.csv", index=False)
    external_processed[cols1].to_csv("../dataset/外院_135年.csv", index=False)

    internal_df_1yr = internal_df_processed[cols1][internal_df_processed['esrd_1y']!=-1]
    internal_df_1yr.drop(columns=['esrd_3y','esrd_5y'],inplace=True)
    internal_df_3yr = internal_df_processed[cols1][internal_df_processed['esrd_3y']!=-1]
    internal_df_3yr.drop(columns=['esrd_5y','esrd_1y'],inplace=True)
    internal_df_5yr = internal_df_processed[cols1][internal_df_processed['esrd_5y']!=-1]
    internal_df_5yr.drop(columns=['esrd_1y','esrd_3y'],inplace=True)

    internal_X_1yr = internal_df_1yr.drop(columns=['esrd_1y'])
    internal_y_1yr = internal_df_1yr['esrd_1y']
    internal_X_3yr = internal_df_3yr.drop(columns=['esrd_3y'])
    internal_y_3yr = internal_df_3yr['esrd_3y']
    internal_X_5yr = internal_df_5yr.drop(columns=['esrd_5y'])
    internal_y_5yr = internal_df_5yr['esrd_5y']