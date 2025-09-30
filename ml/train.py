#!/usr/bin/env python

from utils.data_loading import load_dataset
import pandas as pd
from models.xgb import train_xgboost_classifier
from models.rf import train_rf_classifier
from models.svm import train_svm_classifier
from models.knn_classifier import train_knn_classifier
from models.ann import train_ann_classifier
from models.gbdt import train_gbdt_classifier
from models.gbm import train_gbm_classifier
from models.lightgbm import train_lightgbm_classifier
from models.adaboost import train_adaboost_classifier
from models.ngboost import train_ngboost_classifier
from models.catboost_classifier import train_catboost_classifier
from utils.eval import eval_model
from sklearn.model_selection import train_test_split
import joblib

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
    
numerical_categories = ['age_first_diagnose']
    
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

if __name__ == "__main__": 
    internal_dataset = './dataset/复旦儿科_135年_特征5.csv'
    external_dataset = './dataset/外院_135年_特征5.csv'
    use_selected_cols = True
    with_gene = True


    model_dict = {'xgb': train_xgboost_classifier,
                'rf': train_rf_classifier,
                'svm': train_svm_classifier,
                'knn': train_knn_classifier,
                'ann': train_ann_classifier,
                'catboost': train_catboost_classifier,
                'gbm': train_gbm_classifier,
                'adaboost': train_adaboost_classifier,
                }

    internal_results = []
    external_results = [] 

    for year in [1,3,5]:
        for model_name, train_methods in model_dict.items():
            internal_X, internal_y = load_dataset(internal_dataset, year, cols)
            external_X, external_y = load_dataset(external_dataset, year, cols)

            print(year)
            X_train, X_test, y_train, y_test = train_test_split(internal_X, internal_y, test_size=0.3, random_state=42, stratify=internal_y)
            if model_name == 'gbdt':
                categorical_features = []
                for i, col in enumerate(X_train.columns):
                    if col not in numerical_categories:
                        categorical_features.append(i)
                best_model = train_methods(X_train, y_train, categorical_features)
            else:
                best_model = train_methods(X_train, y_train)
            joblib.dump(best_model, f'./output/all/{model_name}_{year}yr.pkl')
            
            internal_metrics = eval_model(best_model, X_test, y_test)
            external_metrics = eval_model(best_model, external_X, external_y)
            internal_metrics['Model'] = model_name
            internal_metrics['Year'] = year
            external_metrics['Model'] = model_name
            external_metrics['Year'] = year
            internal_results.append(internal_metrics)
            external_results.append(external_metrics)
            print("*"*50)

    internal_results_df = pd.DataFrame(internal_results)
    external_results_df = pd.DataFrame(external_results)


    internal_results_df.to_excel("internal_results_df.xlsx", index=False)
    external_results_df.to_excel("external_results_df.xlsx", index=False)

