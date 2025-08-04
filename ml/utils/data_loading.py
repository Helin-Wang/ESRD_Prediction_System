import pandas as pd

cols = ['gender (1/0)', 'family_history (1/0)', 'preterm_birth (1/0)',
       'prenatal_phenotype (1/0)', 'cakut_subphenotype',
       'behavioral_cognitive_abnormalities (1/0)', 'motor_retardation (1/0)',
       'congenital_heart_disease (1/0)', 'skeletal_anormalies (1/0)',
       'genitoreproductive (1/0)', 'central_nervous_system (1/0)',
       'lung_diaphragm (1/0)', 'face (1/0)', 'hearing (1/0)', 'ocular (1/0)',
       'external_ear (1/0)', 'gastrointestinal_tract (1/0)',
       'diabetes_mellitus (1/0)', 'gene_trioplp (1/0)', 'age_first_diagnose',
       'ckd_stage_first_diagnose', 'short_stature (1/0)', 'hyperuricemia(1/0)']

selected_cols = ['age_first_diagnose',
                'behavioral_cognitive_abnormalities (1/0)',
                'cakut_subphenotype',
                'ckd_stage_first_diagnose',
                'congenital_heart_disease (1/0)',
                'family_history (1/0)',
                'gender (1/0)',
                'gene_trioplp (1/0)',
                'ocular (1/0)',
                'prenatal_phenotype (1/0)',
                'preterm_birth (1/0)',
                'short_stature (1/0)']
y_cols = ['esrd_1y', 'esrd_3y', 'esrd_5y']

def load_dataset(filepath, year, cols=None):
    df = pd.read_csv(filepath)
    
    if cols:
        df = df[cols]
        
    y_col = f'esrd_{year}y'
    df = df[df[y_col]!=-1]
    for col in y_cols:
        if col == y_col:
            y = df[col]
        df.drop(columns=[col], inplace=True)

    return df, y


