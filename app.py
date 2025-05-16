import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sksurv.ensemble import RandomSurvivalForest
import matplotlib.pyplot as plt

model = joblib.load('./models/with_gene/rsf.pkl')

st.set_page_config(page_title="Clinical Decision Support System", layout="wide")
st.title("🩺 Clinical Decision Support System")
st.markdown("<hr>", unsafe_allow_html=True)

left_col, right_col = st.columns([3, 2], gap="large")

cakut_subphenotype_list = {
    'renal hypodysplasia associated with puv': 1,
    'solitary kidney': 2,
    'bilateral renal hypodysplasia': 3,
    'unilateral renal hypodysplasia': 4,
    'multicystic dysplastic kidney': 5,
    'horseshoe kidney': 6,
    'others': 7
}

with left_col:
    st.subheader("🏥 Patient Characteristics")
    col1, col2, col3 = st.columns(3, gap='medium')
    with col1:
        age_first_diagnose = st.number_input("Age at first diagnose(yr)", min_value=0.0, max_value=18.0, value=0.0)
        gender = st.selectbox("gender", ["female", "male"])
        family_history = st.selectbox("Family history", ["No", "Yes"])
        
    with col2:
        ckd_stage_first_diagnose = st.selectbox("CKD stage at first diagnose", [1, 2, 3, 4, 5])
        short_stature = st.selectbox("Short stature", ["No", "Yes"])
        preterm_birth = st.selectbox("Preterm birth", ["No", "Yes"])
    with col3:
        gene_trioplp = st.selectbox("Gene diagnosis", ["No", "Yes"])
        extrarenal_anomalies = st.selectbox("Extrarenal anomalies", ["No", "Yes"])
        prenatal_phenotype = st.selectbox("Prenatal phenotype", ["No", "Yes"])
        
    last_col1, last_col2 = st.columns(2, gap='small')
    with last_col1:
        cakut_subphenotype = st.selectbox("CAKUT subphenotype", cakut_subphenotype_list.keys())
    
    predict_btn = st.button("PREDICT")


input_data = pd.DataFrame({
    'gender (1/0)': [0 if gender == 'female' else 1],
    'age_first_diagnose': [age_first_diagnose],
    'ckd_stage_first_diagnose': [ckd_stage_first_diagnose],
    'gene_trioplp (1/0)': [0 if gene_trioplp=='No' else 1],
    'cakut_subphenotype': [cakut_subphenotype_list[cakut_subphenotype]],
    'family_history (1/0)': [0 if family_history=='No' else 1],
    'preterm_birth (1/0)': [0 if preterm_birth=='No' else 1],
    'prenatal_phenotype (1/0)': [0 if prenatal_phenotype=='No' else 1],
    'extrarenal_anomalies (1/0)': [0 if extrarenal_anomalies=='No' else 1],
    'short_stature (1/0)': [0 if short_stature=='No' else 1],
})

with right_col:
    st.subheader("🤖 Predicted Results")
    if predict_btn:
        try:
            surv_func = model.predict_survival_function(input_data)[0]
            
            st.write(f"Probability of ESRD within 1 year: {1 - surv_func([1]):.2%}")
            st.write(f"Probability of ESRD within 3 years: {1 - surv_func([3]):.2%}")
            st.write(f"Probability of ESRD within 5 years: {1 - surv_func([5]):.2%}")
            
                        
            times = surv_func.x
            probs = surv_func(times)

            # st.markdown("##### 📈 Predicted Survival Function")
            # mask = times <= 10
            # chart_data = pd.DataFrame({
            #     "time": times[mask],
            #     "Survival Probability": probs[mask]
            # })
            
            # small_col1, _ = st.columns([10, 1])
            # with small_col1:
            #     st.line_chart(chart_data, x='time', y='Survival Probability')
        
            
            st.markdown("##### 📈 Predicted Line chart of ESRD probability over time")

            mask = times <= 10
            chart_data = pd.DataFrame({
                "time(year)": times[mask],
                "ESRD Probability": [1-prob for prob in probs[mask]]
            })
            
            small_col1, _ = st.columns([10, 1])
            with small_col1:
                st.line_chart(chart_data, x='time(year)', y='ESRD Probability')
                    
            
                
        except Exception as e:
            st.error(f"Error: {e}")