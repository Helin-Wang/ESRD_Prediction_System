import pandas as pd
from xgboost import XGBClassifier
import shap
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'


def compute_esrd_status(df, years=[1, 3, 5]):
    df = df.copy()
    
    for year in years:
        col_name = f'esrd_{year}y'
        
        # 初始化为 -1
        df[col_name] = -1
        
        # 情况 1: disease_course < year 且 esrd == 0 → 无法判断，保持 NaN
        # 情况 2: disease_course < year 且 esrd == 1 → 肯定在 year 前发生了 → 赋值 1
        cond2 = (df['disease_course（y）'] < year) & (df['esrd (1/0)'] == 1)
        df.loc[cond2, col_name] = 1

        # 情况 3: disease_course == year → 使用现有 esrd 值
        cond3 = df['disease_course（y）'] == year
        df.loc[cond3, col_name] = df.loc[cond3, 'esrd (1/0)']

        # 情况 4: disease_course > year 且 esrd == 0 → 赋值 0
        cond4 = (df['disease_course（y）'] > year) & (df['esrd (1/0)'] == 0)
        df.loc[cond4, col_name] = 0
        # 情况 5: disease_course > year 且 esrd == 1 → 未知
    
    # 删除任何一列 esrd_xx y 为 NA 的行（说明该年无法判断）
    # valid_mask = df[[f'esrd_{y}y' for y in years]].notna().all(axis=1)
    # df = df[valid_mask].reset_index(drop=True)
    
    return df

def feature_importance_selector(X, y, colors='', return_data=False):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    feature_names = np.array([feature_name_dict[col] for col in shap_values.feature_names])
    
    sorted_idx = np.argsort(mean_abs_shap)[::-1]
    top_idx = sorted_idx[:11]
    others_idx = sorted_idx[11:]
    
    top_features = feature_names[top_idx]
    top_values = mean_abs_shap[top_idx]
    others_value = mean_abs_shap[others_idx].sum()
    
    all_features = np.append(top_features, "Others")
    all_values = np.append(top_values, others_value)
    
    if return_data:
        return all_features, all_values, colors
    
    if colors == 'orange':
        colors = plt.cm.Oranges_r(np.linspace(0, 1, len(all_features)))
    elif colors == 'green':
        colors = plt.cm.Greens_r(np.linspace(0, 1, len(all_features)))
    else:
        colors = plt.cm.Blues_r(np.linspace(0, 1, len(all_features)))

    # 画图
    plt.figure(figsize=(7, 5))
    plt.barh(all_features[::-1], all_values[::-1], color=colors[::-1])
    plt.xlabel("mean(|SHAP value|)", fontsize=14, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.title("Top 12 SHAP Feature Importances", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    shap_df = pd.DataFrame({
        'feature': X.columns,
        'importance': mean_abs_shap
    }).sort_values(by='importance', ascending=False)

    top_features = shap_df['feature'].tolist()
    X_selected = X[top_features]
    
    # 只保留 importance > 0 的特征
    non_zero_shap_df = shap_df[shap_df['importance'] > 0].sort_values(by='importance', ascending=False)

    # 查看非零特征名
    selected_features = non_zero_shap_df['feature'].tolist()
    return selected_features
