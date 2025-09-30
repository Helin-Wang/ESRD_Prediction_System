from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
import numpy as np


def train_gbdt_classifier(X_train, y_train, categorical_features='auto'):
    # 网格参数（你可以按需扩展）
    param_grid = {
        'learning_rate': [0.005, 0.01, 0.05],
        'max_iter': [100, 200],
        'l2_regularization': [0.5, 0.8, 1.0],
        'max_depth': [3, 4, 5]
    }
    
    # 构建模型对象
    base_model = HistGradientBoostingClassifier(
        early_stopping=True,
        scoring='roc_auc',
        validation_fraction=0.125,
        categorical_features=categorical_features,
        random_state=42
    )

    # 用 GridSearchCV 包装
    grid_search = GridSearchCV(
        base_model,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=10,  # dummy cv，因为我们使用外部验证集
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Best Params:", grid_search.best_params_)
    
    return grid_search.best_estimator_
