from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_lightgbm_classifier(X_train, y_train):
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    scale_pos_weight = num_neg / num_pos  # 类别不平衡处理

    # 超参数搜索空间（适当控制复杂度，防止过拟合）
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 1],        # L1正则
        'reg_lambda': [1, 5, 10],        # L2正则
    }

    base_model = LGBMClassifier(
        objective='binary',
        scale_pos_weight=scale_pos_weight,
        metric='binary_logloss',
        random_state=42,
        n_jobs=-1  # 并行加速
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=10,
        scoring='roc_auc',  # 可改为 accuracy / f1 等
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(best_params)

    return best_model
