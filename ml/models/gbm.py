from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_gbm_classifier(X_train, y_train):
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    
    # 虽然GBM没有scale_pos_weight参数，但我们可以用class_weight模拟
    class_weight = {0: 1.0, 1: num_neg / num_pos}

    # 超参数搜索空间
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'max_features': [0.8, 1.0],  # 类似于 colsample_bytree
    }

    base_model = GradientBoostingClassifier(
        loss='log_loss',  # 对应binary logistic loss
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=10,
        scoring='roc_auc',  # 可换为 accuracy / f1 等
        verbose=1,
        n_jobs=-1
    )

    # 拟合时通过 sample_weight 处理类别不平衡
    sample_weight = np.array([class_weight[label] for label in y_train])

    grid_search.fit(X_train, y_train, sample_weight=sample_weight)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(best_params)

    return best_model
