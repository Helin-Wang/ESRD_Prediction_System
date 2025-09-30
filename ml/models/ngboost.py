from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_ngboost_classifier(X_train, y_train):
    y_train = np.asarray(y_train).reshape(-1)
    y_train = y_train.astype(int)
    assert set(np.unique(y_train)).issubset({0, 1})
    
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)
    class_weight = {0: 1.0, 1: num_neg / num_pos}

    # NGBoost 目前不支持 sample_weight/class_weight，类别不平衡需通过其他方式处理
    # 你可以尝试下采样、上采样等方式额外处理

    # 超参数搜索空间
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'minibatch_frac': [0.5, 0.8, 1.0],  # 类似subsample
        'col_sample': [0.5, 0.8, 1.0],      # 类似colsample_bytree
        'natural_gradient': [True],        # 是否使用Natural Gradient（一般建议True）
    }

    base_model = NGBClassifier(
        Dist=Bernoulli,  # 用于二分类
        verbose=False,
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(best_params)

    return best_model
