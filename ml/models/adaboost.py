from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_adaboost_classifier(X_train, y_train):
    num_pos = np.sum(y_train == 1)
    num_neg = np.sum(y_train == 0)

    # 类别不平衡处理（加权给到弱分类器）
    class_weight = {0: 1.0, 1: num_neg / num_pos}

    # 弱分类器：决策树
    base_estimator = DecisionTreeClassifier(
        #class_weight=class_weight,
        random_state=42
    )

    # 构建 AdaBoost 模型
    base_model = AdaBoostClassifier(
        estimator=base_estimator,
        algorithm='SAMME',  # 使用概率版本提升，适用于二分类
        random_state=42
    )

    # 超参数搜索空间
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'estimator__max_depth': [1, 3, 5],  # 控制每棵树的复杂度
    }

    # 网格搜索
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=10,
        scoring='roc_auc',  # 可换为 accuracy / f1 等
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(best_params)

    return best_model
