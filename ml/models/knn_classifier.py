from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

def train_knn_classifier(X_train, y_train):
    """
    不使用 Pipeline，先用 SMOTENC 过采样，再用 GridSearchCV 调参训练 KNN。

    参数:
        X_train: np.ndarray 或 pandas.DataFrame（类别特征需整数编码）
        y_train: 标签数组
        categorical_features: 类别特征列索引列表
        monotone_constraints: ⚠️ KNN 不支持，将被忽略

    返回:
        best_model: 最优 KNN 模型
        best_params: 最佳参数组合
    """

    # KNN模型
    model = KNeighborsClassifier()

    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]  # p=1为曼哈顿距离，p=2为欧氏距离
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=10,
        scoring='roc_auc',
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_
