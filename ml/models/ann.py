from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTENC
import numpy as np
import warnings

def train_ann_classifier(X_train, y_train):
    """
    先用SMOTENC做过采样，再用 sklearn MLPClassifier 训练ANN，带早停。

    参数:
        X_train: np.ndarray 或 pd.DataFrame
        y_train: 标签数组
        categorical_features: 类别特征列索引列表
        monotone_constraints: ANN 不支持，忽略

    返回:
        best_model: 最优模型
        best_params: 最佳参数
    """
    if not isinstance(X_train, np.ndarray):
        X_train = X_train.to_numpy()

    # 2. 建立 MLPClassifier 模型，启用早停 early_stopping=True
    model = MLPClassifier(
        early_stopping=True,    # 早停
        validation_fraction=0.125, # 10%做验证集早停判断
        n_iter_no_change=10,    # 连续10轮loss不降就停止
        random_state=42,
        max_iter=200
    )

    # 3. 网格调参
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,)],  # 简单的隐藏层结构
        'alpha': [0.0001, 0.001],                # L2正则
        'learning_rate_init': [0.001, 0.01]
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
