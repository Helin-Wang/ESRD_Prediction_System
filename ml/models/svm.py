from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import warnings

def train_svm_classifier(X_train, y_train):
    """
    使用 SVM (RBF kernel) 进行二分类训练，使用 GridSearchCV 进行调参。
    
    参数:
        X_train: 特征矩阵
        y_train: 标签
        monotone_constraints: ⚠️ SVM 不支持，参数将被忽略

    返回:
        best_model: 最优 SVM 模型
        best_params: 最佳参数组合
    """

    param_grid = {
        'C': [0.1, 1, 10],                # 正则项，越大越容易过拟合
        'gamma': ['scale', 0.01, 0.001],  # 核函数参数，越大越复杂
        'kernel': ['rbf']                 # 你也可以改为 'linear' 看效果
    }

    model = SVC(
        probability=True,                  # 便于输出概率，e.g. roc_auc
        class_weight='balanced',          # ✅ 防止类别不平衡
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=10,
        scoring='roc_auc',                # 或 'accuracy'、'f1' 等
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(best_params)

    return best_model