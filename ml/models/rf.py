from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import warnings

def train_rf_classifier(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
    }

    base_model = RandomForestClassifier(
        class_weight='balanced',  # 自动调整样本权重以应对类别不平衡
        random_state=42,
        n_jobs=-1
    )

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=10,
        scoring='roc_auc',  # 可根据任务调整为 accuracy、f1 等
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(best_params)

    return best_model
