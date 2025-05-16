from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
import joblib
import pandas as pd
import numpy as np
import os
from config import MODEL_DIR

class ModelTrainer:
    def __init__(self, model_type='bayesian'):
        self.model = None
        self.feature_importance = None
        self.model_type = model_type.lower()

    def train_model(self, X, y):
        if self.model_type == 'random_forest':
            base_model = RandomForestRegressor(random_state=42)
            param_grid = {
                'regressor__n_estimators': [100, 200],
                'regressor__max_depth': [None, 10, 20],
                'regressor__min_samples_split': [2, 5],
                'regressor__min_samples_leaf': [1, 2]
            }
        elif self.model_type == 'gradient_boosting':
            base_model = GradientBoostingRegressor(random_state=42)
            param_grid = {
                'regressor__n_estimators': [100, 200],
                'regressor__learning_rate': [0.05, 0.1],
                'regressor__max_depth': [3, 5],
                'regressor__min_samples_split': [2, 5],
                'regressor__min_samples_leaf': [1, 2]
            }
        else:
            base_model = BayesianRidge()
            param_grid = None

        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', base_model)
        ])

        if param_grid:
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
            grid_search.fit(X, y)
            pipeline = grid_search.best_estimator_
            print("Melhores parâmetros:", grid_search.best_params_)
            print(f"Melhor R²: {grid_search.best_score_:.3f}")
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='r2')
            print(f"Performance CV: R² médio = {np.mean(cv_scores):.3f} (±{np.std(cv_scores):.3f})")
            pipeline.fit(X, y)

        self.model = pipeline
        self._calculate_feature_importance(X.columns)
        return pipeline

    def _calculate_feature_importance(self, feature_names):
        reg = self.model.named_steps['regressor']
        if hasattr(reg, 'feature_importances_'):
            importances = reg.feature_importances_
            self.feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
        elif hasattr(reg, 'coef_'):
            coefs = reg.coef_
            self.feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefs
            }).sort_values('Coefficient', ascending=False)
        else:
            self.feature_importance = None

    def get_ideal_settings(self, features, df):
        """
        Calcula configurações ideais baseadas na média ponderada das features importantes,
        excluindo variáveis de saída ('PDI' e 'Finos').
        """
        if self.feature_importance is None:
            return {}

        # Remove 'PDI' e 'Finos' das features para considerar só entradas válidas
        valid_features = [f for f in features if f not in ['PDI', 'Finos'] and f in df.columns]

        weighted_avgs = {}
        total_weight = self.feature_importance.iloc[:, 1].abs().sum()

        for _, row in self.feature_importance.iterrows():
            feature = row['Feature']
            if feature in valid_features:
                weight = abs(row[self.feature_importance.columns[1]]) / total_weight
                weighted_avgs[feature] = (df[feature] * weight).mean()

        return weighted_avgs

    def save_model(self, filename):
        if self.model is not None:
            os.makedirs(MODEL_DIR, exist_ok=True)
            filepath = os.path.join(MODEL_DIR, filename)
            joblib.dump(self.model, filepath)
            print(f"Modelo salvo em {filepath}")
        else:
            raise ValueError("Nenhum modelo treinado para salvar")

    def load_model(self, filepath):
        self.model = joblib.load(filepath)
        return self.model

    def get_feature_importance(self):
        return self.feature_importance.copy() if self.feature_importance is not None else None

    def _print_feature_importance(self):
        if self.feature_importance is not None:
            print("\nImportância das Features:")
            print(self.feature_importance.head(10))
        else:
            print("\nNão foi possível determinar a importância das features")
