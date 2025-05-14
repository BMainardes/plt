from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import LeaveOneOut, cross_val_score
import joblib
import pandas as pd
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.feature_importance = None
    
    def train_model(self, X, y):
        """Treina o modelo preditivo"""
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('regressor', BayesianRidge())
        ])
        
        # Validação cruzada Leave-One-Out para pequenos datasets
        loo = LeaveOneOut()
        cv_scores = cross_val_score(pipeline, X, y, cv=loo, scoring='r2')
        print(f"Performance CV: R² médio = {np.mean(cv_scores):.2f} (±{np.std(cv_scores):.2f})")
        
        # Treinamento final
        pipeline.fit(X, y)
        self.model = pipeline
        
        # Calcula importância das features
        self._calculate_feature_importance(X.columns)
        
        return pipeline
    
    def _calculate_feature_importance(self, feature_names):
        """Calcula a importância das features para o modelo treinado"""
        if hasattr(self.model.named_steps['regressor'], 'coef_'):
            coefs = self.model.named_steps['regressor'].coef_
            self.feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefs
            }).sort_values('Coefficient', ascending=False)
    
    def save_model(self, filepath):
        """Salva o modelo treinado em disco"""
        if self.model is not None:
            joblib.dump(self.model, filepath)
            print(f"Modelo salvo em {filepath}")
        else:
            raise ValueError("Nenhum modelo treinado para salvar")
    
    def load_model(self, filepath):
        """Carrega um modelo salvo"""
        self.model = joblib.load(filepath)
        return self.model
    
    def get_feature_importance(self):
        """Retorna a importância das features"""
        return self.feature_importance.copy()