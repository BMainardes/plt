import pandas as pd
import joblib
from config import CONTROL_LIMITS
import numpy as np

class PDIMonitor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.control_limits = CONTROL_LIMITS
    
    def check_parameters(self, current_params):
        """Verifica se os parâmetros estão dentro dos limites recomendados"""
        alerts = []
        for param, (lower, upper) in self.control_limits.items():
            value = current_params.get(param)
            if value is None:
                alerts.append(f"Parâmetro {param} não informado.")
                continue
            if value < lower:
                alerts.append(f"{param} abaixo do mínimo ({value} < {lower})")
            elif value > upper:
                alerts.append(f"{param} acima do máximo ({value} > {upper})")
        return alerts
    
    def predict_pdi(self, current_params):
        """Faz a previsão do PDI com os parâmetros atuais"""
        input_df = pd.DataFrame([current_params])
        if 'Finos' in input_df.columns:
            input_df = input_df.drop(columns=['Finos'])
        return self.model.predict(input_df)[0]
    
    def predict_pdi_and_finos(self, current_params):
        """Tenta prever PDI e Finos, tratando caso o modelo retorne apenas PDI"""
        input_df = pd.DataFrame([current_params])
        if 'Finos' in input_df.columns:
            input_df = input_df.drop(columns=['Finos'])
        preds = self.model.predict(input_df)
        
        # Se for uma predição multi-output com dois valores, retorne ambos
        if isinstance(preds, (list, tuple, pd.Series, np.ndarray)):
            first_pred = preds[0]
            if hasattr(first_pred, '__len__') and len(first_pred) == 2:
                return first_pred[0], first_pred[1]
            else:
                # Retorna PDI e None para Finos
                return preds[0], None
        else:
            # Caso inesperado, retorna PDI e None
            return preds, None
    
    def generate_recommendations(self, current_params, predicted_pdi):
        """Gera recomendações para melhorar o PDI"""
        recs = []
        
        if predicted_pdi is not None and predicted_pdi < 80:
            if current_params.get('Amperagem_Peletizadora', 0) < 640:
                recs.append("Aumentar amperagem da peletizadora para >640A")
            if current_params.get('Taxa_Compressao', 0) < 18.9:
                recs.append("Aumentar taxa de compressão para ~18.94")
            if current_params.get('Velocidade_Alimentador', 0) > 60:
                recs.append("Reduzir velocidade do alimentador para <60")
        
        elif predicted_pdi is not None and predicted_pdi > 85:
            recs.append("Parâmetros ótimos - manter configuração atual")
        
        return recs
    
    def full_analysis(self, current_params):
        """Executa análise completa e retorna relatório"""
        alerts = self.check_parameters(current_params)
        predicted_pdi, predicted_finos = self.predict_pdi_and_finos(current_params)
        recommendations = self.generate_recommendations(current_params, predicted_pdi)
        
        report = {
            'predicted_pdi': round(predicted_pdi, 2) if predicted_pdi is not None else None,
            'predicted_finos': round(predicted_finos, 2) if predicted_finos is not None else None,
            'alerts': alerts,
            'recommendations': recommendations,
            'status': 'OK' if not alerts and (predicted_pdi is not None and predicted_pdi >= 80) else 'ALERT'
        }
        
        return report
