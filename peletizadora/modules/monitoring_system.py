import pandas as pd
import joblib

class PDIMonitor:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.control_limits = {
            'Amperagem_Peletizadora': (630, 660),
            'Taxa_Compressao': (18.8, 19.0),
            'Velocidade_Alimentador': (35, 60)
        }
    
    def check_parameters(self, current_params):
        """Verifica se os parâmetros estão dentro dos limites recomendados"""
        alerts = []
        for param, (lower, upper) in self.control_limits.items():
            value = current_params[param]
            if value < lower:
                alerts.append(f"{param} abaixo do mínimo ({value} < {lower})")
            elif value > upper:
                alerts.append(f"{param} acima do máximo ({value} > {upper})")
        return alerts
    
    def predict_pdi(self, current_params):
        """Faz a previsão do PDI com os parâmetros atuais"""
        input_df = pd.DataFrame([current_params])
        return self.model.predict(input_df)[0]
    
    def generate_recommendations(self, current_params, predicted_pdi):
        """Gera recomendações para melhorar o PDI"""
        recs = []
        
        if predicted_pdi < 80:
            if current_params['Amperagem_Peletizadora'] < 640:
                recs.append("Aumentar amperagem da peletizadora para >640A")
            if current_params['Taxa_Compressao'] < 18.9:
                recs.append("Aumentar taxa de compressão para ~18.94")
            if current_params['Velocidade_Alimentador'] > 60:
                recs.append("Reduzir velocidade do alimentador para <60")
        
        elif predicted_pdi > 85:
            recs.append("Parâmetros ótimos - manter configuração atual")
        
        return recs
    
    def full_analysis(self, current_params):
        """Executa análise completa e retorna relatório"""
        alerts = self.check_parameters(current_params)
        predicted_pdi = self.predict_pdi(current_params)
        recommendations = self.generate_recommendations(current_params, predicted_pdi)
        
        report = {
            'predicted_pdi': round(predicted_pdi, 2),
            'alerts': alerts,
            'recommendations': recommendations,
            'status': 'OK' if not alerts and predicted_pdi >= 80 else 'ALERT'
        }
        
        return report