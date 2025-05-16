from pathlib import Path
from config import REPORT_DIR
import pandas as pd

class PDIReport:
    @staticmethod
    def generate_recommendation(feature_importance_df, threshold=0.05):
        recommendations = []
        if 'Importance' in feature_importance_df.columns:
            top_vars = feature_importance_df[feature_importance_df['Importance'] > threshold]
            for _, row in top_vars.iterrows():
                direction = "aumentar" if row['Importance'] > 0 else "reduzir"
                recommendations.append(f"Tente {direction} a variável '{row['Feature']}' para melhorar o PDI.")
        elif 'Coefficient' in feature_importance_df.columns:
            top_vars = feature_importance_df[feature_importance_df['Coefficient'].abs() > threshold]
            for _, row in top_vars.iterrows():
                direction = "aumentar" if row['Coefficient'] > 0 else "reduzir"
                recommendations.append(f"Tente {direction} a variável '{row['Feature']}' para melhorar o PDI.")
        return recommendations

    @staticmethod
    def save_report(recommendations, model_name):
        Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
        report_path = Path(REPORT_DIR) / f"recomendacoes_{model_name}.txt"
        with open(report_path, 'w', encoding='utf-8') as file:
            file.write("Recomendações para melhorar o PDI:\n\n")
            if recommendations:
                for rec in recommendations:
                    file.write(f"- {rec}\n")
            else:
                file.write("Nenhuma recomendação gerada.\n")
        print(f"Relatório salvo em {report_path}")

    @staticmethod
    def save_ideal_settings(settings_dict, model_name):
        """
        Salva as configurações ideais em um arquivo txt, excluindo PDI e Finos.
        """
        if not settings_dict:
            print("Nenhuma configuração ideal para salvar.")
            return
        Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
        report_path = Path(REPORT_DIR) / f"configuracoes_ideais_{model_name}.txt"
        with open(report_path, 'w', encoding='utf-8') as file:
            file.write("Configurações Ideais para PDI Alto e Finos Baixos:\n\n")
            for key, value in settings_dict.items():
                if key not in ['PDI', 'Finos']:
                    file.write(f"{key}: {value:.4f}\n")
        print(f"Configurações ideais salvas em {report_path}")
