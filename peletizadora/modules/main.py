from data_processor import DataProcessor
from exploratory_analysis import ExploratoryAnalysis
from model_trainer import ModelTrainer
from visualization import Visualization
from monitoring_system import PDIMonitor
from report import PDIReport
from config import DATA_PATH, IMPORTANCE_THRESHOLD, MODEL_DIR
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model', choices=['bayesian', 'random_forest', 'gradient_boosting'], default='random_forest')
args = parser.parse_args()

def find_ideal_settings(df, target_pdi='PDI', target_finos='Finos'):
    # Filtro de linhas onde o PDI é alto e Finos são baixos
    filtered = df[(df[target_pdi] >= 82) & (df[target_finos] <= 10)]
    if filtered.empty:
        print("Nenhum dado encontrado com PDI >= 82 e Finos <= 10 para definir configurações ideais.")
        return {}

    # Calcular a mediana das variáveis de controle (excluir PDI e Finos)
    cols = [col for col in df.columns if col not in [target_pdi, target_finos]]
    ideal_settings = filtered[cols].median().to_dict()
    return ideal_settings

def run_training_and_evaluation(model_type, X, y, df, features):
    print(f"\n=== Treinando e avaliando modelo: {model_type} ===")
    trainer = ModelTrainer(model_type=model_type)
    model = trainer.train_model(X, y)

    trainer.save_model(f'pdi_model_{model_type}.pkl')

    feature_importance = trainer.get_feature_importance()
    if feature_importance is not None:
        print(feature_importance.head())
    trainer._print_feature_importance()

    df_vis = df.copy()
    Visualization.plot_combined_results(model, df_vis, features, model_step_name='regressor')
    Visualization._plot_feature_importance(model, features, model_step_name='regressor')

    y_pred = model.predict(X)
    Visualization.plot_actual_vs_predicted(y, y_pred)
    Visualization.plot_residuals(y, y_pred)

    recommendations = PDIReport.generate_recommendation(feature_importance, threshold=IMPORTANCE_THRESHOLD)
    PDIReport.save_report(recommendations, model_type)

    # Gerar e salvar configurações ideais com base nos dados
    ideal_settings = find_ideal_settings(df)
    PDIReport.save_ideal_settings(ideal_settings, model_type)

def main():
    data_processor = DataProcessor(DATA_PATH)
    if not data_processor.load_data():
        return

    df = data_processor.preprocess_data()

    explorer = ExploratoryAnalysis(df)
    explorer.plot_distributions(save=True)
    explorer.plot_correlation_matrix(save=True)
    explorer.plot_key_relationships(save=True)

    X = df.drop(['PDI', 'Finos'], axis=1)
    y = df['PDI']
    features = X.columns.tolist()

    run_training_and_evaluation(args.model, X, y, df, features)

    model_path = os.path.join(MODEL_DIR, f'pdi_model_{args.model}.pkl')
    monitor = PDIMonitor(model_path)
    current_operation = {
        'Pressao_Caldeira': 8.9,
        'Taxa_Compressao': 18.75,
        'Afastamento_Rolos': 0,
        'Amperagem_Condicionador': 38,
        'Velocidade_Alimentador': 76,
        'Temp_Condicionador': 80,
        'Pressao_Vapor': 1.4,
        'Amperagem_Peletizadora': 630,
        'Finos': 11.0
    }
    report = monitor.full_analysis(current_operation)

    print("\nRelatório de Monitoramento:")
    print(f"PDI Previsto: {report['predicted_pdi']}%")
    print("\nAlertas:")
    print("\n".join(report['alerts']) if report['alerts'] else "Nenhum alerta")
    print("\nRecomendações:")
    print("\n".join(report['recommendations']) if report['recommendations'] else "Nenhuma recomendação")

    ideal_pdi = report['predicted_pdi'] > 82
    ideal_finos = current_operation['Finos'] < 10
    print("\nCenário ideal atingido:", "Sim" if ideal_pdi and ideal_finos else "Não")

if __name__ == "__main__":
    main()
