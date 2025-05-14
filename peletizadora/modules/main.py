from data_processor import DataProcessor
from exploratory_analysis import ExploratoryAnalysis
from model_trainer import ModelTrainer
from visualization import Visualization
from monitoring_system import PDIMonitor

def main():
    # 1. Carregar e processar dados
    data_processor = DataProcessor(r'D:\Pioneiro\Desktop\New.csv')
    if not data_processor.load_data():
        return
    
    df = data_processor.preprocess_data()
    
    # 2. Análise exploratória
    explorer = ExploratoryAnalysis(df)
    explorer.plot_distributions()
    explorer.plot_correlation_matrix()
    explorer.plot_key_relationships()
    
    # 3. Preparar dados para modelagem
    X = df.drop(['PDI', 'Finos'], axis=1)
    y = df['PDI']
    
    # 4. Treinar modelo
    trainer = ModelTrainer()
    model = trainer.train_model(X, y)
    trainer.save_model('pdi_model.pkl')
    
    # Obter feature importance
    feature_importance = trainer.get_feature_importance()
    if feature_importance is not None:
        print(feature_importance.head())

    # Ou usar o método auxiliar para imprimir
    trainer._print_feature_importance()  # Também funciona!

    # Preparar features para visualização
    features = X.columns.tolist()  # Usa os nomes reais das colunas
    
    # Cria cópia do DF com nomes específicos para os gráficos
    df_vis = df.rename(columns={
        'Temp_Condicionador': 'Porc Temp Condicionador',
        'Amperagem_Peletizadora': 'Amperagem Peletizadora'
    })
    
    # 5. Visualizar resultados combinados
    Visualization.plot_combined_results(model, df_vis, features)
    
    # Visualizações adicionais
    feature_importance = trainer.get_feature_importance()
    Visualization.plot_feature_importance(feature_importance)
    
    y_pred = model.predict(X)
    Visualization.plot_actual_vs_predicted(y, y_pred)
    Visualization.plot_residuals(y, y_pred)
    
    # 6. Análise de sensibilidade (usando DF original)
    Visualization.plot_sensitivity_analysis(
        df, 'Amperagem_Peletizadora', 'Taxa_Compressao', 'PDI', model
    )
    
    # 7. Exemplo de uso do sistema de monitoramento
    monitor = PDIMonitor('pdi_model.pkl')
    
    current_operation = {
        'Pressao_Caldeira': 8.9,
        'Taxa_Compressao': 18.75,
        'Afastamento_Rolos': 0,
        'Amperagem_Condicionador': 38,
        'Velocidade_Alimentador': 76,
        'Temp_Condicionador': 80,
        'Pressao_Vapor': 1.4,
        'Amperagem_Peletizadora': 630
    }
    
    report = monitor.full_analysis(current_operation)
    print("\nRelatório de Monitoramento:")
    print(f"PDI Previsto: {report['predicted_pdi']}%")
    print("\nAlertas:")
    print("\n".join(report['alerts']) if report['alerts'] else "Nenhum alerta")
    print("\nRecomendações:")
    print("\n".join(report['recommendations']) if report['recommendations'] else "Nenhuma recomendação")

if __name__ == "__main__":
    main()