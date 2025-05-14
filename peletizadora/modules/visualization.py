import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualization:
    @staticmethod
    def plot_combined_results(model, df, features, model_step_name='model'):
        """
        Plota os 4 gráficos combinados com nome do passo configurável
        """
        plt.figure(figsize=(20, 15))
        
        # Gráfico 1: Importância das Variáveis
        plt.subplot(2, 2, 1)
        Visualization._plot_feature_importance(model, features, model_step_name)
        
        # Gráfico 2: Relação Temperatura vs PDI
        plt.subplot(2, 2, 2)
        Visualization._plot_temp_vs_pdi(df)
        
        # Gráfico 3: Relação Amperagem vs PDI
        plt.subplot(2, 2, 3)
        Visualization._plot_amperage_vs_pdi(df)
        
        # Gráfico 4: Matriz de Correlação
        plt.subplot(2, 2, 4)
        Visualization._plot_correlation_matrix(df, features)
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_feature_importance(model, features, model_step_name):
        """Gráfico de importância com nome do passo configurável"""
        try:
            if hasattr(model.named_steps[model_step_name], 'feature_importances_'):
                importances = model.named_steps[model_step_name].feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.title('Importância das Variáveis', fontsize=14)
                bars = plt.barh(range(len(indices)), importances[indices], color='royalblue')
                plt.yticks(range(len(indices)), [features[i] for i in indices])
                plt.xlabel('Importância Relativa', fontsize=12)
                
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{width:.2f}', ha='left', va='center')
        except Exception as e:
            plt.title('Importância não disponível')
            print(f"Aviso: Não foi possível plotar importância - {e}")

    # ... (outros métodos permanecem iguais)

    @staticmethod
    def _plot_temp_vs_pdi(df):
        """Relação entre temperatura do condicionador e PDI"""
        sns.regplot(x='Porc Temp Condicionador', y='PDI', data=df, 
                   scatter_kws={'alpha':0.6})
        plt.axvline(x=78, color='r', linestyle='--')
        plt.title('Relação entre Temperatura e PDI')

    @staticmethod
    def _plot_amperage_vs_pdi(df):
        """Relação entre amperagem da peletizadora e PDI"""
        sns.regplot(x='Amperagem Peletizadora', y='PDI', data=df, 
                   scatter_kws={'alpha':0.6})
        plt.axvline(x=650, color='r', linestyle='--')
        plt.title('Relação entre Amperagem e PDI')

    @staticmethod
    def _plot_correlation_matrix(df, features):
        """Matriz de correlação entre variáveis"""
        corr = df[features + ['PDI']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlação')