import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Visualization:
    @staticmethod
    def plot_combined_results(model, df, features):
        """
        Plota os 4 gráficos combinados conforme especificado
        
        Args:
            model: Modelo treinado (deve ter feature_importances_)
            df: DataFrame com os dados (já com nomes de colunas corrigidos)
            features: Lista de nomes das features originais
        """
        # Mapeamento de nomes para os gráficos
        feature_names_for_plot = {
            'Temp_Condicionador': 'Porc Temp Condicionador',
            'Amperagem_Peletizadora': 'Amperagem Peletizadora'
        }
        
        # Converter nomes das features para exibição
        display_features = [feature_names_for_plot.get(f, f) for f in features]
        
        plt.figure(figsize=(18, 12))
        
        # Gráfico 1: Importância das Variáveis
        plt.subplot(2, 2, 1)
        if hasattr(model.named_steps['regressor'], 'feature_importances_'):
            importancias = model.named_steps['regressor'].feature_importances_
            indices = np.argsort(importancias)[::-1]
            
            plt.title('Importância das Variáveis')
            plt.barh(range(len(indices)), importancias[indices], color='b', align='center')
            plt.yticks(range(len(indices)), [display_features[i] for i in indices])
            plt.xlabel('Importância Relativa')
        
        # Gráfico 2: Relação Temperatura vs PDI
        plt.subplot(2, 2, 2)
        if 'Porc Temp Condicionador' in df.columns:
            sns.regplot(x='Porc Temp Condicionador', y='PDI', data=df, scatter_kws={'alpha':0.6})
            plt.axvline(x=78, color='r', linestyle='--')
            plt.title('Relação entre Temperatura e PDI')
        
        # Gráfico 3: Relação Amperagem vs PDI
        plt.subplot(2, 2, 3)
        if 'Amperagem Peletizadora' in df.columns:
            sns.regplot(x='Amperagem Peletizadora', y='PDI', data=df, scatter_kws={'alpha':0.6})
            plt.axvline(x=650, color='r', linestyle='--')
            plt.title('Relação entre Amperagem e PDI')
        
        # Gráfico 4: Matriz de Correlação
        plt.subplot(2, 2, 4)
        cols_for_corr = [feature_names_for_plot.get(f, f) for f in features] + ['PDI']
        cols_for_corr = [c for c in cols_for_corr if c in df.columns]
        
        if len(cols_for_corr) > 1:
            corr = df[cols_for_corr].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
            plt.title('Matriz de Correlação')
        
        plt.tight_layout()
        plt.show()

    # ... (mantenha os outros métodos existentes)

    @staticmethod
    def _plot_feature_importance(model, features):
        """Gráfico de importância das variáveis"""
        importancias = model.named_steps['regressor'].feature_importances_
        indices = np.argsort(importancias)[::-1]
        
        plt.title('Importância das Variáveis')
        plt.barh(range(len(indices)), importancias[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Importância Relativa')

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