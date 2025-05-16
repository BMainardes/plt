import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from config import PLOT_DIR

class Visualization:
    @staticmethod
    def plot_combined_results(model, df, features, model_step_name='regressor', save=False):
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
        if save:
            Visualization.save_plot("combined_results")
        else:
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
            elif hasattr(model.named_steps[model_step_name], 'coef_'):
                coefs = model.named_steps[model_step_name].coef_
                indices = np.argsort(np.abs(coefs))[::-1]
                plt.title('Importância das Variáveis (coeficientes)', fontsize=14)
                bars = plt.barh(range(len(indices)), coefs[indices], color='royalblue')
                plt.yticks(range(len(indices)), [features[i] for i in indices])
                plt.xlabel('Coeficiente', fontsize=12)
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                             f'{width:.2f}', ha='left', va='center')
            else:
                plt.title('Importância não disponível')
        except Exception as e:
            plt.title('Importância não disponível')
            print(f"Aviso: Não foi possível plotar importância - {e}")

    @staticmethod
    def _plot_temp_vs_pdi(df):
        """Relação entre temperatura do condicionador e PDI"""
        sns.regplot(x='Temp_Condicionador', y='PDI', data=df, scatter_kws={'alpha':0.6})
        plt.axvline(x=78, color='r', linestyle='--')
        plt.title('Relação entre Temperatura e PDI')

    @staticmethod
    def _plot_amperage_vs_pdi(df):
        """Relação entre amperagem da peletizadora e PDI"""
        sns.regplot(x='Amperagem_Peletizadora', y='PDI', data=df, scatter_kws={'alpha':0.6})
        plt.axvline(x=650, color='r', linestyle='--')
        plt.title('Relação entre Amperagem e PDI')

    @staticmethod
    def _plot_correlation_matrix(df, features):
        """Matriz de correlação entre variáveis"""
        corr = df[features + ['PDI']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlação')

    @staticmethod
    def plot_actual_vs_predicted(y_true, y_pred, save=False):
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.xlabel('Valores Reais')
        plt.ylabel('Valores Preditos')
        plt.title('Valores Reais vs Preditos')
        if save:
            Visualization.save_plot("actual_vs_predicted")
        else:
            plt.show()

    @staticmethod
    def plot_residuals(y_true, y_pred, save=False):
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel('Resíduos')
        plt.title('Distribuição dos Resíduos')
        if save:
            Visualization.save_plot("residuals")
        else:
            plt.show()

    @staticmethod
    def save_plot(filename):
        Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{PLOT_DIR}/{filename}.png")
        plt.close()
