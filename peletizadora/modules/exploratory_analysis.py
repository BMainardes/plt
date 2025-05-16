import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from config import PLOT_DIR

class ExploratoryAnalysis:
    
    def __init__(self, dataframe):
        self.df = dataframe
    
    def generate_summary(self):
        summary = {
            'info': self.df.info(),
            'describe': self.df.describe(),
            'null_values': self.df.isnull().sum(),
            'correlation': self.df.corr()
        }
        return summary
    
    def plot_distributions(self, save=False):
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
        
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(self.df[col], kde=True)
            plt.title(f'Distribuição de {col}')
            if save:
                plt.savefig(f"{PLOT_DIR}/distribution_{col}.png")
                plt.close()
            else:
                plt.show()
    
    def plot_correlation_matrix(self, save=False):
        plt.figure(figsize=(12, 8))
        corr = self.df.corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlação')
        if save:
            plt.savefig(f"{PLOT_DIR}/correlation_matrix.png")
            plt.close()
        else:
            plt.show()
    
    def plot_key_relationships(self, target_var='PDI', save=False):
        key_vars = ['Finos', 'Amperagem_Peletizadora', 'Taxa_Compressao', 'Velocidade_Alimentador']
        
        Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)
        
        for var in key_vars:
            plt.figure(figsize=(8, 4))
            sns.regplot(x=var, y=target_var, data=self.df)
            plt.title(f'Relação entre {var} e {target_var}')
            if save:
                plt.savefig(f"{PLOT_DIR}/relation_{var}_vs_{target_var}.png")
                plt.close()
            else:
                plt.show()
