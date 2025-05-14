# Importa a biblioteca matplotlib, usada para criar gráficos. Aqui usamos a parte chamada "pyplot".
import matplotlib.pyplot as plt

# Importa a biblioteca seaborn, que deixa os gráficos mais bonitos e mais fáceis de interpretar
import seaborn as sns

# Importa a biblioteca pandas, que permite trabalhar com dados em formato de tabela
import pandas as pd

# Define uma classe chamada ExploratoryAnalysis (análise exploratória)
# Uma classe é como um conjunto de funções que trabalham com os mesmos dados
class ExploratoryAnalysis:
    
    # Método que é executado quando a classe é criada.
    # Recebe um "dataframe", que é uma tabela com dados (vinda do pandas)
    def __init__(self, dataframe):
        # Salva a tabela recebida dentro da classe para uso nos outros métodos
        self.df = dataframe
    
    # Método para gerar um resumo estatístico dos dados
    def generate_summary(self):
        """Gera um resumo estatístico dos dados"""
        summary = {
            # Retorna informações básicas sobre os dados, como nomes das colunas e tipos
            'info': self.df.info(),
            # Calcula estatísticas como média, mínimo, máximo, desvio padrão etc.
            'describe': self.df.describe(),
            # Conta quantos valores estão ausentes (vazios) em cada coluna
            'null_values': self.df.isnull().sum(),
            # Calcula a correlação entre colunas numéricas (como elas se relacionam)
            'correlation': self.df.corr()
        }
        # Retorna o dicionário com o resumo
        return summary
    
    # Método que mostra gráficos com a distribuição dos valores de cada variável numérica
    def plot_distributions(self):
        """Plota distribuições das variáveis"""
        # Pega somente as colunas que têm números do tipo float (decimal) ou int (inteiro)
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        
        # Para cada coluna numérica encontrada:
        for col in numeric_cols:
            # Cria uma nova figura de tamanho 8x4 (polegadas)
            plt.figure(figsize=(8, 4))
            # Cria um histograma com curva KDE para mostrar a distribuição dos valores
            sns.histplot(self.df[col], kde=True)
            # Coloca um título no gráfico com o nome da coluna
            plt.title(f'Distribuição de {col}')
            # Exibe o gráfico na tela
            plt.show()
    
    # Método que mostra a matriz de correlação (gráfico colorido com a relação entre variáveis)
    def plot_correlation_matrix(self):
        """Plota matriz de correlação"""
        # Cria uma nova figura de tamanho 12x8
        plt.figure(figsize=(12, 8))
        # Calcula a correlação entre as colunas numéricas
        corr = self.df.corr()
        # Cria o gráfico de calor (heatmap) com os valores da correlação
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        # Coloca um título no gráfico
        plt.title('Matriz de Correlação')
        # Mostra o gráfico na tela
        plt.show()
    
    # Método que mostra gráficos com a relação entre algumas variáveis importantes e a variável "PDI"
    def plot_key_relationships(self, target_var='PDI'):
        """Plota relações entre variáveis-chave e o target"""
        # Lista de variáveis importantes que queremos comparar com o PDI
        key_vars = ['Finos', 'Amperagem_Peletizadora', 'Taxa_Compressao', 'Velocidade_Alimentador']
        
        # Para cada uma dessas variáveis:
        for var in key_vars:
            # Cria uma nova figura de tamanho 8x4
            plt.figure(figsize=(8, 4))
            # Cria um gráfico de dispersão com linha de tendência (reta de regressão)
            sns.regplot(x=var, y=target_var, data=self.df)
            # Coloca um título no gráfico explicando o que está sendo comparado
            plt.title(f'Relação entre {var} e {target_var}')
            # Mostra o gráfico na tela
            plt.show()
