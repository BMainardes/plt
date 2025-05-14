# Importa a biblioteca pandas, que é usada para trabalhar com dados em tabelas (como planilhas do Excel)
import pandas as pd

# Importa a biblioteca numpy, que é usada para fazer cálculos numéricos com dados
import numpy as np

# Importa uma ferramenta do scikit-learn (biblioteca de aprendizado de máquina)
# que será usada para preencher dados faltantes (vazios) automaticamente
from sklearn.impute import SimpleImputer

# Define uma classe chamada DataProcessor.
# Uma "classe" é como uma receita para criar objetos que lidam com dados.
class DataProcessor:

    # Este é o método construtor da classe. Ele é chamado quando criamos um novo DataProcessor.
    def __init__(self, filepath):
        # Salva o caminho do arquivo que será carregado, como por exemplo "dados.csv"
        self.filepath = filepath
        # Cria um espaço vazio (None) onde os dados serão guardados depois de carregados
        self.df = None
        
    # Define um método que serve para carregar os dados do arquivo
    def load_data(self):
        """Carrega e prepara os dados iniciais"""
        try:
            # Tenta ler o arquivo CSV (separado por ponto e vírgula, com vírgula como separador decimal e codificação utf-8)
            self.df = pd.read_csv(self.filepath, sep=';', decimal=',', encoding='utf-8')
            # Chama um outro método da classe para renomear as colunas dos dados
            self._rename_columns()
            # Retorna True para indicar que deu tudo certo
            return True
        except Exception as e:
            # Se acontecer algum erro ao carregar o arquivo, mostra uma mensagem de erro
            print(f"Erro ao carregar dados: {e}")
            # Retorna False para indicar que houve falha
            return False
    
    # Método privado (interno) que padroniza os nomes das colunas para algo mais fácil de usar
    def _rename_columns(self):
        """Padroniza os nomes das colunas"""
        # Cria um dicionário (lista de pares chave-valor) com os nomes antigos e os novos nomes desejados
        column_mapping = {
            'Porc Pdi Tyl6': 'PDI',
            'Pres Vapor Caldeira': 'Pressao_Caldeira',
            'Tx Compressao Matriz': 'Taxa_Compressao',
            'Afastamento Rolos': 'Afastamento_Rolos',
            'Amperagem Condicionador': 'Amperagem_Condicionador',
            'Velocidade Alimentador': 'Velocidade_Alimentador',
            'Porc Temp Condicionador': 'Temp_Condicionador',
            'Pressao Vapor': 'Pressao_Vapor',
            'Amperagem Peletizadora': 'Amperagem_Peletizadora',
            'Porc Finos Tyl6': 'Finos'
        }
        # Renomeia as colunas do DataFrame usando o dicionário acima
        self.df = self.df.rename(columns=column_mapping)
    
    # Método que faz o pré-processamento dos dados: tratamento e preparação antes de usar
    def preprocess_data(self):
        """Executa todo o pré-processamento dos dados"""
        # Verifica se os dados foram carregados. Se não foram, mostra erro.
        if self.df is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
        
        # Chama o método para tratar os outliers (valores fora do normal)
        self._handle_outliers()
        # Chama o método para preencher valores ausentes (vazios)
        self._impute_missing_values()
        # Retorna os dados prontos (preprocessados)
        return self.df
    
    # Método que trata valores fora do normal (outliers) usando o método IQR
    def _handle_outliers(self):
        """Trata outliers usando o método IQR"""
        # Seleciona somente as colunas que têm números
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        # Para cada coluna numérica:
        for col in numeric_cols:
            # Calcula o primeiro quartil (25% dos dados)
            Q1 = self.df[col].quantile(0.25)
            # Calcula o terceiro quartil (75% dos dados)
            Q3 = self.df[col].quantile(0.75)
            # Calcula o intervalo interquartil (diferença entre Q3 e Q1)
            IQR = Q3 - Q1
            # Define o limite inferior como Q1 - 1.5 * IQR
            lower_bound = Q1 - 1.5 * IQR
            # Define o limite superior como Q3 + 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Substitui valores menores que o limite inferior por esse próprio limite
            # e valores maiores que o limite superior pelo próprio limite superior
            self.df[col] = np.where(
                self.df[col] < lower_bound, lower_bound, 
                np.where(
                    self.df[col] > upper_bound, upper_bound, self.df[col]
                )
            )
    
    # Método que preenche valores ausentes (nulos) com a mediana da coluna
    def _impute_missing_values(self):
        """Preenche valores faltantes com a mediana"""
        # Cria um objeto "imputador" que vai substituir valores ausentes pela mediana
        imputer = SimpleImputer(strategy='median')
        # Aplica esse imputador nos dados e recria o DataFrame com os mesmos nomes de colunas
        self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)
    
    # Método que retorna uma cópia dos dados processados (para que o original não seja alterado sem querer)
    def get_processed_data(self):
        """Retorna os dados processados"""
        return self.df.copy()
