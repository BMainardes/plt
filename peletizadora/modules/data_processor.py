import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from config import DATA_PATH

class DataProcessor:

    def __init__(self, filepath=DATA_PATH):
        self.filepath = filepath
        self.df = None
        
    def load_data(self):
        """Carrega e prepara os dados iniciais"""
        try:
            self.df = pd.read_csv(self.filepath, sep=';', decimal=',', encoding='utf-8')
            self._rename_columns()
            print(f"Dados carregados com sucesso de {self.filepath}")
            return True
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return False
    
    def _rename_columns(self):
        """Padroniza os nomes das colunas"""
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
        self.df = self.df.rename(columns=column_mapping)
    
    def preprocess_data(self):
        """Executa todo o pré-processamento dos dados"""
        if self.df is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
        
        self._handle_outliers()
        self._impute_missing_values()

        # Verifica se as colunas essenciais estão presentes
        required_columns = ['PDI', 'Finos']
        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(f"Coluna obrigatória '{col}' não encontrada nos dados.")

        print("Pré-processamento concluído.")
        return self.df
    
    def _handle_outliers(self):
        """Trata outliers usando o método IQR"""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            self.df[col] = np.where(
                self.df[col] < lower_bound, lower_bound, 
                np.where(
                    self.df[col] > upper_bound, upper_bound, self.df[col]
                )
            )
    
    def _impute_missing_values(self):
        """Preenche valores faltantes com a mediana"""
        imputer = SimpleImputer(strategy='median')
        self.df = pd.DataFrame(imputer.fit_transform(self.df), columns=self.df.columns)
    
    def get_processed_data(self):
        """Retorna os dados processados"""
        return self.df.copy()
