# Caminhos padrão e limites de controle

# Caminho para os dados de entrada
DATA_PATH = "data/Novo.csv"

# Caminho onde os modelos serão salvos
MODEL_DIR = "output/models"

# Caminho para salvar relatórios e gráficos
REPORT_DIR = "output/reports"
PLOT_DIR = "output/plots"

# Limites de controle operacionais (podem ser usados no monitoramento)
CONTROL_LIMITS = {
    'Amperagem_Peletizadora': (630, 660),
    'Taxa_Compressao': (18.8, 19.0),
    'Velocidade_Alimentador': (35, 60)
}

# Threshold para considerar uma variável importante
IMPORTANCE_THRESHOLD = 0.05
