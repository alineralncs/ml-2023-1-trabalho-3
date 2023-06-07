import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder


dataset = pd.read_csv('ds_salaries.csv')
THRESHOLD = 0.3

salario = "salary_in_usd"
COL_SALARIO = dataset[salario]

def transformar_econder(dataset):
    encoder = LabelEncoder()

    for coluna in dataset.columns:
        if dataset[coluna].dtypes == 'object':
            valores_antes = dataset[coluna].values
            tipo_antes_encoder =  dataset[coluna].dtypes
            dataset[coluna] = dataset[coluna].astype(str)
            dataset[coluna] = encoder.fit_transform(dataset[coluna].values)
            valores_depois = dataset[coluna].values
            # tipo_depois_encoder =  dataset[coluna].dtypes
            print(f'\n Coluna {coluna}: valores antes  {valores_antes} e valores depois {valores_depois} \n tipo antes {tipo_antes_encoder} e tipo depois {dataset[coluna].dtypes}')
    return dataset
def colunas_faltantes(dataset):
    percentual_dfaltantes = dataset.isna().sum() / len(dataset) * 100
    # Selecionando as colunas com mais de 90% de dados faltantes
    colunas_dfaltantes = percentual_dfaltantes[percentual_dfaltantes > 90].index
    cols_dfaltantes = dataset[colunas_dfaltantes]
    print(f'\n Colunas com mais de 90% de dados faltantes: { colunas_dfaltantes }')
    return cols_dfaltantes
def preencher_mediana(dataset):
    return dataset.fillna(dataset.median())
def correlacao_forte(dataset):
    result_correlacao = dataset.corrwith(COL_SALARIO, numeric_only=True)
    colunas_correlacionadas = result_correlacao[result_correlacao.abs() > THRESHOLD].index
    print(f"Colunas com correlação acima de {THRESHOLD}: {colunas_correlacionadas}")    
    print(f"Colunas com correlação acima de {THRESHOLD}: {colunas_correlacionadas}") 
    result_correlacao.plot(kind='bar', figsize=(10, 6))
    # Configuração dos rótulos dos eixos x e y e título do gráfico
    plt.xlabel('Variáveis', fontsize=12)
    plt.ylabel('Correlação', fontsize=12)
    plt.title(f'Correlação das colunas com {COL_SALARIO}', fontsize=14)
    # Adicionar legenda
    plt.legend(['Correlação'], loc='best', fontsize=10)
    # Exibir o gráfico
    plt.show()
    return colunas_correlacionadas


transformar_econder(dataset)
colunas_faltantes(dataset)
preencher_mediana(dataset)
dataset_correlacionado = correlacao_forte(dataset)


dataset = dataset.drop(columns=set(dataset.columns) - set(dataset_correlacionado))

print(f'dataset columns {dataset.columns}')

# Divisão do dataset em features (X) e target (y)
X = dataset.drop(columns=["salary_in_usd"])
y = dataset["salary_in_usd"]

# Divisão do dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pré-processamento dos dados com escalonamento
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criação e treinamento do modelo MLPRegressor
model = MLPRegressor(hidden_layer_sizes=(100, 100), activation='relu', random_state=42, max_iter=500)
model.fit(X_train_scaled, y_train)

# Predição do salário para os dados de teste
y_pred = model.predict(X_test_scaled)

# Cálculo da métrica de avaliação (mean absolute error)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")

# Example Profiles
example_profiles = X_test_scaled[:3]  # Selecting the first three data points from the test set

# Predicted Salaries for Example Profiles
predicted_salaries = model.predict(example_profiles)

# Displaying Example Profiles and Predicted Salaries
for i, example_profile in enumerate(example_profiles):
    print(f"\nExample Profile {i+1}:")
    for j, coluna in enumerate(X.columns):
        print(f"{coluna}: {example_profile[j]}")
    print(f"Predicted Salary: {predicted_salaries[i]:.2f} USD")

# Displaying Example Profiles and Predicted Salaries
for i, example_profile in enumerate(example_profiles):
    print(f"\nExample Profile {i+1}:")
    for j, coluna in enumerate(X.columns):
        original_value = X_test.iloc[i][coluna]  # Get the original value from the X_test dataframe
        print(f"{coluna}: {original_value}")
    print(f"Predicted Salary: {predicted_salaries[i]:.2f} USD")
