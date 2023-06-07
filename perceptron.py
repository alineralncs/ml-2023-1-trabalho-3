# import pandas as pd

# import matplotlib.pyplot as plt

# import locale

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.neural_network import MLPRegressor
# from sklearn.metrics import mean_absolute_error
# from sklearn.preprocessing import LabelEncoder
# from sklearn.neural_network import MLPClassifier


# locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

# dataset = pd.read_csv('ds_salaries.csv')

# THRESHOLD = 0.4

# salario = "salary_in_usd"
# COL_SALARIO = dataset[salario]

# def transformar_econder(dataset):
#     encoder = LabelEncoder()

#     for coluna in dataset.columns:
#         if dataset[coluna].dtypes == 'object':
#             valores_antes = dataset[coluna].values
#             tipo_antes_encoder =  dataset[coluna].dtypes
#             dataset[coluna] = dataset[coluna].astype(str)
#             dataset[coluna] = encoder.fit_transform(dataset[coluna].values)
#             valores_depois = dataset[coluna].values
#             # tipo_depois_encoder =  dataset[coluna].dtypes
#             print(f'\n Coluna {coluna}: valores antes  {valores_antes} e valores depois {valores_depois} \n tipo antes {tipo_antes_encoder} e tipo depois {dataset[coluna].dtypes}')
#     return dataset
    
# def colunas_faltantes(dataset):
#     percentual_dfaltantes = dataset.isna().sum() / len(dataset) * 100
#     # Selecionando as colunas com mais de 90% de dados faltantes
#     colunas_dfaltantes = percentual_dfaltantes[percentual_dfaltantes > 90].index
#     cols_dfaltantes = dataset[colunas_dfaltantes]
#     print(f'\n Colunas com mais de 90% de dados faltantes: { colunas_dfaltantes }')
#     return cols_dfaltantes
    
# def preencher_mediana(dataset):
#     return dataset.fillna(dataset.median())

# def correlacao_forte(dataset):
#     result_correlacao = dataset.corrwith(COL_SALARIO, numeric_only=True)
#     colunas_correlacionadas = result_correlacao[result_correlacao.abs() > THRESHOLD].index
#     #print(f"Colunas com correlação acima de {THRESHOLD}: {colunas_correlacionadas}")    
#     print(f"Colunas com correlação acima de {THRESHOLD}: {colunas_correlacionadas}") 
#     result_correlacao.plot(kind='bar', figsize=(10, 6))
#     # Configuração dos rótulos dos eixos x e y e título do gráfico
#     plt.xlabel('Variáveis', fontsize=12)
#     plt.ylabel('Correlação', fontsize=12)
#     plt.title(f'Correlação das colunas com {COL_SALARIO}', fontsize=14)
#     # Adicionar legenda
#     plt.legend(['Correlação'], loc='best', fontsize=10)
#     # Exibir o gráfico
#     plt.show()
#     return colunas_correlacionadas


# transformar_econder(dataset)
# colunas_faltantes(dataset)
# preencher_mediana(dataset)
# correlacao_forte(dataset)

# dataset = dataset.drop(columns=["job_title", "salary", "salary_currency"])

# print(f'dataset columns {dataset.columns}')

# X = dataset.drop(columns=["salary_in_usd"])
# y = dataset["salary_in_usd"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaler = MinMaxScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# model = MLPClassifier(hidden_layer_sizes=(400, 400), activation='relu', random_state=42, max_iter=500)

# model.fit(X_train_scaled, y_train)

# y_pred = model.predict(X_test_scaled)


# mae = mean_absolute_error(y_test, y_pred)
# print(f"Mean Absolute Error: {mae:.2f}")

# example_profiles = X_test_scaled[:3000] 

# print(f'exemplos: {example_profiles}')
# predicted_salaries = model.predict(example_profiles)

# for i, example_profile in enumerate(example_profiles):
#     print(f"\nExample Profile {i+1}:")

#     print(f"Predicted Salary: {locale.currency(predicted_salaries[i], grouping=True)}")


# plt.scatter(range(len(y_test)), y_test, color='pink', label='Valores Reais')
# plt.scatter(range(len(predicted_salaries)), predicted_salaries, color='blue', label='Valores Previstos')

# plt.xlabel('Exemplo')
# plt.ylabel('Salário em USD')
# plt.title('Valores Reais vs. Valores Previstos')

# # Adicionar legenda
# plt.legend(loc='best')

# # Exibir o gráfico
# plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import locale
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

dataset = pd.read_csv('ds_salaries.csv')

THRESHOLD = 0.4

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
            print(f"{coluna}: {list(zip(encoder.classes_, encoder.transform(encoder.classes_)))}")
            # tipo_depois_encoder =  dataset[coluna].dtypes
            print(f'\n Coluna {coluna}: valores antes  {valores_antes} e valores depois {valores_depois} \n tipo antes {tipo_antes_encoder} e tipo depois {dataset[coluna].dtypes}')
    return dataset

def colunas_faltantes(dataset):
    percentual_dfaltantes = dataset.isna().sum() / len(dataset) * 100
    # Selecionando as colunas com mais de 90% de dados faltantes
    colunas_dfaltantes = percentual_dfaltantes[percentual_dfaltantes > 90].index
    cols_dfaltantes = dataset[colunas_dfaltantes]
    print(f'\n Colunas com mais de 70% de dados faltantes: { colunas_dfaltantes }')
    return cols_dfaltantes

def preencher_mediana(dataset):
    return dataset.fillna(dataset.median())

def correlacao_forte(dataset):
    result_correlacao = dataset.corrwith(COL_SALARIO, numeric_only=True)
    colunas_correlacionadas = result_correlacao[result_correlacao.abs() > THRESHOLD].index
    #print(f"Colunas com correlação acima de {THRESHOLD}: {colunas_correlacionadas}")    
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

dataset = dataset.drop(columns=["job_title", "salary", "salary_currency"])

transformar_econder(dataset)
colunas_faltantes(dataset)
preencher_mediana(dataset)
correlacao_forte(dataset)

print(f'dataset columns {dataset.columns}')

X = dataset.drop(columns=["salary_in_usd"])
y = dataset["salary_in_usd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(400, 400), activation='relu', random_state=42, max_iter=500)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
accuracy = model.score(X_test_scaled, y_test)  

print(f"Mean Absolute Error: {mae:.2f}")
print(f"Acurácia: {accuracy:.2f}")



def create_user_input():
    work_year = input("Enter work year: ")
    experience_level = input("Enter experience level: ")
    employment_type = input("Enter employment type: ")
    # job_title = input("Enter job title: ")
    # salary = input("Enter salary: ")
    # salary_currency = input("Enter salary currency: ")
    # salary_in_usd = input("Enter salary in USD: ")
    employee_residence = input("Enter employee residence: ")
    remote_ratio = input("Enter remote ratio: ")
    company_location = input("Enter company location: ")
    company_size = input("Enter company size: ")

    user_input = {
        'work_year': work_year,
        'experience_level': experience_level,
        'employment_type': employment_type,
        # 'job_title': job_title,
        # 'salary': salary,
        # 'salary_currency': salary_currency,
        # 'salary_in_usd': salary_in_usd,
        'employee_residence': employee_residence,
        'remote_ratio': remote_ratio,
        'company_location': company_location,
        'company_size': company_size,
    }

    # Create a DataFrame using the user input dictionary
    user_input_df = pd.DataFrame(user_input, index=[0])

    # Reorder the columns to match the training dataset
    user_input_df = user_input_df[[
        'work_year', 'experience_level', 'employment_type','employee_residence',
        'remote_ratio', 'company_location', 'company_size'
    ]]

    return user_input_df

user_input = create_user_input()

user_input_scaled = scaler.transform(user_input)

predicted_salary = model.predict(user_input_scaled)

formatted_salary = locale.currency(predicted_salary[0], grouping=True)
print(f"Predicted Salary: {formatted_salary}")

plt.scatter(range(len(y_test)), y_test, color='pink', label='Valores Reais')
plt.scatter(range(len(predicted_salary)), predicted_salary, color='blue', label='Valores Previstos')

plt.xlabel('Exemplo')
plt.ylabel('Salário em USD')
plt.title('Valores Reais vs. Valores Previstos')

plt.legend(loc='best')
plt.show()
