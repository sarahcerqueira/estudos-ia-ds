import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


''' Carregamento dos dados '''

base_credit = pd.read_csv('credit_data.csv')
base_credit.head() # Trás os 5 primeiros registros por padrão, mas pode ser passada a quantidade por parametro
base_credit.tail() # Trás os ultimos regitros
base_credit.describe() # Trás uma descrição dos dados com média, mínimo, máximo, desvio padrçao, etc
base_credit[base_credit['income']>= 69995] # Busca um dado de acordo com o filtro da condição

np.unique(base_credit['default']) #Retorna as labels que existem em determinada coluna
np.unique(base_credit['default'], return_counts=True) #Retorna as labels que existem em determinada coluna e a quantidades que tem esses valores

sns.countplot(x= base_credit['default']) #Gera um gráfico de barra com a quantidade de cada label
plt.show() # Mostra o gráfico

plt.hist(x = base_credit['loan']) #Cria um histograma, pode-se verificar a distribuição de frequencia
plt.show() # Mostra o histograma

grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default') # plota um gráfico de distribuição
grafico.show()

base_credit.loc[base_credit['age']< 0]
base_credit[base_credit['age']< 0]


base_credit.loc[(base_credit['clientid'] == 29) | (base_credit['clientid'] == 31) | (base_credit['clientid'] == 32)] #Retorna clientes com ids especificicos
base_credit.loc[base_credit['clientid'].isin([29, 31, 32])] #Outra maneira de fazer a consulta à cima
