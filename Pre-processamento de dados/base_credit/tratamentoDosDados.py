import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


''' Carregamento dos dados '''

base_credit = pd.read_csv('credit_data.csv')

''' Tratamento de dados'''

#Em algumas situações a melhor forma de resolver o problema é removendo toda a coluna
base_credit2 = base_credit.drop('age', axis=1) # Remove a coluna, caso quisesse remover a linha axis = 0

# Remover os clientes que tem os registros com problema
base_credit3 = base_credit.drop(base_credit[base_credit['age']< 0].index)

# Preencher os valores faltantes com a média

base_credit.mean() # retorna a média de todos os atributos
base_credit['age'].mean() #Retorna a média de uma label

base_credit['age'][base_credit['age']> 0].mean() #Retorna a média apenas das idades válidas
base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92 #Atualiza os registros com a média

#Geração de gráficos após tratamento das idades negativas (alores inválidos)
sns.countplot(x= base_credit['default'])
plt.show() 

plt.hist(x = base_credit['loan'])
plt.show() 

grafico = px.scatter_matrix(base_credit, dimensions=['age', 'income', 'loan'], color='default')
grafico.show()