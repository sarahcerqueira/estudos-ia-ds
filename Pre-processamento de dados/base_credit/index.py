import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


''' Carregamento dos dados '''
base_credit = pd.read_csv('credit_data.csv')

''' Tratamento de dados'''

# Preencher os valores faltantes com a média

base_credit['age'].mean() #Retorna a média de uma label
base_credit.loc[base_credit['age'] < 0, 'age'] = 40.92 #Atualiza os registros com a média


''' Tratamento de dados Faltantes'''
base_credit.isnull() #Retorna a tabela com todos os campos como true caso o valor for faltante e false casi o valor exista
base_credit.isnull().sum() #Mostra a soma do valores true para cada coluna
base_credit.loc[pd.isnull(base_credit['age'])] #Retorna as linhas que tem a idade como dado faltante
base_credit['age'].fillna(base_credit['age'].mean(), inplace= True) #Preenche todos os campos faltantes na coluna age com a média. O parametro inplace salva a
# alteração, com ele false os valores serão aletrado apenas na memmória
base_credit.loc[pd.isnull(base_credit['age'])]#testa de se foi corrigido

''' Divisão entre previsores e classe'''
# As váriaveis previsoras geralmente são chamadas de X e as de classse de Y
# Ao fazer uma previsão as colunas escolhidas escolhidas precisam fazer sentido pra previsão, por exemplo se queremos prever se é confiável ou não fazer 
# emprestimo a determinado cliente, o nome por exemlo dos clientes não seria um dado útil

x_credit = base_credit.iloc[:, 1:4].values #Para criar a váriavel de previsores vamos precisar de todas as linhas, assim o primeiro paramentro passado : quer 
# dizer que queremos todas as linhas.  Já  o segundo parametro indica as colunas nesse caso queremos as colunas de 1 à 3, o que exclui somente a coluna 
# clientid e o default. O '.values' converte os dados para o formato do numpy.

type(x_credit) #Vela que agora  valo está no formato do numpy
type(base_credit) # É possível verificar que antes estava no formato do pandas

y_credit = base_credit.iloc[:, 4].values #pega a coluna default que indica se o cliente pagou ou não o empréstimo
type(y_credit)

'''Escalonamento dos valores'''

#Mostra o mínimo e o máxim para cada dado
print(x_credit[:, 0].min(), x_credit[:, 1].min(), x_credit[:, 2].min())  
print(x_credit[:, 0].max(), x_credit[:, 1].max(), x_credit[:, 1].max())

#Como a escalda dos dados tem uma discrepância muito grande uns em relação aos outros, por exemplo a idade máxima é muito menor que a renda máxima, isso pode
#fazer com que o algoritmo considere uma coluna como muito mais relevante que a outra. Assim é preciso realizar uma padronização para que isso não ocorra.

from sklearn.preprocessing import StandardScaler
scale_credit = StandardScaler()
x_credit = scale_credit.fit_transform(x_credit)

# Os mínimos e máximos padronizados
x_credit[:, 0].min(), x_credit[:, 1].min(), x_credit[:, 2].min()  
x_credit[:, 0].max(), x_credit[:, 1].max(), x_credit[:, 1].max()

''' Separação em dados de treinameto e teste '''

from sklearn.model_selection import train_test_split

x_credit_treinamento, x_credit_teste, y_credit_treinamento, y_credit_teste = train_test_split(x_credit, y_credit, test_size=0.25, random_state=0) #O parametro 
#random state, garante que toda vez que esse algoritmo executar os mesmos dados serão escolhidos para teste e para treino

print(y_credit_treinamento.shape)

''' Salvar os dados pré-processados '''

import pickle

with open('credit.pkl', mode='wb') as f:

    pickle.dump([x_credit_treinamento, y_credit_treinamento, x_credit_teste, y_credit_teste], f)