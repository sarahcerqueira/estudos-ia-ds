import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


''' Carregamento dos dados '''

base_census = pd.read_csv('census.csv')

print(base_census)

base_census.describe() #Visualização geral dos dados 

''' Tratamento de dados Faltantes'''
base_census.isnull().sum() # Verifica se tem alguma coluna com dados faltantes. Como nessa base não tem, não precisará de tratamento

''' Visualização dos dados'''
np.unique(base_census['income'], return_counts=True)

#sns.countplot(x= base_census['income'])
#plt.show()

#plt.hist(x= base_census['age'])
#plt.show()

#plt.hist(x= base_census['education-num'])
#plt.show()

#plt.hist(x= base_census['hour-per-week'])
#plt.show()

#grafico = px.treemap(base_census, path=['workclass', 'age']) #Esse tipo de gráfico é muito bom para visualizar agrupamentos
#grafico.show()

#grafico = px.treemap(base_census, path=['occupation', 'relationship'])
#grafico.show()

#grafico = px.parallel_categories(base_census, dimensions=['occupation', 'relationship'])
#grafico.show()

#grafico = px.parallel_categories(base_census, dimensions=['workclass', 'occupation', 'income'])
#grafico.show()

#grafico = px.parallel_categories(base_census, dimensions=['education', 'income'])
#grafico.show()

''' Divisão entre previsores e classes'''
base_census.columns #Pega o nome de todas as colunas
x_census = base_census.iloc[:, 0:14].values #Os previsores, todas as linhas e todas as colunas exceto income (de o à 13)
y_census = base_census.iloc[:, 14].values #Classes

''' Tratamento dos dados categoricos '''

# Os algoritmos de machine leaning não entendem de categorias apenas de números, por isso é preciso transfrorma-los em números.

#LABEL ENCODER

from sklearn.preprocessing import LabelEncoder

label_encoder_workclass = LabelEncoder()
label_encoder_education = LabelEncoder()
label_encoder_marital = LabelEncoder()
label_encoder_occupation = LabelEncoder()
label_encoder_relationship = LabelEncoder()
label_encoder_race = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_country = LabelEncoder()

x_census[:, 1] = label_encoder_workclass.fit_transform(x_census[:, 1])
x_census[:, 3] = label_encoder_education.fit_transform(x_census[:, 3])
x_census[:, 5] = label_encoder_marital.fit_transform(x_census[:, 5])
x_census[:, 6] = label_encoder_occupation.fit_transform(x_census[:, 6])
x_census[:, 7] = label_encoder_relationship.fit_transform(x_census[:, 7])
x_census[:, 8] = label_encoder_race.fit_transform(x_census[:, 8])
x_census[:, 9] = label_encoder_sex.fit_transform(x_census[:, 9])
x_census[:, 13] = label_encoder_country.fit_transform(x_census[:, 13])

#ONE HOT ENCODER

len(np.unique(base_census['workclass'])) # Ver a quantidade de lables diferentes

#Mesmo transformando os atributos categoricos em números, o atributo de número 1 pode ser considerado mais importante que o atributo de 
# número 15 por causa do valor do número. Por isso seria importante tranformar cada label em uma categoria separada, e então onde ele estivesse 
# presente colocar 1 ou 0, como de fosse true ou false em binário. 

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 

onehotencoder_census = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [1,3,5,6,7,8,9,13])], 
                                        remainder='passthrough') # O atributo remainder garantes que as outras olunas que não precisam de transformação não sejam removidas.

x_census = onehotencoder_census.fit_transform(x_census).toarray()

#print(x_census.shape)

''' Escalonamento dos atributos'''

from sklearn.preprocessing import StandardScaler 

scaler_census = StandardScaler()
x_census = scaler_census.fit_transform(x_census)

''' Separação em dados de treinameto e teste '''

from sklearn.model_selection import train_test_split

x_census_treinamento, x_census_teste, y_census_treinamento, y_census_teste = train_test_split(x_census, y_census, test_size=0.15, random_state=0)

print(x_census_teste.shape)

''' Salvar os dados pré-processados '''

import pickle

with open('census.pkl', mode='wb') as f:

    pickle.dump([x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste], f)