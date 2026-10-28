import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#Ler os dados
dados = pd.read_csv('data/creditcard.csv')
print(dados.shape)
print(dados.head())

#Verifica nulos
null = dados.isnull().sum()
print(null)

#Verifica duplicatas
duplicatas = dados.duplicated().sum()
print(f'Duplicatas: {duplicatas}')

#Boxplot para visualizar possiveis outliers
sb.boxplot(dados['Amount'])
plt.show()

#Quantidade de fraudes
fraude = (dados['Class'] == 1).sum()
print(f'Quantidade de Fraudes: {fraude}')

#Quantidade de não fraudes
nao_fraude = (dados['Class'] == 0).sum()
print(f'Quantidade de Fraudes: {nao_fraude}')

#Verifica a correlação das colunas em relação a coluna Class
correlacao = dados.corr()[['Class']].sort_values(by='Class', ascending=False)




