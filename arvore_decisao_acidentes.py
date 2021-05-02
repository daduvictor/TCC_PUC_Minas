"""
Árvore de Decisão

Daniel Victor Silva de Oliveira


Dataset: Acidentes 2017-2020 (atributos nominais)

Descrição dos atributos:

1. regiao {CO, N, NE, S, SE}
2. mes {janeiro, fevereiro, março, abril, maio, junho, julho, agosto, setembro, outubro, novembro, dezembro}
3. dia_semana {domingo, segunda-feira, terça-feira, quarta-feira, quinta-feira, sexta-feira, sábado}
4. fase_dia {amanhecer, pleno dia, anoitecer, plena noite}
5. faixa_horaria {01:00-04:00, 04:00-07:00, 07:00-10:00, 10:00-13:00, 13:00-16:00, 16:00-19:00, 19:00-22:00, 22:00-01:00}
6. condicao_meteorologica {céu claro, chuva, garoa/chuvisco, granizo, neve, nevoeiro/neblina, nublado, sol, vento}
7. tipo_acidente {atropelamento de animal, atropelamento de pedestre, capotamento, colisão com objeto, colisão frontal,
                  colisão lateral, colisão transversal, colisão traseira, danos eventuais, derramamento de carga, engavetamento,
                  incêndio, queda de ocupante de veículo, saída de leito carroçável, tombamento}
8. causa_acidente {agressão externa, animais na pista, carga excessiva e/ou mal acondicionada, condutor dormindo, defeito na via,
                   defeito no veículo, deficiência ou não acionamento do sistema de iluminação/sinalização do veículo,
                   desobediência às normas de trânsito, falta de atenção, fenômenos da natureza,
                   ingestão de álcool e/ou substâncias psicoativas, mal súbito, não guardar distância de segurança,
                   objeto estático sobre o leito carroçável, pista escorregadia, restrição de visibilidade,
                   ultrapassagem indevida, velocidade incompatível}
9. tipo_pista {simples, dupla, múltipla}
10. tracado_via {curva, desvio temporário, interseção de vias, ponte, reta, retorno regulamentado, rotatória, túnel, viaduto}
11. uso_solo {rural, urbano}
12. veiculos {=1, =2, =3, >=4}
13. moto_similares {não, sim}
14. caminhao_similares {não, sim}
15. onibus_similares {não, sim}
16. pessoas {=1, =2, =3, =4, =5, =6, >=7}
17. classificacao_acidente {com vítimas, sem vítimas}
"""

# Inicialização
# Importação dos pacotes necessários.

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Preparação da base de treinamento (entradas)
# Carregamento do dataset, conversão em um dicionário e transformação para um vetor binário.

a1_nominal = pd.read_excel("acidentes17-19.xlsx") 
print("\nDimensões do dataset: {0}".format(a1_nominal.shape))
print("\nCampos do dataset: {0}".format(a1_nominal.keys()))
print("\n", a1_nominal.describe())

X_dict = a1_nominal.iloc[:,1:17].T.to_dict().values()
vect = DictVectorizer(sparse=False)
X_treino = vect.fit_transform(X_dict)
print("\nDimensões da base de treinamento (entradas): {0}".format(X_treino.shape))

# Preparação da base de treinamento (saída)
# Conversão das classes em labels.

le = LabelEncoder()
y_treino = le.fit_transform(a1_nominal.iloc[:,17])
print("\nLabels da base de treinamento (saída):", y_treino)
print("\nDimensão da base de treinamento (saída):", y_treino.shape[0])

# Preparação da base de testes (entradas e saída)

a2_nominal = pd.read_excel("acidentes20.xlsx") 
print("\nDimensões do dataset: {0}".format(a2_nominal.shape))
print("\nCampos do dataset: {0}".format(a2_nominal.keys()))
print("\n", a2_nominal.describe())

X2_dict = a2_nominal.iloc[:,1:17].T.to_dict().values()
vect = DictVectorizer(sparse=False)
X_teste = vect.fit_transform(X2_dict)
print("\nDimensões da base de testes (entradas): {0}".format(X_teste.shape))

y_teste = le.fit_transform(a2_nominal.iloc[:,17])
print("\nLabels da base de testes (saída):", y_teste)
print("\nDimensão da base de testes (saída):", y_teste.shape[0])

# Treinamento e aplicação do modelo
# Definição dos melhores hiperparâmetros e predição da saída da base de testes.

gridAcidentes = DecisionTreeClassifier(random_state=0)
parametros = {'criterion':('gini', 'entropy'), 'splitter':('best', 'random'), 'min_samples_leaf':[15, 25, 35]}

treeAcidentes = GridSearchCV(gridAcidentes, param_grid=parametros)
treeAcidentes = treeAcidentes.fit(X_treino, y_treino)
y_pred_tree = treeAcidentes.predict(X_teste)
print("\n", treeAcidentes.best_params_)

# Avaliação do modelo
# Medidas e matriz de confusão.

print("\nAcurácia na base de treinamento:", treeAcidentes.score(X_treino, y_treino))
print("Acurácia de predição:", treeAcidentes.score(X_teste, y_teste))
print("Número de erros de classificação: {0} de {1}".format((y_teste != y_pred_tree).sum(), y_teste.shape[0]))
print("\n", classification_report(y_teste, y_pred_tree, digits=4, target_names=["c/ vítimas", "s/ vítimas"]))

cnf_matrix = confusion_matrix(y_pred_tree, y_teste)
cnf_table = pd.DataFrame(data=cnf_matrix, columns=["c/ vítimas", "s/ vítimas"], index=["c/ vítimas (pred)", "s/ vítimas (pred)"])
print(cnf_table)

# Gravação do resultado
# Transformação inversa dos labels preditos nas respectivas classes, junção com a base de testes e salvamento do dataset.

treeclass_pred = le.inverse_transform(y_pred_tree)
treeclass_pred = pd.DataFrame(treeclass_pred, columns=["classificacao_predita"])
treeclass_acidentes = pd.concat([a2_nominal, treeclass_pred], axis=1)
treeclass_acidentes.to_csv("treeclass_acidentes20.csv", encoding='iso-8859-1', index=False, sep=';', date_format='%d/%m/%Y')