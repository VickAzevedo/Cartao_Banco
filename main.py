import pandas as   pd
from  sklearn.model_selection import  train_test_split
from sklearn.compose   import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from  sklearn.preprocessing  import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble  import RandomForestClassifier
from  sklearn.linear_model import LogisticRegression
from sklearn.tree  import  DecisionTreeClassifier
from   sklearn.metrics import accuracy_score
from sklearn.metrics    import classification_report
from sklearn.metrics import   roc_auc_score
                                              #vou carregar a base aqui
df = pd.read_csv(
 "default_of_credit_card_clients.csv",
   sep=";"
)
                                        #e aqui tentar remover aquele id que eu acho que tava com problemas 
df = df.drop("ID", axis=1)
                                                          #vou separar as entradas e saidas 
X = df.drop("default payment next month", axis=1)
y = df["default payment next month"]
                                                               
X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.2,
random_state=42,
stratify=y
)
                                 #tentar fazer uma verificação
print("X:", X.shape)
print("y:", y.shape)
print("\nTreino:")
print(X_train.shape)
print(y_train.shape)
print("\nTeste:")
print(X_test.shape)
print(y_test.shape)
                       #colunas pot aqui
categoricas = [
"SEX",
"EDUCATION",
"MARRIAGE"
]
numericas = [
col for col in X.columns
if col not in categoricas
]

preprocessador = ColumnTransformer(
transformers=[
    (
    "cat",
    OneHotEncoder(handle_unknown="ignore"),
    categoricas
    ),
(
    "num",
    StandardScaler(),
    numericas
)
]
)

print("\nPré-processador criado com sucesso.")
print("Colunas categóricas:", categoricas)
print("Quantidade de colunas numéricas:", len(numericas))

#aqui começa modelos, primeiro o random forest

pipeline_rf = Pipeline([
("prep", preprocessador),
("modelo", RandomForestClassifier(
random_state=42
))
])
pipeline_rf.fit(X_train, y_train)
previsoes_rf = pipeline_rf.predict(X_test)
prob_rf = pipeline_rf.predict_proba(X_test)[:, 1]
acc_rf = accuracy_score(y_test, previsoes_rf)
auc_rf = roc_auc_score(y_test, prob_rf)

print("\nRANDOM FOREST")
print(f"Accuracy: {acc_rf:.4f}")
print(f"ROC-AUC: {auc_rf:.4f}")
print("\nClassification Report")
print(classification_report(
    y_test,
    previsoes_rf
))
#o proximo que vai ser o logistic regression
pipeline_lr = Pipeline([
("prep", preprocessador),
("modelo", LogisticRegression(
max_iter=1000,
random_state=42
))
])
pipeline_lr.fit(X_train, y_train)
previsoes_lr = pipeline_lr.predict(X_test)
prob_lr = pipeline_lr.predict_proba(X_test)[:, 1]
acc_lr = accuracy_score(y_test, previsoes_lr)
auc_lr = roc_auc_score(y_test, prob_lr)
print("\nLOGISTIC REGRESSION")
print(f"Accuracy: {acc_lr:.4f}")
print(f"ROC-AUC: {auc_lr:.4f}")
print("\nClassification Report")
print(classification_report(
y_test,
previsoes_lr
))
#e o decison tree, nunca usei muito esse entâo decisão arriscada kkkk
pipeline_dt = Pipeline([
("prep", preprocessador),
("modelo", DecisionTreeClassifier(
random_state=42
))
])
pipeline_dt.fit(X_train, y_train)
previsoes_dt = pipeline_dt.predict(X_test)
prob_dt = pipeline_dt.predict_proba(X_test)[:, 1]
acc_dt = accuracy_score(y_test, previsoes_dt)
auc_dt = roc_auc_score(y_test, prob_dt)
print("\nDECISION TREE")
print(f"Accuracy: {acc_dt:.4f}")
print(f"ROC-AUC: {auc_dt:.4f}")
print("\nClassification Report")
print(classification_report(
y_test,
previsoes_dt
))
#aqui vou começar a comparar os modelos 
print("\nCOMPARAÇÃO DOS MODELOS")
print(f"Random Forest       -> Accuracy: {acc_rf:.4f} | ROC-AUC: {auc_rf:.4f}")
print(f"Logistic Regression -> Accuracy: {acc_lr:.4f} | ROC-AUC: {auc_lr:.4f}")
print(f"Decision Tree       -> Accuracy: {acc_dt:.4f} | ROC-AUC: {auc_dt:.4f}")
#tentar mostrar uma seleção do melhor 
melhor_modelo = pipeline_rf
print("\nMELHOR MODELO")
print("Random Forest")
print(f"Accuracy: {acc_rf:.4f}")
print(f"ROC-AUC: {auc_rf:.4f}")
# ea inferencia 
cliente = X_test.iloc[[0]]
previsao = melhor_modelo.predict(cliente)[0]
probabilidades = melhor_modelo.predict_proba(cliente)[0]
print("\nINFERÊNCIA")
print("Classe prevista:", previsao)
print(f"Probabilidade de NÃO inadimplência: {probabilidades[0]:.4f}")
print(f"Probabilidade de inadimplência: {probabilidades[1]:.4f}")