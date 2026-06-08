import pandas as pd

df = pd.read_csv("default_of_credit_card_clients.csv", sep=";")

print("===== DIMENSÕES =====")
print(df.shape)

print("\n===== COLUNAS =====")
print(df.columns)

print("\n===== INFORMAÇÕES =====")
print(df.info())

print("\n===== VALORES NULOS =====")
print(df.isnull().sum())

print("\n===== DUPLICADOS =====")
print(df.duplicated().sum())

print("\n===== DISTRIBUIÇÃO DA CLASSE =====")
print(df["default payment next month"].value_counts())

print("\n===== DISTRIBUIÇÃO (%) =====")
print(df["default payment next month"].value_counts(normalize=True) * 100)

print("\n===== SEX =====")
print(df["SEX"].value_counts())

print("\n===== EDUCATION =====")
print(df["EDUCATION"].value_counts())

print("\n===== MARRIAGE =====")
print(df["MARRIAGE"].value_counts())

print("\n===== ID =====")
print(df["ID"].head())