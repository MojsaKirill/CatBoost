import pandas as pd

dataset = pd.read_csv("csv/bank.csv", sep=',').to_numpy()

X = dataset

for i in range(len(X)):
    if (X[i][13]=='yes'):
        X[i][13]=1
    if (X[i][13]=='no'):
        X[i][13]=0

df = pd.DataFrame(X)

df.to_csv('csv/bank.csv', sep=',', encoding='utf-8')