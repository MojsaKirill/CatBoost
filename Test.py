from catboost import CatBoostClassifier
import pandas as pd

df_test = pd.read_csv("csv/bank_test.csv", sep=',').to_numpy()

model = CatBoostClassifier().load_model("model.json", format="json")

X, Y = df_test[:, 0:13], df_test[:, 13]
answer = model.predict(X)
loss = 0
for x in range(len(answer)):
    if (Y[x] == 1 and answer[x] == 0):
        loss += 1
    if (Y[x] == 0 and answer[x] == 1):
        loss += 1
print("loss = ", loss)